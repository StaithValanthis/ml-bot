"""Tests for entry fill -> protective exit determinism (orchestrator wiring)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading.execution.order_intent import IntentPurpose, OrderIntent
from trading.runtime.orchestrator import RuntimeOrchestrator
from trading.settings import load_settings
from trading.util.types import OrderSide, OrderStatus, OrderType, RuntimeMode, TimeInForce


def _orch_with_mocks() -> RuntimeOrchestrator:
    mock_rest = MagicMock()
    mock_rest.place_order = AsyncMock()
    mock_ws_public = MagicMock()
    mock_ws_public.subscribe = MagicMock()
    mock_ws_public.run_forever = MagicMock()
    mock_ws_public.close = MagicMock()
    mock_ws_private = MagicMock()
    mock_ws_private.subscribe = MagicMock()
    mock_ws_private.run_forever = MagicMock()
    mock_ws_private.close = MagicMock()
    with (
        patch("trading.runtime.orchestrator.BybitRestClient", return_value=mock_rest),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", return_value=mock_ws_public),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", return_value=mock_ws_private),
    ):
        return RuntimeOrchestrator(load_settings())


@pytest.mark.asyncio
async def test_filled_entry_triggers_protective_submission_and_metrics() -> None:
    """Transition to Filled on a strategy entry order submits protective exit and updates metrics."""
    orch = _orch_with_mocks()
    orch._settings.runtime.mode = RuntimeMode.DEMO
    orch._settings.runtime.dry_run = False
    orch._settings.exchange.bybit_api_key = "k"
    orch._settings.exchange.bybit_api_secret = "s"

    now = datetime.now(UTC)
    intent = OrderIntent(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=Decimal("0.01"),
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.POST_ONLY,
        reduce_only=False,
        price=Decimal("50000"),
        order_link_id="entry_link_1",
        purpose=IntentPurpose.ENTRY,
        created_at=now,
        metadata={"signal_action": "enter_long"},
    )
    await orch._order_manager.register_intent(intent)
    ack = MagicMock()
    ack.order_link_id = "pe_link_1"
    ack.order_id = "ex_oid_1"
    orch._rest.place_order = AsyncMock(return_value=ack)

    from trading.marketdata.normalizers import NormalizedOrderUpdate

    upd = NormalizedOrderUpdate(
        order_id="oid1",
        order_link_id="entry_link_1",
        symbol="BTCUSDT",
        status=OrderStatus.FILLED,
        qty=Decimal("0.01"),
        avg_price=Decimal("50000"),
        ts_event_utc=now,
        raw={},
    )
    await orch._on_private_events([upd])

    assert orch._metrics.snapshot().counters.get("protective_exit_order_submitted_count", 0) >= 1
    assert orch._metrics.snapshot().counters.get("protective_exit_order_ack_received_count", 0) >= 1
    assert "entry_link_1" in orch._protective_exit_done_link_ids
    summary = await orch._build_session_summary()
    ped = summary.get("protective_exit_diagnostics") or {}
    by_entry = ped.get("protective_exit_fill_outcomes_by_entry_link_id") or {}
    assert by_entry.get("entry_link_1", {}).get("outcome") == "submitted"
    assert by_entry.get("entry_link_1", {}).get("acked") is True
    orch._rest.place_order.assert_awaited()


@pytest.mark.asyncio
async def test_duplicate_filled_transition_does_not_double_submit_protective() -> None:
    """Second duplicate Filled WS update does not place a second protective exit."""
    orch = _orch_with_mocks()
    orch._settings.runtime.mode = RuntimeMode.DEMO
    orch._settings.runtime.dry_run = False
    orch._settings.exchange.bybit_api_key = "k"
    orch._settings.exchange.bybit_api_secret = "s"

    now = datetime.now(UTC)
    intent = OrderIntent(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=Decimal("0.01"),
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.POST_ONLY,
        reduce_only=False,
        price=Decimal("50000"),
        order_link_id="entry_link_dup",
        purpose=IntentPurpose.ENTRY,
        created_at=now,
        metadata={"signal_action": "enter_long"},
    )
    await orch._order_manager.register_intent(intent)
    ack = MagicMock()
    ack.order_link_id = "pe_x"
    ack.order_id = "ex_x"
    orch._rest.place_order = AsyncMock(return_value=ack)

    from trading.marketdata.normalizers import NormalizedOrderUpdate

    base = dict(
        order_id="oid1",
        order_link_id="entry_link_dup",
        symbol="BTCUSDT",
        status=OrderStatus.FILLED,
        qty=Decimal("0.01"),
        avg_price=Decimal("50000"),
        ts_event_utc=now,
        raw={},
    )
    await orch._on_private_events([NormalizedOrderUpdate(**base)])
    await orch._on_private_events([NormalizedOrderUpdate(**base)])

    assert orch._rest.place_order.await_count == 1


@pytest.mark.asyncio
async def test_protective_exit_session_summary_diagnostics() -> None:
    """Session summary exposes protective_exit_diagnostics."""
    orch = _orch_with_mocks()
    orch._session_start_time = datetime(2025, 3, 19, 9, 0, 0, tzinfo=UTC)
    orch._metrics.inc("entry_fill_received_count")
    orch._metrics.inc("protective_exit_order_submitted_count")
    summary = await orch._build_session_summary()
    ped = summary.get("protective_exit_diagnostics") or {}
    assert ped.get("fills_with_exit_submission") == 1
    assert ped.get("fills_without_exit_submission") == 0
    assert "protective_exit_skip_reasons_by_type" in ped


@pytest.mark.asyncio
async def test_filled_entry_with_unknown_signal_action_records_explicit_skip() -> None:
    """A filled entry lacking a valid signal action is explicitly counted as skipped."""
    orch = _orch_with_mocks()
    orch._settings.runtime.mode = RuntimeMode.DEMO
    orch._settings.runtime.dry_run = False
    orch._settings.exchange.bybit_api_key = "k"
    orch._settings.exchange.bybit_api_secret = "s"

    now = datetime.now(UTC)
    intent = OrderIntent(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=Decimal("0.01"),
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.POST_ONLY,
        reduce_only=False,
        price=Decimal("50000"),
        order_link_id="entry_link_skip",
        purpose=IntentPurpose.ENTRY,
        created_at=now,
        metadata={"signal_action": "bad_action"},
    )
    await orch._order_manager.register_intent(intent)
    orch._rest.place_order = AsyncMock()

    from trading.marketdata.normalizers import NormalizedOrderUpdate

    upd = NormalizedOrderUpdate(
        order_id="oid_skip",
        order_link_id="entry_link_skip",
        symbol="BTCUSDT",
        status=OrderStatus.FILLED,
        qty=Decimal("0.01"),
        avg_price=Decimal("50000"),
        ts_event_utc=now,
        raw={},
    )
    await orch._on_private_events([upd])

    counters = orch._metrics.snapshot().counters
    assert counters.get("entry_fill_received_count", 0) == 1
    assert counters.get("protective_exit_order_submitted_count", 0) == 0
    assert counters.get("protective_exit_placement_failed_count", 0) == 0
    assert counters.get("protective_exit_placement_skipped_count", 0) == 1
    assert orch._protective_exit_skip_reasons.get("unknown_signal_action", 0) == 1
    orch._rest.place_order.assert_not_awaited()


@pytest.mark.asyncio
async def test_two_fills_one_submission_is_explicitly_attributed() -> None:
    """Two entry fills with one valid and one invalid path have deterministic diagnostics."""
    orch = _orch_with_mocks()
    orch._settings.runtime.mode = RuntimeMode.DEMO
    orch._settings.runtime.dry_run = False
    orch._settings.exchange.bybit_api_key = "k"
    orch._settings.exchange.bybit_api_secret = "s"

    now = datetime.now(UTC)
    valid = OrderIntent(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=Decimal("0.01"),
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.POST_ONLY,
        reduce_only=False,
        price=Decimal("50000"),
        order_link_id="entry_link_valid",
        purpose=IntentPurpose.ENTRY,
        created_at=now,
        metadata={"signal_action": "enter_long"},
    )
    invalid = OrderIntent(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=Decimal("0.01"),
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.POST_ONLY,
        reduce_only=False,
        price=Decimal("50010"),
        order_link_id="entry_link_invalid",
        purpose=IntentPurpose.ENTRY,
        created_at=now,
        metadata={"signal_action": "bad_action"},
    )
    await orch._order_manager.register_intent(valid)
    await orch._order_manager.register_intent(invalid)

    ack = MagicMock()
    ack.order_link_id = "pe_link_two_fill"
    ack.order_id = "ex_oid_two_fill"
    orch._rest.place_order = AsyncMock(return_value=ack)

    from trading.marketdata.normalizers import NormalizedOrderUpdate

    await orch._on_private_events(
        [
            NormalizedOrderUpdate(
                order_id="oid_valid",
                order_link_id="entry_link_valid",
                symbol="BTCUSDT",
                status=OrderStatus.FILLED,
                qty=Decimal("0.01"),
                avg_price=Decimal("50000"),
                ts_event_utc=now,
                raw={},
            ),
            NormalizedOrderUpdate(
                order_id="oid_invalid",
                order_link_id="entry_link_invalid",
                symbol="BTCUSDT",
                status=OrderStatus.FILLED,
                qty=Decimal("0.01"),
                avg_price=Decimal("50010"),
                ts_event_utc=now,
                raw={},
            ),
        ]
    )

    summary = await orch._build_session_summary()
    ped = summary.get("protective_exit_diagnostics") or {}
    assert ped.get("fills_with_exit_submission") == 1
    assert ped.get("fills_without_exit_submission") == 1
    assert (ped.get("protective_exit_skip_reasons_by_type") or {}).get("unknown_signal_action") == 1


@pytest.mark.asyncio
async def test_reduce_only_fill_does_not_increment_entry_fill_received() -> None:
    """Protective exit (reduce-only) fill must not count as entry_fill_received (soak attribution)."""
    orch = _orch_with_mocks()
    orch._settings.runtime.mode = RuntimeMode.DEMO

    now = datetime.now(UTC)
    intent = OrderIntent(
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        qty=Decimal("0.01"),
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.POST_ONLY,
        reduce_only=True,
        price=Decimal("51000"),
        order_link_id="pe_reduce_only_link",
        purpose=IntentPurpose.EXIT,
        created_at=now,
        metadata={},
    )
    await orch._order_manager.register_intent(intent)

    from trading.marketdata.normalizers import NormalizedOrderUpdate

    await orch._on_private_events(
        [
            NormalizedOrderUpdate(
                order_id="oid_ro",
                order_link_id="pe_reduce_only_link",
                symbol="BTCUSDT",
                status=OrderStatus.FILLED,
                qty=Decimal("0.01"),
                avg_price=Decimal("51000"),
                ts_event_utc=now,
                raw={},
            ),
        ]
    )

    assert orch._metrics.snapshot().counters.get("entry_fill_received_count", 0) == 0
    assert orch._strategy_order_outcomes.filled == 1


@pytest.mark.asyncio
async def test_entry_fill_then_protective_exit_fill_diagnostics_aligned() -> None:
    """Entry fill + later reduce-only fill: one entry_fill_received, one PE submit; no false 'missing exit'."""
    orch = _orch_with_mocks()
    orch._settings.runtime.mode = RuntimeMode.DEMO
    orch._settings.runtime.dry_run = False
    orch._settings.exchange.bybit_api_key = "k"
    orch._settings.exchange.bybit_api_secret = "s"

    now = datetime.now(UTC)
    entry = OrderIntent(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=Decimal("0.01"),
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.POST_ONLY,
        reduce_only=False,
        price=Decimal("50000"),
        order_link_id="entry_then_pe_exit",
        purpose=IntentPurpose.ENTRY,
        created_at=now,
        metadata={"signal_action": "enter_long"},
    )
    await orch._order_manager.register_intent(entry)
    ack = MagicMock()
    ack.order_link_id = "ack_pe"
    ack.order_id = "ex_pe_1"
    orch._rest.place_order = AsyncMock(return_value=ack)

    from trading.marketdata.normalizers import NormalizedOrderUpdate

    await orch._on_private_events(
        [
            NormalizedOrderUpdate(
                order_id="oid_ent",
                order_link_id="entry_then_pe_exit",
                symbol="BTCUSDT",
                status=OrderStatus.FILLED,
                qty=Decimal("0.01"),
                avg_price=Decimal("50000"),
                ts_event_utc=now,
                raw={},
            ),
        ]
    )

    open_orders = await orch._order_manager.get_open_orders(None)
    pe_orders = [o for o in open_orders if o.reduce_only and o.symbol == "BTCUSDT"]
    assert len(pe_orders) == 1
    pe_link = pe_orders[0].order_link_id

    await orch._on_private_events(
        [
            NormalizedOrderUpdate(
                order_id="oid_pe_fill",
                order_link_id=pe_link,
                symbol="BTCUSDT",
                status=OrderStatus.FILLED,
                qty=Decimal("0.01"),
                avg_price=Decimal("51000"),
                ts_event_utc=now,
                raw={},
            ),
        ]
    )

    c = orch._metrics.snapshot().counters
    assert c.get("entry_fill_received_count", 0) == 1
    assert c.get("protective_exit_order_submitted_count", 0) == 1

    summary = await orch._build_session_summary()
    ped = summary.get("protective_exit_diagnostics") or {}
    assert ped.get("fills_with_exit_submission") == 1
    assert ped.get("fills_without_exit_submission") == 0
