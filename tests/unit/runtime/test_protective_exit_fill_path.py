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
