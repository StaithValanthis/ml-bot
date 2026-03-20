"""Integration tests for demo execution drill."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading.exchange.schemas import OrderAck, ServerTimeResult
from trading.journal.ledger import LedgerEvent, LedgerSink
from trading.marketdata.normalizers import NormalizedTicker
from trading.runtime.orchestrator import RuntimeOrchestrator
from trading.settings import load_settings
from trading.util.types import RuntimeMode


class _CaptureLedger(LedgerSink):
    def __init__(self) -> None:
        self.events: list[LedgerEvent] = []

    async def write_event(self, event: LedgerEvent) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_drill_refused_outside_demo() -> None:
    """Drill task is not started when mode is not DEMO."""
    with patch.dict(os.environ, {"TRADING_MODE": "paper", "TRADING_DEMO_DRILL_ENABLED": "true"}):
        settings = load_settings()
    settings.runtime.demo_drill.enabled = True
    settings.runtime.mode = RuntimeMode.PAPER

    mock_rest = MagicMock()
    mock_rest.get_server_time = AsyncMock(return_value=ServerTimeResult(time_second="1700000000", time_nano="0"))
    mock_rest.get_wallet = AsyncMock(return_value=[])
    mock_rest.get_positions = AsyncMock(return_value=[])
    mock_rest.get_open_orders = AsyncMock(return_value=[])
    mock_rest.close = AsyncMock()

    async def mock_ws_run() -> None:
        await asyncio.sleep(0.5)

    mock_ws_public = MagicMock()
    mock_ws_public.subscribe = AsyncMock()
    mock_ws_public.run_forever = AsyncMock(side_effect=mock_ws_run)
    mock_ws_public.close = AsyncMock()

    mock_ws_private = MagicMock()
    mock_ws_private.subscribe = AsyncMock()
    mock_ws_private.run_forever = AsyncMock(return_value=None)
    mock_ws_private.close = AsyncMock()

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", return_value=mock_rest),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", return_value=mock_ws_public),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", return_value=mock_ws_private),
    ):
        orch = RuntimeOrchestrator(settings)
        task = asyncio.create_task(orch.run())
        await asyncio.sleep(0.2)
        await orch.stop()
        await task

    task_names = [t.get_name() for t in orch._tasks]
    assert "runtime-demo-drill" not in task_names


@pytest.mark.asyncio
async def test_drill_refused_when_dry_run() -> None:
    """Drill task is not started when dry_run is True."""
    with patch.dict(os.environ, {"TRADING_MODE": "demo", "TRADING_DRY_RUN": "true", "TRADING_DEMO_DRILL_ENABLED": "true"}):
        settings = load_settings()
    settings.runtime.demo_drill.enabled = True
    settings.runtime.mode = RuntimeMode.DEMO
    settings.runtime.dry_run = True

    mock_rest = MagicMock()
    mock_rest.get_server_time = AsyncMock(return_value=ServerTimeResult(time_second="1700000000", time_nano="0"))
    mock_rest.get_wallet = AsyncMock(return_value=[])
    mock_rest.get_positions = AsyncMock(return_value=[])
    mock_rest.get_open_orders = AsyncMock(return_value=[])
    mock_rest.close = AsyncMock()

    async def mock_ws_run() -> None:
        await asyncio.sleep(0.5)

    mock_ws_public = MagicMock()
    mock_ws_public.subscribe = AsyncMock()
    mock_ws_public.run_forever = AsyncMock(side_effect=mock_ws_run)
    mock_ws_public.close = AsyncMock()

    mock_ws_private = MagicMock()
    mock_ws_private.subscribe = AsyncMock()
    mock_ws_private.run_forever = AsyncMock(return_value=None)
    mock_ws_private.close = AsyncMock()

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", return_value=mock_rest),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", return_value=mock_ws_public),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", return_value=mock_ws_private),
    ):
        orch = RuntimeOrchestrator(settings)
        task = asyncio.create_task(orch.run())
        await asyncio.sleep(0.2)
        await orch.stop()
        await task

    task_names = [t.get_name() for t in orch._tasks]
    assert "runtime-demo-drill" not in task_names


@pytest.mark.asyncio
async def test_drill_refused_when_symbol_not_configured() -> None:
    """Drill is aborted when symbol is not in configured trading symbols."""
    with patch.dict(os.environ, {"TRADING_MODE": "demo", "TRADING_DRY_RUN": "false", "TRADING_DEMO_DRILL_ENABLED": "true"}):
        settings = load_settings()
    settings.runtime.demo_drill.enabled = True
    settings.runtime.demo_drill.symbol = "XRPUSDT"
    settings.runtime.mode = RuntimeMode.DEMO
    settings.runtime.dry_run = False

    capture = _CaptureLedger()
    mock_staleness = MagicMock()
    mock_staleness.stale_channels = AsyncMock(return_value=[])
    mock_staleness.set_expected_channels = MagicMock()
    mock_staleness.mark_seen = AsyncMock()

    mock_rest = MagicMock()
    mock_rest.get_server_time = AsyncMock(return_value=ServerTimeResult(time_second="1700000000", time_nano="0"))
    mock_rest.get_wallet = AsyncMock(return_value=[])
    mock_rest.get_positions = AsyncMock(return_value=[])
    mock_rest.get_open_orders = AsyncMock(return_value=[])
    mock_rest.close = AsyncMock()

    async def mock_ws_run() -> None:
        await asyncio.sleep(0.6)

    mock_ws_public = MagicMock()
    mock_ws_public.subscribe = AsyncMock()
    mock_ws_public.run_forever = AsyncMock(side_effect=mock_ws_run)
    mock_ws_public.close = AsyncMock()

    mock_ws_private = MagicMock()
    mock_ws_private.subscribe = AsyncMock()
    mock_ws_private.run_forever = AsyncMock(return_value=None)
    mock_ws_private.close = AsyncMock()

    async def inject_ticker() -> None:
        await asyncio.sleep(2.0)
        ticker = NormalizedTicker(
            symbol="XRPUSDT",
            bid_price=Decimal("1"),
            ask_price=Decimal("1.01"),
            ts_exchange_ms=1700000000000,
            ts_event_utc=datetime.now(UTC),
        )
        await orch._market_state.apply_event(ticker)

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", return_value=mock_rest),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", return_value=mock_ws_public),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", return_value=mock_ws_private),
        patch.object(RuntimeOrchestrator, "_can_place_exchange_orders", return_value=True),
    ):
        orch = RuntimeOrchestrator(settings)
        orch._staleness = mock_staleness
        orch._ledger._sinks.insert(0, capture)

        asyncio.create_task(inject_ticker())
        task = asyncio.create_task(orch.run())
        await asyncio.sleep(20.0)
        await orch.stop()
        await task

    event_types = [e.event_type for e in capture.events]
    assert "drill_aborted" in event_types
    aborted_ev = next(e for e in capture.events if e.event_type == "drill_aborted")
    assert "symbol_not_configured" in aborted_ev.payload.get("reason", "")


@pytest.mark.asyncio
async def test_mocked_successful_demo_drill_ack_flow() -> None:
    """When drill submits and exchange acks, drill_ack_received is recorded."""
    with patch.dict(os.environ, {"TRADING_MODE": "demo", "TRADING_DRY_RUN": "false", "TRADING_DEMO_DRILL_ENABLED": "true"}):
        settings = load_settings()
    settings.runtime.demo_drill.enabled = True
    settings.runtime.mode = RuntimeMode.DEMO
    settings.runtime.dry_run = False

    capture = _CaptureLedger()

    mock_rest = MagicMock()
    mock_rest.get_server_time = AsyncMock(return_value=ServerTimeResult(time_second="1700000000", time_nano="0"))
    mock_rest.get_wallet = AsyncMock(return_value=[])
    mock_rest.get_positions = AsyncMock(return_value=[])
    mock_rest.get_open_orders = AsyncMock(return_value=[])

    def place_order_side_effect(request: object) -> object:
        order_link_id = getattr(request, "order_link_id", "drill-btcu-240101120000")
        return OrderAck(order_id="ex-order-123", order_link_id=order_link_id)

    mock_rest.place_order = AsyncMock(side_effect=place_order_side_effect)
    mock_rest.close = AsyncMock()

    async def mock_ws_run() -> None:
        await asyncio.sleep(0.6)

    mock_ws_public = MagicMock()
    mock_ws_public.subscribe = AsyncMock()
    mock_ws_public.run_forever = AsyncMock(side_effect=mock_ws_run)
    mock_ws_public.close = AsyncMock()

    mock_ws_private = MagicMock()
    mock_ws_private.subscribe = AsyncMock()
    mock_ws_private.run_forever = AsyncMock(return_value=None)
    mock_ws_private.close = AsyncMock()

    async def inject_ticker() -> None:
        await asyncio.sleep(2.0)
        ticker = NormalizedTicker(
            symbol="BTCUSDT",
            bid_price=Decimal("5000"),
            ask_price=Decimal("5001"),
            ts_exchange_ms=1700000000000,
            ts_event_utc=datetime.now(UTC),
        )
        await orch._market_state.apply_event(ticker)

    mock_staleness = MagicMock()
    mock_staleness.stale_channels = AsyncMock(return_value=[])
    mock_staleness.set_expected_channels = MagicMock()
    mock_staleness.mark_seen = AsyncMock()

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", return_value=mock_rest),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", return_value=mock_ws_public),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", return_value=mock_ws_private),
        patch.object(RuntimeOrchestrator, "_can_place_exchange_orders", return_value=True),
    ):
        orch = RuntimeOrchestrator(settings)
        orch._staleness = mock_staleness
        orch._ledger._sinks.insert(0, capture)
        orch._drill_outcome.enabled = True
        orch._settings.runtime.demo_drill.enabled = True
        orch._settings.runtime.mode = RuntimeMode.DEMO
        orch._settings.runtime.dry_run = False

        asyncio.create_task(inject_ticker())
        task = asyncio.create_task(orch.run())
        await asyncio.sleep(20.0)
        await orch.stop()
        await task

    event_types = [e.event_type for e in capture.events]
    assert "drill_requested" in event_types
    assert "drill_submitted" in event_types
    assert "drill_ack_received" in event_types
    assert orch._drill_outcome.ack_received is True


@pytest.mark.asyncio
async def test_drill_waits_for_market_data_before_aborting() -> None:
    """Drill waits for market data, logs waiting/timeout, aborts with structured details."""
    with patch.dict(os.environ, {"TRADING_MODE": "demo", "TRADING_DRY_RUN": "false", "TRADING_DEMO_DRILL_ENABLED": "true"}):
        settings = load_settings()
    settings.runtime.demo_drill.enabled = True
    settings.runtime.mode = RuntimeMode.DEMO
    settings.runtime.dry_run = False

    capture = _CaptureLedger()
    mock_staleness = MagicMock()
    mock_staleness.stale_channels = AsyncMock(return_value=[])
    mock_staleness.set_expected_channels = MagicMock()
    mock_staleness.mark_seen = AsyncMock()

    mock_rest = MagicMock()
    mock_rest.get_server_time = AsyncMock(return_value=ServerTimeResult(time_second="1700000000", time_nano="0"))
    mock_rest.get_wallet = AsyncMock(return_value=[])
    mock_rest.get_positions = AsyncMock(return_value=[])
    mock_rest.get_open_orders = AsyncMock(return_value=[])
    mock_rest.get_ticker = AsyncMock(return_value=None)
    mock_rest.close = AsyncMock()

    async def mock_ws_run() -> None:
        await asyncio.sleep(0.6)

    mock_ws_public = MagicMock()
    mock_ws_public.subscribe = AsyncMock()
    mock_ws_public.run_forever = AsyncMock(side_effect=mock_ws_run)
    mock_ws_public.close = AsyncMock()

    mock_ws_private = MagicMock()
    mock_ws_private.subscribe = AsyncMock()
    mock_ws_private.run_forever = AsyncMock(return_value=None)
    mock_ws_private.close = AsyncMock()

    abort_details = {
        "waited_seconds": 25.0,
        "symbol": "BTCUSDT",
        "ws_public_connected": False,
        "ticker_seen": False,
        "trade_seen": False,
        "rest_fallback_attempted": True,
        "reason": "timeout",
    }

    async def mock_wait(*args: object, **kwargs: object) -> tuple:
        return (None, abort_details)

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", return_value=mock_rest),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", return_value=mock_ws_public),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", return_value=mock_ws_private),
        patch.object(RuntimeOrchestrator, "_can_place_exchange_orders", return_value=True),
        patch.object(RuntimeOrchestrator, "_wait_for_drill_reference_price", side_effect=mock_wait),
    ):
        orch = RuntimeOrchestrator(settings)
        orch._staleness = mock_staleness
        orch._ledger._sinks.insert(0, capture)

        task = asyncio.create_task(orch.run())
        await asyncio.sleep(25.0)
        await orch.stop()
        await task

    assert "drill_aborted" in [e.event_type for e in capture.events]
    aborted_ev = next(e for e in capture.events if e.event_type == "drill_aborted")
    details = aborted_ev.payload.get("details", {})
    assert "waited_seconds" in details
    assert "symbol" in details
    assert "ws_public_connected" in details
    assert "ticker_seen" in details
    assert "trade_seen" in details


@pytest.mark.asyncio
async def test_drill_succeeds_when_ticker_becomes_available_during_wait() -> None:
    """Drill succeeds when ticker is injected during the wait period."""
    with patch.dict(os.environ, {"TRADING_MODE": "demo", "TRADING_DRY_RUN": "false", "TRADING_DEMO_DRILL_ENABLED": "true"}):
        settings = load_settings()
    settings.runtime.demo_drill.enabled = True
    settings.runtime.mode = RuntimeMode.DEMO
    settings.runtime.dry_run = False

    capture = _CaptureLedger()
    mock_staleness = MagicMock()
    mock_staleness.stale_channels = AsyncMock(return_value=[])
    mock_staleness.set_expected_channels = MagicMock()
    mock_staleness.mark_seen = AsyncMock()

    mock_rest = MagicMock()
    mock_rest.get_server_time = AsyncMock(return_value=ServerTimeResult(time_second="1700000000", time_nano="0"))
    mock_rest.get_wallet = AsyncMock(return_value=[])
    mock_rest.get_positions = AsyncMock(return_value=[])
    mock_rest.get_open_orders = AsyncMock(return_value=[])

    def place_order_side_effect(request: object) -> object:
        order_link_id = getattr(request, "order_link_id", "drill-btcu-240101120000")
        return OrderAck(order_id="ex-order-123", order_link_id=order_link_id)

    mock_rest.place_order = AsyncMock(side_effect=place_order_side_effect)
    mock_rest.close = AsyncMock()

    async def mock_ws_run() -> None:
        await asyncio.sleep(0.6)

    mock_ws_public = MagicMock()
    mock_ws_public.subscribe = AsyncMock()
    mock_ws_public.run_forever = AsyncMock(side_effect=mock_ws_run)
    mock_ws_public.close = AsyncMock()

    mock_ws_private = MagicMock()
    mock_ws_private.subscribe = AsyncMock()
    mock_ws_private.run_forever = AsyncMock(return_value=None)
    mock_ws_private.close = AsyncMock()

    async def inject_ticker_late() -> None:
        await asyncio.sleep(17.0)
        ticker = NormalizedTicker(
            symbol="BTCUSDT",
            bid_price=Decimal("5000"),
            ask_price=Decimal("5001"),
            ts_exchange_ms=1700000000000,
            ts_event_utc=datetime.now(UTC),
        )
        await orch._market_state.apply_event(ticker)

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", return_value=mock_rest),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", return_value=mock_ws_public),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", return_value=mock_ws_private),
        patch.object(RuntimeOrchestrator, "_can_place_exchange_orders", return_value=True),
    ):
        orch = RuntimeOrchestrator(settings)
        orch._staleness = mock_staleness
        orch._ledger._sinks.insert(0, capture)

        asyncio.create_task(inject_ticker_late())
        task = asyncio.create_task(orch.run())
        await asyncio.sleep(45.0)
        await orch.stop()
        await task

    event_types = [e.event_type for e in capture.events]
    assert "drill_waiting_for_market_data" in event_types
    assert "drill_market_data_ready" in event_types
    assert "drill_requested" in event_types
    assert "drill_ack_received" in event_types


def test_drill_abort_summary_includes_improved_details() -> None:
    """Session summary includes drill_abort_details when drill aborts with structured reason."""
    settings = load_settings()
    settings.runtime.demo_drill.enabled = True
    settings.runtime.mode = RuntimeMode.DEMO

    orch = RuntimeOrchestrator(settings)
    orch._drill_outcome.enabled = True
    orch._drill_outcome.attempted = True
    orch._drill_outcome.aborted = True
    orch._drill_outcome.refused_reason = "drill_refused_market_data_timeout"
    orch._drill_outcome.abort_details = {
        "waited_seconds": 25.0,
        "symbol": "BTCUSDT",
        "ws_public_connected": False,
        "ticker_seen": False,
        "trade_seen": False,
        "rest_fallback_attempted": True,
        "reason": "timeout",
    }

    summary = orch._build_session_summary()
    assert summary.get("drill_abort_details") is not None
    details = summary["drill_abort_details"]
    assert details.get("waited_seconds") == 25.0
    assert details.get("symbol") == "BTCUSDT"
    assert details.get("ws_public_connected") is False
    assert details.get("ticker_seen") is False
    assert details.get("trade_seen") is False

    md = orch._build_markdown_summary(summary)
    assert "Abort details" in md
    assert "waited_seconds" in md
    assert "25.0" in md


def test_drill_refusal_details_in_summary() -> None:
    """Session summary includes structured refusal details (symbol, qty, min_qty, notional, cap)."""
    settings = load_settings()
    settings.runtime.demo_drill.enabled = True
    settings.runtime.mode = RuntimeMode.DEMO

    orch = RuntimeOrchestrator(settings)
    orch._drill_outcome.enabled = True
    orch._drill_outcome.attempted = True
    orch._drill_outcome.aborted = True
    orch._drill_outcome.refused_reason = "drill_refused_notional_exceeds_cap_10"
    orch._drill_outcome.abort_details = {
        "symbol": "BTCUSDT",
        "qty": "0.001",
        "min_qty": "0.001",
        "estimated_notional_usdt": "50",
        "max_notional_usdt": "10",
    }

    summary = orch._build_session_summary()
    assert summary.get("drill_abort_details") is not None
    details = summary["drill_abort_details"]
    assert details.get("symbol") == "BTCUSDT"
    assert details.get("qty") == "0.001"
    assert details.get("min_qty") == "0.001"
    assert details.get("estimated_notional_usdt") == "50"
    assert details.get("max_notional_usdt") == "10"

    md = orch._build_markdown_summary(summary)
    assert "Abort details" in md
    assert "BTCUSDT" in md
    assert "0.001" in md
    assert "50" in md
    assert "10" in md


def test_drill_summary_fields() -> None:
    """Session summary includes drill fields when drill is enabled."""
    settings = load_settings()
    settings.runtime.demo_drill.enabled = True
    settings.runtime.mode = RuntimeMode.DEMO

    orch = RuntimeOrchestrator(settings)
    orch._drill_outcome.enabled = True
    orch._drill_outcome.attempted = True
    orch._drill_outcome.symbol = "BTCUSDT"
    orch._drill_outcome.side = "Buy"
    orch._drill_outcome.qty = "0.001"
    orch._drill_outcome.ack_received = True
    orch._drill_outcome.completed = True

    summary = orch._build_session_summary()
    assert summary.get("drill_enabled") is True
    assert summary.get("drill_attempted") is True
    assert summary.get("drill_symbol") == "BTCUSDT"
    assert summary.get("drill_side") == "Buy"
    assert summary.get("drill_qty") == "0.001"
    assert summary.get("drill_ack_received") is True
    assert summary.get("drill_outcome") == "completed"

    md = orch._build_markdown_summary(summary)
    assert "Demo Drill" in md
    assert "BTCUSDT" in md
    assert "completed" in md
