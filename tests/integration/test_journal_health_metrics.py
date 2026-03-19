"""Integration tests for journal/health/metrics updates during a minimal runtime cycle."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading.exchange.schemas import ServerTimeResult
from trading.journal.ledger import LedgerEvent, LedgerSink
from trading.monitoring.alerts import AlertEvent, AlertLevel, AlertSink
from trading.runtime.orchestrator import RuntimeOrchestrator
from trading.settings import load_settings


class _CaptureSink(LedgerSink):
    def __init__(self) -> None:
        self.events: list[LedgerEvent] = []

    async def write_event(self, event: LedgerEvent) -> None:
        self.events.append(event)


class _CaptureAlerts(AlertSink):
    def __init__(self) -> None:
        self.alerts: list[AlertEvent] = []

    def emit(self, event: AlertEvent) -> None:
        self.alerts.append(event)


@pytest.mark.asyncio
async def test_watchdog_staleness_journals_and_alerts() -> None:
    """When watchdog detects staleness, ledger records staleness_violation and alert is emitted."""
    settings = load_settings()
    capture_ledger = _CaptureSink()
    capture_alerts = _CaptureAlerts()

    mock_staleness = MagicMock()
    mock_staleness.stale_channels = AsyncMock(return_value=["public:BTCUSDT"])

    mock_rest = MagicMock()
    mock_rest.get_server_time = AsyncMock(return_value=ServerTimeResult(time_second="1700000000", time_nano="0"))
    mock_rest.get_wallet = AsyncMock(return_value=[])
    mock_rest.get_positions = AsyncMock(return_value=[])
    mock_rest.get_open_orders = AsyncMock(return_value=[])
    mock_rest.close = AsyncMock()

    async def mock_ws_run() -> None:
        await asyncio.sleep(0.2)

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
        orch._ledger._sinks.insert(0, capture_ledger)
        orch._alerts = capture_alerts
        orch._staleness = mock_staleness

        task = asyncio.create_task(orch.run())
        await asyncio.sleep(0.35)
        await orch.stop()
        await task

    event_types = [e.event_type for e in capture_ledger.events]
    assert "staleness_violation" in event_types
    staleness_ev = next(e for e in capture_ledger.events if e.event_type == "staleness_violation")
    assert "public:BTCUSDT" in staleness_ev.payload.get("channels", [])

    assert any(a.code == "feed_stale" for a in capture_alerts.alerts)
    feed_stale = next(a for a in capture_alerts.alerts if a.code == "feed_stale")
    assert feed_stale.level == AlertLevel.CRITICAL
    assert "public:BTCUSDT" in feed_stale.context.get("channels", "")

    snap = orch._health.snapshot()
    assert "public:BTCUSDT" in snap.stale_channels
    assert snap.circuit_breaker_tripped is True
