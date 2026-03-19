"""Integration tests for WebSocket lifecycle with mocked ack/reconnect paths."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading.exchange.schemas import ServerTimeResult
from trading.journal.ledger import LedgerSink
from trading.monitoring.health import HealthState
from trading.runtime.orchestrator import RuntimeOrchestrator
from trading.settings import load_settings


@pytest.mark.asyncio
async def test_ws_lifecycle_connection_state_reflected_in_health() -> None:
    """WS connect/disconnect updates health state via connection_state_handler."""
    settings = load_settings()
    mock_ws_public = MagicMock()
    mock_ws_public.subscribe = AsyncMock()
    mock_ws_public.close = AsyncMock()

    async def mock_run_forever() -> None:
        await asyncio.sleep(0.1)

    mock_ws_public.run_forever = AsyncMock(side_effect=mock_run_forever)

    mock_rest = MagicMock()
    mock_rest.get_server_time = AsyncMock(return_value=ServerTimeResult(time_second="1700000000", time_nano="0"))
    mock_rest.get_wallet = AsyncMock(return_value=[])
    mock_rest.get_positions = AsyncMock(return_value=[])
    mock_rest.get_open_orders = AsyncMock(return_value=[])
    mock_rest.close = AsyncMock()

    mock_ws_private = MagicMock()
    mock_ws_private.subscribe = AsyncMock()
    mock_ws_private.run_forever = AsyncMock(return_value=None)
    mock_ws_private.close = AsyncMock()

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", return_value=mock_rest),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", return_value=mock_ws_public) as mock_ws_class,
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", return_value=mock_ws_private),
    ):
        orch = RuntimeOrchestrator(settings)
        handler = mock_ws_class.call_args.kwargs.get("connection_state_handler")
        assert handler is not None

        handler(True)
        snap = orch._health.snapshot()
        assert snap.ws_public_connected is True

        handler(False)
        snap = orch._health.snapshot()
        assert snap.ws_public_connected is False


@pytest.mark.asyncio
async def test_ws_subscribe_ack_path_with_mocked_client() -> None:
    """Orchestrator subscribes to public topics; mock WS client receives subscribe call."""
    settings = load_settings()
    mock_ws_public = MagicMock()
    subscribe_args: list[list[str]] = []

    async def capture_subscribe(topics: list[str]) -> None:
        subscribe_args.append(list(topics))

    mock_ws_public.subscribe = AsyncMock(side_effect=capture_subscribe)
    mock_ws_public.run_forever = AsyncMock(side_effect=lambda: asyncio.sleep(0.05))
    mock_ws_public.close = AsyncMock()

    mock_rest = MagicMock()
    mock_rest.get_server_time = AsyncMock(return_value=ServerTimeResult(time_second="1700000000", time_nano="0"))
    mock_rest.get_wallet = AsyncMock(return_value=[])
    mock_rest.get_positions = AsyncMock(return_value=[])
    mock_rest.get_open_orders = AsyncMock(return_value=[])
    mock_rest.close = AsyncMock()

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
        await asyncio.sleep(0.1)
        await orch.stop()
        await task

    assert len(subscribe_args) >= 1
    topics = subscribe_args[0]
    assert any("tickers" in t or "kline" in t for t in topics)
    assert any(s in " ".join(topics) for s in settings.trading.symbols)
