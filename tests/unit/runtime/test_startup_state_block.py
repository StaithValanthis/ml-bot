"""Unit tests for startup dirty exchange state safety block."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading.execution.reconciler import ReconciliationIssue, ReconciliationReport
from trading.runtime.orchestrator import RuntimeOrchestrator
from trading.settings import load_settings
from trading.util.types import RuntimeMode


@pytest.mark.asyncio
async def test_startup_detects_open_position_and_blocks() -> None:
    """Startup inspects exchange and sets startup_state_blocked when non-flat position exists."""
    settings = load_settings()
    settings.runtime.mode = RuntimeMode.DEMO
    mock_pos = MagicMock()
    mock_pos.symbol = "BTCUSDT"
    mock_pos.side = "Buy"
    mock_pos.size = Decimal("0.05")
    mock_orders: list = []
    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)
        orch._settings.exchange.bybit_api_key = MagicMock()
        orch._settings.exchange.bybit_api_key.get_secret_value = lambda: "test"
        orch._settings.exchange.bybit_api_secret = MagicMock()
        orch._settings.exchange.bybit_api_secret.get_secret_value = lambda: "secret"
        orch._rest.get_positions = AsyncMock(return_value=[mock_pos])
        orch._rest.get_open_orders = AsyncMock(return_value=mock_orders)
        await orch._inspect_startup_exchange_state()
    assert orch._startup_state_blocked is True
    assert len(orch._startup_state_details) >= 1
    assert any(d["symbol"] == "BTCUSDT" and d["position_size"] == 0.05 for d in orch._startup_state_details)


@pytest.mark.asyncio
async def test_startup_detects_stray_orders_and_blocks() -> None:
    """Startup sets startup_state_blocked when exchange has open orders (stray)."""
    settings = load_settings()
    settings.runtime.mode = RuntimeMode.DEMO
    mock_order = MagicMock()
    mock_order.symbol = "BTCUSDT"
    mock_order.reduce_only = False
    mock_order.order_link_id = "stray-1"
    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)
        orch._settings.exchange.bybit_api_key = MagicMock()
        orch._settings.exchange.bybit_api_key.get_secret_value = lambda: "test"
        orch._settings.exchange.bybit_api_secret = MagicMock()
        orch._settings.exchange.bybit_api_secret.get_secret_value = lambda: "secret"
        orch._rest.get_positions = AsyncMock(return_value=[])
        orch._rest.get_open_orders = AsyncMock(return_value=[mock_order])
        await orch._inspect_startup_exchange_state()
    assert orch._startup_state_blocked is True
    assert len(orch._startup_state_details) >= 1
    assert any(d["symbol"] == "BTCUSDT" and d.get("non_reduce_only_order_count", 0) >= 1 for d in orch._startup_state_details)


@pytest.mark.asyncio
async def test_startup_blocks_new_entries_when_blocked() -> None:
    """When startup_state_blocked, _infer_strategy_blocking_stage returns startup_state_blocked."""
    settings = load_settings()
    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)
        orch._startup_state_blocked = True
    assert orch._infer_strategy_blocking_stage({}) == "startup_state_blocked"


@pytest.mark.asyncio
async def test_startup_block_cleared_when_reconcile_ok() -> None:
    """startup_state_blocked is cleared when reconcile reports ok."""
    settings = load_settings()
    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)
        orch._settings.exchange.bybit_api_key = MagicMock()
        orch._settings.exchange.bybit_api_key.get_secret_value = lambda: "test"
        orch._settings.exchange.bybit_api_secret = MagicMock()
        orch._settings.exchange.bybit_api_secret.get_secret_value = lambda: "secret"
        orch._startup_state_blocked = True
        orch._startup_state_details = [{"symbol": "BTCUSDT"}]
        mock_reconciler = MagicMock()
        mock_reconciler.reconcile_orders = AsyncMock(return_value=ReconciliationReport(ok=True, issues=[]))
        mock_reconciler.reconcile_positions = AsyncMock(return_value=ReconciliationReport(ok=True, issues=[]))
        orch._reconciler = mock_reconciler
        await orch._reconcile_cycle()
    assert orch._startup_state_blocked is False
    assert orch._startup_state_details == []


@pytest.mark.asyncio
async def test_reconcile_preserves_startup_block_on_orphan() -> None:
    """Reconcile sets startup_state_blocked when orphan position detected."""
    settings = load_settings()
    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)
        orch._settings.exchange.bybit_api_key = MagicMock()
        orch._settings.exchange.bybit_api_key.get_secret_value = lambda: "test"
        orch._settings.exchange.bybit_api_secret = MagicMock()
        orch._settings.exchange.bybit_api_secret.get_secret_value = lambda: "secret"
        mock_reconciler = MagicMock()
        mock_reconciler.reconcile_orders = AsyncMock(return_value=ReconciliationReport(ok=True, issues=[]))
        mock_reconciler.reconcile_positions = AsyncMock(
            return_value=ReconciliationReport(
                ok=False,
                issues=[
                    ReconciliationIssue(
                        issue_type="missing_reduce_only_exit",
                        symbol="BTCUSDT",
                        details="Non-flat position has no local tracked reduce-only exit order.",
                        position_size=Decimal("0.05"),
                        position_side="Buy",
                    ),
                ],
            )
        )
        orch._reconciler = mock_reconciler
        await orch._reconcile_cycle()
    assert orch._startup_state_blocked is True
    assert orch._orphan_position_blocked is True
    assert len(orch._startup_state_details) >= 1


@pytest.mark.asyncio
async def test_runtime_summary_includes_startup_state_blocked() -> None:
    """Runtime summary includes startup_state_blocked and startup_state_details."""
    settings = load_settings()
    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)
        orch._startup_state_blocked = True
        orch._startup_state_details = [
            {
                "symbol": "BTCUSDT",
                "position_size": 0.05,
                "position_side": "Buy",
                "open_order_count": 0,
                "reduce_only_order_count": 0,
                "non_reduce_only_order_count": 0,
                "local_order_state_empty_or_not": True,
            },
        ]
        summary = await orch._build_session_summary()
    assert summary.get("startup_state_blocked") is True
    assert "startup_state_details" in summary
    assert summary["blocking_stage"] == "startup_state_blocked"
    md = orch._build_markdown_summary(summary)
    assert "## Last Startup Dirty State" in md
