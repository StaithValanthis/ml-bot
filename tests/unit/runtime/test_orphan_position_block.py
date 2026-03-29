"""Unit tests for orphan position safety block."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading.execution.reconciler import ReconciliationIssue, ReconciliationReport
from trading.runtime.orchestrator import RuntimeOrchestrator
from trading.settings import load_settings

def _sym() -> str:
    return load_settings().trading.symbols[0]



@pytest.mark.asyncio
async def test_orphan_position_block_set_on_missing_reduce_only() -> None:
    """Orphan block is set when reconcile detects non-flat position without tracked reduce-only exit."""
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
                        symbol=_sym(),
                        details="Non-flat position has no local tracked reduce-only exit order.",
                        position_size=Decimal("0.05"),
                        position_side="Buy",
                    ),
                ],
            )
        )
        orch._reconciler = mock_reconciler

        await orch._reconcile_cycle()

    assert orch._orphan_position_blocked is True
    assert len(orch._orphan_position_details) == 1
    assert orch._orphan_position_details[0]["symbol"] == _sym()
    assert orch._orphan_position_details[0]["position_size"] == 0.05
    assert orch._orphan_position_details[0]["side"] == "Buy"


@pytest.mark.asyncio
async def test_orphan_position_block_cleared_when_reconcile_ok() -> None:
    """Orphan block is cleared when reconciliation reports positions ok."""
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
        orch._orphan_position_blocked = True
        orch._orphan_position_details = [{"symbol": _sym(), "position_size": 0.05, "side": "Buy", "reason": "test"}]

        mock_reconciler = MagicMock()
        mock_reconciler.reconcile_orders = AsyncMock(return_value=ReconciliationReport(ok=True, issues=[]))
        mock_reconciler.reconcile_positions = AsyncMock(return_value=ReconciliationReport(ok=True, issues=[]))
        orch._reconciler = mock_reconciler

        await orch._reconcile_cycle()

    assert orch._orphan_position_blocked is False
    assert orch._orphan_position_details == []


@pytest.mark.asyncio
async def test_orphan_block_not_set_for_order_issues_only() -> None:
    """Orphan block is NOT set when only order issues exist (no position issues)."""
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
        mock_reconciler.reconcile_orders = AsyncMock(
            return_value=ReconciliationReport(
                ok=False,
                issues=[
                    ReconciliationIssue(
                        issue_type="missing_on_exchange",
                        symbol=_sym(),
                        details="Local open order not found remotely",
                        order_link_id="link-1",
                    ),
                ],
            )
        )
        mock_reconciler.reconcile_positions = AsyncMock(return_value=ReconciliationReport(ok=True, issues=[]))
        orch._reconciler = mock_reconciler

        await orch._reconcile_cycle()

    assert orch._orphan_position_blocked is False


@pytest.mark.asyncio
async def test_session_summary_includes_orphan_position_blocked() -> None:
    """Session summary includes orphan_position_blocked and details when blocked."""
    settings = load_settings()
    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)
        orch._orphan_position_blocked = True
        orch._orphan_position_details = [
            {"symbol": _sym(), "position_size": 0.05, "side": "Buy", "reason": "non_flat_position_no_tracked_reduce_only_exit"},
        ]

        summary = await orch._build_session_summary()

    assert summary.get("orphan_position_blocked") is True
    details = summary.get("orphan_position_details", [])
    assert len(details) == 1
    assert details[0]["symbol"] == _sym()
    assert details[0]["position_size"] == 0.05


@pytest.mark.asyncio
async def test_runtime_summary_includes_orphan_position_blocked() -> None:
    """Runtime summary log includes orphan_position_blocked when set."""
    settings = load_settings()
    captured: list[dict] = []

    def capture_log(event: str, **kwargs: object) -> None:
        if event == "runtime_summary":
            captured.append(dict(kwargs))

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)
        orch._orphan_position_blocked = True
        orch._orphan_position_details = [{"symbol": _sym(), "position_size": 0.05, "side": "Buy", "reason": "test"}]
        orch._logger.info = capture_log

        await orch._runtime_summary_cycle()

    assert len(captured) == 1
    assert captured[0].get("orphan_position_blocked") is True
    assert "orphan_position_details" in captured[0]


@pytest.mark.asyncio
async def test_markdown_summary_includes_orphan_position_section() -> None:
    """Markdown summary includes Orphan Position Blocked section when blocked."""
    settings = load_settings()
    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)
        orch._orphan_position_blocked = True
        orch._orphan_position_details = [
            {"symbol": "ETHUSDT", "position_size": 0.1, "side": "Sell", "reason": "non_flat_position_no_tracked_reduce_only_exit"},
        ]
        summary = await orch._build_session_summary()
        md = orch._build_markdown_summary(summary)

    assert "## Orphan Position Blocked (SAFETY)" in md
    assert "ETHUSDT" in md
    assert "0.1" in md
    assert "Sell" in md
