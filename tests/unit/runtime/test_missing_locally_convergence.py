"""Unit tests for missing_locally reconciliation and startup convergence."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading.execution.reconciler import ReconciliationIssue, ReconciliationReport
from trading.runtime.orchestrator import RuntimeOrchestrator
from trading.settings import load_settings


@pytest.mark.asyncio
async def test_startup_block_clears_after_missing_locally_reduce_only_sync() -> None:
    """When only missing_locally (reduce-only) and positions ok, startup block clears."""
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
        orch._startup_state_details = [{"reason": "reconcile_order_mismatch"}]

        mock_reconciler = MagicMock()
        mock_reconciler.reconcile_orders = AsyncMock(
            return_value=ReconciliationReport(
                ok=False,
                issues=[
                    ReconciliationIssue(
                        issue_type="missing_locally",
                        symbol="BTCUSDT",
                        details="Exchange open order not tracked locally",
                        order_link_id="exch-ro-1",
                        order_id="ord-1",
                        reduce_only=True,
                        qty=Decimal("0.05"),
                    ),
                ],
            )
        )
        mock_reconciler.reconcile_positions = AsyncMock(
            return_value=ReconciliationReport(ok=True, issues=[]),
        )
        orch._reconciler = mock_reconciler

        await orch._reconcile_cycle()

        assert orch._startup_state_blocked is False
        assert orch._startup_state_details == []
        assert orch._consecutive_reconcile_mismatches == 0


@pytest.mark.asyncio
async def test_missing_locally_non_reduce_only_stays_blocked_but_no_escalation() -> None:
    """Non-reduce-only missing_locally: sync, keep startup block, reset consecutive (no loop)."""
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
        orch._startup_state_details = [{"reason": "reconcile_order_mismatch"}]
        orch._consecutive_reconcile_mismatches = 2

        mock_reconciler = MagicMock()
        mock_reconciler.reconcile_orders = AsyncMock(
            return_value=ReconciliationReport(
                ok=False,
                issues=[
                    ReconciliationIssue(
                        issue_type="missing_locally",
                        symbol="BTCUSDT",
                        details="Exchange open order not tracked locally",
                        order_link_id="exch-entry-1",
                        order_id="ord-2",
                        reduce_only=False,
                        qty=Decimal("0.001"),
                    ),
                ],
            )
        )
        mock_reconciler.reconcile_positions = AsyncMock(
            return_value=ReconciliationReport(ok=True, issues=[]),
        )
        orch._reconciler = mock_reconciler

        await orch._reconcile_cycle()

        assert orch._startup_state_blocked is True
        assert orch._consecutive_reconcile_mismatches == 0


@pytest.mark.asyncio
async def test_no_repeated_mismatch_after_missing_locally_sync() -> None:
    """After missing_locally reduce-only sync, second reconcile yields ok (no churn)."""
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

        mock_reconciler = MagicMock()
        mock_reconciler.reconcile_orders = AsyncMock(
            return_value=ReconciliationReport(
                ok=False,
                issues=[
                    ReconciliationIssue(
                        issue_type="missing_locally",
                        symbol="BTCUSDT",
                        details="Exchange open order not tracked locally",
                        order_link_id="exch-ro-2",
                        order_id="ord-3",
                        reduce_only=True,
                        qty=Decimal("0.05"),
                    ),
                ],
            )
        )
        mock_reconciler.reconcile_positions = AsyncMock(
            return_value=ReconciliationReport(ok=True, issues=[]),
        )
        orch._reconciler = mock_reconciler

        await orch._reconcile_cycle()

        assert orch._startup_state_blocked is False
        assert orch._consecutive_reconcile_mismatches == 0

        mock_reconciler.reconcile_orders = AsyncMock(
            return_value=ReconciliationReport(ok=True, issues=[]),
        )
        mock_reconciler.reconcile_positions = AsyncMock(
            return_value=ReconciliationReport(ok=True, issues=[]),
        )

        await orch._reconcile_cycle()

        assert orch._startup_state_blocked is False
        assert orch._consecutive_reconcile_mismatches == 0


@pytest.mark.asyncio
async def test_qty_mismatch_only_clears_startup() -> None:
    """Only qty_mismatch (no missing_locally) clears startup block."""
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

        mock_reconciler = MagicMock()
        mock_reconciler.reconcile_orders = AsyncMock(
            return_value=ReconciliationReport(
                ok=False,
                issues=[
                    ReconciliationIssue(
                        issue_type="qty_mismatch",
                        symbol="BTCUSDT",
                        details="qty mismatch local=0.05 remote=0.04",
                        order_link_id="link-1",
                        order_id="ord-1",
                        reduce_only=True,
                        qty=Decimal("0.04"),
                    ),
                ],
            )
        )
        mock_reconciler.reconcile_positions = AsyncMock(
            return_value=ReconciliationReport(ok=True, issues=[]),
        )
        orch._reconciler = mock_reconciler

        await orch._reconcile_cycle()

        assert orch._startup_state_blocked is False
        assert orch._consecutive_reconcile_mismatches == 0


@pytest.mark.asyncio
async def test_soak_diagnostics_reflect_missing_locally_resolved() -> None:
    """After missing_locally reduce-only sync, session summary reflects startup cleared."""
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

        mock_reconciler = MagicMock()
        mock_reconciler.reconcile_orders = AsyncMock(
            return_value=ReconciliationReport(
                ok=False,
                issues=[
                    ReconciliationIssue(
                        issue_type="missing_locally",
                        symbol="BTCUSDT",
                        details="Exchange open order not tracked locally",
                        order_link_id="exch-ro-diag",
                        order_id="ord-diag",
                        reduce_only=True,
                        qty=Decimal("0.05"),
                    ),
                ],
            )
        )
        mock_reconciler.reconcile_positions = AsyncMock(
            return_value=ReconciliationReport(ok=True, issues=[]),
        )
        orch._reconciler = mock_reconciler

        await orch._reconcile_cycle()

        assert orch._startup_state_blocked is False
        summary = await orch._build_session_summary()
        assert summary.get("startup_state_blocked") is not True
        assert orch._metrics.snapshot().counters.get("startup_state_block_cleared_count", 0) >= 1


@pytest.mark.asyncio
async def test_mixed_reduce_only_and_non_reduce_only_keeps_block() -> None:
    """If any missing_locally is non-reduce-only, startup block stays."""
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

        mock_reconciler = MagicMock()
        mock_reconciler.reconcile_orders = AsyncMock(
            return_value=ReconciliationReport(
                ok=False,
                issues=[
                    ReconciliationIssue(
                        issue_type="missing_locally",
                        symbol="BTCUSDT",
                        details="Exchange reduce-only not tracked",
                        order_link_id="exch-ro-1",
                        order_id="ord-1",
                        reduce_only=True,
                        qty=Decimal("0.05"),
                    ),
                    ReconciliationIssue(
                        issue_type="missing_locally",
                        symbol="BTCUSDT",
                        details="Exchange entry not tracked",
                        order_link_id="exch-entry-1",
                        order_id="ord-2",
                        reduce_only=False,
                        qty=Decimal("0.001"),
                    ),
                ],
            )
        )
        mock_reconciler.reconcile_positions = AsyncMock(
            return_value=ReconciliationReport(ok=True, issues=[]),
        )
        orch._reconciler = mock_reconciler

        await orch._reconcile_cycle()

        assert orch._startup_state_blocked is True
        assert orch._consecutive_reconcile_mismatches == 0
