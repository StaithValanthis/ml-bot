"""Unit tests for missing_on_exchange local state convergence."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading.execution.order_intent import IntentPurpose, OrderIntent
from trading.execution.reconciler import ReconciliationIssue, ReconciliationReport
from trading.runtime.orchestrator import RuntimeOrchestrator
from trading.settings import load_settings

def _sym() -> str:
    return load_settings().trading.symbols[0]

from trading.util.types import OrderSide, RuntimeMode, OrderType, TimeInForce


@pytest.mark.asyncio
async def test_missing_on_exchange_resolved_locally_order_no_longer_open() -> None:
    """Local tracked open order + exchange missing => resolved locally, order no longer open."""
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

        await orch._order_manager.register_intent(
            OrderIntent(
                symbol=_sym(),
                side=OrderSide.BUY,
                qty=Decimal("0.001"),
                order_type=OrderType.LIMIT,
                time_in_force=TimeInForce.GTC,
                reduce_only=False,
                price=Decimal("60000"),
                order_link_id="v1alph-btcu-260321235000-c2def87b",
                purpose=IntentPurpose.ENTRY,
                created_at=datetime.now(UTC),
                metadata={},
            )
        )
        await orch._order_manager.ack_exchange_order(
            order_link_id="v1alph-btcu-260321235000-c2def87b",
            order_id="ord-xyz",
            updated_at=datetime.now(UTC),
        )

        open_before = await orch._order_manager.get_open_orders(None)
        assert len(open_before) == 1

        mock_reconciler = MagicMock()
        mock_reconciler.reconcile_orders = AsyncMock(
            return_value=ReconciliationReport(
                ok=False,
                issues=[
                    ReconciliationIssue(
                        issue_type="missing_on_exchange",
                        symbol=_sym(),
                        details="Local open order not found remotely: link_id=v1alph-btcu-260321235000-c2def87b",
                        order_link_id="v1alph-btcu-260321235000-c2def87b",
                        order_id="ord-xyz",
                    ),
                ],
            )
        )
        mock_reconciler.reconcile_positions = AsyncMock(
            return_value=ReconciliationReport(ok=True, issues=[]),
        )
        orch._reconciler = mock_reconciler

        await orch._reconcile_cycle()

        open_after = await orch._order_manager.get_open_orders(None)
        assert len(open_after) == 0

        by_link = await orch._order_manager.get_by_link_id("v1alph-btcu-260321235000-c2def87b")
        assert by_link is not None
        assert by_link.metadata.get("reconcile_terminal_reason") == "closed_missing_on_exchange"


@pytest.mark.asyncio
async def test_startup_state_block_clears_after_missing_on_exchange_resolution() -> None:
    """When only missing_on_exchange and positions ok, startup block clears."""
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
        orch._startup_state_details = [{"reason": "prior"}]

        await orch._order_manager.register_intent(
            OrderIntent(
                symbol=_sym(),
                side=OrderSide.BUY,
                qty=Decimal("0.001"),
                order_type=OrderType.LIMIT,
                time_in_force=TimeInForce.GTC,
                reduce_only=False,
                price=Decimal("60000"),
                order_link_id="link-missing",
                purpose=IntentPurpose.ENTRY,
                created_at=datetime.now(UTC),
                metadata={},
            )
        )
        await orch._order_manager.ack_exchange_order(
            order_link_id="link-missing",
            order_id="ord-1",
            updated_at=datetime.now(UTC),
        )

        mock_reconciler = MagicMock()
        mock_reconciler.reconcile_orders = AsyncMock(
            return_value=ReconciliationReport(
                ok=False,
                issues=[
                    ReconciliationIssue(
                        issue_type="missing_on_exchange",
                        symbol=_sym(),
                        details="Local open order not found remotely",
                        order_link_id="link-missing",
                        order_id="ord-1",
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


@pytest.mark.asyncio
async def test_no_repeated_mismatch_after_resolution() -> None:
    """Second reconcile after missing_on_exchange resolution yields ok (no repeated mismatch)."""
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

        await orch._order_manager.register_intent(
            OrderIntent(
                symbol=_sym(),
                side=OrderSide.BUY,
                qty=Decimal("0.001"),
                order_type=OrderType.LIMIT,
                time_in_force=TimeInForce.GTC,
                reduce_only=False,
                price=Decimal("60000"),
                order_link_id="link-missing",
                purpose=IntentPurpose.ENTRY,
                created_at=datetime.now(UTC),
                metadata={},
            )
        )
        await orch._order_manager.ack_exchange_order(
            order_link_id="link-missing",
            order_id="ord-1",
            updated_at=datetime.now(UTC),
        )

        mock_reconciler = MagicMock()
        mock_reconciler.reconcile_orders = AsyncMock(
            return_value=ReconciliationReport(
                ok=False,
                issues=[
                    ReconciliationIssue(
                        issue_type="missing_on_exchange",
                        symbol=_sym(),
                        details="Local open order not found remotely",
                        order_link_id="link-missing",
                        order_id="ord-1",
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
async def test_soak_report_includes_missing_on_exchange_counts() -> None:
    """Session summary and soak report include missing_on_exchange_detected/resolved."""
    settings = load_settings()
    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)
        orch._settings.runtime.mode = RuntimeMode.DEMO
        orch._metrics.inc("missing_on_exchange_detected_count", 2)
        orch._metrics.inc("missing_on_exchange_resolved_count", 2)

        summary = await orch._build_session_summary()
        assert summary.get("missing_on_exchange_detected_count") == 2
        assert summary.get("missing_on_exchange_resolved_count") == 2

        from trading.runtime.soak_report import build_soak_report

        report = build_soak_report(summary, orch._metrics.snapshot())
        safety = report.get("safety_summary") or {}
        assert safety.get("missing_on_exchange_detected_count") == 2
        assert safety.get("missing_on_exchange_resolved_count") == 2
