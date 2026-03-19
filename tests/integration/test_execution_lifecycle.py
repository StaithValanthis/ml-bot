"""Integration tests for mocked execution lifecycle."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from trading.exchange.schemas import ServerTimeResult
from trading.settings import load_settings
from trading.execution.order_intent import IntentPurpose, OrderIntent
from trading.execution.order_manager import OrderManager
from trading.journal.ledger import LedgerEvent, LedgerSink
from trading.util.types import OrderSide, OrderType, TimeInForce


class _CaptureSink(LedgerSink):
    def __init__(self) -> None:
        self.events: list[LedgerEvent] = []

    async def write_event(self, event: LedgerEvent) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_mocked_place_order_ack_flow() -> None:
    """Mocked place_order returns ack; orchestrator records order_submission_attempt and order_ack."""
    from trading.exchange.schemas import OrderAck
    from trading.runtime.orchestrator import RuntimeOrchestrator
    from trading.util.types import RuntimeMode
    from trading.settings import load_settings

    settings = load_settings()
    capture = _CaptureSink()

    mock_ack = OrderAck.model_validate(
        {"orderId": "ex-ack-123", "orderLinkId": "v1alpha-BTCUSDT-001"}
    )
    mock_rest = MagicMock()
    mock_rest.get_server_time = AsyncMock(return_value=ServerTimeResult(time_second="1700000000", time_nano="0"))
    mock_rest.get_wallet = AsyncMock(return_value=[])
    mock_rest.get_positions = AsyncMock(return_value=[])
    mock_rest.get_open_orders = AsyncMock(return_value=[])
    mock_rest.place_order = AsyncMock(return_value=mock_ack)
    mock_rest.close = AsyncMock()

    mock_ws_public = MagicMock()
    mock_ws_public.subscribe = AsyncMock()
    mock_ws_public.run_forever = AsyncMock(side_effect=lambda: asyncio.sleep(0.2))
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
        orch._ledger._sinks.insert(0, capture)
        orch._settings.runtime.dry_run = False
        orch._settings.runtime.mode = RuntimeMode.DEMO
        orch._settings.exchange.bybit_api_key = SecretStr("test-key")
        orch._settings.exchange.bybit_api_secret = SecretStr("test-secret")

        intent = OrderIntent(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            qty=Decimal("0.01"),
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.POST_ONLY,
            reduce_only=False,
            price=Decimal("60000"),
            order_link_id="v1alpha-BTCUSDT-001",
            purpose=IntentPurpose.ENTRY,
            created_at=datetime.now(UTC),
            metadata={},
        )
        await orch._order_manager.register_intent(intent)
        await orch._submit_intent(intent)

    event_types = [e.event_type for e in capture.events]
    assert "order_submission_attempt" in event_types
    assert "order_ack" in event_types
    sub_ev = next(e for e in capture.events if e.event_type == "order_submission_attempt")
    assert sub_ev.payload["order_link_id"] == "v1alpha-BTCUSDT-001"
    ack_ev = next(e for e in capture.events if e.event_type == "order_ack")
    assert ack_ev.payload["order_id"] == "ex-ack-123"
    mock_rest.place_order.assert_called_once()


@pytest.mark.asyncio
async def test_order_manager_state_transition_flow() -> None:
    """OrderManager ack and apply_order_update produce expected state transitions."""
    from trading.util.types import OrderStatus

    mgr = OrderManager()
    intent = OrderIntent(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=Decimal("0.01"),
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.GTC,
        reduce_only=False,
        price=Decimal("60000"),
        order_link_id="link-1",
        purpose=IntentPurpose.ENTRY,
        created_at=datetime.now(UTC),
        metadata={},
    )
    await mgr.register_intent(intent)
    order = await mgr.get_by_link_id("link-1")
    assert order is not None
    assert order.order_id is None
    assert order.status == OrderStatus.NEW

    await mgr.ack_exchange_order(order_link_id="link-1", order_id="ex-1", updated_at=datetime.now(UTC))
    order = await mgr.get_by_link_id("link-1")
    assert order.order_id == "ex-1"
    assert order.status == OrderStatus.NEW

    await mgr.apply_order_update(
        order_id="ex-1",
        order_link_id="link-1",
        status=OrderStatus.PARTIALLY_FILLED,
        filled_qty=Decimal("0.005"),
        avg_price=Decimal("60000"),
        updated_at=datetime.now(UTC),
    )
    order = await mgr.get_by_link_id("link-1")
    assert order.status == OrderStatus.PARTIALLY_FILLED
    assert order.filled_qty == Decimal("0.005")


@pytest.mark.asyncio
async def test_reconciliation_mismatch_detection_and_reporting() -> None:
    """Reconciler detects missing_on_exchange and records to ledger via orchestrator."""
    from trading.execution.reconciler import Reconciler
    from trading.runtime.orchestrator import RuntimeOrchestrator
    from trading.settings import load_settings
    from trading.util.types import OrderStatus

    settings = load_settings()
    capture = _CaptureSink()

    mock_rest = MagicMock()
    mock_rest.get_server_time = AsyncMock(return_value=ServerTimeResult(time_second="1700000000", time_nano="0"))
    mock_rest.get_wallet = AsyncMock(return_value=[])
    mock_rest.get_positions = AsyncMock(return_value=[])
    mock_rest.get_open_orders = AsyncMock(return_value=[])
    mock_rest.close = AsyncMock()

    mock_ws_public = MagicMock()
    mock_ws_public.subscribe = AsyncMock()
    mock_ws_public.run_forever = AsyncMock(side_effect=lambda: asyncio.sleep(0.5))
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
        orch._ledger._sinks.insert(0, capture)
        orch._settings.exchange.bybit_api_key = SecretStr("test-key")
        orch._settings.exchange.bybit_api_secret = SecretStr("test-secret")

        await orch._order_manager.register_intent(
            OrderIntent(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                qty=Decimal("0.01"),
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
            order_link_id="link-missing", order_id="ex-missing", updated_at=datetime.now(UTC)
        )
        mock_rest.get_open_orders = AsyncMock(return_value=[])

        task = asyncio.create_task(orch.run())
        await asyncio.sleep(0.35)
        await orch.stop()
        await task

    event_types = [e.event_type for e in capture.events]
    assert "reconcile_mismatch_detected" in event_types
    mismatch_ev = next(e for e in capture.events if e.event_type == "reconcile_mismatch_detected")
    assert "missing_on_exchange" in str(mismatch_ev.payload.get("issue_types", {}))
    assert "reconcile_recovery_action" in event_types
    recovery_ev = next(e for e in capture.events if e.event_type == "reconcile_recovery_action")
    assert recovery_ev.payload.get("auto_cancel_implemented") is False


@pytest.mark.asyncio
async def test_startup_mode_warning_for_demo_order_capable() -> None:
    """Startup logs execution_mode_warning when demo/live with order placement enabled."""
    from trading.runtime.orchestrator import RuntimeOrchestrator
    from trading.settings import load_settings
    from trading.util.types import RuntimeMode

    settings = load_settings()
    log_calls: list[tuple[str, dict]] = []

    def capture_warning(msg: str, **kwargs: object) -> None:
        log_calls.append((msg, dict(kwargs)))

    mock_rest = MagicMock()
    mock_rest.get_server_time = AsyncMock(return_value=ServerTimeResult(time_second="1700000000", time_nano="0"))
    mock_rest.get_wallet = AsyncMock(return_value=[])
    mock_rest.get_positions = AsyncMock(return_value=[])
    mock_rest.get_open_orders = AsyncMock(return_value=[])
    mock_rest.close = AsyncMock()

    mock_ws_public = MagicMock()
    mock_ws_public.subscribe = AsyncMock()
    mock_ws_public.run_forever = AsyncMock(side_effect=lambda: asyncio.sleep(0.05))
    mock_ws_public.close = AsyncMock()

    mock_ws_private = MagicMock()
    mock_ws_private.subscribe = AsyncMock()
    mock_ws_private.run_forever = AsyncMock(return_value=None)
    mock_ws_private.close = AsyncMock()

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", return_value=mock_rest),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", return_value=mock_ws_public),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", return_value=mock_ws_private),
        patch("trading.runtime.orchestrator.get_logger") as mock_get_logger,
    ):
        mock_logger = MagicMock()
        mock_logger.info = MagicMock()
        mock_logger.warning = capture_warning
        mock_logger.exception = MagicMock()
        mock_get_logger.return_value = mock_logger

        orch = RuntimeOrchestrator(settings)
        orch._settings.runtime.mode = RuntimeMode.DEMO
        orch._settings.runtime.dry_run = False
        orch._settings.exchange.bybit_api_key = SecretStr("test-key")
        orch._settings.exchange.bybit_api_secret = SecretStr("test-secret")

        orch._log_execution_mode_warning()

    assert any(msg == "execution_mode_warning" for msg, _ in log_calls)
    assert any(msg == "execution_mode_banner" for msg, _ in log_calls)
    ev = next((kw for msg, kw in log_calls if msg == "execution_mode_warning"), {})
    assert ev.get("dry_run") is False
    assert ev.get("mode") == "demo"


@pytest.mark.asyncio
async def test_portfolio_refresh_uses_symbol_scoped_positions() -> None:
    """Portfolio refresh calls get_positions with symbol for each trading symbol."""
    from trading.runtime.orchestrator import RuntimeOrchestrator

    settings = load_settings()
    settings.exchange.bybit_api_key = SecretStr("test-key")
    settings.exchange.bybit_api_secret = SecretStr("test-secret")
    symbols = settings.trading.symbols

    mock_rest = MagicMock()
    mock_rest.get_wallet = AsyncMock(return_value=[])
    mock_rest.get_positions = AsyncMock(return_value=[])

    with patch("trading.runtime.orchestrator.BybitRestClient", return_value=mock_rest):
        orch = RuntimeOrchestrator(settings)
        await orch._refresh_portfolio_snapshot()

    assert mock_rest.get_positions.await_count == len(symbols)
    calls = mock_rest.get_positions.await_args_list
    symbols_called = {c[1].get("symbol") for c in calls}
    assert symbols_called == set(symbols)
