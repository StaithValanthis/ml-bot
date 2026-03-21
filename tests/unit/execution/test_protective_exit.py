"""Unit tests for protective exit placement after entry fill."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading.execution.execution_engine import ExecutionEngine, ExecutionPolicyConfig
from trading.util.types import PositionSide


def test_build_protective_limit_exit_long() -> None:
    """Protective limit exit for LONG places sell limit above entry."""
    engine = ExecutionEngine(strategy_id="v1alpha")
    intent = engine.build_protective_limit_exit(
        symbol="BTCUSDT",
        side_to_close=PositionSide.LONG,
        qty=Decimal("0.01"),
        entry_avg_price=Decimal("50000"),
        price_tick=Decimal("0.1"),
        qty_step=Decimal("0.001"),
        now=datetime.now(UTC),
    )
    assert intent is not None
    assert intent.reduce_only is True
    assert intent.symbol == "BTCUSDT"
    assert intent.qty == Decimal("0.01")
    assert intent.price is not None
    assert intent.price > Decimal("50000")
    assert intent.side.value == "Sell"


def test_build_protective_limit_exit_short() -> None:
    """Protective limit exit for SHORT places buy limit below entry."""
    engine = ExecutionEngine(strategy_id="v1alpha")
    intent = engine.build_protective_limit_exit(
        symbol="ETHUSDT",
        side_to_close=PositionSide.SHORT,
        qty=Decimal("0.1"),
        entry_avg_price=Decimal("3000"),
        price_tick=Decimal("0.01"),
        qty_step=Decimal("0.01"),
        now=datetime.now(UTC),
    )
    assert intent is not None
    assert intent.reduce_only is True
    assert intent.symbol == "ETHUSDT"
    assert intent.price is not None
    assert intent.price < Decimal("3000")
    assert intent.side.value == "Buy"


def test_build_protective_limit_exit_returns_none_for_flat() -> None:
    """Returns None when side_to_close is FLAT."""
    engine = ExecutionEngine(strategy_id="v1alpha")
    intent = engine.build_protective_limit_exit(
        symbol="BTCUSDT",
        side_to_close=PositionSide.FLAT,
        qty=Decimal("0.01"),
        entry_avg_price=Decimal("50000"),
        price_tick=Decimal("0.1"),
        qty_step=Decimal("0.001"),
        now=datetime.now(UTC),
    )
    assert intent is None


@pytest.mark.asyncio
async def test_place_protective_exit_after_fill_creates_and_tracks() -> None:
    """Entry fill triggers protective exit: plan created, intent registered, order submitted."""
    from trading.marketdata.normalizers import NormalizedOrderUpdate
    from trading.runtime.orchestrator import RuntimeOrchestrator
    from trading.settings import load_settings
    from trading.util.types import OrderSide, OrderStatus

    settings = load_settings()
    capture_events: list[tuple[str, dict]] = []

    async def capture_record(event_type: str, payload: dict) -> None:
        capture_events.append((event_type, dict(payload)))

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
        orch._ledger.record = capture_record
        orch._symbol_specs["BTCUSDT"] = MagicMock()
        orch._symbol_specs["BTCUSDT"].price_tick = Decimal("0.1")
        orch._symbol_specs["BTCUSDT"].qty_step = Decimal("0.001")
        orch._symbol_specs["BTCUSDT"].min_qty = Decimal("0.001")

        from trading.execution.order_intent import IntentPurpose, OrderIntent
        from trading.util.types import OrderType, TimeInForce

        orch._rest.place_order = AsyncMock(return_value=MagicMock(order_link_id="link-exit", order_id="ord-exit"))
        orch._can_place_exchange_orders = lambda: True

        await orch._order_manager.register_intent(
            OrderIntent(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                qty=Decimal("0.01"),
                order_type=OrderType.LIMIT,
                time_in_force=TimeInForce.POST_ONLY,
                reduce_only=False,
                price=Decimal("50000"),
                order_link_id="link-entry",
                purpose=IntentPurpose.ENTRY,
                created_at=datetime.now(UTC),
                metadata={"signal_action": "enter_long"},
            )
        )
        await orch._order_manager.ack_exchange_order(
            order_link_id="link-entry", order_id="ord-1", updated_at=datetime.now(UTC)
        )
        await orch._order_manager.apply_order_update(
            order_id="ord-1",
            order_link_id="link-entry",
            status=OrderStatus.FILLED,
            filled_qty=Decimal("0.01"),
            avg_price=Decimal("50100"),
            updated_at=datetime.now(UTC),
        )

        event = NormalizedOrderUpdate(
            order_id="ord-1",
            order_link_id="link-entry",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.FILLED,
            tif=TimeInForce.POST_ONLY,
            qty=Decimal("0.01"),
            leaves_qty=Decimal("0"),
            price=Decimal("50000"),
            avg_price=Decimal("50100"),
            reduce_only=False,
            update_time_ms=1710000000000,
            ts_event_utc=datetime.now(UTC),
            raw={},
        )
        prev = await orch._order_manager.get_by_link_id("link-entry")
        assert prev is not None

        await orch._place_protective_exit_after_fill(link_id="link-entry", event=event, prev=prev)

    event_types = [e[0] for e in capture_events]
    assert "protective_exit_plan_created" in event_types
    assert "protective_exit_tracking_registered" in event_types
    assert "protective_exit_order_submitted" in event_types
    open_orders = await orch._order_manager.get_open_orders(None)
    reduce_only = [o for o in open_orders if o.reduce_only]
    assert len(reduce_only) >= 1
    assert any(o.symbol == "BTCUSDT" for o in reduce_only)
