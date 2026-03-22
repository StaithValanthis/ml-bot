"""Unit tests for order manager."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading.execution.order_intent import IntentPurpose, OrderIntent
from trading.execution.order_manager import OrderManager
from trading.util.types import OrderSide, OrderStatus, OrderType, TimeInForce


def _intent(
    order_link_id: str = "link-1",
    symbol: str = "BTCUSDT",
    qty: Decimal = Decimal("0.1"),
) -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        side=OrderSide.BUY,
        qty=qty,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.POST_ONLY,
        reduce_only=False,
        price=Decimal("60000"),
        order_link_id=order_link_id,
        purpose=IntentPurpose.ENTRY,
        created_at=datetime.now(UTC),
        metadata={},
    )


@pytest.mark.asyncio
async def test_register_intent_creates_managed_order() -> None:
    mgr = OrderManager()
    intent = _intent(order_link_id="link-1")
    managed = await mgr.register_intent(intent)
    assert managed.order_link_id == "link-1"
    assert managed.symbol == "BTCUSDT"
    assert managed.status == OrderStatus.NEW
    assert managed.qty == Decimal("0.1")
    assert managed.filled_qty == Decimal("0")


@pytest.mark.asyncio
async def test_register_intent_idempotent_same_link_id() -> None:
    mgr = OrderManager()
    intent = _intent(order_link_id="link-1")
    m1 = await mgr.register_intent(intent)
    m2 = await mgr.register_intent(intent)
    assert m1 is m2


@pytest.mark.asyncio
async def test_ack_exchange_order_updates_order_id() -> None:
    mgr = OrderManager()
    intent = _intent(order_link_id="link-1")
    await mgr.register_intent(intent)
    await mgr.ack_exchange_order(
        order_link_id="link-1",
        order_id="ord-123",
        updated_at=datetime.now(UTC),
    )
    by_link = await mgr.get_by_link_id("link-1")
    assert by_link is not None
    assert by_link.order_id == "ord-123"


@pytest.mark.asyncio
async def test_mark_closed_missing_on_exchange_resolves_local_order() -> None:
    """Local open order + mark_closed_missing_on_exchange => no longer in get_open_orders."""
    mgr = OrderManager()
    intent = _intent(order_link_id="link-1")
    await mgr.register_intent(intent)
    await mgr.ack_exchange_order(
        order_link_id="link-1",
        order_id="ord-123",
        updated_at=datetime.now(UTC),
    )
    open_before = await mgr.get_open_orders(None)
    assert len(open_before) == 1
    assert open_before[0].order_link_id == "link-1"
    assert open_before[0].status == OrderStatus.NEW

    ok = await mgr.mark_closed_missing_on_exchange(order_link_id="link-1")
    assert ok is True

    open_after = await mgr.get_open_orders(None)
    assert len(open_after) == 0

    by_link = await mgr.get_by_link_id("link-1")
    assert by_link is not None
    assert by_link.status == OrderStatus.CANCELLED
    assert by_link.metadata.get("reconcile_terminal_reason") == "closed_missing_on_exchange"


@pytest.mark.asyncio
async def test_mark_closed_missing_on_exchange_idempotent_already_terminal() -> None:
    """mark_closed_missing_on_exchange returns False for already terminal order."""
    mgr = OrderManager()
    intent = _intent(order_link_id="link-1")
    await mgr.register_intent(intent)
    await mgr.ack_exchange_order(
        order_link_id="link-1",
        order_id="ord-123",
        updated_at=datetime.now(UTC),
    )
    order = await mgr.get_by_link_id("link-1")
    assert order is not None
    order.status = OrderStatus.CANCELLED

    ok = await mgr.mark_closed_missing_on_exchange(order_link_id="link-1")
    assert ok is False


@pytest.mark.asyncio
async def test_apply_order_update_updates_status_and_filled() -> None:
    mgr = OrderManager()
    intent = _intent(order_link_id="link-1")
    await mgr.register_intent(intent)
    await mgr.ack_exchange_order(
        order_link_id="link-1",
        order_id="ord-123",
        updated_at=datetime.now(UTC),
    )
    await mgr.apply_order_update(
        order_id="ord-123",
        order_link_id="link-1",
        status=OrderStatus.PARTIALLY_FILLED,
        filled_qty=Decimal("0.05"),
        avg_price=Decimal("60100"),
        updated_at=datetime.now(UTC),
    )
    by_link = await mgr.get_by_link_id("link-1")
    assert by_link is not None
    assert by_link.status == OrderStatus.PARTIALLY_FILLED
    assert by_link.filled_qty == Decimal("0.05")
    assert by_link.avg_price == Decimal("60100")


@pytest.mark.asyncio
async def test_get_open_orders_excludes_filled_and_cancelled() -> None:
    mgr = OrderManager()
    await mgr.register_intent(_intent(order_link_id="link-1"))
    await mgr.register_intent(_intent(order_link_id="link-2"))
    await mgr.ack_exchange_order(order_link_id="link-1", order_id="ord-1", updated_at=datetime.now(UTC))
    await mgr.ack_exchange_order(order_link_id="link-2", order_id="ord-2", updated_at=datetime.now(UTC))
    await mgr.apply_order_update(
        order_id="ord-1",
        order_link_id="link-1",
        status=OrderStatus.FILLED,
        filled_qty=Decimal("0.1"),
        avg_price=Decimal("60100"),
        updated_at=datetime.now(UTC),
    )
    open_orders = await mgr.get_open_orders()
    assert len(open_orders) == 1
    assert open_orders[0].order_link_id == "link-2"


@pytest.mark.asyncio
async def test_upsert_from_exchange_snapshot_creates_if_missing() -> None:
    mgr = OrderManager()
    await mgr.upsert_from_exchange_snapshot(
        order_id="ord-123",
        order_link_id="link-1",
        symbol="BTCUSDT",
        status=OrderStatus.NEW,
        qty=Decimal("0.1"),
        filled_qty=Decimal("0"),
        avg_price=None,
        reduce_only=False,
        updated_at=datetime.now(UTC),
    )
    by_link = await mgr.get_by_link_id("link-1")
    assert by_link is not None
    assert by_link.order_id == "ord-123"
    assert by_link.symbol == "BTCUSDT"
    assert "reconciler" in str(by_link.metadata.get("source", ""))


@pytest.mark.asyncio
async def test_upsert_from_exchange_snapshot_updates_existing() -> None:
    mgr = OrderManager()
    await mgr.register_intent(_intent(order_link_id="link-1"))
    await mgr.ack_exchange_order(order_link_id="link-1", order_id="ord-123", updated_at=datetime.now(UTC))
    await mgr.upsert_from_exchange_snapshot(
        order_id="ord-123",
        order_link_id="link-1",
        symbol="BTCUSDT",
        status=OrderStatus.PARTIALLY_FILLED,
        qty=Decimal("0.1"),
        filled_qty=Decimal("0.05"),
        avg_price=Decimal("60100"),
        reduce_only=False,
        updated_at=datetime.now(UTC),
    )
    by_link = await mgr.get_by_link_id("link-1")
    assert by_link is not None
    assert by_link.status == OrderStatus.PARTIALLY_FILLED
    assert by_link.filled_qty == Decimal("0.05")
