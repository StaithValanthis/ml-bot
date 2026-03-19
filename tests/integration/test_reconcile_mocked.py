"""Integration tests for reconcile loop with mocked REST."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from trading.exchange.schemas import OpenOrderItem
from trading.execution.order_intent import IntentPurpose, OrderIntent
from trading.execution.order_manager import OrderManager
from trading.execution.reconciler import Reconciler
from trading.util.types import OrderSide, OrderStatus, OrderType, TimeInForce


def _open_order(
    order_id: str = "ord-1",
    order_link_id: str = "link-1",
    qty: Decimal = Decimal("0.1"),
) -> OpenOrderItem:
    return OpenOrderItem.model_validate(
        {
            "orderId": order_id,
            "orderLinkId": order_link_id,
            "symbol": "BTCUSDT",
            "side": "Buy",
            "orderType": "Limit",
            "orderStatus": OrderStatus.NEW.value,
            "price": "60000",
            "qty": str(qty),
            "cumExecQty": "0",
            "cumExecValue": "6000",
            "reduceOnly": False,
            "timeInForce": "GTC",
            "createdTime": "1700000000000",
            "updatedTime": "1700000000000",
        }
    )


@pytest.fixture
def mock_rest() -> MagicMock:
    """Mock REST client for reconcile tests."""
    client = MagicMock()
    client.get_open_orders = AsyncMock(return_value=[])
    client.get_positions = AsyncMock(return_value=[])
    return client


@pytest.fixture
def reconciler(mock_rest: MagicMock) -> Reconciler:
    """Reconciler with mocked REST client."""
    return Reconciler(
        rest_client=mock_rest,
        order_manager=OrderManager(),
        category="linear",
    )


@pytest.mark.asyncio
async def test_reconcile_orders_ok_when_both_empty(reconciler: Reconciler) -> None:
    """Reconcile returns ok when local and exchange have no open orders."""
    report = await reconciler.reconcile_orders(symbol="BTCUSDT")
    assert report.ok
    assert len(report.issues) == 0


@pytest.mark.asyncio
async def test_reconcile_orders_detects_missing_on_exchange(
    reconciler: Reconciler,
    mock_rest: MagicMock,
) -> None:
    """Reconcile detects when local has order not on exchange."""
    mgr = reconciler._order_manager
    await mgr.register_intent(
        OrderIntent(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            qty=Decimal("0.01"),
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.GTC,
            reduce_only=False,
            price=Decimal("60000"),
            order_link_id="test-link-123",
            purpose=IntentPurpose.ENTRY,
            created_at=datetime.now(UTC),
            metadata={},
        )
    )
    await mgr.ack_exchange_order(order_link_id="test-link-123", order_id="ex-1", updated_at=datetime.now(UTC))
    mock_rest.get_open_orders = AsyncMock(return_value=[])

    report = await reconciler.reconcile_orders(symbol="BTCUSDT")
    assert not report.ok
    assert any(i.issue_type == "missing_on_exchange" for i in report.issues)


@pytest.mark.asyncio
async def test_reconcile_orders_syncs_when_exchange_has_order(
    reconciler: Reconciler,
    mock_rest: MagicMock,
) -> None:
    """Reconcile upserts local state when exchange returns matching order."""
    mock_rest.get_open_orders = AsyncMock(
        return_value=[_open_order(order_id="ex-123", order_link_id="link-456", qty=Decimal("0.01"))]
    )

    mgr = reconciler._order_manager
    await mgr.register_intent(
        OrderIntent(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            qty=Decimal("0.01"),
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.GTC,
            reduce_only=False,
            price=Decimal("60000"),
            order_link_id="link-456",
            purpose=IntentPurpose.ENTRY,
            created_at=datetime.now(UTC),
            metadata={},
        )
    )

    report = await reconciler.reconcile_orders(symbol="BTCUSDT")
    assert report.ok
    open_orders = await mgr.get_open_orders("BTCUSDT")
    assert len(open_orders) == 1
    assert open_orders[0].order_id == "ex-123"
