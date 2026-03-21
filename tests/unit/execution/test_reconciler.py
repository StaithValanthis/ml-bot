"""Unit tests for reconciler."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from trading.exchange.bybit_rest import BybitRestClient
from trading.exchange.schemas import OpenOrderItem, PositionItem
from trading.execution.order_intent import IntentPurpose, OrderIntent
from trading.execution.order_manager import OrderManager
from trading.execution.reconciler import Reconciler
from trading.util.types import OrderSide, OrderStatus, OrderType, TimeInForce


def _open_order(
    order_id: str = "ord-123",
    order_link_id: str = "link-1",
    symbol: str = "BTCUSDT",
    order_status: OrderStatus = OrderStatus.NEW,
    qty: Decimal = Decimal("0.1"),
    cum_exec_qty: Decimal = Decimal("0"),
    price: Decimal = Decimal("60000"),
    reduce_only: bool = False,
    updated_time: str = "1710000000000",
) -> OpenOrderItem:
    return OpenOrderItem.model_validate(
        {
            "orderId": order_id,
            "orderLinkId": order_link_id,
            "symbol": symbol,
            "side": "Buy",
            "orderType": "Limit",
            "orderStatus": order_status.value,
            "price": str(price),
            "qty": str(qty),
            "cumExecQty": str(cum_exec_qty),
            "cumExecValue": "6000",
            "reduceOnly": reduce_only,
            "timeInForce": "GTC",
            "createdTime": "1710000000000",
            "updatedTime": updated_time,
        }
    )


def _position(
    symbol: str = "BTCUSDT",
    size: Decimal = Decimal("0.1"),
) -> PositionItem:
    return PositionItem.model_validate(
        {
            "symbol": symbol,
            "side": "Buy",
            "size": str(size),
            "avgPrice": "60000",
            "markPrice": "60100",
            "positionValue": "6010",
            "leverage": "10",
            "liqPrice": "54000",
            "unrealisedPnl": "10",
            "updatedTime": "1710000000000",
        }
    )


@pytest.mark.asyncio
async def test_reconcile_orders_ok_when_in_sync() -> None:
    mock_rest = AsyncMock(spec=BybitRestClient)
    mock_rest.get_open_orders.return_value = [
        _open_order(order_id="ord-1", order_link_id="link-1"),
    ]
    mgr = OrderManager()
    await mgr.register_intent(
        OrderIntent(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            qty=Decimal("0.1"),
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.GTC,
            reduce_only=False,
            price=Decimal("60000"),
            order_link_id="link-1",
            purpose=IntentPurpose.ENTRY,
            created_at=datetime.now(UTC),
            metadata={},
        )
    )
    await mgr.ack_exchange_order(order_link_id="link-1", order_id="ord-1", updated_at=datetime.now(UTC))

    reconciler = Reconciler(rest_client=mock_rest, order_manager=mgr, category="linear")
    report = await reconciler.reconcile_orders()

    assert report.ok is True
    assert len(report.issues) == 0


@pytest.mark.asyncio
async def test_reconcile_orders_detects_missing_on_exchange() -> None:
    mock_rest = AsyncMock(spec=BybitRestClient)
    mock_rest.get_open_orders.return_value = []
    mgr = OrderManager()
    await mgr.register_intent(
        OrderIntent(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            qty=Decimal("0.1"),
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.GTC,
            reduce_only=False,
            price=Decimal("60000"),
            order_link_id="link-1",
            purpose=IntentPurpose.ENTRY,
            created_at=datetime.now(UTC),
            metadata={},
        )
    )
    await mgr.ack_exchange_order(order_link_id="link-1", order_id="ord-1", updated_at=datetime.now(UTC))

    reconciler = Reconciler(rest_client=mock_rest, order_manager=mgr, category="linear")
    report = await reconciler.reconcile_orders()

    assert report.ok is False
    assert any(i.issue_type == "missing_on_exchange" for i in report.issues)


@pytest.mark.asyncio
async def test_reconcile_orders_detects_missing_locally() -> None:
    mock_rest = AsyncMock(spec=BybitRestClient)
    mock_rest.get_open_orders.return_value = [
        _open_order(order_id="ord-1", order_link_id="link-1"),
    ]
    mgr = OrderManager()

    reconciler = Reconciler(rest_client=mock_rest, order_manager=mgr, category="linear")
    report = await reconciler.reconcile_orders()

    assert report.ok is False
    assert any(i.issue_type == "missing_locally" for i in report.issues)
    by_link = await mgr.get_by_link_id("link-1")
    assert by_link is not None
    assert by_link.order_id == "ord-1"


@pytest.mark.asyncio
async def test_reconcile_orders_detects_qty_mismatch() -> None:
    mock_rest = AsyncMock(spec=BybitRestClient)
    mock_rest.get_open_orders.return_value = [
        _open_order(order_id="ord-1", order_link_id="link-1", qty=Decimal("0.2")),
    ]
    mgr = OrderManager()
    await mgr.register_intent(
        OrderIntent(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            qty=Decimal("0.1"),
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.GTC,
            reduce_only=False,
            price=Decimal("60000"),
            order_link_id="link-1",
            purpose=IntentPurpose.ENTRY,
            created_at=datetime.now(UTC),
            metadata={},
        )
    )
    await mgr.ack_exchange_order(order_link_id="link-1", order_id="ord-1", updated_at=datetime.now(UTC))

    reconciler = Reconciler(rest_client=mock_rest, order_manager=mgr, category="linear")
    report = await reconciler.reconcile_orders()

    assert report.ok is False
    assert any(i.issue_type == "qty_mismatch" for i in report.issues)


@pytest.mark.asyncio
async def test_reconcile_positions_detects_missing_reduce_only() -> None:
    mock_rest = AsyncMock(spec=BybitRestClient)
    mock_rest.get_positions.return_value = [_position(symbol="BTCUSDT", size=Decimal("0.1"))]
    mgr = OrderManager()

    reconciler = Reconciler(rest_client=mock_rest, order_manager=mgr, category="linear")
    report = await reconciler.reconcile_positions()

    assert report.ok is False
    issue = next(i for i in report.issues if i.issue_type == "missing_reduce_only_exit")
    assert issue.symbol == "BTCUSDT"
    assert issue.position_size == Decimal("0.1")
    assert issue.position_side == "Buy"


@pytest.mark.asyncio
async def test_reconcile_orders_symbol_scoped_when_symbols_configured() -> None:
    """Reconciler with symbols uses symbol-scoped get_open_orders per symbol."""
    mock_rest = AsyncMock(spec=BybitRestClient)
    mock_rest.get_open_orders = AsyncMock(return_value=[])
    mgr = OrderManager()

    reconciler = Reconciler(
        rest_client=mock_rest,
        order_manager=mgr,
        category="linear",
        symbols=["BTCUSDT", "ETHUSDT"],
    )
    await reconciler.reconcile_orders()

    assert mock_rest.get_open_orders.await_count == 2
    calls = mock_rest.get_open_orders.await_args_list
    symbols_seen = {c[1].get("symbol") for c in calls}
    assert symbols_seen == {"BTCUSDT", "ETHUSDT"}


@pytest.mark.asyncio
async def test_reconcile_positions_symbol_scoped_when_symbols_configured() -> None:
    """Reconciler with symbols uses symbol-scoped get_positions per symbol."""
    mock_rest = AsyncMock(spec=BybitRestClient)
    mock_rest.get_positions = AsyncMock(return_value=[])
    mgr = OrderManager()

    reconciler = Reconciler(
        rest_client=mock_rest,
        order_manager=mgr,
        category="linear",
        symbols=["BTCUSDT", "ETHUSDT"],
    )
    await reconciler.reconcile_positions()

    assert mock_rest.get_positions.await_count == 2
    calls = mock_rest.get_positions.await_args_list
    symbols_seen = {c[1].get("symbol") for c in calls}
    assert symbols_seen == {"BTCUSDT", "ETHUSDT"}


@pytest.mark.asyncio
async def test_reconcile_positions_ok_when_reduce_only_has_exit() -> None:
    mock_rest = AsyncMock(spec=BybitRestClient)
    mock_rest.get_positions.return_value = [_position(symbol="BTCUSDT", size=Decimal("0.1"))]
    mock_rest.get_open_orders.return_value = [_open_order(order_link_id="link-exit", symbol="BTCUSDT", reduce_only=True)]
    mgr = OrderManager()
    await mgr.register_intent(
        OrderIntent(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            qty=Decimal("0.1"),
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
            reduce_only=True,
            price=None,
            order_link_id="link-exit",
            purpose=IntentPurpose.EXIT,
            created_at=datetime.now(UTC),
            metadata={},
        )
    )
    await mgr.ack_exchange_order(order_link_id="link-exit", order_id="ord-exit", updated_at=datetime.now(UTC))

    reconciler = Reconciler(rest_client=mock_rest, order_manager=mgr, category="linear")
    report = await reconciler.reconcile_positions()

    assert report.ok is True


@pytest.mark.asyncio
async def test_reconcile_positions_requires_exchange_presence_for_protective_exit() -> None:
    """Protective exit must exist on exchange; local-only is insufficient."""
    mock_rest = AsyncMock(spec=BybitRestClient)
    mock_rest.get_positions.return_value = [_position(symbol="BTCUSDT", size=Decimal("0.1"))]
    mock_rest.get_open_orders.return_value = []
    mgr = OrderManager()
    await mgr.register_intent(
        OrderIntent(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            qty=Decimal("0.1"),
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
            reduce_only=True,
            price=None,
            order_link_id="link-exit",
            purpose=IntentPurpose.EXIT,
            created_at=datetime.now(UTC),
            metadata={},
        )
    )

    reconciler = Reconciler(rest_client=mock_rest, order_manager=mgr, category="linear")
    report = await reconciler.reconcile_positions()

    assert report.ok is False
    assert any(i.issue_type == "missing_reduce_only_exit" for i in report.issues)
