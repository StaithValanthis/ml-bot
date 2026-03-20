"""Unit tests for market data normalizers."""

from __future__ import annotations

from decimal import Decimal

from trading.marketdata.normalizers import (
    normalize_private_message,
    normalize_public_message,
)
from trading.util.types import OrderSide, OrderStatus, OrderType


def test_normalize_public_ticker() -> None:
    payload = {
        "topic": "tickers.BTCUSDT",
        "data": [
            {
                "symbol": "BTCUSDT",
                "lastPrice": "60100.5",
                "markPrice": "60102",
                "ts": 1710000000000,
            }
        ],
    }
    events = normalize_public_message(payload)
    assert len(events) == 1
    ticker = events[0]
    assert ticker.symbol == "BTCUSDT"
    assert ticker.last_price == Decimal("60100.5")
    assert ticker.mark_price == Decimal("60102")
    assert ticker.ts_exchange_ms == 1710000000000


def test_normalize_public_trade() -> None:
    payload = {
        "topic": "publicTrade.BTCUSDT",
        "data": [
            {
                "s": "BTCUSDT",
                "S": "Buy",
                "p": "60050",
                "v": "0.1",
                "T": 1710000001000,
            }
        ],
    }
    events = normalize_public_message(payload)
    assert len(events) == 1
    trade = events[0]
    assert trade.symbol == "BTCUSDT"
    assert trade.side == OrderSide.BUY
    assert trade.price == Decimal("60050")
    assert trade.size == Decimal("0.1")
    assert trade.ts_exchange_ms == 1710000001000


def test_normalize_public_kline() -> None:
    payload = {
        "topic": "kline.5.BTCUSDT",
        "data": [
            {
                "symbol": "BTCUSDT",
                "interval": "5",
                "start": 1710000000000,
                "end": 1710000300000,
                "open": "60000",
                "high": "60200",
                "low": "59900",
                "close": "60100",
                "volume": "123.4",
                "turnover": "7412340",
                "confirm": True,
            }
        ],
    }
    events = normalize_public_message(payload)
    assert len(events) == 1
    kline = events[0]
    assert kline.symbol == "BTCUSDT"
    assert kline.interval == "5"
    assert kline.open == Decimal("60000")
    assert kline.close == Decimal("60100")
    assert kline.confirmed is True


def test_normalize_public_kline_symbol_from_topic_when_missing_in_data() -> None:
    """Bybit kline WS payload: symbol is in topic kline.{interval}.{symbol}, not in data items."""
    payload = {
        "topic": "kline.60.BTCUSDT",
        "data": [
            {
                "start": 1710000000000,
                "end": 1710003600000,
                "interval": "60",
                "open": "60000",
                "high": "60200",
                "low": "59900",
                "close": "60100",
                "volume": "123.4",
                "turnover": "7412340",
                "confirm": True,
            }
        ],
    }
    events = normalize_public_message(payload)
    assert len(events) == 1
    kline = events[0]
    assert kline.symbol == "BTCUSDT"
    assert kline.interval == "60"


def test_normalize_public_empty_topic_returns_empty() -> None:
    payload = {"topic": "unknown.xyz", "data": [{}]}
    events = normalize_public_message(payload)
    assert events == []


def test_normalize_public_missing_topic_returns_empty() -> None:
    payload = {"data": []}
    events = normalize_public_message(payload)
    assert events == []


def test_normalize_private_order_update() -> None:
    payload = {
        "topic": "order",
        "data": [
            {
                "orderId": "ord-123",
                "orderLinkId": "link-456",
                "symbol": "BTCUSDT",
                "side": "Buy",
                "orderType": "Limit",
                "orderStatus": "New",
                "timeInForce": "GTC",
                "qty": "0.1",
                "leavesQty": "0.1",
                "avgPrice": "0",
                "updatedTime": 1710000000000,
            }
        ],
    }
    events = normalize_private_message(payload)
    assert len(events) == 1
    order = events[0]
    assert order.order_id == "ord-123"
    assert order.order_link_id == "link-456"
    assert order.symbol == "BTCUSDT"
    assert order.side == OrderSide.BUY
    assert order.order_type == OrderType.LIMIT
    assert order.status == OrderStatus.NEW
    assert order.qty == Decimal("0.1")


def test_normalize_private_execution() -> None:
    payload = {
        "topic": "execution",
        "data": [
            {
                "execId": "exec-789",
                "orderId": "ord-123",
                "orderLinkId": "link-456",
                "symbol": "BTCUSDT",
                "side": "Buy",
                "execPrice": "60100",
                "execQty": "0.05",
                "execFee": "-0.15",
                "execTime": 1710000001000,
            }
        ],
    }
    events = normalize_private_message(payload)
    assert len(events) == 1
    exec_event = events[0]
    assert exec_event.exec_id == "exec-789"
    assert exec_event.order_id == "ord-123"
    assert exec_event.exec_price == Decimal("60100")
    assert exec_event.exec_qty == Decimal("0.05")
    assert exec_event.exec_fee == Decimal("-0.15")


def test_normalize_private_wallet() -> None:
    payload = {
        "topic": "wallet",
        "data": [
            {
                "accountType": "UNIFIED",
                "totalEquity": "10000.5",
                "totalAvailableBalance": "9500",
                "creationTime": 1710000000000,
            }
        ],
    }
    events = normalize_private_message(payload)
    assert len(events) == 1
    wallet = events[0]
    assert wallet.account_type == "UNIFIED"
    assert wallet.total_equity == Decimal("10000.5")
    assert wallet.total_available_balance == Decimal("9500")


def test_normalize_private_position() -> None:
    payload = {
        "topic": "position",
        "data": [
            {
                "symbol": "BTCUSDT",
                "side": "Buy",
                "size": "0.1",
                "avgPrice": "60000",
                "markPrice": "60100",
                "leverage": "10",
                "unrealisedPnl": "10.5",
                "liqPrice": "54000",
                "updatedTime": 1710000000000,
            }
        ],
    }
    events = normalize_private_message(payload)
    assert len(events) == 1
    pos = events[0]
    assert pos.symbol == "BTCUSDT"
    assert pos.side == "Buy"
    assert pos.size == Decimal("0.1")
    assert pos.avg_price == Decimal("60000")
    assert pos.leverage == Decimal("10")


def test_normalize_private_unknown_topic_returns_empty() -> None:
    payload = {"topic": "unknown", "data": [{}]}
    events = normalize_private_message(payload)
    assert events == []
