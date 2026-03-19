from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from trading.util.types import OrderSide, OrderStatus, OrderType, TimeInForce


class NormalizedBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class NormalizedTicker(NormalizedBaseModel):
    symbol: str
    mark_price: Decimal | None = None
    index_price: Decimal | None = None
    last_price: Decimal | None = None
    bid_price: Decimal | None = None
    ask_price: Decimal | None = None
    ts_exchange_ms: int
    ts_event_utc: datetime


class NormalizedTrade(NormalizedBaseModel):
    symbol: str
    trade_id: str | None = None
    side: OrderSide
    price: Decimal
    size: Decimal
    ts_exchange_ms: int
    ts_event_utc: datetime


class NormalizedKline(NormalizedBaseModel):
    symbol: str
    interval: str
    start_ms: int
    end_ms: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    turnover: Decimal
    confirmed: bool = Field(alias="confirm")
    ts_event_utc: datetime


class NormalizedOrderUpdate(NormalizedBaseModel):
    order_id: str | None = None
    order_link_id: str | None = None
    symbol: str | None = None
    side: OrderSide | None = None
    order_type: OrderType | None = None
    status: OrderStatus | None = None
    tif: TimeInForce | None = None
    qty: Decimal | None = None
    leaves_qty: Decimal | None = None
    price: Decimal | None = None
    avg_price: Decimal | None = None
    reduce_only: bool | None = None
    update_time_ms: int | None = None
    ts_event_utc: datetime
    raw: dict[str, Any]


class NormalizedExecution(NormalizedBaseModel):
    exec_id: str | None = None
    order_id: str | None = None
    order_link_id: str | None = None
    symbol: str | None = None
    side: OrderSide | None = None
    exec_price: Decimal | None = None
    exec_qty: Decimal | None = None
    exec_fee: Decimal | None = None
    exec_time_ms: int | None = None
    is_maker: bool | None = None
    ts_event_utc: datetime
    raw: dict[str, Any]


class NormalizedWalletUpdate(NormalizedBaseModel):
    account_type: str | None = None
    total_equity: Decimal | None = None
    total_available_balance: Decimal | None = None
    ts_event_utc: datetime
    raw: dict[str, Any]


class NormalizedPositionUpdate(NormalizedBaseModel):
    symbol: str | None = None
    side: str | None = None
    size: Decimal | None = None
    avg_price: Decimal | None = None
    mark_price: Decimal | None = None
    leverage: Decimal | None = None
    unrealised_pnl: Decimal | None = None
    liq_price: Decimal | None = None
    update_time_ms: int | None = None
    ts_event_utc: datetime
    raw: dict[str, Any]


NormalizedEvent = (
    NormalizedTicker
    | NormalizedTrade
    | NormalizedKline
    | NormalizedOrderUpdate
    | NormalizedExecution
    | NormalizedWalletUpdate
    | NormalizedPositionUpdate
)


def normalize_public_message(payload: dict[str, Any]) -> list[NormalizedEvent]:
    topic = payload.get("topic")
    if not isinstance(topic, str):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []

    if topic.startswith("tickers."):
        return [_normalize_ticker(item) for item in data if isinstance(item, dict)]
    if topic.startswith("publicTrade."):
        return [_normalize_trade(item) for item in data if isinstance(item, dict)]
    if topic.startswith("kline."):
        interval = topic.split(".")[1] if len(topic.split(".")) > 1 else ""
        return [_normalize_kline(item, interval) for item in data if isinstance(item, dict)]
    return []


def normalize_private_message(payload: dict[str, Any]) -> list[NormalizedEvent]:
    topic = payload.get("topic")
    if not isinstance(topic, str):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []

    if topic == "order":
        return [_normalize_order_update(item) for item in data if isinstance(item, dict)]
    if topic == "execution":
        return [_normalize_execution(item) for item in data if isinstance(item, dict)]
    if topic == "wallet":
        return [_normalize_wallet(item) for item in data if isinstance(item, dict)]
    if topic == "position":
        return [_normalize_position(item) for item in data if isinstance(item, dict)]
    return []


def _normalize_ticker(raw: dict[str, Any]) -> NormalizedTicker:
    ts_ms = _coerce_int(raw.get("ts"), fallback=0)
    return NormalizedTicker(
        symbol=str(raw.get("symbol", "")),
        mark_price=_coerce_decimal_optional(raw.get("markPrice")),
        index_price=_coerce_decimal_optional(raw.get("indexPrice")),
        last_price=_coerce_decimal_optional(raw.get("lastPrice")),
        bid_price=_coerce_decimal_optional(raw.get("bid1Price")),
        ask_price=_coerce_decimal_optional(raw.get("ask1Price")),
        ts_exchange_ms=ts_ms,
        ts_event_utc=_ms_to_utc(ts_ms),
    )


def _normalize_trade(raw: dict[str, Any]) -> NormalizedTrade:
    ts_ms = _coerce_int(raw.get("T"), fallback=0)
    side_raw = str(raw.get("S", "Buy"))
    return NormalizedTrade(
        symbol=str(raw.get("s", "")),
        trade_id=str(raw.get("i")) if raw.get("i") is not None else None,
        side=OrderSide.BUY if side_raw == "Buy" else OrderSide.SELL,
        price=_coerce_decimal(raw.get("p"), default=Decimal("0")),
        size=_coerce_decimal(raw.get("v"), default=Decimal("0")),
        ts_exchange_ms=ts_ms,
        ts_event_utc=_ms_to_utc(ts_ms),
    )


def _normalize_kline(raw: dict[str, Any], interval_hint: str) -> NormalizedKline:
    start_ms = _coerce_int(raw.get("start"), fallback=0)
    end_ms = _coerce_int(raw.get("end"), fallback=0)
    return NormalizedKline(
        symbol=str(raw.get("symbol", "")),
        interval=str(raw.get("interval", interval_hint)),
        start_ms=start_ms,
        end_ms=end_ms,
        open=_coerce_decimal(raw.get("open"), default=Decimal("0")),
        high=_coerce_decimal(raw.get("high"), default=Decimal("0")),
        low=_coerce_decimal(raw.get("low"), default=Decimal("0")),
        close=_coerce_decimal(raw.get("close"), default=Decimal("0")),
        volume=_coerce_decimal(raw.get("volume"), default=Decimal("0")),
        turnover=_coerce_decimal(raw.get("turnover"), default=Decimal("0")),
        confirm=bool(raw.get("confirm", False)),
        ts_event_utc=_ms_to_utc(end_ms if end_ms > 0 else start_ms),
    )


def _normalize_order_update(raw: dict[str, Any]) -> NormalizedOrderUpdate:
    side = raw.get("side")
    order_type = raw.get("orderType")
    tif = raw.get("timeInForce")
    status = raw.get("orderStatus")
    ts_ms = _coerce_int_optional(raw.get("updatedTime"))
    return NormalizedOrderUpdate(
        order_id=_string_optional(raw.get("orderId")),
        order_link_id=_string_optional(raw.get("orderLinkId")),
        symbol=_string_optional(raw.get("symbol")),
        side=OrderSide(side) if isinstance(side, str) and side in {"Buy", "Sell"} else None,
        order_type=OrderType(order_type)
        if isinstance(order_type, str) and order_type in {"Market", "Limit"}
        else None,
        status=OrderStatus(status)
        if isinstance(status, str) and status in {"New", "PartiallyFilled", "Filled", "Cancelled", "Rejected"}
        else None,
        tif=TimeInForce(tif) if isinstance(tif, str) and tif in {"GTC", "IOC", "FOK", "PostOnly"} else None,
        qty=_coerce_decimal_optional(raw.get("qty")),
        leaves_qty=_coerce_decimal_optional(raw.get("leavesQty")),
        price=_coerce_decimal_optional(raw.get("price")),
        avg_price=_coerce_decimal_optional(raw.get("avgPrice")),
        reduce_only=_coerce_bool_optional(raw.get("reduceOnly")),
        update_time_ms=ts_ms,
        ts_event_utc=_ms_to_utc(ts_ms or 0),
        raw=raw,
    )


def _normalize_execution(raw: dict[str, Any]) -> NormalizedExecution:
    side = raw.get("side")
    exec_time_ms = _coerce_int_optional(raw.get("execTime"))
    is_maker_raw = raw.get("isMaker")
    return NormalizedExecution(
        exec_id=_string_optional(raw.get("execId")),
        order_id=_string_optional(raw.get("orderId")),
        order_link_id=_string_optional(raw.get("orderLinkId")),
        symbol=_string_optional(raw.get("symbol")),
        side=OrderSide(side) if isinstance(side, str) and side in {"Buy", "Sell"} else None,
        exec_price=_coerce_decimal_optional(raw.get("execPrice")),
        exec_qty=_coerce_decimal_optional(raw.get("execQty")),
        exec_fee=_coerce_decimal_optional(raw.get("execFee")),
        exec_time_ms=exec_time_ms,
        is_maker=bool(is_maker_raw) if isinstance(is_maker_raw, bool) else None,
        ts_event_utc=_ms_to_utc(exec_time_ms or 0),
        raw=raw,
    )


def _normalize_wallet(raw: dict[str, Any]) -> NormalizedWalletUpdate:
    ts_ms = _coerce_int_optional(raw.get("creationTime"))
    return NormalizedWalletUpdate(
        account_type=_string_optional(raw.get("accountType")),
        total_equity=_coerce_decimal_optional(raw.get("totalEquity")),
        total_available_balance=_coerce_decimal_optional(raw.get("totalAvailableBalance")),
        ts_event_utc=_ms_to_utc(ts_ms or 0),
        raw=raw,
    )


def _normalize_position(raw: dict[str, Any]) -> NormalizedPositionUpdate:
    ts_ms = _coerce_int_optional(raw.get("updatedTime"))
    return NormalizedPositionUpdate(
        symbol=_string_optional(raw.get("symbol")),
        side=_string_optional(raw.get("side")),
        size=_coerce_decimal_optional(raw.get("size")),
        avg_price=_coerce_decimal_optional(raw.get("avgPrice")),
        mark_price=_coerce_decimal_optional(raw.get("markPrice")),
        leverage=_coerce_decimal_optional(raw.get("leverage")),
        unrealised_pnl=_coerce_decimal_optional(raw.get("unrealisedPnl")),
        liq_price=_coerce_decimal_optional(raw.get("liqPrice")),
        update_time_ms=ts_ms,
        ts_event_utc=_ms_to_utc(ts_ms or 0),
        raw=raw,
    )


def _ms_to_utc(ms: int) -> datetime:
    if ms <= 0:
        return datetime.now(UTC)
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def _coerce_decimal(value: Any, *, default: Decimal) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:
        return default


def _coerce_decimal_optional(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _coerce_int(value: Any, *, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def _coerce_int_optional(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _coerce_bool_optional(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
    return None


def _string_optional(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
