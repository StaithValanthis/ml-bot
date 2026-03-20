from __future__ import annotations

from decimal import Decimal
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from trading.util.types import OrderSide, OrderStatus, OrderType, TimeInForce


def _empty_str_to_decimal(v: object) -> Decimal:
    """Convert empty string or None to Decimal('0'); otherwise parse to Decimal."""
    if v is None or v == "":
        return Decimal("0")
    return Decimal(str(v))


class BybitBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


T = TypeVar("T")


class BybitEnvelope(BybitBaseModel, Generic[T]):
    ret_code: int = Field(alias="retCode")
    ret_msg: str = Field(alias="retMsg")
    result: T
    time_ms: int = Field(alias="time")


class BybitListResult(BybitBaseModel, Generic[T]):
    category: str | None = None
    symbol: str | None = None
    items: list[T] = Field(default_factory=list, alias="list")
    next_page_cursor: str | None = Field(default=None, alias="nextPageCursor")


class ServerTimeResult(BybitBaseModel):
    time_second: str = Field(alias="timeSecond")
    time_nano: str = Field(alias="timeNano")


class TickerItem(BybitBaseModel):
    """Bybit v5 market ticker; bid/ask/last for drill reference price fallback."""

    symbol: str
    last_price: str = Field(alias="lastPrice")
    bid1_price: str = Field(alias="bid1Price")
    ask1_price: str = Field(alias="ask1Price")


class FeeRateItem(BybitBaseModel):
    symbol: str
    taker_fee_rate: Decimal = Field(alias="takerFeeRate")
    maker_fee_rate: Decimal = Field(alias="makerFeeRate")


class OpenOrderItem(BybitBaseModel):
    order_id: str = Field(alias="orderId")
    order_link_id: str = Field(alias="orderLinkId")
    symbol: str
    side: OrderSide
    order_type: OrderType = Field(alias="orderType")
    order_status: OrderStatus = Field(alias="orderStatus")
    price: Decimal
    qty: Decimal
    cum_exec_qty: Decimal = Field(alias="cumExecQty")
    cum_exec_value: Decimal = Field(alias="cumExecValue")
    reduce_only: bool = Field(alias="reduceOnly")
    time_in_force: TimeInForce = Field(alias="timeInForce")
    created_time: str = Field(alias="createdTime")
    updated_time: str = Field(alias="updatedTime")


class PositionItem(BybitBaseModel):
    """Bybit position; tolerates empty-string payloads for flat/empty positions."""

    symbol: str
    side: str
    size: Decimal
    avg_price: Decimal = Field(alias="avgPrice")
    mark_price: Decimal = Field(alias="markPrice")
    position_value: Decimal = Field(alias="positionValue")
    leverage: Decimal
    liq_price: str = Field(alias="liqPrice")
    unrealised_pnl: Decimal = Field(alias="unrealisedPnl")
    updated_time: str = Field(alias="updatedTime")

    @field_validator(
        "size", "avg_price", "mark_price", "position_value", "leverage", "unrealised_pnl",
        mode="before",
    )
    @classmethod
    def _coerce_empty_to_zero(cls, v: object) -> Decimal:
        return _empty_str_to_decimal(v)

    @field_validator("liq_price", mode="before")
    @classmethod
    def _coerce_liq_price(cls, v: object) -> str:
        if v is None:
            return ""
        return str(v)


class WalletCoinItem(BybitBaseModel):
    coin: str
    wallet_balance: Decimal = Field(alias="walletBalance")
    equity: Decimal
    usd_value: Decimal = Field(alias="usdValue")
    unrealised_pnl: Decimal = Field(alias="unrealisedPnl")


class WalletBalanceItem(BybitBaseModel):
    account_type: str = Field(alias="accountType")
    total_equity: Decimal = Field(alias="totalEquity")
    total_wallet_balance: Decimal = Field(alias="totalWalletBalance")
    total_available_balance: Decimal = Field(alias="totalAvailableBalance")
    coin: list[WalletCoinItem] = Field(default_factory=list)


class OrderAck(BybitBaseModel):
    order_id: str = Field(alias="orderId")
    order_link_id: str = Field(alias="orderLinkId")


class KlineItem(BybitBaseModel):
    start_time_ms: int
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: Decimal
    turnover: Decimal

    @classmethod
    def from_raw(cls, raw: list[str]) -> KlineItem:
        if len(raw) < 7:
            raise ValueError(f"Unexpected kline row size: {raw}")
        return cls(
            start_time_ms=int(raw[0]),
            open_price=Decimal(raw[1]),
            high_price=Decimal(raw[2]),
            low_price=Decimal(raw[3]),
            close_price=Decimal(raw[4]),
            volume=Decimal(raw[5]),
            turnover=Decimal(raw[6]),
        )


class FundingHistoryItem(BybitBaseModel):
    symbol: str
    funding_rate: Decimal = Field(alias="fundingRate")
    funding_rate_timestamp: str = Field(alias="fundingRateTimestamp")


class OpenInterestItem(BybitBaseModel):
    symbol: str
    open_interest: Decimal = Field(alias="openInterest")
    timestamp: str


class PlaceOrderRequest(BybitBaseModel):
    category: str = "linear"
    symbol: str
    side: OrderSide
    order_type: OrderType = Field(alias="orderType")
    qty: Decimal
    price: Decimal | None = None
    time_in_force: TimeInForce = Field(alias="timeInForce")
    order_link_id: str = Field(alias="orderLinkId")
    reduce_only: bool = Field(default=False, alias="reduceOnly")
    close_on_trigger: bool = Field(default=False, alias="closeOnTrigger")
    position_idx: int = Field(default=0, alias="positionIdx")


class AmendOrderRequest(BybitBaseModel):
    category: str = "linear"
    symbol: str
    order_id: str | None = Field(default=None, alias="orderId")
    order_link_id: str | None = Field(default=None, alias="orderLinkId")
    qty: Decimal | None = None
    price: Decimal | None = None


class CancelOrderRequest(BybitBaseModel):
    category: str = "linear"
    symbol: str
    order_id: str | None = Field(default=None, alias="orderId")
    order_link_id: str | None = Field(default=None, alias="orderLinkId")


