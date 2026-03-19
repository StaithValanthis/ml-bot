from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum


class RuntimeMode(str, Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    DEMO = "demo"
    LIVE = "live"


class ExchangeType(str, Enum):
    BYBIT = "bybit"


class OrderSide(str, Enum):
    BUY = "Buy"
    SELL = "Sell"


class OrderType(str, Enum):
    MARKET = "Market"
    LIMIT = "Limit"


class TimeInForce(str, Enum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"
    POST_ONLY = "PostOnly"


class OrderStatus(str, Enum):
    NEW = "New"
    PARTIALLY_FILLED = "PartiallyFilled"
    FILLED = "Filled"
    CANCELLED = "Cancelled"
    REJECTED = "Rejected"


class PositionSide(str, Enum):
    LONG = "Long"
    SHORT = "Short"
    FLAT = "Flat"


@dataclass(slots=True, frozen=True)
class MarketSymbol:
    symbol: str
    qty_step: Decimal
    min_qty: Decimal
    price_tick: Decimal
    max_leverage: Decimal


@dataclass(slots=True, frozen=True)
class OHLCVBar:
    symbol: str
    timeframe: str
    open_time: datetime
    close_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    turnover: Decimal
    confirmed: bool


@dataclass(slots=True, frozen=True)
class PositionSnapshot:
    symbol: str
    side: PositionSide
    qty: Decimal
    entry_price: Decimal
    mark_price: Decimal
    leverage: Decimal
    unrealized_pnl: Decimal
    timestamp: datetime
