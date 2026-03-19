from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from trading.util.types import OrderSide, OrderType, TimeInForce


class IntentPurpose(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"


@dataclass(slots=True, frozen=True)
class OrderIntent:
    symbol: str
    side: OrderSide
    qty: Decimal
    order_type: OrderType
    time_in_force: TimeInForce
    reduce_only: bool
    price: Decimal | None
    order_link_id: str
    purpose: IntentPurpose
    created_at: datetime
    metadata: dict[str, Any]
