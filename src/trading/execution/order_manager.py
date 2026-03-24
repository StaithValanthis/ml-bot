from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from trading.execution.order_intent import IntentPurpose, OrderIntent
from trading.util.types import OrderStatus


@dataclass(slots=True)
class ManagedOrder:
    order_id: str | None
    order_link_id: str
    symbol: str
    status: OrderStatus
    qty: Decimal
    filled_qty: Decimal
    avg_price: Decimal | None
    reduce_only: bool
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any]


class OrderManager:
    """Tracks local order lifecycle and supports idempotent link-id lookup."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._by_link_id: dict[str, ManagedOrder] = {}
        self._by_order_id: dict[str, ManagedOrder] = {}

    async def register_intent(self, intent: OrderIntent) -> ManagedOrder:
        async with self._lock:
            existing = self._by_link_id.get(intent.order_link_id)
            if existing is not None:
                return existing
            meta = dict(intent.metadata)
            if intent.purpose == IntentPurpose.ENTRY:
                meta.setdefault("entry_side", intent.side.value)
            managed = ManagedOrder(
                order_id=None,
                order_link_id=intent.order_link_id,
                symbol=intent.symbol,
                status=OrderStatus.NEW,
                qty=intent.qty,
                filled_qty=Decimal("0"),
                avg_price=None,
                reduce_only=intent.reduce_only,
                created_at=intent.created_at,
                updated_at=intent.created_at,
                metadata=meta,
            )
            self._by_link_id[intent.order_link_id] = managed
            return managed

    async def ack_exchange_order(self, *, order_link_id: str, order_id: str, updated_at: datetime) -> None:
        async with self._lock:
            order = self._by_link_id.get(order_link_id)
            if order is None:
                return
            order.order_id = order_id
            order.updated_at = updated_at
            self._by_order_id[order_id] = order

    async def apply_order_update(
        self,
        *,
        order_id: str | None,
        order_link_id: str | None,
        status: OrderStatus | None,
        filled_qty: Decimal | None,
        avg_price: Decimal | None,
        updated_at: datetime,
    ) -> None:
        async with self._lock:
            order = None
            if order_id is not None:
                order = self._by_order_id.get(order_id)
            if order is None and order_link_id is not None:
                order = self._by_link_id.get(order_link_id)
            if order is None:
                return
            if status is not None:
                order.status = status
            if filled_qty is not None:
                order.filled_qty = filled_qty
            if avg_price is not None:
                order.avg_price = avg_price
            order.updated_at = updated_at
            if order_id is not None:
                order.order_id = order_id
                self._by_order_id[order_id] = order

    async def mark_closed_missing_on_exchange(
        self, *, order_link_id: str, updated_at: datetime | None = None
    ) -> bool:
        """
        Mark a locally tracked open order as terminal because it is no longer on exchange.

        Local state convergence only. Does not cancel on exchange.
        Status set to CANCELLED (terminal); metadata stores reconcile_terminal_reason.
        Returns True if order was updated, False if not found or already terminal.
        """
        from datetime import datetime, timezone

        when = updated_at or datetime.now(timezone.utc)
        async with self._lock:
            order = self._by_link_id.get(order_link_id)
            if order is None:
                return False
            if order.status not in {OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED}:
                return False
            order.status = OrderStatus.CANCELLED
            order.updated_at = when
            meta = dict(order.metadata)
            meta["reconcile_terminal_reason"] = "closed_missing_on_exchange"
            order.metadata = meta
            return True

    async def get_open_orders(self, symbol: str | None = None) -> list[ManagedOrder]:
        async with self._lock:
            results = [
                order
                for order in self._by_link_id.values()
                if order.status in {OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED}
            ]
            if symbol is not None:
                results = [order for order in results if order.symbol == symbol]
            return list(results)

    async def get_by_link_id(self, order_link_id: str) -> ManagedOrder | None:
        async with self._lock:
            return self._by_link_id.get(order_link_id)

    async def upsert_from_exchange_snapshot(
        self,
        *,
        order_id: str,
        order_link_id: str,
        symbol: str,
        status: OrderStatus,
        qty: Decimal,
        filled_qty: Decimal,
        avg_price: Decimal | None,
        reduce_only: bool,
        updated_at: datetime,
    ) -> None:
        """
        Ensure local order state reflects authoritative exchange snapshot.

        This method is used by reconciliation loops when local process missed
        private stream events or restarted.
        """
        async with self._lock:
            existing = self._by_link_id.get(order_link_id) or self._by_order_id.get(order_id)
            if existing is None:
                managed = ManagedOrder(
                    order_id=order_id,
                    order_link_id=order_link_id,
                    symbol=symbol,
                    status=status,
                    qty=qty,
                    filled_qty=filled_qty,
                    avg_price=avg_price,
                    reduce_only=reduce_only,
                    created_at=updated_at,
                    updated_at=updated_at,
                    metadata={"source": "reconciler_snapshot"},
                )
                self._by_link_id[order_link_id] = managed
                self._by_order_id[order_id] = managed
                return

            existing.order_id = order_id
            existing.order_link_id = order_link_id
            existing.symbol = symbol
            existing.status = status
            existing.qty = qty
            existing.filled_qty = filled_qty
            existing.avg_price = avg_price
            existing.reduce_only = reduce_only
            existing.updated_at = updated_at
            self._by_link_id[order_link_id] = existing
            self._by_order_id[order_id] = existing
