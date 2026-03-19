from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from trading.exchange.bybit_rest import BybitRestClient
from trading.execution.order_manager import OrderManager


@dataclass(slots=True, frozen=True)
class ReconciliationIssue:
    issue_type: str
    symbol: str | None
    details: str
    order_link_id: str | None = None
    order_id: str | None = None


@dataclass(slots=True, frozen=True)
class ReconciliationReport:
    ok: bool
    issues: list[ReconciliationIssue]


class Reconciler:
    """
    Compares local in-memory order tracking with authoritative exchange snapshots.

    Uses symbol-scoped REST requests for DEMO compatibility (avoids unscoped
    category-only calls that can fail on demo API).
    """

    def __init__(
        self,
        *,
        rest_client: BybitRestClient,
        order_manager: OrderManager,
        category: str,
        symbols: list[str] | None = None,
    ) -> None:
        self._rest_client = rest_client
        self._order_manager = order_manager
        self._category = category
        self._symbols = symbols or []

    async def reconcile_orders(self, symbol: str | None = None) -> ReconciliationReport:
        if symbol is not None:
            local_open = await self._order_manager.get_open_orders(symbol)
            exchange_open = await self._rest_client.get_open_orders(category=self._category, symbol=symbol)
        elif self._symbols:
            local_open = await self._order_manager.get_open_orders(None)
            exchange_open: list = []
            for s in self._symbols:
                orders = await self._rest_client.get_open_orders(category=self._category, symbol=s)
                exchange_open.extend(orders)
        else:
            local_open = await self._order_manager.get_open_orders(None)
            exchange_open = await self._rest_client.get_open_orders(category=self._category, symbol=None)

        issues: list[ReconciliationIssue] = []
        local_by_link = {order.order_link_id: order for order in local_open}
        exchange_by_link = {order.order_link_id: order for order in exchange_open if order.order_link_id}

        for link_id, local in local_by_link.items():
            remote = exchange_by_link.get(link_id)
            if remote is None:
                issues.append(
                    ReconciliationIssue(
                        issue_type="missing_on_exchange",
                        symbol=local.symbol,
                        details=f"Local open order not found remotely: link_id={link_id}",
                        order_link_id=link_id,
                        order_id=local.order_id,
                    )
                )
                continue
            qty_diff = abs(local.qty - remote.qty)
            if qty_diff > Decimal("0"):
                issues.append(
                    ReconciliationIssue(
                        issue_type="qty_mismatch",
                        symbol=local.symbol,
                        details=f"qty mismatch link_id={link_id} local={local.qty} remote={remote.qty}",
                        order_link_id=link_id,
                        order_id=remote.order_id,
                    )
                )
            await self._order_manager.upsert_from_exchange_snapshot(
                order_id=remote.order_id,
                order_link_id=remote.order_link_id,
                symbol=remote.symbol,
                status=remote.order_status,
                qty=remote.qty,
                filled_qty=remote.cum_exec_qty,
                avg_price=remote.price,
                reduce_only=remote.reduce_only,
                updated_at=_parse_exchange_ms(remote.updated_time),
            )

        for link_id, remote in exchange_by_link.items():
            if link_id not in local_by_link:
                issues.append(
                    ReconciliationIssue(
                        issue_type="missing_locally",
                        symbol=remote.symbol,
                        details=f"Exchange open order not tracked locally: link_id={link_id}",
                        order_link_id=link_id,
                        order_id=remote.order_id,
                    )
                )
                await self._order_manager.upsert_from_exchange_snapshot(
                    order_id=remote.order_id,
                    order_link_id=remote.order_link_id,
                    symbol=remote.symbol,
                    status=remote.order_status,
                    qty=remote.qty,
                    filled_qty=remote.cum_exec_qty,
                    avg_price=remote.price,
                    reduce_only=remote.reduce_only,
                    updated_at=_parse_exchange_ms(remote.updated_time),
                )
        return ReconciliationReport(ok=len(issues) == 0, issues=issues)

    async def reconcile_positions(self, symbol: str | None = None) -> ReconciliationReport:
        # v1 scaffold: verify protective reduce-only exits exist for non-flat positions.
        if symbol is not None:
            positions = await self._rest_client.get_positions(category=self._category, symbol=symbol)
        elif self._symbols:
            positions = []
            for s in self._symbols:
                pos_list = await self._rest_client.get_positions(category=self._category, symbol=s)
                positions.extend(pos_list)
        else:
            positions = await self._rest_client.get_positions(category=self._category, symbol=None)
        local_open = await self._order_manager.get_open_orders(None)
        reduce_only_open = [order for order in local_open if order.reduce_only]
        issues: list[ReconciliationIssue] = []

        for position in positions:
            if position.size <= 0:
                continue
            has_protective_exit = any(order.symbol == position.symbol for order in reduce_only_open)
            if not has_protective_exit:
                issues.append(
                    ReconciliationIssue(
                        issue_type="missing_reduce_only_exit",
                        symbol=position.symbol,
                        details="Non-flat position has no local tracked reduce-only exit order.",
                    )
                )
        return ReconciliationReport(ok=len(issues) == 0, issues=issues)


def _parse_exchange_ms(raw_ms: str) -> datetime:
    # Parsing faults should never crash reconciliation loop.
    try:
        return datetime.fromtimestamp(int(raw_ms) / 1000, tz=UTC)
    except Exception:
        return datetime.now(UTC)
