from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from trading.util.types import PositionSide


@dataclass(slots=True)
class PositionRiskView:
    symbol: str
    side: PositionSide
    qty: Decimal
    entry_price: Decimal
    mark_price: Decimal
    leverage: Decimal
    liquidation_price: Decimal | None

    @property
    def notional(self) -> Decimal:
        return abs(self.qty) * self.mark_price

    @property
    def distance_to_liq_bps(self) -> Decimal | None:
        if self.liquidation_price is None or self.mark_price <= 0:
            return None
        if self.side == PositionSide.LONG:
            return ((self.mark_price - self.liquidation_price) / self.mark_price) * Decimal("10000")
        if self.side == PositionSide.SHORT:
            return ((self.liquidation_price - self.mark_price) / self.mark_price) * Decimal("10000")
        return None


@dataclass(slots=True)
class PortfolioState:
    equity_usdt: Decimal
    available_balance_usdt: Decimal
    positions: dict[str, PositionRiskView] = field(default_factory=dict)
    realized_pnl_today_usdt: Decimal = Decimal("0")
    pnl_date: date | None = None
    safe_mode: bool = False

    def total_notional(self) -> Decimal:
        return sum((p.notional for p in self.positions.values()), start=Decimal("0"))

    def max_effective_leverage(self) -> Decimal:
        if self.equity_usdt <= 0:
            return Decimal("0")
        return self.total_notional() / self.equity_usdt

    def position_for(self, symbol: str) -> PositionRiskView | None:
        return self.positions.get(symbol)

    def update_realized_pnl(self, pnl_delta: Decimal, as_of: date) -> None:
        if self.pnl_date != as_of:
            self.realized_pnl_today_usdt = Decimal("0")
            self.pnl_date = as_of
        self.realized_pnl_today_usdt += pnl_delta
