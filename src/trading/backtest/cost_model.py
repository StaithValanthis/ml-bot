"""Cost model for backtest: maker/taker fees, spread, slippage."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class CostModelConfig:
    maker_fee_bps: Decimal = Decimal("2")
    taker_fee_bps: Decimal = Decimal("6")
    spread_bps: Decimal = Decimal("2")
    slippage_bps: Decimal = Decimal("5")
    slippage_per_unit_usdt: Decimal = Decimal("0")


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    fee_usdt: Decimal
    slippage_usdt: Decimal
    spread_usdt: Decimal
    total_usdt: Decimal
    is_maker: bool


class CostModel:
    """
    Typed cost model for simulated execution.

    Applies maker/taker fees, spread (half per side), and slippage.
    """

    def __init__(self, config: CostModelConfig | None = None) -> None:
        self._cfg = config or CostModelConfig()

    def compute(
        self,
        *,
        notional_usdt: Decimal,
        is_maker: bool,
        qty: Decimal,
        price: Decimal,
    ) -> CostBreakdown:
        fee_bps = self._cfg.maker_fee_bps if is_maker else self._cfg.taker_fee_bps
        fee_usdt = (notional_usdt * fee_bps) / Decimal("10000")

        slippage_bps = self._cfg.slippage_bps
        slippage_usdt = (notional_usdt * slippage_bps) / Decimal("10000")
        if self._cfg.slippage_per_unit_usdt > 0:
            slippage_usdt += qty * self._cfg.slippage_per_unit_usdt

        spread_half_bps = self._cfg.spread_bps / 2
        spread_usdt = (notional_usdt * spread_half_bps) / Decimal("10000")

        total = fee_usdt + slippage_usdt + spread_usdt
        return CostBreakdown(
            fee_usdt=fee_usdt,
            slippage_usdt=slippage_usdt,
            spread_usdt=spread_usdt,
            total_usdt=total,
            is_maker=is_maker,
        )

    def compute_fee_only(self, notional_usdt: Decimal, is_maker: bool) -> Decimal:
        fee_bps = self._cfg.maker_fee_bps if is_maker else self._cfg.taker_fee_bps
        return (notional_usdt * fee_bps) / Decimal("10000")
