from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN

from trading.util.types import MarketSymbol


@dataclass(slots=True, frozen=True)
class SizingInputs:
    equity_usdt: Decimal
    confidence: Decimal
    volatility_bps: Decimal
    reference_price: Decimal
    max_leverage: Decimal


class VolatilityAwareSizer:
    """
    Volatility-aware sizing:
    base_risk_fraction scales with confidence and inversely with volatility.
    """

    def __init__(self, *, base_risk_fraction: Decimal = Decimal("0.01"), min_confidence: Decimal = Decimal("0.2")) -> None:
        self._base_risk_fraction = base_risk_fraction
        self._min_confidence = min_confidence

    def size_qty(self, inputs: SizingInputs, symbol_info: MarketSymbol) -> Decimal:
        if inputs.reference_price <= 0 or inputs.equity_usdt <= 0:
            return Decimal("0")
        if inputs.confidence < self._min_confidence:
            return Decimal("0")
        vol_factor = Decimal("1") / max(Decimal("1"), inputs.volatility_bps / Decimal("100"))
        confidence_factor = max(Decimal("0.1"), min(Decimal("1"), inputs.confidence))
        notional_budget = (
            inputs.equity_usdt
            * self._base_risk_fraction
            * confidence_factor
            * vol_factor
            * min(inputs.max_leverage, symbol_info.max_leverage)
        )
        raw_qty = notional_budget / inputs.reference_price
        stepped_qty = self._round_down_to_step(raw_qty, symbol_info.qty_step)
        if stepped_qty < symbol_info.min_qty:
            return Decimal("0")
        return stepped_qty

    @staticmethod
    def _round_down_to_step(value: Decimal, step: Decimal) -> Decimal:
        if step <= 0:
            return value
        units = (value / step).to_integral_value(rounding=ROUND_DOWN)
        return units * step
