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

    Optional demo_min_notional_floor_usdt: when set (DEMO-only), if computed qty
    would be below min_qty, use this notional floor to meet exchange minimum.
    Capped at max_equity_fraction_for_floor of equity for safety.
    """

    def __init__(
        self,
        *,
        base_risk_fraction: Decimal = Decimal("0.01"),
        min_confidence: Decimal = Decimal("0.2"),
        demo_min_notional_floor_usdt: Decimal | None = None,
        max_equity_fraction_for_floor: Decimal = Decimal("0.1"),
    ) -> None:
        self._base_risk_fraction = base_risk_fraction
        self._min_confidence = min_confidence
        self._demo_min_notional_floor_usdt = demo_min_notional_floor_usdt
        self._max_equity_fraction_for_floor = max_equity_fraction_for_floor
        self._last_floor_applied: bool = False
        self._last_floor_details: dict[str, object] | None = None

    def size_qty(self, inputs: SizingInputs, symbol_info: MarketSymbol) -> Decimal:
        self._last_floor_applied = False
        self._last_floor_details = None
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
        if stepped_qty >= symbol_info.min_qty:
            return stepped_qty
        if self._demo_min_notional_floor_usdt is not None:
            effective_notional = max(notional_budget, self._demo_min_notional_floor_usdt)
            cap = inputs.equity_usdt * self._max_equity_fraction_for_floor
            effective_notional = min(effective_notional, cap)
            raw_qty_floor = effective_notional / inputs.reference_price
            stepped_qty_floor = self._round_down_to_step(raw_qty_floor, symbol_info.qty_step)
            if stepped_qty_floor >= symbol_info.min_qty:
                self._last_floor_applied = True
                self._last_floor_details = {
                    "original_notional": float(notional_budget),
                    "effective_notional": float(effective_notional),
                    "qty": float(stepped_qty_floor),
                }
                return stepped_qty_floor
        return Decimal("0")

    def reject_reason(
        self, inputs: SizingInputs, symbol_info: MarketSymbol
    ) -> str | None:
        """
        Return concise reason when size_qty would return 0. For visibility only.
        Does not change sizing behavior.
        """
        if inputs.reference_price <= 0 or inputs.equity_usdt <= 0:
            return "reference_price_or_equity_zero"
        if inputs.confidence < self._min_confidence:
            return "confidence_below_min"
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
            return "qty_below_min_after_rounding"
        return None

    @staticmethod
    def _round_down_to_step(value: Decimal, step: Decimal) -> Decimal:
        if step <= 0:
            return value
        units = (value / step).to_integral_value(rounding=ROUND_DOWN)
        return units * step
