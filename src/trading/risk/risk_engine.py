from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from trading.risk.circuit_breaker import CircuitBreaker
from trading.risk.portfolio_state import PortfolioState
from trading.strategy.signal_engine import SignalDecision
from trading.util.types import OrderSide, PositionSide


@dataclass(slots=True, frozen=True)
class PerSymbolLimit:
    """Per-symbol risk limit (avoids coupling to settings)."""

    max_notional_usdt: Decimal
    max_position_abs: Decimal


@dataclass(slots=True, frozen=True)
class RiskDecision:
    approved: bool
    reason: str
    adjusted_confidence: Decimal
    metadata: dict[str, Any]


class RiskEngine:
    """
    Hard-limits-first risk checker.

    Evaluation order:
    1) circuit breaker / safe mode
    2) daily loss limit
    3) leverage and notional hard caps
    4) liquidation distance buffer
    5) soft confidence scaling
    """

    def __init__(
        self,
        *,
        max_total_notional_usdt: Decimal,
        max_leverage: Decimal,
        daily_loss_limit_usdt: Decimal,
        liquidation_buffer_bps: int,
        circuit_breaker: CircuitBreaker,
        per_symbol_limits: dict[str, PerSymbolLimit] | None = None,
    ) -> None:
        self._max_total_notional_usdt = max_total_notional_usdt
        self._max_leverage = max_leverage
        self._daily_loss_limit_usdt = daily_loss_limit_usdt
        self._liquidation_buffer_bps = Decimal(liquidation_buffer_bps)
        self._circuit_breaker = circuit_breaker
        self._per_symbol_limits = per_symbol_limits or {}

    def evaluate(
        self,
        *,
        signal: SignalDecision,
        portfolio: PortfolioState,
        expected_order_notional: Decimal,
    ) -> RiskDecision:
        if self._circuit_breaker.is_tripped():
            return RiskDecision(
                False,
                self._circuit_breaker.status_reason() or "circuit_breaker",
                Decimal("0"),
                {"stage": "hard_limits"},
            )
        if portfolio.safe_mode:
            return RiskDecision(False, "safe_mode_enabled", Decimal("0"), {"stage": "hard_limits"})
        if portfolio.realized_pnl_today_usdt <= -self._daily_loss_limit_usdt:
            return RiskDecision(
                False,
                "daily_loss_limit_reached",
                Decimal("0"),
                {"stage": "hard_limits", "realized_pnl_today_usdt": portfolio.realized_pnl_today_usdt},
            )

        projected_notional = portfolio.total_notional() + expected_order_notional
        if projected_notional > self._max_total_notional_usdt:
            return RiskDecision(
                False,
                "max_total_notional_exceeded",
                Decimal("0"),
                {
                    "stage": "hard_limits",
                    "projected_notional": projected_notional,
                    "max_total_notional_usdt": self._max_total_notional_usdt,
                },
            )

        if portfolio.max_effective_leverage() > self._max_leverage:
            return RiskDecision(
                False,
                "portfolio_max_leverage_exceeded",
                Decimal("0"),
                {"stage": "hard_limits", "effective_leverage": portfolio.max_effective_leverage()},
            )
        position = portfolio.position_for(signal.symbol)
        if position is not None and position.leverage > self._max_leverage:
            return RiskDecision(
                False,
                "position_max_leverage_exceeded",
                Decimal("0"),
                {"stage": "hard_limits", "position_leverage": position.leverage},
            )

        if position is not None:
            distance = position.distance_to_liq_bps
            if distance is not None and distance < self._liquidation_buffer_bps:
                return RiskDecision(
                    False,
                    "liquidation_buffer_violation",
                    Decimal("0"),
                    {
                        "stage": "hard_limits",
                        "distance_to_liq_bps": distance,
                        "min_distance_bps": self._liquidation_buffer_bps,
                    },
                )

        limit = self._per_symbol_limits.get(signal.symbol)
        if limit is not None:
            pos = portfolio.position_for(signal.symbol)
            pos_notional = pos.notional if pos is not None else Decimal("0")
            symbol_projected_notional = pos_notional + expected_order_notional
            if symbol_projected_notional > limit.max_notional_usdt:
                return RiskDecision(
                    False,
                    "per_symbol_max_notional_exceeded",
                    Decimal("0"),
                    {
                        "stage": "hard_limits",
                        "symbol": signal.symbol,
                        "projected_notional": symbol_projected_notional,
                        "max_notional_usdt": limit.max_notional_usdt,
                    },
                )
            if signal.side is not None and signal.reference_price and signal.reference_price > 0:
                order_qty = expected_order_notional / signal.reference_price
                order_delta = order_qty if signal.side == OrderSide.BUY else -order_qty
                current_signed = (
                    (pos.qty if pos.side == PositionSide.LONG else -pos.qty) if pos is not None else Decimal("0")
                )
                new_signed = current_signed + order_delta
                new_size = abs(new_signed)
                if new_size > limit.max_position_abs:
                    return RiskDecision(
                        False,
                        "per_symbol_max_position_exceeded",
                        Decimal("0"),
                        {
                            "stage": "hard_limits",
                            "symbol": signal.symbol,
                            "projected_position_abs": new_size,
                            "max_position_abs": limit.max_position_abs,
                        },
                    )

        confidence = min(Decimal("1"), max(Decimal("0"), signal.confidence))
        if projected_notional > (self._max_total_notional_usdt * Decimal("0.85")):
            confidence *= Decimal("0.75")
        if confidence < Decimal("0.15"):
            return RiskDecision(
                False,
                "confidence_below_threshold",
                confidence,
                {"stage": "soft_scaling"},
            )
        return RiskDecision(
            True,
            "approved",
            confidence,
            {
                "stage": "approved",
                "projected_notional": projected_notional,
                "effective_leverage": portfolio.max_effective_leverage(),
            },
        )
