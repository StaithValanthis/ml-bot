from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from trading.risk.circuit_breaker import CircuitBreaker
from trading.risk.portfolio_state import PortfolioState
from trading.strategy.signal_engine import SignalDecision
from trading.util.types import OrderSide, PositionSide


@dataclass(slots=True, frozen=True)
class RiskDecisionReport:
    """Visibility report for risk engine decision (no strategy changes)."""

    symbol: str
    allow: bool
    reason: str
    failed_conditions: tuple[str, ...]
    side: str | None
    candidate_type: str | None
    entry_price: Decimal | None
    stop_price: Decimal | None
    tp_price: Decimal | None
    qty: Decimal | None
    notional: Decimal | None
    current_position_size: Decimal | None
    current_position_side: str | None
    current_open_orders_count: int | None
    orphan_position_blocked: bool | None
    cooldown_active: bool | None
    duplicate_side_block: bool | None
    max_position_limit: Decimal | None
    max_notional_limit: Decimal | None
    min_rr_required: Decimal | None
    actual_rr: Decimal | None
    circuit_breaker_tripped: bool
    daily_loss_limit_usdt: Decimal
    realized_pnl_today_usdt: Decimal
    projected_notional: Decimal | None
    max_total_notional_usdt: Decimal
    max_leverage: Decimal
    effective_leverage: Decimal | None
    position_leverage: Decimal | None
    distance_to_liq_bps: Decimal | None
    liquidation_buffer_bps: Decimal
    min_confidence_threshold: Decimal
    confidence: Decimal

    def to_log_dict(self) -> dict[str, Any]:
        """Operator-friendly structured dict for logging."""
        d: dict[str, Any] = {
            "symbol": self.symbol,
            "allow": self.allow,
            "reason": self.reason,
            "failed_conditions": list(self.failed_conditions),
            "side": self.side,
            "candidate_type": self.candidate_type,
            "entry_price": float(self.entry_price) if self.entry_price is not None else None,
            "stop_price": float(self.stop_price) if self.stop_price is not None else None,
            "tp_price": float(self.tp_price) if self.tp_price is not None else None,
            "qty": float(self.qty) if self.qty is not None else None,
            "notional": float(self.notional) if self.notional is not None else None,
            "current_position_size": float(self.current_position_size) if self.current_position_size is not None else None,
            "current_position_side": self.current_position_side,
            "current_open_orders_count": self.current_open_orders_count,
            "orphan_position_blocked": self.orphan_position_blocked,
            "cooldown_active": self.cooldown_active,
            "duplicate_side_block": self.duplicate_side_block,
            "max_position_limit": float(self.max_position_limit) if self.max_position_limit is not None else None,
            "max_notional_limit": float(self.max_notional_limit) if self.max_notional_limit is not None else None,
            "min_rr_required": float(self.min_rr_required) if self.min_rr_required is not None else None,
            "actual_rr": float(self.actual_rr) if self.actual_rr is not None else None,
            "circuit_breaker_tripped": self.circuit_breaker_tripped,
            "daily_loss_limit_usdt": float(self.daily_loss_limit_usdt),
            "realized_pnl_today_usdt": float(self.realized_pnl_today_usdt),
            "projected_notional": float(self.projected_notional) if self.projected_notional is not None else None,
            "max_total_notional_usdt": float(self.max_total_notional_usdt),
            "max_leverage": float(self.max_leverage),
            "effective_leverage": float(self.effective_leverage) if self.effective_leverage is not None else None,
            "position_leverage": float(self.position_leverage) if self.position_leverage is not None else None,
            "distance_to_liq_bps": float(self.distance_to_liq_bps) if self.distance_to_liq_bps is not None else None,
            "liquidation_buffer_bps": float(self.liquidation_buffer_bps),
            "min_confidence_threshold": float(self.min_confidence_threshold),
            "confidence": float(self.confidence),
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(slots=True)
class RiskEvaluationContext:
    """Optional context from caller for report enrichment (e.g. orchestrator)."""

    candidate_type: str | None = None
    orphan_position_blocked: bool | None = None
    current_open_orders_count: int | None = None
    duplicate_side_block: bool | None = None


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

    MIN_CONFIDENCE = Decimal("0.15")

    def evaluate(
        self,
        *,
        signal: SignalDecision,
        portfolio: PortfolioState,
        expected_order_notional: Decimal,
    ) -> RiskDecision:
        decision, _ = self.evaluate_with_report(
            signal=signal,
            portfolio=portfolio,
            expected_order_notional=expected_order_notional,
        )
        return decision

    def _base_report(
        self,
        *,
        signal: SignalDecision,
        portfolio: PortfolioState,
        expected_order_notional: Decimal,
        allow: bool,
        reason: str,
        failed_conditions: tuple[str, ...],
        confidence: Decimal,
        ctx: RiskEvaluationContext | None = None,
    ) -> RiskDecisionReport:
        pos = portfolio.position_for(signal.symbol)
        ref = signal.reference_price or Decimal("0")
        qty = expected_order_notional / ref if ref > 0 else None
        pos_size = pos.qty if pos is not None else None
        pos_side = pos.side.value if pos is not None else None
        limit = self._per_symbol_limits.get(signal.symbol)
        proj = portfolio.total_notional() + expected_order_notional
        return RiskDecisionReport(
            symbol=signal.symbol,
            allow=allow,
            reason=reason,
            failed_conditions=failed_conditions,
            side=signal.side.value if signal.side else None,
            candidate_type=(ctx.candidate_type if ctx else None),
            entry_price=ref if ref else None,
            stop_price=signal.stop_price,
            tp_price=None,
            qty=qty,
            notional=expected_order_notional,
            current_position_size=pos_size,
            current_position_side=pos_side,
            current_open_orders_count=(ctx.current_open_orders_count if ctx else None),
            orphan_position_blocked=(ctx.orphan_position_blocked if ctx else None),
            cooldown_active=self._circuit_breaker.is_tripped(),
            duplicate_side_block=(ctx.duplicate_side_block if ctx else None),
            max_position_limit=limit.max_position_abs if limit else None,
            max_notional_limit=limit.max_notional_usdt if limit else None,
            min_rr_required=None,
            actual_rr=None,
            circuit_breaker_tripped=self._circuit_breaker.is_tripped(),
            daily_loss_limit_usdt=self._daily_loss_limit_usdt,
            realized_pnl_today_usdt=portfolio.realized_pnl_today_usdt,
            projected_notional=proj,
            max_total_notional_usdt=self._max_total_notional_usdt,
            max_leverage=self._max_leverage,
            effective_leverage=portfolio.max_effective_leverage() if portfolio.equity_usdt > 0 else None,
            position_leverage=pos.leverage if pos is not None else None,
            distance_to_liq_bps=pos.distance_to_liq_bps if pos is not None else None,
            liquidation_buffer_bps=self._liquidation_buffer_bps,
            min_confidence_threshold=self.MIN_CONFIDENCE,
            confidence=confidence,
        )

    def evaluate_with_report(
        self,
        *,
        signal: SignalDecision,
        portfolio: PortfolioState,
        expected_order_notional: Decimal,
        context: RiskEvaluationContext | None = None,
    ) -> tuple[RiskDecision, RiskDecisionReport]:
        ctx = context
        cb_tripped = self._circuit_breaker.is_tripped()
        if cb_tripped:
            d = RiskDecision(False, self._circuit_breaker.status_reason() or "circuit_breaker", Decimal("0"), {"stage": "hard_limits"})
            r = self._base_report(signal=signal, portfolio=portfolio, expected_order_notional=expected_order_notional, allow=False, reason=d.reason, failed_conditions=("circuit_breaker",), confidence=Decimal("0"), ctx=ctx)
            return (d, r)
        if portfolio.safe_mode:
            d = RiskDecision(False, "safe_mode_enabled", Decimal("0"), {"stage": "hard_limits"})
            r = self._base_report(signal=signal, portfolio=portfolio, expected_order_notional=expected_order_notional, allow=False, reason=d.reason, failed_conditions=("safe_mode_enabled",), confidence=Decimal("0"), ctx=ctx)
            return (d, r)
        if portfolio.realized_pnl_today_usdt <= -self._daily_loss_limit_usdt:
            d = RiskDecision(False, "daily_loss_limit_reached", Decimal("0"), {"stage": "hard_limits", "realized_pnl_today_usdt": portfolio.realized_pnl_today_usdt})
            r = self._base_report(signal=signal, portfolio=portfolio, expected_order_notional=expected_order_notional, allow=False, reason=d.reason, failed_conditions=("daily_loss_limit_reached",), confidence=Decimal("0"), ctx=ctx)
            return (d, r)

        projected_notional = portfolio.total_notional() + expected_order_notional
        if projected_notional > self._max_total_notional_usdt:
            d = RiskDecision(False, "max_total_notional_exceeded", Decimal("0"), {"stage": "hard_limits", "projected_notional": projected_notional, "max_total_notional_usdt": self._max_total_notional_usdt})
            r = self._base_report(signal=signal, portfolio=portfolio, expected_order_notional=expected_order_notional, allow=False, reason=d.reason, failed_conditions=("max_total_notional_exceeded",), confidence=Decimal("0"), ctx=ctx)
            return (d, r)

        if portfolio.max_effective_leverage() > self._max_leverage:
            d = RiskDecision(False, "portfolio_max_leverage_exceeded", Decimal("0"), {"stage": "hard_limits", "effective_leverage": portfolio.max_effective_leverage()})
            r = self._base_report(signal=signal, portfolio=portfolio, expected_order_notional=expected_order_notional, allow=False, reason=d.reason, failed_conditions=("portfolio_max_leverage_exceeded",), confidence=Decimal("0"), ctx=ctx)
            return (d, r)
        position = portfolio.position_for(signal.symbol)
        if position is not None and position.leverage > self._max_leverage:
            d = RiskDecision(False, "position_max_leverage_exceeded", Decimal("0"), {"stage": "hard_limits", "position_leverage": position.leverage})
            r = self._base_report(signal=signal, portfolio=portfolio, expected_order_notional=expected_order_notional, allow=False, reason=d.reason, failed_conditions=("position_max_leverage_exceeded",), confidence=Decimal("0"), ctx=ctx)
            return (d, r)

        if position is not None:
            distance = position.distance_to_liq_bps
            if distance is not None and distance < self._liquidation_buffer_bps:
                d = RiskDecision(False, "liquidation_buffer_violation", Decimal("0"), {"stage": "hard_limits", "distance_to_liq_bps": distance, "min_distance_bps": self._liquidation_buffer_bps})
                r = self._base_report(signal=signal, portfolio=portfolio, expected_order_notional=expected_order_notional, allow=False, reason=d.reason, failed_conditions=("liquidation_buffer_violation",), confidence=Decimal("0"), ctx=ctx)
                return (d, r)

        limit = self._per_symbol_limits.get(signal.symbol)
        if limit is not None:
            pos = portfolio.position_for(signal.symbol)
            pos_notional = pos.notional if pos is not None else Decimal("0")
            symbol_projected_notional = pos_notional + expected_order_notional
            if symbol_projected_notional > limit.max_notional_usdt:
                d = RiskDecision(False, "per_symbol_max_notional_exceeded", Decimal("0"), {"stage": "hard_limits", "symbol": signal.symbol, "projected_notional": symbol_projected_notional, "max_notional_usdt": limit.max_notional_usdt})
                r = self._base_report(signal=signal, portfolio=portfolio, expected_order_notional=expected_order_notional, allow=False, reason=d.reason, failed_conditions=("per_symbol_max_notional_exceeded",), confidence=Decimal("0"), ctx=ctx)
                return (d, r)
            if signal.side is not None and signal.reference_price and signal.reference_price > 0:
                order_qty = expected_order_notional / signal.reference_price
                order_delta = order_qty if signal.side == OrderSide.BUY else -order_qty
                current_signed = (pos.qty if pos.side == PositionSide.LONG else -pos.qty) if pos is not None else Decimal("0")
                new_signed = current_signed + order_delta
                new_size = abs(new_signed)
                if new_size > limit.max_position_abs:
                    d = RiskDecision(False, "per_symbol_max_position_exceeded", Decimal("0"), {"stage": "hard_limits", "symbol": signal.symbol, "projected_position_abs": new_size, "max_position_abs": limit.max_position_abs})
                    r = self._base_report(signal=signal, portfolio=portfolio, expected_order_notional=expected_order_notional, allow=False, reason=d.reason, failed_conditions=("per_symbol_max_position_exceeded",), confidence=Decimal("0"), ctx=ctx)
                    return (d, r)

        confidence = min(Decimal("1"), max(Decimal("0"), signal.confidence))
        if projected_notional > (self._max_total_notional_usdt * Decimal("0.85")):
            confidence *= Decimal("0.75")
        if confidence < self.MIN_CONFIDENCE:
            d = RiskDecision(False, "confidence_below_threshold", confidence, {"stage": "soft_scaling"})
            r = self._base_report(signal=signal, portfolio=portfolio, expected_order_notional=expected_order_notional, allow=False, reason=d.reason, failed_conditions=("confidence_below_threshold",), confidence=confidence, ctx=ctx)
            return (d, r)
        d = RiskDecision(True, "approved", confidence, {"stage": "approved", "projected_notional": projected_notional, "effective_leverage": portfolio.max_effective_leverage()})
        r = self._base_report(signal=signal, portfolio=portfolio, expected_order_notional=expected_order_notional, allow=True, reason=d.reason, failed_conditions=(), confidence=confidence, ctx=ctx)
        return (d, r)
