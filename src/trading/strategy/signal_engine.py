from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol

from trading.strategy.base_alpha import AlphaCandidate, CandidateType
from trading.strategy.regime_filter import RegimeDecision
from trading.util.types import OrderSide


class SignalAction(str, Enum):
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    HOLD = "hold"


@dataclass(slots=True, frozen=True)
class SignalDecision:
    symbol: str
    action: SignalAction
    side: OrderSide | None
    confidence: Decimal
    reference_price: Decimal | None
    stop_price: Decimal | None
    reason: str
    generated_at: datetime
    metadata: dict[str, Any]


class OptionalMetaGate(Protocol):
    """
    Optional integration point for future meta-labeling gate.

    Return Decimal in [0, 1], where 0 blocks and 1 fully passes.
    """

    def score(self, candidate: AlphaCandidate, regime: RegimeDecision) -> Decimal: ...


class SignalEngine:
    """Maps candidate + regime into action-oriented signal decisions."""

    def __init__(self, *, meta_gate: OptionalMetaGate | None = None) -> None:
        self._meta_gate = meta_gate

    def evaluate(self, candidate: AlphaCandidate, regime: RegimeDecision) -> SignalDecision:
        if not regime.allow:
            return SignalDecision(
                symbol=candidate.symbol,
                action=SignalAction.HOLD,
                side=None,
                confidence=Decimal("0"),
                reference_price=None,
                stop_price=None,
                reason=regime.reason,
                generated_at=candidate.signal_time,
                metadata={"regime_state": regime.state.value},
            )
        gate_score = self._meta_gate.score(candidate, regime) if self._meta_gate is not None else Decimal("1")
        gate_score = max(Decimal("0"), min(Decimal("1"), gate_score))
        if gate_score <= Decimal("0"):
            return SignalDecision(
                symbol=candidate.symbol,
                action=SignalAction.HOLD,
                side=None,
                confidence=Decimal("0"),
                reference_price=None,
                stop_price=None,
                reason="meta_gate_blocked",
                generated_at=candidate.signal_time,
                metadata={"regime_state": regime.state.value},
            )

        side = self._candidate_side(candidate.candidate_type)
        action = SignalAction.ENTER_LONG if side == OrderSide.BUY else SignalAction.ENTER_SHORT
        combined_confidence = min(
            Decimal("0.99"),
            candidate.confidence * Decimal("0.8") + Decimal("0.2"),
        ) * gate_score
        return SignalDecision(
            symbol=candidate.symbol,
            action=action,
            side=side,
            confidence=combined_confidence,
            reference_price=candidate.reference_price,
            stop_price=candidate.stop_price,
            reason="candidate_and_regime_approved",
            generated_at=candidate.signal_time,
            metadata={
                **candidate.metadata,
                "regime_state": regime.state.value,
                "regime_vol_bps": regime.volatility_bps,
                "regime_trend_bps": regime.trend_bps,
            },
        )

    @staticmethod
    def _candidate_side(candidate_type: CandidateType) -> OrderSide:
        if candidate_type in {CandidateType.BREAKOUT_LONG, CandidateType.TREND_CONTINUATION_LONG}:
            return OrderSide.BUY
        return OrderSide.SELL
