from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from trading.strategy.base_alpha import AlphaCandidate, CandidateType
from trading.util.types import OHLCVBar


class RegimeState(str, Enum):
    RISK_ON_LONG = "risk_on_long"
    RISK_ON_SHORT = "risk_on_short"
    RANGE = "range"
    HIGH_VOLATILITY_BLOCK = "high_volatility_block"


@dataclass(slots=True, frozen=True)
class RegimeDecision:
    state: RegimeState
    allow: bool
    reason: str
    volatility_bps: Decimal
    trend_bps: Decimal


@dataclass(slots=True, frozen=True)
class RegimeFilterConfig:
    trend_threshold_bps: Decimal = Decimal("12")
    max_volatility_bps: Decimal = Decimal("180")
    min_volatility_bps: Decimal = Decimal("5")
    max_abs_funding_bps: Decimal = Decimal("8")


class RegimeFilter:
    """
    1h regime filter with volatility-aware gate and funding-aware placeholder.

    TODO(phase-8): Replace funding placeholder with feature store integration.
    """

    def __init__(self, config: RegimeFilterConfig | None = None) -> None:
        self._cfg = config or RegimeFilterConfig()

    def evaluate(
        self,
        *,
        candidate: AlphaCandidate,
        bars_1h: list[OHLCVBar],
        funding_rate_bps: Decimal | None = None,
    ) -> RegimeDecision:
        # Need one full day of 1h context to avoid unstable early-session gating.
        if len(bars_1h) < 24:
            return RegimeDecision(
                state=RegimeState.RANGE,
                allow=False,
                reason="insufficient_1h_context",
                volatility_bps=Decimal("0"),
                trend_bps=Decimal("0"),
            )
        if any(not bar.confirmed for bar in bars_1h[-24:]):
            return RegimeDecision(
                state=RegimeState.RANGE,
                allow=False,
                reason="unconfirmed_1h_candles",
                volatility_bps=Decimal("0"),
                trend_bps=Decimal("0"),
            )

        closes = [bar.close for bar in bars_1h[-24:]]
        returns_abs_bps = [
            (abs(closes[i] - closes[i - 1]) / max(closes[i - 1], Decimal("1"))) * Decimal("10000")
            for i in range(1, len(closes))
        ]
        volatility_bps = sum(returns_abs_bps, start=Decimal("0")) / Decimal(len(returns_abs_bps))
        trend_bps = ((closes[-1] - closes[0]) / max(closes[0], Decimal("1"))) * Decimal("10000")
        # Adaptive threshold: require stronger trend when volatility increases.
        adaptive_trend_threshold = self._cfg.trend_threshold_bps + (volatility_bps / Decimal("20"))

        if volatility_bps > self._cfg.max_volatility_bps:
            return RegimeDecision(
                state=RegimeState.HIGH_VOLATILITY_BLOCK,
                allow=False,
                reason="volatility_above_limit",
                volatility_bps=volatility_bps,
                trend_bps=trend_bps,
            )
        if volatility_bps < self._cfg.min_volatility_bps:
            return RegimeDecision(
                state=RegimeState.RANGE,
                allow=False,
                reason="volatility_too_low",
                volatility_bps=volatility_bps,
                trend_bps=trend_bps,
            )

        if funding_rate_bps is not None and abs(funding_rate_bps) > self._cfg.max_abs_funding_bps:
            return RegimeDecision(
                state=RegimeState.RANGE,
                allow=False,
                reason="funding_extreme",
                volatility_bps=volatility_bps,
                trend_bps=trend_bps,
            )

        state = self._infer_state(trend_bps, adaptive_trend_threshold)
        allow = self._candidate_matches_regime(candidate.candidate_type, state)
        return RegimeDecision(
            state=state,
            allow=allow,
            reason="regime_match" if allow else "regime_mismatch",
            volatility_bps=volatility_bps,
            trend_bps=trend_bps,
        )

    def _infer_state(self, trend_bps: Decimal, threshold_bps: Decimal) -> RegimeState:
        if trend_bps >= threshold_bps:
            return RegimeState.RISK_ON_LONG
        if trend_bps <= -threshold_bps:
            return RegimeState.RISK_ON_SHORT
        return RegimeState.RANGE

    @staticmethod
    def _candidate_matches_regime(candidate_type: CandidateType, state: RegimeState) -> bool:
        long_candidate = candidate_type in {
            CandidateType.BREAKOUT_LONG,
            CandidateType.TREND_CONTINUATION_LONG,
        }
        short_candidate = candidate_type in {
            CandidateType.BREAKOUT_SHORT,
            CandidateType.TREND_CONTINUATION_SHORT,
        }
        if state == RegimeState.RISK_ON_LONG:
            return long_candidate
        if state == RegimeState.RISK_ON_SHORT:
            return short_candidate
        return False
