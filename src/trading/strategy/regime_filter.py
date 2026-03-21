from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any

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
class RegimeFilterReport:
    """Visibility report for regime filter decision (no strategy changes)."""

    symbol: str
    candidate_type: str
    allow: bool
    reason: str
    state: str
    volatility_bps: Decimal
    trend_bps: Decimal
    adaptive_trend_threshold_bps: Decimal
    trend_threshold_bps: Decimal
    max_volatility_bps: Decimal
    min_volatility_bps: Decimal
    max_abs_funding_bps: Decimal
    funding_rate_bps: Decimal | None
    bars_1h_count: int
    failed_conditions: tuple[str, ...]

    def to_log_dict(self) -> dict[str, Any]:
        """Operator-friendly structured dict for logging."""
        d: dict[str, Any] = {
            "symbol": self.symbol,
            "candidate_type": self.candidate_type,
            "allow": self.allow,
            "reason": self.reason,
            "state": self.state,
            "volatility_bps": float(self.volatility_bps),
            "trend_bps": float(self.trend_bps),
            "adaptive_trend_threshold_bps": float(self.adaptive_trend_threshold_bps),
            "trend_threshold_bps": float(self.trend_threshold_bps),
            "max_volatility_bps": float(self.max_volatility_bps),
            "min_volatility_bps": float(self.min_volatility_bps),
            "max_abs_funding_bps": float(self.max_abs_funding_bps),
            "bars_1h_count": self.bars_1h_count,
            "failed_conditions": list(self.failed_conditions),
        }
        if self.funding_rate_bps is not None:
            d["funding_rate_bps"] = float(self.funding_rate_bps)
        return d


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
        decision, _ = self.evaluate_with_report(
            candidate=candidate,
            bars_1h=bars_1h,
            funding_rate_bps=funding_rate_bps,
        )
        return decision

    def evaluate_with_report(
        self,
        *,
        candidate: AlphaCandidate,
        bars_1h: list[OHLCVBar],
        funding_rate_bps: Decimal | None = None,
    ) -> tuple[RegimeDecision, RegimeFilterReport]:
        """Evaluate regime filter and return decision plus diagnostic report. No behavior change."""
        bars_1h_count = len(bars_1h)
        volatility_bps = Decimal("0")
        trend_bps = Decimal("0")
        adaptive_trend_threshold = self._cfg.trend_threshold_bps

        if bars_1h_count < 24:
            decision = RegimeDecision(
                state=RegimeState.RANGE,
                allow=False,
                reason="insufficient_1h_context",
                volatility_bps=volatility_bps,
                trend_bps=trend_bps,
            )
            report = RegimeFilterReport(
                symbol=candidate.symbol,
                candidate_type=candidate.candidate_type.value,
                allow=False,
                reason="insufficient_1h_context",
                state=RegimeState.RANGE.value,
                volatility_bps=volatility_bps,
                trend_bps=trend_bps,
                adaptive_trend_threshold_bps=adaptive_trend_threshold,
                trend_threshold_bps=self._cfg.trend_threshold_bps,
                max_volatility_bps=self._cfg.max_volatility_bps,
                min_volatility_bps=self._cfg.min_volatility_bps,
                max_abs_funding_bps=self._cfg.max_abs_funding_bps,
                funding_rate_bps=funding_rate_bps,
                bars_1h_count=bars_1h_count,
                failed_conditions=("insufficient_1h_context",),
            )
            return (decision, report)

        if any(not bar.confirmed for bar in bars_1h[-24:]):
            decision = RegimeDecision(
                state=RegimeState.RANGE,
                allow=False,
                reason="unconfirmed_1h_candles",
                volatility_bps=volatility_bps,
                trend_bps=trend_bps,
            )
            report = RegimeFilterReport(
                symbol=candidate.symbol,
                candidate_type=candidate.candidate_type.value,
                allow=False,
                reason="unconfirmed_1h_candles",
                state=RegimeState.RANGE.value,
                volatility_bps=volatility_bps,
                trend_bps=trend_bps,
                adaptive_trend_threshold_bps=adaptive_trend_threshold,
                trend_threshold_bps=self._cfg.trend_threshold_bps,
                max_volatility_bps=self._cfg.max_volatility_bps,
                min_volatility_bps=self._cfg.min_volatility_bps,
                max_abs_funding_bps=self._cfg.max_abs_funding_bps,
                funding_rate_bps=funding_rate_bps,
                bars_1h_count=bars_1h_count,
                failed_conditions=("unconfirmed_1h_candles",),
            )
            return (decision, report)

        closes = [bar.close for bar in bars_1h[-24:]]
        returns_abs_bps = [
            (abs(closes[i] - closes[i - 1]) / max(closes[i - 1], Decimal("1"))) * Decimal("10000")
            for i in range(1, len(closes))
        ]
        volatility_bps = sum(returns_abs_bps, start=Decimal("0")) / Decimal(len(returns_abs_bps))
        trend_bps = ((closes[-1] - closes[0]) / max(closes[0], Decimal("1"))) * Decimal("10000")
        adaptive_trend_threshold = self._cfg.trend_threshold_bps + (volatility_bps / Decimal("20"))

        if volatility_bps > self._cfg.max_volatility_bps:
            decision = RegimeDecision(
                state=RegimeState.HIGH_VOLATILITY_BLOCK,
                allow=False,
                reason="volatility_above_limit",
                volatility_bps=volatility_bps,
                trend_bps=trend_bps,
            )
            report = RegimeFilterReport(
                symbol=candidate.symbol,
                candidate_type=candidate.candidate_type.value,
                allow=False,
                reason="volatility_above_limit",
                state=RegimeState.HIGH_VOLATILITY_BLOCK.value,
                volatility_bps=volatility_bps,
                trend_bps=trend_bps,
                adaptive_trend_threshold_bps=adaptive_trend_threshold,
                trend_threshold_bps=self._cfg.trend_threshold_bps,
                max_volatility_bps=self._cfg.max_volatility_bps,
                min_volatility_bps=self._cfg.min_volatility_bps,
                max_abs_funding_bps=self._cfg.max_abs_funding_bps,
                funding_rate_bps=funding_rate_bps,
                bars_1h_count=bars_1h_count,
                failed_conditions=("volatility_above_limit",),
            )
            return (decision, report)

        if volatility_bps < self._cfg.min_volatility_bps:
            decision = RegimeDecision(
                state=RegimeState.RANGE,
                allow=False,
                reason="volatility_too_low",
                volatility_bps=volatility_bps,
                trend_bps=trend_bps,
            )
            report = RegimeFilterReport(
                symbol=candidate.symbol,
                candidate_type=candidate.candidate_type.value,
                allow=False,
                reason="volatility_too_low",
                state=RegimeState.RANGE.value,
                volatility_bps=volatility_bps,
                trend_bps=trend_bps,
                adaptive_trend_threshold_bps=adaptive_trend_threshold,
                trend_threshold_bps=self._cfg.trend_threshold_bps,
                max_volatility_bps=self._cfg.max_volatility_bps,
                min_volatility_bps=self._cfg.min_volatility_bps,
                max_abs_funding_bps=self._cfg.max_abs_funding_bps,
                funding_rate_bps=funding_rate_bps,
                bars_1h_count=bars_1h_count,
                failed_conditions=("volatility_too_low",),
            )
            return (decision, report)

        if funding_rate_bps is not None and abs(funding_rate_bps) > self._cfg.max_abs_funding_bps:
            decision = RegimeDecision(
                state=RegimeState.RANGE,
                allow=False,
                reason="funding_extreme",
                volatility_bps=volatility_bps,
                trend_bps=trend_bps,
            )
            report = RegimeFilterReport(
                symbol=candidate.symbol,
                candidate_type=candidate.candidate_type.value,
                allow=False,
                reason="funding_extreme",
                state=RegimeState.RANGE.value,
                volatility_bps=volatility_bps,
                trend_bps=trend_bps,
                adaptive_trend_threshold_bps=adaptive_trend_threshold,
                trend_threshold_bps=self._cfg.trend_threshold_bps,
                max_volatility_bps=self._cfg.max_volatility_bps,
                min_volatility_bps=self._cfg.min_volatility_bps,
                max_abs_funding_bps=self._cfg.max_abs_funding_bps,
                funding_rate_bps=funding_rate_bps,
                bars_1h_count=bars_1h_count,
                failed_conditions=("funding_extreme",),
            )
            return (decision, report)

        state = self._infer_state(trend_bps, adaptive_trend_threshold)
        allow = self._candidate_matches_regime(candidate.candidate_type, state)
        reason = "regime_match" if allow else "regime_mismatch"

        if allow:
            failed: tuple[str, ...] = ()
        else:
            if state == RegimeState.RANGE:
                failed = ("state_range",)
            elif state == RegimeState.RISK_ON_LONG:
                failed = ("candidate_short_vs_long_regime",)
            else:
                failed = ("candidate_long_vs_short_regime",)

        decision = RegimeDecision(
            state=state,
            allow=allow,
            reason=reason,
            volatility_bps=volatility_bps,
            trend_bps=trend_bps,
        )
        report = RegimeFilterReport(
            symbol=candidate.symbol,
            candidate_type=candidate.candidate_type.value,
            allow=allow,
            reason=reason,
            state=state.value,
            volatility_bps=volatility_bps,
            trend_bps=trend_bps,
            adaptive_trend_threshold_bps=adaptive_trend_threshold,
            trend_threshold_bps=self._cfg.trend_threshold_bps,
            max_volatility_bps=self._cfg.max_volatility_bps,
            min_volatility_bps=self._cfg.min_volatility_bps,
            max_abs_funding_bps=self._cfg.max_abs_funding_bps,
            funding_rate_bps=funding_rate_bps,
            bars_1h_count=bars_1h_count,
            failed_conditions=failed,
        )
        return (decision, report)

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
