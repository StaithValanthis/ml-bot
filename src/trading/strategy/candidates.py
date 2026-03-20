from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from trading.strategy.base_alpha import AlphaCandidate, BaseAlpha, CandidateType
from trading.util.types import OHLCVBar


@dataclass(slots=True, frozen=True)
class BreakoutPreconditionReport:
    """Visibility report for breakout candidate preconditions (no strategy changes)."""

    symbol: str
    breakout_up_bps: Decimal
    breakout_dn_bps: Decimal
    candle_move_bps: Decimal
    vol_multiplier: Decimal
    lookback_high: Decimal
    lookback_low: Decimal
    last_close: Decimal
    min_breakout_bps: Decimal
    min_trend_bps: Decimal
    min_volume_multiplier: Decimal
    breakout_up_ok: bool
    breakout_dn_ok: bool
    trend_long_ok: bool
    trend_short_ok: bool
    failed_conditions: tuple[str, ...]

    def to_log_dict(self) -> dict[str, Any]:
        """Operator-friendly structured dict for logging."""
        return {
            "symbol": self.symbol,
            "breakout_up_bps": float(self.breakout_up_bps),
            "breakout_dn_bps": float(self.breakout_dn_bps),
            "candle_move_bps": float(self.candle_move_bps),
            "vol_multiplier": float(self.vol_multiplier),
            "lookback_high": float(self.lookback_high),
            "lookback_low": float(self.lookback_low),
            "last_close": float(self.last_close),
            "min_breakout_bps": float(self.min_breakout_bps),
            "min_trend_bps": float(self.min_trend_bps),
            "min_volume_multiplier": float(self.min_volume_multiplier),
            "breakout_up_ok": self.breakout_up_ok,
            "breakout_dn_ok": self.breakout_dn_ok,
            "trend_long_ok": self.trend_long_ok,
            "trend_short_ok": self.trend_short_ok,
            "failed_conditions": list(self.failed_conditions),
        }


def get_candidate_readiness(
    symbol: str,
    bars_5m: list[OHLCVBar],
    bars_1h: list[OHLCVBar],
    *,
    lookback_bars: int = 20,
) -> dict[str, Any]:
    """
    Compute candidate-generation readiness for operator visibility.

    Returns dict with: bars_5m, bars_1h, has_enough_5m, has_enough_1h,
    unconfirmed_in_5m_window, unconfirmed_in_1h_window, reason.
    Does not change strategy behavior.
    """
    min_5m = lookback_bars + 2
    min_1h = 24
    bars_5m_count = len(bars_5m)
    bars_1h_count = len(bars_1h)
    has_enough_5m = bars_5m_count >= min_5m
    has_enough_1h = bars_1h_count >= min_1h

    unconfirmed_in_5m_window = False
    if has_enough_5m:
        window_5m = bars_5m[-(lookback_bars + 2) :]
        unconfirmed_in_5m_window = any(not bar.confirmed for bar in window_5m)

    unconfirmed_in_1h_window = False
    if has_enough_1h:
        window_1h = bars_1h[-24:]
        unconfirmed_in_1h_window = any(not bar.confirmed for bar in window_1h)

    reason: str
    if not has_enough_5m:
        reason = "insufficient_5m_history"
    elif unconfirmed_in_5m_window:
        reason = "unconfirmed_5m_in_window"
    else:
        reason = "ready"

    return {
        "symbol": symbol,
        "bars_5m": bars_5m_count,
        "bars_1h": bars_1h_count,
        "min_5m_required": min_5m,
        "min_1h_required": min_1h,
        "has_enough_5m": has_enough_5m,
        "has_enough_1h": has_enough_1h,
        "unconfirmed_in_5m_window": unconfirmed_in_5m_window,
        "unconfirmed_in_1h_window": unconfirmed_in_1h_window,
        "reason": reason,
    }


@dataclass(slots=True, frozen=True)
class CandidateGeneratorConfig:
    lookback_bars: int = 20
    min_breakout_bps: Decimal = Decimal("5")
    min_trend_bps: Decimal = Decimal("8")
    min_volume_multiplier: Decimal = Decimal("1.1")
    stop_atr_multiplier: Decimal = Decimal("1.2")


class BreakoutTrendCandidateGenerator(BaseAlpha):
    """
    5-minute trend + breakout candidate generator.

    Rules:
    - breakout: close exceeds lookback high/low by threshold
    - trend continuation: directional move + stronger volume than baseline
    """

    def __init__(self, config: CandidateGeneratorConfig | None = None) -> None:
        self._cfg = config or CandidateGeneratorConfig()

    def get_readiness(
        self, symbol: str, bars_5m: list[OHLCVBar], bars_1h: list[OHLCVBar]
    ) -> dict[str, Any]:
        """Return candidate-generation readiness for operator visibility."""
        return get_candidate_readiness(
            symbol, bars_5m, bars_1h, lookback_bars=self._cfg.lookback_bars
        )

    def get_precondition_report(
        self, symbol: str, bars_5m: list[OHLCVBar]
    ) -> BreakoutPreconditionReport | None:
        """
        Return visibility report for breakout preconditions when no candidate produced.
        Uses same computation as on_closed_candle; does not change strategy behavior.
        Returns None if bars insufficient for evaluation.
        """
        if len(bars_5m) < self._cfg.lookback_bars + 2:
            return None
        if any(not bar.confirmed for bar in bars_5m[-(self._cfg.lookback_bars + 2) :]):
            return None

        window = bars_5m[-(self._cfg.lookback_bars + 1) : -1]
        last = bars_5m[-1]
        prev = bars_5m[-2]
        lookback_high = max(bar.high for bar in window)
        lookback_low = min(bar.low for bar in window)
        avg_volume = sum((bar.volume for bar in window), start=Decimal("0")) / Decimal(
            len(window)
        )
        vol_multiplier = (last.volume / avg_volume) if avg_volume > 0 else Decimal("0")
        range_basis = max(last.close, Decimal("1"))
        breakout_up_bps = ((last.close - lookback_high) / range_basis) * Decimal("10000")
        breakout_dn_bps = ((lookback_low - last.close) / range_basis) * Decimal("10000")
        candle_move_bps = (
            (last.close - prev.close) / max(prev.close, Decimal("1"))
        ) * Decimal("10000")

        breakout_up_ok = breakout_up_bps >= self._cfg.min_breakout_bps
        breakout_dn_ok = breakout_dn_bps >= self._cfg.min_breakout_bps
        trend_long_ok = (
            candle_move_bps >= self._cfg.min_trend_bps
            and vol_multiplier >= self._cfg.min_volume_multiplier
        )
        trend_short_ok = (
            candle_move_bps <= -self._cfg.min_trend_bps
            and vol_multiplier >= self._cfg.min_volume_multiplier
        )

        failed: list[str] = []
        if not breakout_up_ok:
            failed.append("breakout_up")
        if not breakout_dn_ok:
            failed.append("breakout_dn")
        if not trend_long_ok:
            failed.append("trend_long")
        if not trend_short_ok:
            failed.append("trend_short")

        return BreakoutPreconditionReport(
            symbol=symbol,
            breakout_up_bps=breakout_up_bps,
            breakout_dn_bps=breakout_dn_bps,
            candle_move_bps=candle_move_bps,
            vol_multiplier=vol_multiplier,
            lookback_high=lookback_high,
            lookback_low=lookback_low,
            last_close=last.close,
            min_breakout_bps=self._cfg.min_breakout_bps,
            min_trend_bps=self._cfg.min_trend_bps,
            min_volume_multiplier=self._cfg.min_volume_multiplier,
            breakout_up_ok=breakout_up_ok,
            breakout_dn_ok=breakout_dn_ok,
            trend_long_ok=trend_long_ok,
            trend_short_ok=trend_short_ok,
            failed_conditions=tuple(failed),
        )

    def on_closed_candle(self, symbol: str, bars_5m: list[OHLCVBar]) -> list[AlphaCandidate]:
        # Strategy rule: absolutely no decisions on partially formed candles.
        if len(bars_5m) < self._cfg.lookback_bars + 2:
            return []
        if any(not bar.confirmed for bar in bars_5m[-(self._cfg.lookback_bars + 2) :]):
            return []

        window = bars_5m[-(self._cfg.lookback_bars + 1) : -1]
        last = bars_5m[-1]
        prev = bars_5m[-2]
        lookback_high = max(bar.high for bar in window)
        lookback_low = min(bar.low for bar in window)
        avg_volume = sum((bar.volume for bar in window), start=Decimal("0")) / Decimal(len(window))
        vol_multiplier = (last.volume / avg_volume) if avg_volume > 0 else Decimal("0")
        range_basis = max(last.close, Decimal("1"))
        breakout_up_bps = ((last.close - lookback_high) / range_basis) * Decimal("10000")
        breakout_dn_bps = ((lookback_low - last.close) / range_basis) * Decimal("10000")
        candle_move_bps = ((last.close - prev.close) / max(prev.close, Decimal("1"))) * Decimal("10000")
        atr = self._estimate_atr(window + [last])

        candidates: list[AlphaCandidate] = []
        if breakout_up_bps >= self._cfg.min_breakout_bps:
            candidates.append(
                AlphaCandidate(
                    symbol=symbol,
                    candidate_type=CandidateType.BREAKOUT_LONG,
                    confidence=min(Decimal("0.99"), Decimal("0.5") + breakout_up_bps / Decimal("100")),
                    reference_price=last.close,
                    stop_price=last.close - (atr * self._cfg.stop_atr_multiplier),
                    timeframe="5",
                    signal_time=last.close_time,
                    metadata={
                        "breakout_bps": breakout_up_bps,
                        "volume_multiplier": vol_multiplier,
                    },
                )
            )
        if breakout_dn_bps >= self._cfg.min_breakout_bps:
            candidates.append(
                AlphaCandidate(
                    symbol=symbol,
                    candidate_type=CandidateType.BREAKOUT_SHORT,
                    confidence=min(Decimal("0.99"), Decimal("0.5") + breakout_dn_bps / Decimal("100")),
                    reference_price=last.close,
                    stop_price=last.close + (atr * self._cfg.stop_atr_multiplier),
                    timeframe="5",
                    signal_time=last.close_time,
                    metadata={
                        "breakout_bps": breakout_dn_bps,
                        "volume_multiplier": vol_multiplier,
                    },
                )
            )

        if candle_move_bps >= self._cfg.min_trend_bps and vol_multiplier >= self._cfg.min_volume_multiplier:
            candidates.append(
                AlphaCandidate(
                    symbol=symbol,
                    candidate_type=CandidateType.TREND_CONTINUATION_LONG,
                    confidence=min(Decimal("0.95"), Decimal("0.45") + candle_move_bps / Decimal("120")),
                    reference_price=last.close,
                    stop_price=last.close - (atr * self._cfg.stop_atr_multiplier),
                    timeframe="5",
                    signal_time=last.close_time,
                    metadata={
                        "trend_bps": candle_move_bps,
                        "volume_multiplier": vol_multiplier,
                    },
                )
            )
        if candle_move_bps <= -self._cfg.min_trend_bps and vol_multiplier >= self._cfg.min_volume_multiplier:
            candidates.append(
                AlphaCandidate(
                    symbol=symbol,
                    candidate_type=CandidateType.TREND_CONTINUATION_SHORT,
                    confidence=min(Decimal("0.95"), Decimal("0.45") + abs(candle_move_bps) / Decimal("120")),
                    reference_price=last.close,
                    stop_price=last.close + (atr * self._cfg.stop_atr_multiplier),
                    timeframe="5",
                    signal_time=last.close_time,
                    metadata={
                        "trend_bps": candle_move_bps,
                        "volume_multiplier": vol_multiplier,
                    },
                )
            )
        return self._dedupe_candidates(candidates)

    @staticmethod
    def _estimate_atr(bars: list[OHLCVBar]) -> Decimal:
        if len(bars) < 2:
            return Decimal("0")
        trs: list[Decimal] = []
        prev_close = bars[0].close
        for bar in bars[1:]:
            tr = max(bar.high - bar.low, abs(bar.high - prev_close), abs(bar.low - prev_close))
            trs.append(tr)
            prev_close = bar.close
        return sum(trs, start=Decimal("0")) / Decimal(len(trs))

    @staticmethod
    def _dedupe_candidates(candidates: list[AlphaCandidate]) -> list[AlphaCandidate]:
        """
        Keep strongest candidate per direction to avoid conflicting duplicate intents.
        """
        best_long: AlphaCandidate | None = None
        best_short: AlphaCandidate | None = None
        for candidate in candidates:
            if candidate.candidate_type in {
                CandidateType.BREAKOUT_LONG,
                CandidateType.TREND_CONTINUATION_LONG,
            }:
                if best_long is None or candidate.confidence > best_long.confidence:
                    best_long = candidate
            else:
                if best_short is None or candidate.confidence > best_short.confidence:
                    best_short = candidate
        output: list[AlphaCandidate] = []
        if best_long is not None:
            output.append(best_long)
        if best_short is not None:
            output.append(best_short)
        return output
