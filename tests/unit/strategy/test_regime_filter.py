"""Unit tests for regime filter visibility and diagnostics."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading.strategy.base_alpha import AlphaCandidate, CandidateType
from trading.strategy.regime_filter import (
    RegimeFilter,
    RegimeFilterConfig,
    RegimeFilterReport,
    RegimeState,
)
from trading.util.types import OHLCVBar


def _make_bar_1h(
    close: Decimal = Decimal("40000"),
    confirmed: bool = True,
) -> OHLCVBar:
    t = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    return OHLCVBar(
        symbol="BTCUSDT",
        timeframe="60",
        open_time=t,
        close_time=t,
        open=close,
        high=close + Decimal("10"),
        low=close - Decimal("10"),
        close=close,
        volume=Decimal("100"),
        turnover=close * Decimal("100"),
        confirmed=confirmed,
    )


def _make_candidate(
    symbol: str = "BTCUSDT",
    candidate_type: CandidateType = CandidateType.BREAKOUT_LONG,
) -> AlphaCandidate:
    return AlphaCandidate(
        symbol=symbol,
        candidate_type=candidate_type,
        confidence=Decimal("0.6"),
        reference_price=Decimal("40000"),
        stop_price=Decimal("39500"),
        timeframe="5",
        signal_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
        metadata={},
    )


def test_evaluate_with_report_insufficient_1h_context() -> None:
    """Report includes failed_conditions when bars_1h < 24."""
    filt = RegimeFilter()
    bars = [_make_bar_1h() for _ in range(20)]
    candidate = _make_candidate()
    decision, report = filt.evaluate_with_report(candidate=candidate, bars_1h=bars)
    assert not decision.allow
    assert decision.reason == "insufficient_1h_context"
    assert report.symbol == "BTCUSDT"
    assert report.reason == "insufficient_1h_context"
    assert report.failed_conditions == ("insufficient_1h_context",)
    assert report.bars_1h_count == 20


def test_evaluate_with_report_unconfirmed_1h_candles() -> None:
    """Report includes failed_conditions when last 24 bars have unconfirmed."""
    filt = RegimeFilter()
    bars = [_make_bar_1h(confirmed=True) for _ in range(24)]
    bars[-1] = _make_bar_1h(confirmed=False)
    candidate = _make_candidate()
    decision, report = filt.evaluate_with_report(candidate=candidate, bars_1h=bars)
    assert not decision.allow
    assert decision.reason == "unconfirmed_1h_candles"
    assert report.failed_conditions == ("unconfirmed_1h_candles",)


def test_evaluate_with_report_volatility_above_limit() -> None:
    """Report shows volatility_above_limit when volatility exceeds max."""
    filt = RegimeFilter(RegimeFilterConfig(max_volatility_bps=Decimal("50")))
    base = Decimal("40000")
    bars: list[OHLCVBar] = []
    for i in range(24):
        # Large alternating swings: ~100bps per bar to exceed 50bps average
        c = base + Decimal(i * 500) if i % 2 == 0 else base + Decimal(i * 500) - Decimal("450")
        bars.append(_make_bar_1h(close=c))
    candidate = _make_candidate()
    decision, report = filt.evaluate_with_report(candidate=candidate, bars_1h=bars)
    assert not decision.allow
    assert decision.reason == "volatility_above_limit"
    assert report.failed_conditions == ("volatility_above_limit",)
    assert report.state == RegimeState.HIGH_VOLATILITY_BLOCK.value
    assert float(report.volatility_bps) > 50


def test_evaluate_with_report_volatility_too_low() -> None:
    """Report shows volatility_too_low when volatility below min."""
    filt = RegimeFilter(RegimeFilterConfig(min_volatility_bps=Decimal("50")))
    base = Decimal("40000")
    bars = [_make_bar_1h(close=base + Decimal(i)) for i in range(24)]
    candidate = _make_candidate()
    decision, report = filt.evaluate_with_report(candidate=candidate, bars_1h=bars)
    assert not decision.allow
    assert decision.reason == "volatility_too_low"
    assert report.failed_conditions == ("volatility_too_low",)


def test_evaluate_with_report_regime_mismatch_range_state() -> None:
    """Report shows state_range when trend is below threshold (RANGE blocks all)."""
    filt = RegimeFilter(
        RegimeFilterConfig(
            trend_threshold_bps=Decimal("12"),
            min_volatility_bps=Decimal("1"),
            max_volatility_bps=Decimal("500"),
        )
    )
    base = Decimal("40000")
    bars: list[OHLCVBar] = []
    for i in range(24):
        c = base + Decimal("10") if i % 2 == 0 else base - Decimal("10")
        bars.append(_make_bar_1h(close=c))
    bars[-1] = _make_bar_1h(close=base)
    candidate = _make_candidate(candidate_type=CandidateType.BREAKOUT_LONG)
    decision, report = filt.evaluate_with_report(candidate=candidate, bars_1h=bars)
    assert not decision.allow
    assert decision.reason == "regime_mismatch"
    assert "state_range" in report.failed_conditions
    assert report.state == RegimeState.RANGE.value


def test_evaluate_with_report_regime_mismatch_short_vs_long() -> None:
    """Report shows candidate_short_vs_long_regime when short candidate in long regime."""
    filt = RegimeFilter(
        RegimeFilterConfig(
            trend_threshold_bps=Decimal("5"),
            min_volatility_bps=Decimal("1"),
            max_volatility_bps=Decimal("500"),
        )
    )
    base = Decimal("40000")
    bars = [_make_bar_1h(close=base + Decimal(i * 20)) for i in range(24)]
    candidate = _make_candidate(candidate_type=CandidateType.BREAKOUT_SHORT)
    decision, report = filt.evaluate_with_report(candidate=candidate, bars_1h=bars)
    assert not decision.allow
    assert decision.reason == "regime_mismatch"
    assert "candidate_short_vs_long_regime" in report.failed_conditions
    assert report.state == RegimeState.RISK_ON_LONG.value


def test_evaluate_with_report_regime_match_long() -> None:
    """Report shows allow=True and empty failed_conditions when long candidate in long regime."""
    filt = RegimeFilter(
        RegimeFilterConfig(
            trend_threshold_bps=Decimal("5"),
            min_volatility_bps=Decimal("1"),
            max_volatility_bps=Decimal("500"),
        )
    )
    base = Decimal("40000")
    bars = [_make_bar_1h(close=base + Decimal(i * 20)) for i in range(24)]
    candidate = _make_candidate(candidate_type=CandidateType.BREAKOUT_LONG)
    decision, report = filt.evaluate_with_report(candidate=candidate, bars_1h=bars)
    assert decision.allow
    assert report.allow
    assert report.reason == "regime_match"
    assert report.failed_conditions == ()
    assert report.state == RegimeState.RISK_ON_LONG.value


def test_evaluate_unchanged_behavior() -> None:
    """evaluate() returns same decision as evaluate_with_report (no behavior change)."""
    filt = RegimeFilter()
    bars = [_make_bar_1h() for _ in range(24)]
    candidate = _make_candidate()
    decision_direct = filt.evaluate(candidate=candidate, bars_1h=bars)
    decision_with_report, _ = filt.evaluate_with_report(candidate=candidate, bars_1h=bars)
    assert decision_direct.allow == decision_with_report.allow
    assert decision_direct.reason == decision_with_report.reason
    assert decision_direct.state == decision_with_report.state


def test_regime_filter_report_to_log_dict() -> None:
    """to_log_dict returns operator-friendly dict with expected keys."""
    filt = RegimeFilter()
    bars = [_make_bar_1h() for _ in range(20)]
    candidate = _make_candidate(symbol="ETHUSDT", candidate_type=CandidateType.BREAKOUT_SHORT)
    _, report = filt.evaluate_with_report(candidate=candidate, bars_1h=bars)
    d = report.to_log_dict()
    assert d["symbol"] == "ETHUSDT"
    assert d["candidate_type"] == "breakout_short"
    assert d["allow"] is False
    assert d["reason"] == "insufficient_1h_context"
    assert "failed_conditions" in d
    assert isinstance(d["failed_conditions"], list)
    assert "volatility_bps" in d
    assert "trend_bps" in d
    assert "adaptive_trend_threshold_bps" in d
    assert "trend_threshold_bps" in d
    assert "max_volatility_bps" in d
    assert "min_volatility_bps" in d
    assert "max_abs_funding_bps" in d
    assert "bars_1h_count" in d


def test_regime_filter_report_includes_funding_when_provided() -> None:
    """to_log_dict includes funding_rate_bps when provided."""
    filt = RegimeFilter()
    bars = [_make_bar_1h() for _ in range(24)]
    candidate = _make_candidate()
    _, report = filt.evaluate_with_report(
        candidate=candidate,
        bars_1h=bars,
        funding_rate_bps=Decimal("5"),
    )
    d = report.to_log_dict()
    assert "funding_rate_bps" in d
    assert d["funding_rate_bps"] == 5.0


def test_funding_extreme_rejection() -> None:
    """Report shows funding_extreme when |funding| > max_abs_funding_bps."""
    filt = RegimeFilter(
        RegimeFilterConfig(
            max_abs_funding_bps=Decimal("5"),
            min_volatility_bps=Decimal("1"),
            max_volatility_bps=Decimal("500"),
        )
    )
    base = Decimal("40000")
    bars: list[OHLCVBar] = []
    for i in range(24):
        c = base + Decimal("10") if i % 2 == 0 else base - Decimal("10")
        bars.append(_make_bar_1h(close=c))
    bars[-1] = _make_bar_1h(close=base)
    candidate = _make_candidate()
    decision, report = filt.evaluate_with_report(
        candidate=candidate,
        bars_1h=bars,
        funding_rate_bps=Decimal("10"),
    )
    assert not decision.allow
    assert decision.reason == "funding_extreme"
    assert report.failed_conditions == ("funding_extreme",)
    assert report.funding_rate_bps == Decimal("10")


@pytest.mark.asyncio
async def test_session_summary_includes_last_regime_rejection() -> None:
    """Session summary includes last_regime_rejection when regime rejected a candidate."""
    from unittest.mock import MagicMock, patch

    from trading.runtime.orchestrator import RuntimeOrchestrator
    from trading.settings import load_settings

    settings = load_settings()
    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)
        orch._last_regime_rejection = {
            "symbol": "BTCUSDT",
            "reason": "regime_mismatch",
            "failed_conditions": ["state_range"],
            "state": "range",
            "candidate_type": "breakout_long",
            "volatility_bps": 25.0,
            "trend_bps": 3.0,
            "adaptive_trend_threshold_bps": 13.25,
            "bars_1h_count": 24,
        }
        summary = await orch._build_session_summary()
        md = orch._build_markdown_summary(summary)

    assert "last_regime_rejection" in summary
    lrr = summary["last_regime_rejection"]
    assert lrr["symbol"] == "BTCUSDT"
    assert lrr["reason"] == "regime_mismatch"
    assert lrr["failed_conditions"] == ["state_range"]
    assert "## Last Regime Rejection" in md
    assert "BTCUSDT" in md
    assert "regime_mismatch" in md
    assert "state_range" in md


@pytest.mark.asyncio
async def test_candidate_readiness_includes_regime_rejection_in_markdown() -> None:
    """Markdown summary shows regime_rejection under Candidate Readiness when present."""
    from unittest.mock import MagicMock, patch

    from trading.runtime.orchestrator import RuntimeOrchestrator
    from trading.settings import load_settings

    settings = load_settings()
    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)
        orch._last_candidate_readiness["BTCUSDT"] = {
            "symbol": "BTCUSDT",
            "bars_5m": 25,
            "bars_1h": 24,
            "has_enough_5m": True,
            "has_enough_1h": True,
            "reason": "ready",
            "candidate_count": 2,
            "regime_rejection": {
                "symbol": "BTCUSDT",
                "reason": "regime_mismatch",
                "failed_conditions": ["candidate_short_vs_long_regime"],
                "state": "risk_on_long",
                "trend_bps": 18.5,
                "volatility_bps": 42.0,
            },
        }
        summary = await orch._build_session_summary()
        md = orch._build_markdown_summary(summary)

    assert "regime_rejection" in md
    assert "regime_mismatch" in md
    assert "candidate_short_vs_long_regime" in md
    assert "risk_on_long" in md
