"""Unit tests for breakout precondition visibility and no-pattern-match reporting."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading.strategy.candidates import (
    BreakoutPreconditionReport,
    BreakoutTrendCandidateGenerator,
    CandidateGeneratorConfig,
)
from trading.util.types import OHLCVBar


def _make_bar(
    close: Decimal = Decimal("40000"),
    high: Decimal | None = None,
    low: Decimal | None = None,
    volume: Decimal = Decimal("100"),
    confirmed: bool = True,
) -> OHLCVBar:
    t = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    h = high if high is not None else close + Decimal("10")
    l = low if low is not None else close - Decimal("10")
    return OHLCVBar(
        symbol="BTCUSDT",
        timeframe="5",
        open_time=t,
        close_time=t,
        open=close,
        high=h,
        low=l,
        close=close,
        volume=volume,
        turnover=close * volume,
        confirmed=confirmed,
    )


def test_get_precondition_report_returns_none_when_insufficient_bars() -> None:
    """When bars_5m < lookback+2, returns None."""
    gen = BreakoutTrendCandidateGenerator()
    bars_5m = [_make_bar() for _ in range(20)]
    assert gen.get_precondition_report("BTCUSDT", bars_5m) is None


def test_get_precondition_report_returns_none_when_unconfirmed_in_window() -> None:
    """When last lookback+2 bars have unconfirmed, returns None."""
    gen = BreakoutTrendCandidateGenerator()
    bars_5m = [_make_bar(confirmed=True) for _ in range(25)]
    bars_5m[-1] = _make_bar(confirmed=False)
    assert gen.get_precondition_report("BTCUSDT", bars_5m) is None


def test_get_precondition_report_all_conditions_failed() -> None:
    """When no breakout/trend, all conditions fail; report shows evaluated values."""
    gen = BreakoutTrendCandidateGenerator()
    base = Decimal("40000")
    bars_5m: list[OHLCVBar] = []
    for i in range(22):
        bars_5m.append(
            _make_bar(
                close=base + Decimal(i),
                high=base + Decimal(i) + Decimal("5"),
                low=base + Decimal(i) - Decimal("5"),
                volume=Decimal("100"),
            )
        )
    bars_5m[-1] = _make_bar(
        close=base + Decimal("1"),
        high=base + Decimal("6"),
        low=base - Decimal("4"),
        volume=Decimal("50"),
    )

    report = gen.get_precondition_report("BTCUSDT", bars_5m)
    assert report is not None
    assert report.symbol == "BTCUSDT"
    assert report.breakout_up_ok is False
    assert report.breakout_dn_ok is False
    assert report.trend_long_ok is False
    assert report.trend_short_ok is False
    assert set(report.failed_conditions) == {"breakout_up", "breakout_dn", "trend_long", "trend_short"}
    assert report.min_breakout_bps == Decimal("5")
    assert report.min_trend_bps == Decimal("8")
    assert report.min_volume_multiplier == Decimal("1.1")


def test_get_precondition_report_to_log_dict() -> None:
    """to_log_dict returns operator-friendly dict with float values."""
    gen = BreakoutTrendCandidateGenerator()
    bars_5m = [_make_bar() for _ in range(25)]
    report = gen.get_precondition_report("ETHUSDT", bars_5m)
    assert report is not None
    d = report.to_log_dict()
    assert d["symbol"] == "ETHUSDT"
    assert "breakout_up_bps" in d
    assert "breakout_dn_bps" in d
    assert "candle_move_bps" in d
    assert "vol_multiplier" in d
    assert "failed_conditions" in d
    assert isinstance(d["breakout_up_bps"], float)
    assert isinstance(d["failed_conditions"], list)


def test_get_precondition_report_breakout_up_passes() -> None:
    """When close exceeds lookback_high by >=5bps, breakout_up_ok is True."""
    gen = BreakoutTrendCandidateGenerator()
    base = Decimal("40000")
    bars_5m: list[OHLCVBar] = []
    for i in range(21):
        bars_5m.append(
            _make_bar(
                close=base + Decimal(i * 10),
                high=base + Decimal(i * 10) + Decimal("5"),
                low=base + Decimal(i * 10) - Decimal("5"),
            )
        )
    lookback_high = base + Decimal("200")
    bars_5m.append(
        _make_bar(
            close=lookback_high + Decimal("30"),
            high=lookback_high + Decimal("35"),
            low=lookback_high,
            volume=Decimal("150"),
        )
    )

    report = gen.get_precondition_report("BTCUSDT", bars_5m)
    assert report is not None
    assert report.breakout_up_ok is True
    assert "breakout_up" not in report.failed_conditions


def test_get_precondition_report_uses_config_thresholds() -> None:
    """Report reflects config min_breakout_bps, min_trend_bps, min_volume_multiplier."""
    cfg = CandidateGeneratorConfig(
        min_breakout_bps=Decimal("10"),
        min_trend_bps=Decimal("12"),
        min_volume_multiplier=Decimal("1.5"),
    )
    gen = BreakoutTrendCandidateGenerator(config=cfg)
    bars_5m = [_make_bar() for _ in range(25)]
    report = gen.get_precondition_report("BTCUSDT", bars_5m)
    assert report is not None
    assert report.min_breakout_bps == Decimal("10")
    assert report.min_trend_bps == Decimal("12")
    assert report.min_volume_multiplier == Decimal("1.5")


@pytest.mark.asyncio
async def test_session_summary_includes_breakout_precondition_when_no_pattern_match() -> None:
    """Session summary includes breakout_precondition when reason is no_pattern_match."""
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
            "reason": "no_pattern_match",
            "candidate_count": 0,
            "breakout_precondition": {
                "symbol": "BTCUSDT",
                "breakout_up_bps": 2.5,
                "breakout_dn_bps": -1.0,
                "candle_move_bps": 3.0,
                "vol_multiplier": 0.9,
                "failed_conditions": ["breakout_up", "breakout_dn", "trend_long", "trend_short"],
            },
        }
        summary = await orch._build_session_summary()
        md = orch._build_markdown_summary(summary)

    assert "breakout_precondition" in md
    assert "failed=" in md
    assert "breakout_up" in md or "breakout_dn" in md
