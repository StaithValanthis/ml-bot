"""Unit tests for DEMO-only candidate threshold overrides."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from trading.settings import load_settings
from trading.strategy.candidates import (
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


def test_default_config_thresholds() -> None:
    """Default config uses conservative thresholds."""
    cfg = CandidateGeneratorConfig()
    assert cfg.min_breakout_bps == Decimal("5")
    assert cfg.min_trend_bps == Decimal("8")
    assert cfg.min_volume_multiplier == Decimal("1.1")


def test_relaxed_config_produces_more_candidates() -> None:
    """Relaxed DEMO thresholds allow candidates that default would reject."""
    default_gen = BreakoutTrendCandidateGenerator()
    relaxed_cfg = CandidateGeneratorConfig(
        min_breakout_bps=Decimal("3"),
        min_trend_bps=Decimal("5"),
        min_volume_multiplier=Decimal("1.05"),
    )
    relaxed_gen = BreakoutTrendCandidateGenerator(config=relaxed_cfg)

    base = Decimal("40000")
    bars_5m: list[OHLCVBar] = []
    for i in range(21):
        bars_5m.append(
            _make_bar(
                close=base + Decimal(i * 10),
                high=base + Decimal(i * 10) + Decimal("5"),
                low=base + Decimal(i * 10) - Decimal("5"),
                volume=Decimal("110"),
            )
        )
    bars_5m.append(
        _make_bar(
            close=base + Decimal("218"),
            high=base + Decimal("220"),
            low=base + Decimal("215"),
            volume=Decimal("116"),
        )
    )

    default_candidates = default_gen.on_closed_candle("BTCUSDT", bars_5m)
    relaxed_candidates = relaxed_gen.on_closed_candle("BTCUSDT", bars_5m)

    assert len(default_candidates) == 0
    assert len(relaxed_candidates) >= 1


def test_demo_overrides_loaded_from_bybit_demo_config() -> None:
    """When env=bybit_demo and mode=demo, overrides are loaded."""
    import os

    orig_env = os.environ.get("TRADING_ENV")
    orig_mode = os.environ.get("TRADING_MODE")
    try:
        os.environ["TRADING_ENV"] = "bybit_demo"
        os.environ["TRADING_MODE"] = "demo"
        settings = load_settings()
        assert settings.runtime.mode.value == "demo"
        o = settings.runtime.demo_candidate_overrides
        assert o is not None
        assert o.min_breakout_bps == 3.0
        assert o.min_trend_bps == 5.0
        assert o.min_volume_multiplier == 1.05
    finally:
        if orig_env is not None:
            os.environ["TRADING_ENV"] = orig_env
        elif "TRADING_ENV" in os.environ:
            del os.environ["TRADING_ENV"]
        if orig_mode is not None:
            os.environ["TRADING_MODE"] = orig_mode
        elif "TRADING_MODE" in os.environ:
            del os.environ["TRADING_MODE"]


@pytest.mark.asyncio
async def test_orchestrator_applies_demo_overrides_when_mode_demo() -> None:
    """Orchestrator uses relaxed config when mode is DEMO and overrides present."""
    from trading.runtime.orchestrator import RuntimeOrchestrator
    from trading.settings import AppSettings
    from trading.util.types import RuntimeMode

    settings = load_settings()
    settings.runtime.mode = RuntimeMode.DEMO
    from trading.settings import DemoCandidateOverrides

    settings.runtime.demo_candidate_overrides = DemoCandidateOverrides(
        min_breakout_bps=3.0,
        min_trend_bps=5.0,
        min_volume_multiplier=1.05,
    )

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)

    gen = orch._candidate_generator
    assert gen._cfg.min_breakout_bps == Decimal("3")
    assert gen._cfg.min_trend_bps == Decimal("5")
    assert gen._cfg.min_volume_multiplier == Decimal("1.05")


@pytest.mark.asyncio
async def test_orchestrator_uses_default_config_when_mode_paper() -> None:
    """Orchestrator uses default config when mode is PAPER (overrides ignored)."""
    from trading.runtime.orchestrator import RuntimeOrchestrator
    from trading.settings import DemoCandidateOverrides
    from trading.util.types import RuntimeMode

    settings = load_settings()
    settings.runtime.mode = RuntimeMode.PAPER
    settings.runtime.demo_candidate_overrides = DemoCandidateOverrides(
        min_breakout_bps=3.0,
        min_trend_bps=5.0,
        min_volume_multiplier=1.05,
    )

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)

    gen = orch._candidate_generator
    assert gen._cfg.min_breakout_bps == Decimal("5")
    assert gen._cfg.min_trend_bps == Decimal("8")
    assert gen._cfg.min_volume_multiplier == Decimal("1.1")


@pytest.mark.asyncio
async def test_orchestrator_applies_more_opportunities_profile_when_enabled() -> None:
    """Orchestrator uses more opportunities profile when DEMO and demo_more_opportunities_enabled."""
    from trading.runtime.orchestrator import RuntimeOrchestrator
    from trading.settings import DemoCandidateOverrides
    from trading.util.types import RuntimeMode

    settings = load_settings()
    settings.runtime.mode = RuntimeMode.DEMO
    settings.runtime.demo_candidate_overrides = DemoCandidateOverrides(
        min_breakout_bps=3.0,
        min_trend_bps=5.0,
        min_volume_multiplier=1.05,
    )
    settings.runtime.demo_more_opportunities_enabled = True

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)

    gen = orch._candidate_generator
    assert gen._cfg.min_breakout_bps == Decimal("1")
    assert gen._cfg.min_trend_bps == Decimal("2")
    assert gen._cfg.min_volume_multiplier == Decimal("1.0")
    assert gen._cfg.lookback_bars == 15


@pytest.mark.asyncio
async def test_orchestrator_uses_demo_overrides_when_more_opportunities_disabled() -> None:
    """Orchestrator uses demo_candidate_overrides when more_opportunities is disabled."""
    from trading.runtime.orchestrator import RuntimeOrchestrator
    from trading.settings import DemoCandidateOverrides
    from trading.util.types import RuntimeMode

    settings = load_settings()
    settings.runtime.mode = RuntimeMode.DEMO
    settings.runtime.demo_candidate_overrides = DemoCandidateOverrides(
        min_breakout_bps=3.0,
        min_trend_bps=5.0,
        min_volume_multiplier=1.05,
    )
    settings.runtime.demo_more_opportunities_enabled = False

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)

    gen = orch._candidate_generator
    assert gen._cfg.min_breakout_bps == Decimal("3")
    assert gen._cfg.min_trend_bps == Decimal("5")
    assert gen._cfg.min_volume_multiplier == Decimal("1.05")
    assert gen._cfg.lookback_bars == 20


def test_more_opportunities_profile_produces_more_candidates() -> None:
    """More opportunities profile allows candidates that relaxed overrides would reject."""
    more_opps_cfg = CandidateGeneratorConfig(
        min_breakout_bps=Decimal("1"),
        min_trend_bps=Decimal("2"),
        min_volume_multiplier=Decimal("1.0"),
        lookback_bars=15,
    )
    relaxed_cfg = CandidateGeneratorConfig(
        min_breakout_bps=Decimal("3"),
        min_trend_bps=Decimal("5"),
        min_volume_multiplier=Decimal("1.05"),
    )
    more_opps_gen = BreakoutTrendCandidateGenerator(config=more_opps_cfg)
    relaxed_gen = BreakoutTrendCandidateGenerator(config=relaxed_cfg)

    base = Decimal("40000")
    bars_5m: list[OHLCVBar] = []
    for i in range(22):
        bars_5m.append(
            _make_bar(
                close=base + Decimal(i * 4),
                high=base + Decimal(i * 4) + Decimal("2"),
                low=base + Decimal(i * 4) - Decimal("2"),
                volume=Decimal("102"),
            )
        )
    bars_5m.append(
        _make_bar(
            close=base + Decimal("92"),
            high=base + Decimal("94"),
            low=base + Decimal("89"),
            volume=Decimal("106"),
        )
    )

    relaxed_candidates = relaxed_gen.on_closed_candle("BTCUSDT", bars_5m)
    more_opps_candidates = more_opps_gen.on_closed_candle("BTCUSDT", bars_5m)

    assert len(relaxed_candidates) == 0
    assert len(more_opps_candidates) >= 1


@pytest.mark.asyncio
async def test_session_summary_includes_demo_more_opportunities_when_enabled() -> None:
    """Session summary and markdown include demo_more_opportunities_enabled when DEMO."""
    from trading.runtime.orchestrator import RuntimeOrchestrator
    from trading.util.types import RuntimeMode

    settings = load_settings()
    settings.runtime.mode = RuntimeMode.DEMO
    settings.runtime.demo_more_opportunities_enabled = True

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)

    summary = await orch._build_session_summary()
    assert summary.get("demo_more_opportunities_enabled") is True

    md = orch._build_markdown_summary(summary)
    assert "## Demo Profile" in md
    assert "More opportunities: enabled" in md
