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
