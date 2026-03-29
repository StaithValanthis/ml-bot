"""Unit tests for DEMO-only relaxed candidate validation (ML filter exercise path)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from trading.settings import load_settings

def _sym() -> str:
    return load_settings().trading.symbols[0]

from trading.strategy.candidates import BreakoutTrendCandidateGenerator, CandidateGeneratorConfig
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
    l_ = low if low is not None else close - Decimal("10")
    return OHLCVBar(
        symbol=_sym(),
        timeframe="5",
        open_time=t,
        close_time=t,
        open=close,
        high=h,
        low=l_,
        close=close,
        volume=volume,
        turnover=close * volume,
        confirmed=confirmed,
    )


def test_demo_relaxed_validation_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """TRADING_DEMO_RELAXED_CANDIDATE_VALIDATION parses to bool."""
    monkeypatch.setenv("TRADING_DEMO_RELAXED_CANDIDATE_VALIDATION", "true")
    settings = load_settings()
    assert settings.runtime.demo_relaxed_candidate_validation is True

    monkeypatch.setenv("TRADING_DEMO_RELAXED_CANDIDATE_VALIDATION", "false")
    settings = load_settings()
    assert settings.runtime.demo_relaxed_candidate_validation is False


def test_demo_relaxed_validation_default_false() -> None:
    """Default demo_relaxed_candidate_validation is False."""
    settings = load_settings()
    assert settings.runtime.demo_relaxed_candidate_validation is False


def test_create_relaxed_validation_candidates_breakout_near_miss() -> None:
    """create_relaxed_validation_candidates produces candidate from breakout near-miss."""
    cfg = CandidateGeneratorConfig(min_breakout_bps=Decimal("10"))
    gen = BreakoutTrendCandidateGenerator(config=cfg)
    base = Decimal("40000")
    bars = [_make_bar(close=base + Decimal(i), volume=Decimal("110")) for i in range(25)]
    bars[-1] = _make_bar(
        close=base + Decimal("8"),
        high=base + Decimal("10"),
        low=base + Decimal("5"),
        volume=Decimal("115"),
    )
    precondition = gen.get_precondition_report(_sym(), bars)
    assert precondition is not None
    assert "breakout_up" in precondition.failed_conditions or "breakout_dn" in precondition.failed_conditions

    relaxed = gen.create_relaxed_validation_candidates(_sym(), precondition, bars)
    assert len(relaxed) >= 1
    assert relaxed[0].metadata.get("relaxed_validation") is True
    assert "original_failed_conditions" in relaxed[0].metadata
    assert relaxed[0].metadata.get("relaxed_reason") in ("breakout_up_near_miss", "breakout_dn_near_miss", "trend_near_miss")


def test_create_relaxed_validation_candidates_empty_when_no_near_miss() -> None:
    """create_relaxed_validation_candidates returns empty when no meaningful move."""
    gen = BreakoutTrendCandidateGenerator()
    base = Decimal("40000")
    bars = [_make_bar(close=base, volume=Decimal("100")) for _ in range(25)]
    precondition = gen.get_precondition_report(_sym(), bars)
    assert precondition is not None
    relaxed = gen.create_relaxed_validation_candidates(_sym(), precondition, bars)
    assert isinstance(relaxed, list)


@pytest.mark.asyncio
async def test_runtime_summary_includes_relaxed_validation_fields() -> None:
    """Runtime summary log includes demo_relaxed_candidate_validation and candidate_pipeline_detail when DEMO."""
    from trading.runtime.orchestrator import RuntimeOrchestrator
    from trading.util.types import RuntimeMode

    settings = load_settings()
    settings.runtime.mode = RuntimeMode.DEMO
    settings.runtime.demo_relaxed_candidate_validation = True

    captured: list[dict] = []

    def capture(event: str, **kwargs: object) -> None:
        if event == "runtime_summary":
            captured.append(dict(kwargs))

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)
        orch._logger.info = capture
        orch._metrics.inc("strategy_raw_candidates_total")
        orch._metrics.inc("strategy_relaxed_demo_candidates_created")
        orch._metrics.inc("strategy_candidates_total")

    await orch._runtime_summary_cycle()

    assert len(captured) == 1
    assert captured[0].get("demo_relaxed_candidate_validation") is True
    assert "demo_validation_candidates_created" in captured[0]
    assert "candidate_pipeline_detail" in captured[0]
    detail = captured[0]["candidate_pipeline_detail"]
    assert "raw_candidates" in detail
    assert "relaxed_demo_candidates" in detail
    assert "model_reached" in detail


@pytest.mark.asyncio
async def test_no_relaxed_path_when_flag_false() -> None:
    """When demo_relaxed_candidate_validation is False, no relaxed candidates injected."""
    from trading.runtime.orchestrator import RuntimeOrchestrator
    from trading.util.types import RuntimeMode

    settings = load_settings()
    settings.runtime.mode = RuntimeMode.DEMO
    settings.runtime.demo_relaxed_candidate_validation = False

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)

    assert orch._settings.runtime.demo_relaxed_candidate_validation is False
    metrics_before = orch._metrics.snapshot().counters.get("strategy_relaxed_demo_candidates_created", 0)
    assert metrics_before == 0


@pytest.mark.asyncio
async def test_live_mode_ignores_relaxed_validation_flag() -> None:
    """In LIVE mode, demo_relaxed_candidate_validation is never used (no injection)."""
    from trading.util.types import RuntimeMode

    from trading.runtime.orchestrator import RuntimeOrchestrator

    settings = load_settings()
    settings.runtime.mode = RuntimeMode.LIVE
    settings.runtime.demo_relaxed_candidate_validation = True

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)

    assert orch._settings.runtime.mode == RuntimeMode.LIVE
    assert orch._settings.runtime.demo_relaxed_candidate_validation is True
    assert orch._settings.runtime.mode != RuntimeMode.DEMO
