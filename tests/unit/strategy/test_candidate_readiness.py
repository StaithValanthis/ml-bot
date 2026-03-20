"""Unit tests for candidate-generation readiness visibility."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading.strategy.candidates import (
    BreakoutTrendCandidateGenerator,
    CandidateGeneratorConfig,
    get_candidate_readiness,
)
from trading.util.types import OHLCVBar


def _make_bar(
    symbol: str = "BTCUSDT",
    timeframe: str = "5",
    close: Decimal = Decimal("40000"),
    confirmed: bool = True,
) -> OHLCVBar:
    t = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    return OHLCVBar(
        symbol=symbol,
        timeframe=timeframe,
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


def test_get_candidate_readiness_insufficient_5m_history() -> None:
    """When bars_5m < 22, reason is insufficient_5m_history."""
    bars_5m = [_make_bar() for _ in range(15)]
    bars_1h = [_make_bar(timeframe="60") for _ in range(30)]
    r = get_candidate_readiness("BTCUSDT", bars_5m, bars_1h, lookback_bars=20)
    assert r["bars_5m"] == 15
    assert r["bars_1h"] == 30
    assert r["has_enough_5m"] is False
    assert r["has_enough_1h"] is True
    assert r["reason"] == "insufficient_5m_history"


def test_get_candidate_readiness_unconfirmed_in_window() -> None:
    """When last 22 bars have unconfirmed, reason is unconfirmed_5m_in_window."""
    bars_5m = [_make_bar(confirmed=True) for _ in range(25)]
    bars_5m[-1] = _make_bar(confirmed=False)
    bars_1h = [_make_bar(timeframe="60") for _ in range(30)]
    r = get_candidate_readiness("BTCUSDT", bars_5m, bars_1h, lookback_bars=20)
    assert r["bars_5m"] == 25
    assert r["has_enough_5m"] is True
    assert r["unconfirmed_in_5m_window"] is True
    assert r["reason"] == "unconfirmed_5m_in_window"


def test_get_candidate_readiness_ready() -> None:
    """When enough bars and all confirmed, reason is ready."""
    bars_5m = [_make_bar() for _ in range(25)]
    bars_1h = [_make_bar(timeframe="60") for _ in range(30)]
    r = get_candidate_readiness("BTCUSDT", bars_5m, bars_1h, lookback_bars=20)
    assert r["bars_5m"] == 25
    assert r["has_enough_5m"] is True
    assert r["unconfirmed_in_5m_window"] is False
    assert r["reason"] == "ready"


def test_get_candidate_readiness_insufficient_1h() -> None:
    """When bars_1h < 24, has_enough_1h is False."""
    bars_5m = [_make_bar() for _ in range(25)]
    bars_1h = [_make_bar(timeframe="60") for _ in range(12)]
    r = get_candidate_readiness("BTCUSDT", bars_5m, bars_1h, lookback_bars=20)
    assert r["has_enough_1h"] is False
    assert r["bars_1h"] == 12


def test_generator_get_readiness_uses_config() -> None:
    """Generator get_readiness uses its lookback_bars config."""
    cfg = CandidateGeneratorConfig(lookback_bars=10)
    gen = BreakoutTrendCandidateGenerator(config=cfg)
    bars_5m = [_make_bar() for _ in range(15)]
    bars_1h = [_make_bar(timeframe="60") for _ in range(30)]
    r = gen.get_readiness("ETHUSDT", bars_5m, bars_1h)
    assert r["min_5m_required"] == 12
    assert r["has_enough_5m"] is True
    assert r["symbol"] == "ETHUSDT"


@pytest.mark.asyncio
async def test_session_summary_includes_candidate_readiness() -> None:
    """Session summary includes candidate_readiness when symbols processed."""
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
            "bars_5m": 15,
            "bars_1h": 12,
            "has_enough_5m": False,
            "has_enough_1h": False,
            "reason": "insufficient_5m_history",
            "candidate_count": 0,
        }
        summary = await orch._build_session_summary()

    assert "candidate_readiness" in summary
    readiness = summary["candidate_readiness"]
    assert "BTCUSDT" in readiness
    assert readiness["BTCUSDT"]["bars_5m"] == 15
    assert readiness["BTCUSDT"]["reason"] == "insufficient_5m_history"


@pytest.mark.asyncio
async def test_markdown_summary_includes_candidate_readiness_section() -> None:
    """Markdown summary includes Candidate Readiness section."""
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
            "bars_5m": 22,
            "bars_1h": 24,
            "has_enough_5m": True,
            "has_enough_1h": True,
            "reason": "no_pattern_match",
            "candidate_count": 0,
        }
        summary = await orch._build_session_summary()
        md = orch._build_markdown_summary(summary)

    assert "## Candidate Readiness" in md
    assert "BTCUSDT" in md
    assert "bars_5m=22" in md
    assert "bars_1h=24" in md
    assert "no_pattern_match" in md
