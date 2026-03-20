"""Unit tests for multi-symbol DEMO candidate observation."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from trading.runtime.orchestrator import RuntimeOrchestrator
from trading.settings import load_settings


@pytest.mark.asyncio
async def test_demo_config_loads_multi_symbol_basket() -> None:
    """When env=bybit_demo, trading.symbols includes BTCUSDT, ETHUSDT, SOLUSDT, LINKUSDT."""
    orig_env = os.environ.get("TRADING_ENV")
    orig_mode = os.environ.get("TRADING_MODE")
    try:
        os.environ["TRADING_ENV"] = "bybit_demo"
        os.environ["TRADING_MODE"] = "demo"
        settings = load_settings()
        symbols = settings.trading.symbols
        assert "BTCUSDT" in symbols
        assert "ETHUSDT" in symbols
        assert "SOLUSDT" in symbols
        assert "LINKUSDT" in symbols
        assert len(symbols) >= 4
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
async def test_multi_symbol_readiness_in_session_summary() -> None:
    """Session summary includes per-symbol candidate_readiness for all configured symbols."""
    settings = load_settings()
    settings.trading.symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

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
            "reason": "no_pattern_match",
            "candidate_count": 0,
        }
        orch._last_candidate_readiness["ETHUSDT"] = {
            "symbol": "ETHUSDT",
            "bars_5m": 22,
            "bars_1h": 24,
            "reason": "ready",
            "candidate_count": 0,
        }
        orch._last_candidate_readiness["SOLUSDT"] = {
            "symbol": "SOLUSDT",
            "bars_5m": 20,
            "bars_1h": 18,
            "reason": "insufficient_5m_history",
            "candidate_count": 0,
        }
        summary = await orch._build_session_summary()

    readiness = summary["candidate_readiness"]
    assert "BTCUSDT" in readiness
    assert "ETHUSDT" in readiness
    assert "SOLUSDT" in readiness
    assert readiness["BTCUSDT"]["bars_5m"] == 25
    assert readiness["ETHUSDT"]["reason"] == "ready"
    assert readiness["SOLUSDT"]["reason"] == "insufficient_5m_history"


@pytest.mark.asyncio
async def test_multi_symbol_markdown_summary_per_symbol() -> None:
    """Markdown summary includes Candidate Readiness per symbol."""
    settings = load_settings()
    settings.trading.symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

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
            "reason": "no_pattern_match",
            "candidate_count": 0,
        }
        orch._last_candidate_readiness["ETHUSDT"] = {
            "symbol": "ETHUSDT",
            "bars_5m": 25,
            "bars_1h": 24,
            "reason": "no_pattern_match",
            "candidate_count": 0,
        }
        orch._last_candidate_readiness["SOLUSDT"] = {
            "symbol": "SOLUSDT",
            "bars_5m": 25,
            "bars_1h": 24,
            "reason": "no_pattern_match",
            "candidate_count": 0,
        }
        summary = await orch._build_session_summary()
        md = orch._build_markdown_summary(summary)

    assert "## Candidate Readiness" in md
    assert "BTCUSDT" in md
    assert "ETHUSDT" in md
    assert "SOLUSDT" in md


@pytest.mark.asyncio
async def test_multi_symbol_warmup_post_snapshot_structure() -> None:
    """Bar_history supports all configured symbols; warmup iterates over them."""
    settings = load_settings()
    settings.trading.symbols = ["BTCUSDT", "ETHUSDT"]

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)

    assert orch._settings.trading.symbols == ["BTCUSDT", "ETHUSDT"]
    for sym in settings.trading.symbols:
        for tf in ["5", "60"]:
            _ = orch._bar_history[sym][tf]
        assert len(orch._bar_history[sym]) == 2
