"""Unit tests for multi-symbol DEMO candidate observation."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from trading.runtime.orchestrator import RuntimeOrchestrator
from trading.settings import load_settings


@pytest.mark.asyncio
async def test_demo_config_loads_multi_symbol_basket() -> None:
    """When env=bybit_demo, trading.symbols is demo basket intersected with symbols.yaml metadata."""
    orig_env = os.environ.get("TRADING_ENV")
    orig_mode = os.environ.get("TRADING_MODE")
    try:
        os.environ["TRADING_ENV"] = "bybit_demo"
        os.environ["TRADING_MODE"] = "demo"
        settings = load_settings()
        symbols = settings.trading.symbols
        meta = set(settings.symbols.keys())
        demo_pref = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT"]
        expected = [s for s in demo_pref if s in meta]
        assert symbols == expected
        assert len(symbols) >= 1
        assert set(symbols) <= meta
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
    symbols = list(settings.trading.symbols)
    assert len(symbols) >= 1

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)
        for i, sym in enumerate(symbols):
            orch._last_candidate_readiness[sym] = {
                "symbol": sym,
                "bars_5m": 25 - i,
                "bars_1h": 24 - min(i, 6),
                "reason": "no_pattern_match" if i == 0 else ("ready" if i == 1 else "insufficient_5m_history"),
                "candidate_count": 0,
            }
        summary = await orch._build_session_summary()

    readiness = summary["candidate_readiness"]
    for sym in symbols:
        assert sym in readiness
    assert readiness[symbols[0]]["bars_5m"] == 25
    if len(symbols) > 1:
        assert readiness[symbols[1]]["reason"] == "ready"
    if len(symbols) > 2:
        assert readiness[symbols[2]]["reason"] == "insufficient_5m_history"


@pytest.mark.asyncio
async def test_multi_symbol_markdown_summary_per_symbol() -> None:
    """Markdown summary includes Candidate Readiness per symbol."""
    settings = load_settings()
    symbols = list(settings.trading.symbols)
    assert len(symbols) >= 1

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)
        for sym in symbols:
            orch._last_candidate_readiness[sym] = {
                "symbol": sym,
                "bars_5m": 25,
                "bars_1h": 24,
                "reason": "no_pattern_match",
                "candidate_count": 0,
            }
        summary = await orch._build_session_summary()
        md = orch._build_markdown_summary(summary)

    assert "## Candidate Readiness" in md
    for sym in symbols:
        assert sym in md


@pytest.mark.asyncio
async def test_multi_symbol_warmup_post_snapshot_structure() -> None:
    """Bar_history supports all configured symbols; warmup iterates over them."""
    settings = load_settings()
    symbols = list(settings.trading.symbols)
    assert len(symbols) >= 1

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)

    assert orch._settings.trading.symbols == symbols
    for sym in settings.trading.symbols:
        for tf in ["5", "60"]:
            _ = orch._bar_history[sym][tf]
        assert len(orch._bar_history[sym]) == 2
