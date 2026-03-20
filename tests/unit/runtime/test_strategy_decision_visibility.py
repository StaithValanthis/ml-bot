"""Unit tests for strategy decision flow visibility and blocking stage inference."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from trading.runtime.orchestrator import RuntimeOrchestrator
from trading.settings import load_settings


def test_infer_blocking_stage_no_bars() -> None:
    """When no bars confirmed, blocking stage is no_bars."""
    settings = load_settings()
    with patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()), patch(
        "trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()
    ), patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()):
        orch = RuntimeOrchestrator(settings)
    m = {"decisions_total": 0, "order_intents_total": 0, "strategy_bars_confirmed": 0}
    assert orch._infer_strategy_blocking_stage(m) == "no_bars"


def test_infer_blocking_stage_no_candidates() -> None:
    """When bars but no candidates, blocking stage is no_candidates."""
    settings = load_settings()
    with patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()), patch(
        "trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()
    ), patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()):
        orch = RuntimeOrchestrator(settings)
    m = {
        "decisions_total": 0,
        "order_intents_total": 0,
        "strategy_bars_confirmed": 10,
        "strategy_candidates_total": 0,
    }
    assert orch._infer_strategy_blocking_stage(m) == "no_candidates"


def test_infer_blocking_stage_regime_rejected() -> None:
    """When all candidates rejected by regime, blocking stage is regime_rejected."""
    settings = load_settings()
    with patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()), patch(
        "trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()
    ), patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()):
        orch = RuntimeOrchestrator(settings)
    m = {
        "decisions_total": 0,
        "order_intents_total": 0,
        "strategy_bars_confirmed": 10,
        "strategy_candidates_total": 5,
        "strategy_regime_rejected": 5,
    }
    assert orch._infer_strategy_blocking_stage(m) == "regime_rejected"


def test_infer_blocking_stage_signal_rejected() -> None:
    """When candidates pass regime but all rejected by signal/sizing, blocking is signal_rejected."""
    settings = load_settings()
    with patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()), patch(
        "trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()
    ), patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()):
        orch = RuntimeOrchestrator(settings)
    m = {
        "decisions_total": 0,
        "order_intents_total": 0,
        "strategy_bars_confirmed": 10,
        "strategy_candidates_total": 5,
        "strategy_regime_rejected": 2,
        "strategy_signal_rejected": 3,
    }
    assert orch._infer_strategy_blocking_stage(m) == "signal_rejected"


def test_infer_blocking_stage_risk_rejected() -> None:
    """When risk rejects, blocking stage is risk_rejected."""
    settings = load_settings()
    with patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()), patch(
        "trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()
    ), patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()):
        orch = RuntimeOrchestrator(settings)
    m = {
        "decisions_total": 0,
        "order_intents_total": 0,
        "strategy_bars_confirmed": 10,
        "strategy_candidates_total": 5,
        "strategy_regime_rejected": 0,
        "strategy_signal_rejected": 0,
        "strategy_risk_rejected": 5,
    }
    assert orch._infer_strategy_blocking_stage(m) == "risk_rejected"


def test_infer_blocking_stage_submitted_when_intents() -> None:
    """When intents exist, blocking stage is submitted."""
    settings = load_settings()
    with patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()), patch(
        "trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()
    ), patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()):
        orch = RuntimeOrchestrator(settings)
    m = {
        "decisions_total": 2,
        "order_intents_total": 2,
        "strategy_bars_confirmed": 10,
        "strategy_candidates_total": 5,
    }
    assert orch._infer_strategy_blocking_stage(m) == "submitted"


@pytest.mark.asyncio
async def test_session_summary_includes_strategy_flow_and_blocking_stage() -> None:
    """Session summary includes strategy_flow and blocking_stage."""
    settings = load_settings()
    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)
        orch._metrics.inc("strategy_bars_confirmed", 5)
        orch._metrics.inc("strategy_candidates_total", 3)
        orch._metrics.inc("strategy_regime_rejected", 2)
        orch._metrics.inc("strategy_signal_rejected", 1)

        summary = await orch._build_session_summary()

    assert "strategy_flow" in summary
    flow = summary["strategy_flow"]
    assert flow["bars_confirmed"] == 5
    assert flow["candidates"] == 3
    assert flow["regime_rejected"] == 2
    assert flow["signal_rejected"] == 1
    assert "blocking_stage" in summary


@pytest.mark.asyncio
async def test_markdown_summary_includes_strategy_flow_section() -> None:
    """Markdown summary includes Strategy Flow section with all counters."""
    settings = load_settings()
    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)
        summary = await orch._build_session_summary()
        md = orch._build_markdown_summary(summary)

    assert "## Strategy Flow" in md
    assert "Bars confirmed:" in md
    assert "Candidates:" in md
    assert "Regime rejected:" in md
    assert "Signal rejected:" in md
    assert "Sizing rejected:" in md
    assert "Risk rejected:" in md
    assert "Model filter reached:" in md
    assert "Model blocked:" in md
    assert "Submitted:" in md
    assert "Blocking stage:" in md
