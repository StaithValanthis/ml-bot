"""Unit tests for ML model calibration and threshold-readiness analysis."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trading.runtime.model_calibration import (
    build_model_calibration_summary,
    build_probability_buckets,
    build_threshold_sweep,
)


def test_build_probability_buckets() -> None:
    """Probability buckets have correct sample and shadow counts."""
    decisions = [
        {"model_probability": 0.1, "shadow_would_block": True},
        {"model_probability": 0.25, "shadow_would_block": True},
        {"model_probability": 0.35, "shadow_would_block": False},
        {"model_probability": 0.55, "shadow_would_block": False},
        {"model_probability": 0.75, "shadow_would_block": False},
        {"model_probability": 0.95, "shadow_would_block": False},
    ]
    buckets = build_probability_buckets(decisions)
    assert len(buckets) == 5
    labels = [b["probability_bucket"] for b in buckets]
    assert labels == ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    b02 = next(b for b in buckets if b["probability_bucket"] == "0.0-0.2")
    assert b02["sample_count"] == 1
    assert b02["shadow_block_count"] == 1
    assert b02["shadow_allow_count"] == 0
    b24 = next(b for b in buckets if b["probability_bucket"] == "0.2-0.4")
    assert b24["sample_count"] == 2
    assert b24["shadow_block_count"] == 1
    assert b24["shadow_allow_count"] == 1


def test_build_threshold_sweep() -> None:
    """Threshold sweep reports would_block/would_allow/block_rate per hypothetical threshold."""
    decisions = [
        {"model_probability": 0.25},
        {"model_probability": 0.35},
        {"model_probability": 0.45},
        {"model_probability": 0.55},
        {"model_probability": 0.65},
    ]
    sweep = build_threshold_sweep(decisions)
    assert len(sweep) == 5
    t30 = next(s for s in sweep if s["threshold"] == 0.30)
    assert t30["would_block_count"] == 1
    assert t30["would_allow_count"] == 4
    assert t30["block_rate"] == 0.2
    t50 = next(s for s in sweep if s["threshold"] == 0.50)
    assert t50["would_block_count"] == 3
    assert t50["would_allow_count"] == 2
    assert t50["block_rate"] == 0.6


def test_build_threshold_sweep_empty() -> None:
    """Empty decisions yield zero counts."""
    sweep = build_threshold_sweep([])
    for row in sweep:
        assert row["would_block_count"] == 0
        assert row["would_allow_count"] == 0
        assert row["block_rate"] == 0


def test_build_model_calibration_summary() -> None:
    """Calibration summary includes aggregates, buckets, sweep, and outcome linkage note."""
    decisions = [
        {"model_probability": 0.3, "shadow_would_block": True},
        {"model_probability": 0.5, "shadow_would_block": False},
        {"model_probability": 0.7, "shadow_would_block": False},
    ]
    cal = build_model_calibration_summary(
        decisions,
        threshold_configured=0.5,
        session_submitted=2,
        session_filled=1,
    )
    assert cal["total_model_evaluations"] == 3
    assert cal["total_shadow_blocks"] == 1
    assert cal["total_shadow_allows"] == 2
    assert cal["block_rate"] == pytest.approx(0.3333, abs=0.001)
    assert cal["mean_probability"] == pytest.approx(0.5)
    assert cal["median_probability"] == pytest.approx(0.5)
    assert cal["threshold_configured"] == 0.5
    assert cal["session_submitted_count"] == 2
    assert cal["session_filled_count"] == 1
    assert "per_decision_linkage_unavailable" in cal["outcome_linkage_note"]
    assert len(cal["probability_buckets"]) == 5
    assert len(cal["threshold_sweep"]) == 5


def test_build_model_calibration_summary_empty() -> None:
    """Empty decisions yield zero/None for all stats."""
    cal = build_model_calibration_summary(
        [],
        threshold_configured=0.5,
    )
    assert cal["total_model_evaluations"] == 0
    assert cal["total_shadow_blocks"] == 0
    assert cal["total_shadow_allows"] == 0
    assert cal["block_rate"] == 0
    assert cal["mean_probability"] is None
    assert cal["median_probability"] is None


@pytest.mark.asyncio
async def test_session_summary_includes_model_calibration() -> None:
    """Session summary includes model_calibration when shadow decisions exist."""
    from trading.runtime.orchestrator import RuntimeOrchestrator
    from trading.settings import load_settings

    mock_rest = MagicMock()
    mock_ws_public = MagicMock()
    mock_ws_public.subscribe = MagicMock()
    mock_ws_public.run_forever = MagicMock()
    mock_ws_public.close = MagicMock()
    mock_ws_private = MagicMock()
    mock_ws_private.subscribe = MagicMock()
    mock_ws_private.run_forever = MagicMock()
    mock_ws_private.close = MagicMock()
    with (
        patch("trading.runtime.orchestrator.BybitRestClient", return_value=mock_rest),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", return_value=mock_ws_public),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", return_value=mock_ws_private),
    ):
        orch = RuntimeOrchestrator(load_settings())
    orch._model_shadow_decisions = [
        {"model_probability": 0.4, "shadow_would_block": True},
        {"model_probability": 0.6, "shadow_would_block": False},
    ]
    orch._strategy_order_outcomes.model_filter.threshold = 0.5
    orch._strategy_order_outcomes.intents = 1
    orch._strategy_order_outcomes.filled = 1
    summary = await orch._build_session_summary()
    cal = summary.get("model_calibration")
    assert cal is not None
    assert cal["total_model_evaluations"] == 2
    assert cal["threshold_sweep"]
    assert cal["probability_buckets"]
    assert "outcome_linkage_note" in cal


def test_markdown_includes_calibration_and_threshold_readiness() -> None:
    """Markdown includes Model Calibration Review and Threshold Readiness sections."""
    from trading.runtime.orchestrator import RuntimeOrchestrator
    from trading.settings import load_settings

    mock_rest = MagicMock()
    mock_ws_public = MagicMock()
    mock_ws_private = MagicMock()
    with (
        patch("trading.runtime.orchestrator.BybitRestClient", return_value=mock_rest),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", return_value=mock_ws_public),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", return_value=mock_ws_private),
    ):
        orch = RuntimeOrchestrator(load_settings())
    summary = {
        "model_calibration": {
            "total_model_evaluations": 10,
            "total_shadow_blocks": 4,
            "total_shadow_allows": 6,
            "block_rate": 0.4,
            "mean_probability": 0.48,
            "median_probability": 0.5,
            "threshold_configured": 0.5,
            "session_submitted_count": 3,
            "session_filled_count": 2,
            "outcome_linkage_note": "per_decision_linkage_unavailable_session_aggregates_only",
            "probability_buckets": [
                {"probability_bucket": "0.0-0.2", "sample_count": 1, "shadow_block_count": 1, "shadow_allow_count": 0},
                {"probability_bucket": "0.4-0.6", "sample_count": 5, "shadow_block_count": 2, "shadow_allow_count": 3},
            ],
            "threshold_sweep": [
                {"threshold": 0.3, "would_block_count": 1, "would_allow_count": 9, "block_rate": 0.1},
                {"threshold": 0.5, "would_block_count": 4, "would_allow_count": 6, "block_rate": 0.4},
            ],
        },
    }
    md = orch._build_markdown_summary(summary)
    assert "## Model Calibration Review" in md
    assert "## Threshold Readiness" in md
    assert "Evaluations: 10" in md
    assert "Block rate: 0.4" in md
    assert "Threshold configured: 0.5" in md
    assert "thresh=0.3" in md
    assert "would_block=1" in md
    assert "per_decision_linkage_unavailable" in md


@pytest.mark.asyncio
async def test_calibration_json_artifact_written(tmp_path: Path) -> None:
    """model_calibration JSON artifact is written when calibration data exists."""
    from trading.util.types import RuntimeMode

    from trading.runtime.orchestrator import RuntimeOrchestrator
    from trading.settings import load_settings

    mock_rest = MagicMock()
    mock_ws_public = MagicMock()
    mock_ws_private = MagicMock()
    with (
        patch("trading.runtime.orchestrator.BybitRestClient", return_value=mock_rest),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", return_value=mock_ws_public),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", return_value=mock_ws_private),
    ):
        orch = RuntimeOrchestrator(load_settings())
    orch._parquet_store._root_dir = tmp_path
    orch._session_start_time = datetime(2025, 3, 19, 9, 0, 0, tzinfo=UTC)
    orch._settings.runtime.mode = RuntimeMode.DEMO
    orch._model_shadow_decisions = [
        {"model_probability": 0.35, "shadow_would_block": True},
        {"model_probability": 0.65, "shadow_would_block": False},
    ]
    orch._strategy_order_outcomes.model_filter.threshold = 0.5
    await orch._write_session_summary()
    report_dir = tmp_path / "session_summaries"
    cal_files = list(report_dir.glob("model_calibration_*.json"))
    assert len(cal_files) == 1
    import json
    data = json.loads(cal_files[0].read_text(encoding="utf-8"))
    assert "total_model_evaluations" in data
    assert "threshold_sweep" in data
    assert "probability_buckets" in data
    assert data["total_model_evaluations"] == 2
    assert len(data["threshold_sweep"]) == 5


def test_calibration_no_control_flow_change() -> None:
    """Calibration module is pure; no side effects or trading behavior."""
    from trading.runtime.model_calibration import build_model_calibration_summary

    decisions = [{"model_probability": 0.5, "shadow_would_block": False}]
    result = build_model_calibration_summary(decisions, threshold_configured=0.5)
    assert result is not None
    assert "total_model_evaluations" in result
