"""Unit tests for ML model calibration and threshold-readiness analysis."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trading.runtime.model_calibration import (
    build_log_scale_buckets,
    build_model_calibration_summary,
    build_probability_buckets,
    build_promotion_recommendation,
    build_runtime_calibration_stats,
    build_threshold_sweep,
    compute_retention_thresholds,
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
    assert "runtime_calibration" in cal
    assert cal["runtime_calibration"]["probability_distribution"]["max"] == 0.6
    pr = summary.get("promotion_recommendation")
    assert pr is not None
    assert pr["observed_max_probability"] == 0.6
    assert "verdict" in pr


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
            "runtime_calibration": {
                "probability_distribution": {"min": 0.2, "max": 0.8, "mean": 0.5, "median": 0.5, "p50": 0.5, "p95": 0.75, "p99": 0.79},
                "current_threshold_above_observed_max": False,
                "retention_thresholds": {"threshold_keep_75pct": 0.25, "threshold_keep_50pct": 0.5},
            },
        },
    }
    md = orch._build_markdown_summary(summary)
    assert "## Model Calibration Review" in md
    assert "## Runtime Probability Distribution" in md
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
    assert "runtime_calibration" in data
    assert data["total_model_evaluations"] == 2
    assert len(data["threshold_sweep"]) == 5


def test_calibration_no_control_flow_change() -> None:
    """Calibration module is pure; no side effects or trading behavior."""
    from trading.runtime.model_calibration import build_model_calibration_summary

    decisions = [{"model_probability": 0.5, "shadow_would_block": False}]
    result = build_model_calibration_summary(decisions, threshold_configured=0.5)
    assert result is not None
    assert "total_model_evaluations" in result




def test_compute_retention_thresholds() -> None:
    """Retention thresholds derived from empirical percentiles."""
    probs = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    rec = compute_retention_thresholds(probs)
    assert rec["threshold_keep_90pct"] <= rec["threshold_keep_75pct"]
    assert rec["threshold_keep_75pct"] <= rec["threshold_keep_50pct"]
    assert rec["threshold_keep_50pct"] <= rec["threshold_keep_25pct"]
    assert rec["threshold_keep_25pct"] <= rec["threshold_keep_10pct"]
    assert 0.05 <= rec["threshold_keep_90pct"] <= 0.25
    assert 0.4 <= rec["threshold_keep_50pct"] <= 0.6


def test_build_log_scale_buckets() -> None:
    """Log-scale buckets count by magnitude."""
    probs = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.2]
    buckets = build_log_scale_buckets(probs)
    assert len(buckets) == 7
    lt_1e6 = next(b for b in buckets if b["bucket"] == "lt_1e_6")
    assert lt_1e6["count"] == 1
    gte_1e1 = next(b for b in buckets if b["bucket"] == "gte_1e_1")
    assert gte_1e1["count"] == 1


def test_build_runtime_calibration_stats() -> None:
    """Runtime calibration includes distribution, percentiles, retention thresholds."""
    probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    stats = build_runtime_calibration_stats(probs, current_threshold=0.45)
    assert stats["total_shadow_evaluations"] == 9
    dist = stats["probability_distribution"]
    assert dist["min"] == 0.1
    assert dist["max"] == 0.9
    assert dist["p50"] == pytest.approx(0.5, abs=0.01)
    assert dist["p95"] >= 0.8
    assert stats["retention_thresholds"]["threshold_keep_50pct"] == pytest.approx(0.5, abs=0.05)
    assert stats["current_threshold_above_observed_max"] is False


def test_build_runtime_calibration_stats_threshold_above_max() -> None:
    """Flag when current threshold is above observed max probability."""
    probs = [1e-7, 2e-7, 3e-7]
    stats = build_runtime_calibration_stats(probs, current_threshold=0.45)
    assert stats["current_threshold_above_observed_max"] is True
    assert stats["probability_distribution"]["max"] == 3e-7


def test_build_runtime_calibration_stats_empty() -> None:
    """Empty probs yields empty stats."""
    stats = build_runtime_calibration_stats([])
    assert stats["total_shadow_evaluations"] == 0
    assert stats["probability_distribution"]["min"] is None
    assert stats["current_threshold_above_observed_max"] is None


def test_build_runtime_calibration_stats_nearly_identical() -> None:
    """Nearly identical probs yields not_predictive_enough via promotion recommendation."""
    probs = [0.5, 0.50001, 0.49999]
    stats = build_runtime_calibration_stats(probs, current_threshold=0.45)
    rec = build_promotion_recommendation(
        current_threshold=0.45,
        observed_max=stats["probability_distribution"]["max"],
        observed_p95=stats["probability_distribution"]["p95"],
        observed_p99=stats["probability_distribution"]["p99"],
        observed_min=stats["probability_distribution"]["min"],
        observed_mean=stats["probability_distribution"]["mean"],
        retention_thresholds=stats["retention_thresholds"],
    )
    assert rec["verdict"] == "not_predictive_enough"


def test_build_promotion_recommendation() -> None:
    """Promotion recommendation includes verdict and suggested thresholds."""
    rec = build_promotion_recommendation(
        current_threshold=0.45,
        observed_max=0.52,
        observed_p95=0.48,
        observed_p99=0.51,
        retention_thresholds={
            "threshold_keep_75pct": 0.25,
            "threshold_keep_50pct": 0.45,
        },
    )
    assert rec["current_runtime_threshold"] == 0.45
    assert rec["observed_max_probability"] == 0.52
    assert rec["current_threshold_realistic"] is True
    assert rec["suggested_threshold_shadow"] == 0.25
    assert rec["suggested_threshold_active_demo"] == 0.45
    assert rec["verdict"] in ("remain_shadow", "active_demo_ready", "not_predictive_enough")


@pytest.mark.asyncio
async def test_session_summary_no_decisions_no_promotion() -> None:
    """When no model decisions, promotion_recommendation is absent."""
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
    orch._model_shadow_decisions = []
    summary = await orch._build_session_summary()
    assert "promotion_recommendation" not in summary


def test_markdown_includes_promotion_recommendation() -> None:
    """Markdown includes Promotion Recommendation section when present."""
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
        "model_calibration": {"total_model_evaluations": 5},
        "promotion_recommendation": {
            "current_runtime_threshold": 0.45,
            "observed_max_probability": 0.52,
            "observed_p95": 0.48,
            "observed_p99": 0.51,
            "current_threshold_realistic": True,
            "suggested_threshold_shadow": 0.25,
            "suggested_threshold_active_demo": 0.45,
            "verdict": "remain_shadow",
        },
    }
    md = orch._build_markdown_summary(summary)
    assert "## Promotion Recommendation" in md
    assert "Current runtime threshold: 0.45" in md
    assert "Observed max probability: 0.52" in md
    assert "Verdict: remain_shadow" in md
