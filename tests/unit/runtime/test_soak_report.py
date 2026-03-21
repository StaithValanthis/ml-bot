"""Unit tests for soak report generation and verdict logic."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trading.monitoring.metrics import MetricsSnapshot
from trading.runtime.orchestrator import RuntimeOrchestrator
from trading.runtime.soak_report import (
    VERDICT_FAIL,
    VERDICT_PASS,
    VERDICT_PASS_WITH_WARNINGS,
    build_soak_markdown,
    build_soak_report,
    compute_verdict,
)
from trading.settings import load_settings
from trading.util.types import RuntimeMode


def _minimal_session_summary(
    *,
    session_ended_cleanly: bool = True,
    strategy_filled: int = 0,
    submitted: int = 0,
    model_filter_reached: int = 0,
    model_allowed: int = 0,
    candidates: int = 1,
    total_model_evals: int = 1,
    abort_reasons: list[str] | None = None,
    reconcile_cycles: int = 0,
) -> dict:
    return {
        "session_start": "2025-03-19T10:00:00+00:00",
        "session_end": "2025-03-19T11:00:00+00:00",
        "mode": "demo",
        "symbols": ["BTCUSDT"],
        "session_ended_cleanly": session_ended_cleanly,
        "abort_reasons": abort_reasons or [],
        "strategy_flow": {
            "bars_confirmed": 10,
            "candidates": candidates,
            "regime_rejected": 0,
            "signal_rejected": 0,
            "sizing_rejected": 0,
            "risk_rejected": 0,
            "model_filter_reached": model_filter_reached,
            "model_blocked": 0,
            "submitted": submitted,
        },
        "strategy_order_outcomes": {
            "intents": submitted,
            "submissions": submitted,
            "acks": submitted,
            "filled": strategy_filled,
            "cancelled": 0,
            "rejected": 0,
        },
        "model_filter": {
            "enabled": True,
            "active": True,
            "mode": "hard_block",
            "threshold": 0.5,
            "allowed": model_allowed,
            "blocked": max(0, model_filter_reached - model_allowed),
            "prob_count": total_model_evals,
        },
        "reconcile_mismatch_cycles": reconcile_cycles,
        "blocking_stage": "submitted",
    }


def test_soak_report_verdict_pass() -> None:
    """Healthy run yields PASS verdict."""
    summary = _minimal_session_summary(
        session_ended_cleanly=True,
        strategy_filled=1,
        submitted=1,
        model_filter_reached=1,
        model_allowed=1,
        candidates=1,
    )
    metrics = MetricsSnapshot(
        counters={
            "entry_fill_received_count": 1,
            "protective_exit_plan_created_count": 1,
            "protective_exit_order_ack_received_count": 1,
            "protective_exit_placement_failed_count": 0,
        },
        gauges={},
        histograms={},
    )
    report = build_soak_report(summary, metrics)
    verdict_block = report.get("health_verdict") or {}
    assert verdict_block.get("verdict") == VERDICT_PASS
    assert not verdict_block.get("failures")
    assert not verdict_block.get("warnings")


def test_soak_report_verdict_pass_with_warnings() -> None:
    """No failures but warnings yields PASS_WITH_WARNINGS."""
    summary = _minimal_session_summary(
        session_ended_cleanly=True,
        strategy_filled=0,
        submitted=0,
        model_filter_reached=0,
        model_allowed=0,
        candidates=0,
    )
    report = build_soak_report(summary, None)
    verdict_block = report.get("health_verdict") or {}
    assert verdict_block.get("verdict") == VERDICT_PASS_WITH_WARNINGS
    assert not verdict_block.get("failures")
    assert "no_model_evaluations" in (verdict_block.get("warnings") or [])
    assert "no_candidates_seen" in (verdict_block.get("warnings") or [])


def test_soak_report_verdict_fail_fill_without_protective_exit_ack() -> None:
    """Fill without protective exit ack yields FAIL."""
    summary = _minimal_session_summary(
        session_ended_cleanly=True,
        strategy_filled=2,
        submitted=2,
        model_filter_reached=2,
        model_allowed=2,
    )
    metrics = MetricsSnapshot(
        counters={
            "entry_fill_received_count": 2,
            "protective_exit_plan_created_count": 2,
            "protective_exit_order_ack_received_count": 1,
            "protective_exit_placement_failed_count": 0,
        },
        gauges={},
        histograms={},
    )
    report = build_soak_report(summary, metrics)
    verdict_block = report.get("health_verdict") or {}
    assert verdict_block.get("verdict") == VERDICT_FAIL
    assert "fills_without_protective_exit_ack" in (verdict_block.get("failures") or [])


def test_soak_report_verdict_fail_protective_exit_placement_failed() -> None:
    """protective_exit_placement_failed > 0 yields FAIL."""
    summary = _minimal_session_summary(
        session_ended_cleanly=True,
        strategy_filled=1,
        submitted=1,
        model_filter_reached=1,
        model_allowed=1,
    )
    metrics = MetricsSnapshot(
        counters={
            "entry_fill_received_count": 1,
            "protective_exit_plan_created_count": 1,
            "protective_exit_order_ack_received_count": 1,
            "protective_exit_placement_failed_count": 1,
        },
        gauges={},
        histograms={},
    )
    report = build_soak_report(summary, metrics)
    verdict_block = report.get("health_verdict") or {}
    assert verdict_block.get("verdict") == VERDICT_FAIL
    assert "protective_exit_failures_present" in (verdict_block.get("failures") or [])


def test_soak_report_json_structure() -> None:
    """Report has required top-level keys."""
    summary = _minimal_session_summary(submitted=1, strategy_filled=0)
    report = build_soak_report(summary, None)
    assert "session_metadata" in report
    assert "runtime_pipeline_totals" in report
    assert "model_evaluation_summary" in report
    assert "execution_summary" in report
    assert "safety_summary" in report
    assert "candidate_summary" in report
    assert "health_verdict" in report

    meta = report["session_metadata"]
    assert "session_id" in meta
    assert "mode" in meta
    assert "symbols" in meta
    assert "started_at" in meta
    assert "ended_at" in meta
    assert "duration_seconds" in meta
    assert "session_ended_cleanly" in meta
    assert "abort_reasons" in meta

    exec_s = report["execution_summary"]
    assert "strategy_order_intent_created_count" in exec_s
    assert "protective_exit_plan_created_count" in exec_s
    assert "protective_exit_order_ack_received_count" in exec_s
    assert "protective_exit_placement_failed_count" in exec_s
    assert "entry_fill_received_count" in exec_s


def test_soak_report_markdown_contains_required_sections() -> None:
    """Markdown has Session Overview, Pipeline Totals, Model Summary, etc."""
    summary = _minimal_session_summary(submitted=1)
    report = build_soak_report(summary, None)
    md = build_soak_markdown(report)
    assert "## Session Overview" in md
    assert "## Pipeline Totals" in md
    assert "## Model Summary" in md
    assert "## Execution Summary" in md
    assert "## Safety Summary" in md
    assert "## Final Verdict" in md
    assert "## Recommended Next Action" in md


def _make_orchestrator() -> RuntimeOrchestrator:
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
        return RuntimeOrchestrator(load_settings())


@pytest.mark.asyncio
async def test_soak_report_written_on_session_summary_generation(tmp_path: Path) -> None:
    """Soak report JSON and MD written when session summary is generated."""
    orch = _make_orchestrator()
    orch._settings.runtime.mode = RuntimeMode.DEMO
    orch._parquet_store._root_dir = tmp_path
    orch._session_start_time = datetime(2025, 3, 19, 9, 0, 0, tzinfo=UTC)

    await orch._write_session_summary()

    summaries_dir = tmp_path / "session_summaries"
    soak_jsons = list(summaries_dir.glob("soak_report_*.json"))
    soak_mds = list(summaries_dir.glob("soak_report_*.md"))

    assert len(soak_jsons) == 1
    assert len(soak_mds) == 1

    report = __import__("json").loads(soak_jsons[0].read_text(encoding="utf-8"))
    assert "health_verdict" in report
    assert report["health_verdict"]["verdict"] in (VERDICT_PASS, VERDICT_PASS_WITH_WARNINGS, VERDICT_FAIL)

    md = soak_mds[0].read_text(encoding="utf-8")
    assert "## Final Verdict" in md


def test_compute_verdict_session_aborted_fail() -> None:
    """Session aborted yields FAIL."""
    report = {
        "session_metadata": {"session_ended_cleanly": False, "abort_reasons": ["task_failed:foo"]},
        "execution_summary": {"strategy_order_filled_count": 0},
        "safety_summary": {},
        "runtime_pipeline_totals": {},
        "model_evaluation_summary": {},
    }
    verdict, failures, _ = compute_verdict(report)
    assert verdict == VERDICT_FAIL
    assert "session_aborted" in failures


def test_compute_verdict_repeated_reconcile_fail() -> None:
    """Repeated reconcile mismatch in abort_reasons yields FAIL."""
    report = {
        "session_metadata": {"session_ended_cleanly": False, "abort_reasons": ["repeated_reconcile_mismatch"]},
        "execution_summary": {"strategy_order_filled_count": 0},
        "safety_summary": {"repeated_reconcile_mismatch_triggered": True},
        "runtime_pipeline_totals": {},
        "model_evaluation_summary": {},
    }
    verdict, failures, _ = compute_verdict(report)
    assert verdict == VERDICT_FAIL
    assert "repeated_reconcile_mismatch_abort" in failures
