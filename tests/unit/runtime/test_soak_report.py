"""Unit tests for soak report generation and verdict logic."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trading.monitoring.metrics import MetricsSnapshot
from trading.runtime.orchestrator import RuntimeOrchestrator
from trading.runtime.soak_report import (
    REASON_MODEL_ALLOWED_BUT_NO_SUBMISSIONS,
    VERDICT_FAIL,
    VERDICT_PASS,
    VERDICT_PASS_WITH_WARNINGS,
    build_soak_markdown,
    build_soak_report,
    compute_verdict,
    REASON_MODEL_ALLOWED_BUT_GUARD_BLOCKED_DUE_TO_ACTIVE_POSITION,
    REASON_MODEL_ALLOWED_BUT_NO_SUBMISSIONS_UNEXPLAINED,
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
            "protective_exit_order_submitted_count": 1,
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


def test_soak_report_verdict_fail_fill_without_protective_exit_submit() -> None:
    """Fill without protective exit submitted (hard failure) yields FAIL."""
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
            "protective_exit_order_submitted_count": 1,
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


def test_soak_report_warn_model_allowed_but_guard_blocked_due_to_active_position() -> None:
    """model_filter_allowed>0 with zero submissions is a warning if guard-only existing_position blocked it."""
    summary = _minimal_session_summary(
        session_ended_cleanly=True,
        submitted=0,
        model_filter_reached=55,
        model_allowed=55,
        candidates=55,
    )
    summary["entry_guard_block_reasons"] = {
        "by_type": {"existing_position": 55},
        "by_symbol": {"BTCUSDT": {"existing_position": 55}},
    }
    metrics = MetricsSnapshot(
        counters={
            "entry_fill_received_count": 0,
            "protective_exit_plan_created_count": 0,
            "protective_exit_order_submitted_count": 0,
            "protective_exit_order_ack_received_count": 0,
            "protective_exit_placement_failed_count": 0,
            "position_add_blocked_count": 55,
            "working_entry_blocked_count": 0,
            "startup_state_blocked_count": 0,
        },
        gauges={},
        histograms={},
    )
    report = build_soak_report(summary, metrics)
    verdict_block = report.get("health_verdict") or {}
    assert verdict_block.get("verdict") == VERDICT_PASS_WITH_WARNINGS
    assert (
        REASON_MODEL_ALLOWED_BUT_GUARD_BLOCKED_DUE_TO_ACTIVE_POSITION
        in (verdict_block.get("warnings") or [])
    )
    assert REASON_MODEL_ALLOWED_BUT_NO_SUBMISSIONS_UNEXPLAINED not in (verdict_block.get("failures") or [])


def test_soak_report_exact_session_shape_guard_blocked_active_position_is_warning_only() -> None:
    """Exact provided session shape must be PASS_WITH_WARNINGS, not model-allowed-no-submissions FAIL."""
    summary = _minimal_session_summary(
        session_ended_cleanly=True,
        submitted=0,
        model_filter_reached=55,
        model_allowed=55,
        candidates=106,
    )
    summary["strategy_flow"]["bars_confirmed"] = 106
    summary["strategy_flow"]["regime_rejected"] = 51
    summary["entry_guard_block_reasons"] = {
        "by_type": {"existing_position": 55},
        "by_symbol": {"BTCUSDT": {"existing_position": 55}},
    }
    summary["startup_state_diagnostics"] = {
        "block_reason": "dirty_at_startup",
        "cleared": True,
        "uncleared_at_session_end": False,
    }
    metrics = MetricsSnapshot(
        counters={
            "entry_fill_received_count": 0,
            "protective_exit_order_submitted_count": 0,
            "protective_exit_order_ack_received_count": 0,
            "protective_exit_placement_failed_count": 0,
            "protective_exit_placement_skipped_count": 0,
            "runtime_decision_failures_count": 0,
            "startup_state_blocked_count": 1,
            "startup_state_block_cleared_count": 1,
            "position_add_blocked_count": 55,
            "working_entry_blocked_count": 0,
            "missing_on_exchange_detected_count": 0,
            "missing_on_exchange_resolved_count": 0,
        },
        gauges={},
        histograms={},
    )
    report = build_soak_report(summary, metrics)
    verdict_block = report.get("health_verdict") or {}
    failures = verdict_block.get("failures") or []
    warnings = verdict_block.get("warnings") or []
    assert verdict_block.get("verdict") == VERDICT_PASS_WITH_WARNINGS
    assert REASON_MODEL_ALLOWED_BUT_GUARD_BLOCKED_DUE_TO_ACTIVE_POSITION in warnings
    assert "startup_state_block_triggered" in warnings
    assert REASON_MODEL_ALLOWED_BUT_NO_SUBMISSIONS_UNEXPLAINED not in failures
    assert REASON_MODEL_ALLOWED_BUT_NO_SUBMISSIONS not in failures


def test_soak_report_fail_model_allowed_guard_only_when_runtime_decision_failures() -> None:
    """Guard-only existing_position blocks are not 'explained' if runtime-decision task failed."""
    summary = _minimal_session_summary(
        session_ended_cleanly=True,
        submitted=0,
        model_filter_reached=10,
        model_allowed=10,
        candidates=10,
    )
    summary["entry_guard_block_reasons"] = {
        "by_type": {"existing_position": 10},
        "by_symbol": {"BTCUSDT": {"existing_position": 10}},
    }
    metrics = MetricsSnapshot(
        counters={
            "entry_fill_received_count": 0,
            "protective_exit_order_submitted_count": 0,
            "protective_exit_order_ack_received_count": 0,
            "protective_exit_placement_failed_count": 0,
            "position_add_blocked_count": 10,
            "working_entry_blocked_count": 0,
            "runtime_decision_failures_count": 1,
        },
        gauges={},
        histograms={},
    )
    report = build_soak_report(summary, metrics)
    verdict_block = report.get("health_verdict") or {}
    assert verdict_block.get("verdict") == VERDICT_FAIL
    assert (
        REASON_MODEL_ALLOWED_BUT_NO_SUBMISSIONS_UNEXPLAINED
        in (verdict_block.get("failures") or [])
    )
    assert (
        REASON_MODEL_ALLOWED_BUT_GUARD_BLOCKED_DUE_TO_ACTIVE_POSITION
        not in (verdict_block.get("warnings") or [])
    )


def test_soak_report_fail_model_allowed_guard_only_when_reconcile_mismatch() -> None:
    """Guard-only blocks do not downgrade to guard warning if reconcile mismatches were detected."""
    summary = _minimal_session_summary(
        session_ended_cleanly=True,
        submitted=0,
        model_filter_reached=5,
        model_allowed=5,
        candidates=5,
        reconcile_cycles=3,
    )
    summary["entry_guard_block_reasons"] = {
        "by_type": {"existing_position": 5},
        "by_symbol": {"BTCUSDT": {"existing_position": 5}},
    }
    metrics = MetricsSnapshot(
        counters={
            "entry_fill_received_count": 0,
            "protective_exit_order_submitted_count": 0,
            "protective_exit_order_ack_received_count": 0,
            "protective_exit_placement_failed_count": 0,
            "position_add_blocked_count": 5,
            "working_entry_blocked_count": 0,
        },
        gauges={},
        histograms={},
    )
    report = build_soak_report(summary, metrics)
    verdict_block = report.get("health_verdict") or {}
    assert verdict_block.get("verdict") == VERDICT_FAIL
    assert (
        REASON_MODEL_ALLOWED_BUT_NO_SUBMISSIONS_UNEXPLAINED
        in (verdict_block.get("failures") or [])
    )


def test_soak_report_fail_when_existing_position_is_not_only_nonzero_guard_reason() -> None:
    """Guard-block warning is only allowed when existing_position is the sole non-zero reason."""
    summary = _minimal_session_summary(
        session_ended_cleanly=True,
        submitted=0,
        model_filter_reached=55,
        model_allowed=55,
        candidates=55,
    )
    summary["entry_guard_block_reasons"] = {
        "by_type": {"existing_position": 55, "max_concurrent_entries": 1},
        "by_symbol": {"BTCUSDT": {"existing_position": 55, "max_concurrent_entries": 1}},
    }
    metrics = MetricsSnapshot(
        counters={
            "entry_fill_received_count": 0,
            "protective_exit_order_submitted_count": 0,
            "protective_exit_order_ack_received_count": 0,
            "protective_exit_placement_failed_count": 0,
            "protective_exit_placement_skipped_count": 0,
            "runtime_decision_failures_count": 0,
            "position_add_blocked_count": 55,
            "working_entry_blocked_count": 0,
        },
        gauges={},
        histograms={},
    )
    report = build_soak_report(summary, metrics)
    verdict_block = report.get("health_verdict") or {}
    assert verdict_block.get("verdict") == VERDICT_FAIL
    assert REASON_MODEL_ALLOWED_BUT_GUARD_BLOCKED_DUE_TO_ACTIVE_POSITION not in (verdict_block.get("warnings") or [])
    assert REASON_MODEL_ALLOWED_BUT_NO_SUBMISSIONS_UNEXPLAINED in (verdict_block.get("failures") or [])


def test_soak_report_fail_model_allowed_but_no_submissions_unexplained() -> None:
    """model_filter_allowed>0 with zero submissions is a hard FAIL if guard wasn't explained as existing_position-only."""
    summary = _minimal_session_summary(
        session_ended_cleanly=True,
        submitted=0,
        model_filter_reached=55,
        model_allowed=55,
        candidates=55,
    )
    summary["entry_guard_block_reasons"] = {
        "by_type": {"existing_working_entry": 55},
        "by_symbol": {"BTCUSDT": {"existing_working_entry": 55}},
    }
    metrics = MetricsSnapshot(
        counters={
            "entry_fill_received_count": 0,
            "protective_exit_plan_created_count": 0,
            "protective_exit_order_submitted_count": 0,
            "protective_exit_order_ack_received_count": 0,
            "protective_exit_placement_failed_count": 0,
            "position_add_blocked_count": 0,
            "working_entry_blocked_count": 55,
            "startup_state_blocked_count": 0,
        },
        gauges={},
        histograms={},
    )
    report = build_soak_report(summary, metrics)
    verdict_block = report.get("health_verdict") or {}
    assert verdict_block.get("verdict") == VERDICT_FAIL
    assert (
        REASON_MODEL_ALLOWED_BUT_NO_SUBMISSIONS_UNEXPLAINED
        in (verdict_block.get("failures") or [])
    )


def test_soak_report_verdict_pass_ack_pending_at_session_end() -> None:
    """Fill + submitted protective exit + ack pending at session end does NOT fail."""
    summary = _minimal_session_summary(
        session_ended_cleanly=True,
        strategy_filled=44,
        submitted=44,
        model_filter_reached=71,
        model_allowed=71,
    )
    metrics = MetricsSnapshot(
        counters={
            "entry_fill_received_count": 44,
            "protective_exit_plan_created_count": 44,
            "protective_exit_order_submitted_count": 44,
            "protective_exit_order_ack_received_count": 43,
            "protective_exit_placement_failed_count": 0,
        },
        gauges={},
        histograms={},
    )
    report = build_soak_report(summary, metrics)
    verdict_block = report.get("health_verdict") or {}
    assert "fills_without_protective_exit_ack" not in (verdict_block.get("failures") or [])
    exec_s = report.get("execution_summary") or {}
    assert exec_s.get("protective_exit_ack_pending_count") == 1
    assert exec_s.get("filled_entries_without_exit_ack") == 1




def test_soak_report_verdict_pass_when_one_protective_exit_is_skipped_explicitly() -> None:
    # Explicit protective-exit placement skip is terminal attribution, not a missing submission.
    summary = _minimal_session_summary(
        session_ended_cleanly=True,
        strategy_filled=2,
        submitted=2,
        model_filter_reached=2,
        model_allowed=2,
        candidates=1,
    )
    metrics = MetricsSnapshot(
        counters={
            "entry_fill_received_count": 2,
            "protective_exit_plan_created_count": 2,
            "protective_exit_order_submitted_count": 1,
            "protective_exit_order_ack_received_count": 1,
            "protective_exit_placement_failed_count": 0,
            "protective_exit_placement_skipped_count": 1,
        },
        gauges={},
        histograms={},
    )
    report = build_soak_report(summary, metrics)
    verdict_block = report.get("health_verdict") or {}
    assert verdict_block.get("verdict") in (VERDICT_PASS, VERDICT_PASS_WITH_WARNINGS)
    assert "fills_without_protective_exit_ack" not in (verdict_block.get("failures") or [])
    exec_s = report.get("execution_summary") or {}
    assert exec_s.get("filled_entries_without_exit_ack") == 0


def test_soak_report_does_not_misattribute_pe_fills_as_entry_fills() -> None:
    """strategy_order_filled_count may include reduce-only protective-exit fills.

    The safety checks must be based on entry_fill_received_count, so a session with
    entry_fill=1 and PE ack=1 should not fail even if strategy_filled=2.
    """
    summary = _minimal_session_summary(
        session_ended_cleanly=True,
        # Includes: 1 entry fill + 1 reduce-only PE fill.
        strategy_filled=2,
        submitted=1,
        model_filter_reached=1,
        model_allowed=1,
        candidates=1,
    )
    metrics = MetricsSnapshot(
        counters={
            "entry_fill_received_count": 1,
            "protective_exit_plan_created_count": 1,
            "protective_exit_order_submitted_count": 1,
            "protective_exit_order_ack_received_count": 1,
            "protective_exit_placement_failed_count": 0,
        },
        gauges={},
        histograms={},
    )
    report = build_soak_report(summary, metrics)
    verdict_block = report.get("health_verdict") or {}
    assert verdict_block.get("verdict") == VERDICT_PASS
    assert "fills_without_protective_exit_ack" not in (verdict_block.get("failures") or [])
    exec_s = report.get("execution_summary") or {}
    assert exec_s.get("filled_entries_without_exit_ack") == 0


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


def test_soak_report_includes_reconcile_and_startup_diagnostics() -> None:
    """Soak report includes reconcile_diagnostics and startup_state_diagnostics when present."""
    summary = _minimal_session_summary(submitted=1, strategy_filled=0)
    summary["reconcile_diagnostics"] = {
        "total_occurrences": 5,
        "top_issue_type": "missing_on_exchange",
        "top_symbol": "BTCUSDT",
        "top_3_reason_buckets": [{"reason_bucket": "missing_on_exchange", "count": 5}],
    }
    summary["startup_state_diagnostics"] = {
        "block_reason": "dirty_at_startup",
        "cleared": True,
        "uncleared_at_session_end": False,
    }
    report = build_soak_report(summary, None)
    assert report.get("reconcile_diagnostics") is not None
    assert report["reconcile_diagnostics"]["top_issue_type"] == "missing_on_exchange"
    assert report.get("startup_state_diagnostics") is not None
    assert report["startup_state_diagnostics"]["block_reason"] == "dirty_at_startup"
    md = build_soak_markdown(report)
    assert "## Reconcile Diagnostics" in md
    assert "## Startup State Diagnostics" in md


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
