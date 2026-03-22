"""Unit tests for promotion-readiness evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest

from trading.runtime.promotion_readiness import (
    VERDICT_CONTINUE_DEMO_SOAK,
    VERDICT_NOT_READY,
    VERDICT_READY_FOR_PAPER,
    aggregate_soak_reports,
    build_promotion_assessment,
    build_promotion_markdown,
    collect_paths_from_input,
    compute_promotion_verdict,
    load_soak_reports,
    run_promotion_evaluation,
)


def _minimal_soak_report(
    *,
    verdict: str = "PASS",
    session_ended_cleanly: bool = True,
    abort_reasons: list[str] | None = None,
    filled: int = 1,
    pe_ack: int = 1,
    pe_failed: int = 0,
    duration_seconds: float = 3600.0,
    total_model_evaluations: int = 20,
    model_allowed: int = 10,
    submitted: int = 5,
) -> dict:
    return {
        "session_metadata": {
            "session_id": "session_20250319_100000",
            "mode": "demo",
            "symbols": ["BTCUSDT"],
            "started_at": "2025-03-19T10:00:00+00:00",
            "ended_at": "2025-03-19T11:00:00+00:00",
            "duration_seconds": duration_seconds,
            "session_ended_cleanly": session_ended_cleanly,
            "abort_reasons": abort_reasons or [],
        },
        "runtime_pipeline_totals": {"submitted": submitted},
        "model_evaluation_summary": {
            "total_model_evaluations": total_model_evaluations,
            "model_allowed_count": model_allowed,
            "model_blocked_count": total_model_evaluations - model_allowed,
            "threshold": 0.5,
            "prob_max": 0.65,
        },
        "execution_summary": {
            "strategy_order_submitted_count": submitted,
            "strategy_order_filled_count": filled,
            "entry_fill_received_count": filled,
            "protective_exit_order_ack_received_count": pe_ack,
            "protective_exit_placement_failed_count": pe_failed,
        },
        "safety_summary": {
            "orphan_position_blocked_count": 0,
            "orphan_position_block_cleared_count": 0,
            "startup_state_blocked_count": 0,
            "startup_state_block_cleared_count": 0,
            "reconcile_mismatch_detected_count": 0,
            "repeated_reconcile_mismatch_triggered": False,
            "position_add_blocked_count": 0,
            "working_entry_blocked_count": 0,
        },
        "health_verdict": {
            "verdict": verdict,
            "failures": [],
            "warnings": [],
        },
    }


def test_not_ready_when_single_fail_soak_exists() -> None:
    """NOT_READY when any soak report has FAIL verdict."""
    reports = [
        _minimal_soak_report(verdict="PASS", duration_seconds=4000, total_model_evaluations=15),
        _minimal_soak_report(verdict="FAIL", filled=2, pe_ack=1),
    ]
    assessment = build_promotion_assessment(reports)
    verdict_block = assessment.get("promotion_verdict") or {}
    assert verdict_block.get("verdict") == VERDICT_NOT_READY
    assert "any_fail_soak_report" in (verdict_block.get("reasons") or [])


def test_not_ready_when_fill_gt_protective_exit_ack() -> None:
    """NOT_READY when fills > protective_exit ack."""
    reports = [
        _minimal_soak_report(filled=3, pe_ack=2, verdict="PASS_WITH_WARNINGS"),
    ]
    assessment = build_promotion_assessment(reports)
    verdict_block = assessment.get("promotion_verdict") or {}
    assert verdict_block.get("verdict") == VERDICT_NOT_READY
    assert "fills_without_protective_exit_ack" in (verdict_block.get("reasons") or [])


def test_not_ready_when_protective_exit_placement_failed() -> None:
    """NOT_READY when protective_exit_placement_failed_count > 0."""
    reports = [
        _minimal_soak_report(pe_failed=1, pe_ack=1),
    ]
    assessment = build_promotion_assessment(reports)
    verdict_block = assessment.get("promotion_verdict") or {}
    assert verdict_block.get("verdict") == VERDICT_NOT_READY
    assert "protective_exit_placement_failed_count_gt_zero" in (verdict_block.get("reasons") or [])


def test_continue_demo_soak_when_sample_size_too_small() -> None:
    """CONTINUE_DEMO_SOAK when no hard failures but passing sessions < minimum."""
    reports = [
        _minimal_soak_report(duration_seconds=1000, total_model_evaluations=5, filled=1),
        _minimal_soak_report(duration_seconds=1000, total_model_evaluations=5, filled=1),
    ]
    assessment = build_promotion_assessment(
        reports,
        minimum_passing_sessions=3,
        minimum_total_duration_seconds=7200,
        minimum_total_fills=3,
        minimum_model_evaluations=10,
    )
    verdict_block = assessment.get("promotion_verdict") or {}
    assert verdict_block.get("verdict") == VERDICT_CONTINUE_DEMO_SOAK
    reasons = verdict_block.get("reasons") or []
    assert any("passing_sessions" in r for r in reasons) or any("total_duration" in r for r in reasons) or any("total_fills" in r for r in reasons) or any("model_evaluations" in r for r in reasons)


def test_ready_for_paper_when_all_minimum_criteria_met() -> None:
    """READY_FOR_PAPER when no failures and minimum coverage met."""
    reports = [
        _minimal_soak_report(duration_seconds=2500, total_model_evaluations=15, filled=2, pe_ack=2),
        _minimal_soak_report(duration_seconds=2500, total_model_evaluations=15, filled=2, pe_ack=2),
        _minimal_soak_report(duration_seconds=2500, total_model_evaluations=15, filled=2, pe_ack=2),
    ]
    assessment = build_promotion_assessment(
        reports,
        minimum_passing_sessions=3,
        minimum_total_duration_seconds=7200,
        minimum_total_fills=3,
        minimum_model_evaluations=10,
    )
    verdict_block = assessment.get("promotion_verdict") or {}
    verdict = verdict_block.get("verdict")
    reasons = verdict_block.get("reasons") or []
    assert verdict == VERDICT_READY_FOR_PAPER, f"Expected READY_FOR_PAPER, got {verdict}, reasons: {reasons}"
    assert "minimum_coverage_met" in reasons


def test_promotion_json_structure() -> None:
    """Assessment has required top-level keys."""
    reports = [_minimal_soak_report()]
    assessment = build_promotion_assessment(reports)
    assert "assessment_metadata" in assessment
    assert "promotion_verdict" in assessment
    assert "session_coverage" in assessment
    assert "execution_health" in assessment
    assert "model_activity" in assessment
    assert "safety_checks" in assessment


def test_promotion_markdown_contains_required_sections() -> None:
    """Markdown has Overview, Coverage, Execution Health, etc."""
    reports = [_minimal_soak_report()]
    assessment = build_promotion_assessment(reports)
    md = build_promotion_markdown(assessment)
    assert "## Overview" in md
    assert "## Coverage Summary" in md
    assert "## Execution Health" in md
    assert "## Model Activity" in md
    assert "## Safety Findings" in md
    assert "## Final Recommendation" in md
    assert "## Recommended Next Action" in md


def test_load_soak_reports_valid_json() -> None:
    """load_soak_reports parses valid JSON."""
    reports = load_soak_reports([])
    assert reports == []


def test_collect_paths_from_dir(tmp_path: Path) -> None:
    """collect_paths_from_input finds soak_report_*.json in directory."""
    (tmp_path / "soak_report_session_20250319.json").write_text("{}")
    (tmp_path / "other.json").write_text("{}")
    paths = collect_paths_from_input(None, tmp_path)
    assert len(paths) == 1
    assert paths[0].name == "soak_report_session_20250319.json"


def test_collect_paths_from_input_file(tmp_path: Path) -> None:
    """collect_paths_from_input accepts explicit file path."""
    p = tmp_path / "soak_report_custom.json"
    p.write_text('{"session_metadata":{}}')
    paths = collect_paths_from_input([str(p)], None)
    assert len(paths) == 1
    assert paths[0].name == "soak_report_custom.json"


def test_directory_input_aggregation(tmp_path: Path) -> None:
    """Multiple soak reports in directory are aggregated."""
    for i in range(3):
        (tmp_path / f"soak_report_session_20250319_10000{i}.json").write_text(
            '{"session_metadata":{"session_id":"s' + str(i) + '"},'
            '"health_verdict":{"verdict":"PASS"},"execution_summary":{},'
            '"safety_summary":{},"model_evaluation_summary":{},"runtime_pipeline_totals":{}}'
        )
    paths = collect_paths_from_input(None, tmp_path)
    assert len(paths) == 3
    reports = load_soak_reports(paths)
    assert len(reports) == 3


def test_aggregate_soak_reports_session_coverage() -> None:
    """Aggregation computes pass/fail counts and duration."""
    reports = [
        _minimal_soak_report(verdict="PASS", duration_seconds=100),
        _minimal_soak_report(verdict="PASS_WITH_WARNINGS", duration_seconds=200),
        _minimal_soak_report(verdict="FAIL", duration_seconds=300),
    ]
    agg = aggregate_soak_reports(reports)
    cov = agg.get("session_coverage") or {}
    assert cov.get("total_sessions") == 3
    assert cov.get("pass_count") == 1
    assert cov.get("pass_with_warnings_count") == 1
    assert cov.get("fail_count") == 1
    assert cov.get("total_duration_seconds") == 600


def test_compute_verdict_not_ready_no_reports() -> None:
    """Empty aggregated yields NOT_READY."""
    verdict, reasons = compute_promotion_verdict({})
    assert verdict == VERDICT_NOT_READY
    assert "no_valid_soak_reports" in reasons


def test_run_promotion_evaluation_writes_files(tmp_path: Path) -> None:
    """run_promotion_evaluation writes JSON and markdown."""
    inp = tmp_path / "soak_report_test.json"
    inp.write_text(
        '{"session_metadata":{"session_id":"s1","duration_seconds":4000},'
        '"health_verdict":{"verdict":"PASS"},'
        '"execution_summary":{"strategy_order_filled_count":2,"protective_exit_order_ack_received_count":2,'
        '"protective_exit_placement_failed_count":0},"safety_summary":{},'
        '"model_evaluation_summary":{"total_model_evaluations":15,"model_allowed_count":5,"model_blocked_count":10}}'
    )
    out_dir = tmp_path / "promotion_out"
    json_path, md_path, assessment = run_promotion_evaluation(
        input_specs=[str(inp)],
        dir_path=None,
        output_dir=out_dir,
        minimum_passing_sessions=1,
        minimum_total_duration_seconds=1,
        minimum_total_fills=1,
        minimum_model_evaluations=1,
    )
    assert json_path.exists()
    assert md_path.exists()
    assert json_path.read_text()
    assert "promotion_verdict" in assessment
