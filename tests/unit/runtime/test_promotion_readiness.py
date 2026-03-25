"""Unit tests for promotion-readiness evaluation."""

from __future__ import annotations

import subprocess
import sys
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
    pe_submitted: int | None = None,
    pe_failed: int = 0,
    duration_seconds: float = 3600.0,
    total_model_evaluations: int = 20,
    model_allowed: int = 10,
    submitted: int = 5,
    session_id: str | None = None,
) -> dict:
    pe_sub = pe_submitted if pe_submitted is not None else filled
    sid = session_id or "session_20250319_100000"
    return {
        "session_metadata": {
            "session_id": sid,
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
            "protective_exit_order_submitted_count": pe_sub,
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
    """NOT_READY when fills > protective_exit submitted (hard failure)."""
    reports = [
        _minimal_soak_report(filled=3, pe_ack=2, pe_submitted=2, verdict="PASS_WITH_WARNINGS"),
    ]
    assessment = build_promotion_assessment(reports)
    verdict_block = assessment.get("promotion_verdict") or {}
    assert verdict_block.get("verdict") == VERDICT_NOT_READY
    assert "fills_without_protective_exit_ack" in (verdict_block.get("reasons") or [])




def test_no_false_missing_protective_exit_when_terminal_skip_present() -> None:
    # A recorded protective-exit placement skip counts as terminal attribution.
    reports = [
        _minimal_soak_report(filled=2, pe_ack=1, pe_submitted=1, session_id="skip_terminal"),
    ]
    # Simulate explicit terminal skip attribution.
    reports[0]["execution_summary"]["protective_exit_placement_skipped_count"] = 1
    reports[0]["execution_summary"]["filled_entries_without_exit_ack"] = 0

    assessment = build_promotion_assessment(
        reports,
        minimum_passing_sessions=0,
        minimum_total_duration_seconds=0,
        minimum_total_fills=0,
        minimum_model_evaluations=0,
    )
    verdict_block = assessment.get("promotion_verdict") or {}
    reasons = verdict_block.get("reasons") or []
    assert "fills_without_protective_exit_ack" not in reasons
    assert verdict_block.get("verdict") != VERDICT_NOT_READY


def test_no_false_missing_protective_exit_when_strategy_fills_include_pe() -> None:
    """If strategy_filled includes reduce-only PE fills, safety checks must still use entry_fill_received_count."""
    reports = [
        _minimal_soak_report(filled=2, pe_ack=1, pe_submitted=1, session_id="pe_in_strategy_fills", verdict="PASS"),
    ]
    # entry_fill_received_count represents only opening fills; strategy_filled includes PE fills too.
    reports[0]["execution_summary"]["entry_fill_received_count"] = 1
    reports[0]["execution_summary"]["filled_entries_without_exit_ack"] = 0

    assessment = build_promotion_assessment(
        reports,
        minimum_passing_sessions=0,
        minimum_total_duration_seconds=0,
        minimum_total_fills=0,
        minimum_model_evaluations=0,
    )
    verdict_block = assessment.get("promotion_verdict") or {}
    assert verdict_block.get("verdict") != VERDICT_NOT_READY
    assert "fills_without_protective_exit_ack" not in (verdict_block.get("reasons") or [])


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
        _minimal_soak_report(duration_seconds=1000, total_model_evaluations=5, filled=1, session_id="session_a"),
        _minimal_soak_report(duration_seconds=1000, total_model_evaluations=5, filled=1, session_id="session_b"),
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


def test_promotion_report_includes_dominant_reconcile_fields() -> None:
    """Promotion assessment includes dominant_reconcile_issue_type and top_3 buckets when present."""
    reports = [
        _minimal_soak_report(duration_seconds=2500, filled=2, pe_ack=2, total_model_evaluations=15),
        _minimal_soak_report(duration_seconds=2500, filled=2, pe_ack=2, total_model_evaluations=15),
        _minimal_soak_report(duration_seconds=2500, filled=2, pe_ack=2, total_model_evaluations=15),
    ]
    for r in reports:
        r["reconcile_diagnostics"] = {
            "by_issue_type": {"missing_on_exchange": 3},
            "by_symbol": {"BTCUSDT": 3},
            "by_reason_bucket": {"missing_on_exchange": 3},
        }
        r["startup_state_diagnostics"] = {
            "block_reason": "dirty_at_startup",
            "cleared": True,
            "blocked_count": 1,
            "cleared_count": 1,
            "uncleared_at_session_end": False,
        }
    assessment = build_promotion_assessment(reports, minimum_passing_sessions=3, minimum_total_duration_seconds=7200, minimum_total_fills=3, minimum_model_evaluations=10)
    rsd = assessment.get("reconcile_and_startup_diagnostics") or {}
    assert "dominant_reconcile_issue_type" in rsd
    assert "dominant_reconcile_symbol" in rsd
    assert "top_3_reconcile_issue_buckets" in rsd
    assert rsd.get("dominant_reconcile_issue_type") == "missing_on_exchange"
    assert rsd.get("startup_block_clear_rate") == 1.0


def test_promotion_handles_empty_reconcile_diagnostics() -> None:
    """Promotion assessment handles sessions with no reconcile_diagnostics."""
    reports = [_minimal_soak_report(duration_seconds=8000, filled=5, pe_ack=5, total_model_evaluations=20)]
    assessment = build_promotion_assessment(reports, minimum_passing_sessions=1, minimum_total_duration_seconds=1, minimum_total_fills=1, minimum_model_evaluations=1)
    rsd = assessment.get("reconcile_and_startup_diagnostics") or {}
    assert rsd.get("dominant_reconcile_issue_type") is None
    assert rsd.get("sessions_with_uncleared_startup_state") == 0


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


def test_cli_valid_directory_writes_json_and_markdown(tmp_path: Path) -> None:
    """CLI with valid --dir discovers reports, writes JSON and markdown."""
    inp_dir = tmp_path / "summaries"
    inp_dir.mkdir()
    (inp_dir / "soak_report_session_20250319_100000.json").write_text(
        '{"session_metadata":{"session_id":"s1","duration_seconds":4000},'
        '"health_verdict":{"verdict":"PASS"},'
        '"execution_summary":{"strategy_order_filled_count":2,"protective_exit_order_ack_received_count":2,'
        '"protective_exit_placement_failed_count":0},"safety_summary":{},'
        '"model_evaluation_summary":{"total_model_evaluations":15}}'
    )
    out_dir = tmp_path / "promotion"
    result = subprocess.run(
        [sys.executable, "-m", "trading.runtime.promotion_cli", "--dir", str(inp_dir), "--output-dir", str(out_dir)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[3],
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "Loaded 1 soak report" in result.stdout
    assert "Verdict:" in result.stdout
    assert "Wrote promotion report:" in result.stdout
    assert out_dir.exists()
    jsons = list(out_dir.glob("promotion_readiness_*.json"))
    mds = list(out_dir.glob("promotion_readiness_*.md"))
    assert len(jsons) == 1
    assert len(mds) == 1
    assert "promotion_verdict" in jsons[0].read_text()


def test_cli_empty_directory_exits_nonzero(tmp_path: Path) -> None:
    """CLI with empty directory exits non-zero."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    out_dir = tmp_path / "promotion"
    result = subprocess.run(
        [sys.executable, "-m", "trading.runtime.promotion_cli", "--dir", str(empty_dir), "--output-dir", str(out_dir)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[3],
    )
    assert result.returncode != 0
    assert "No soak report" in result.stderr


def test_cli_creates_output_directory_if_missing(tmp_path: Path) -> None:
    """CLI creates output directory when it does not exist."""
    inp_dir = tmp_path / "summaries"
    inp_dir.mkdir()
    (inp_dir / "soak_report_session_20250319.json").write_text(
        '{"session_metadata":{},"health_verdict":{"verdict":"PASS"},"execution_summary":{},'
        '"safety_summary":{},"model_evaluation_summary":{}}'
    )
    out_dir = tmp_path / "new" / "nested" / "promotion"
    assert not out_dir.exists()
    result = subprocess.run(
        [sys.executable, "-m", "trading.runtime.promotion_cli", "--dir", str(inp_dir), "--output-dir", str(out_dir)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[3],
    )
    assert result.returncode == 0
    assert out_dir.exists()
    assert list(out_dir.glob("promotion_readiness_*.json"))


def test_unresolved_startup_state_not_emitted_when_clear_rate_one_and_uncleared_zero() -> None:
    """unresolved_startup_state must not be present when startup_block_clear_rate==1.0 and sessions_with_uncleared_startup_state==0."""
    reports = [
        _minimal_soak_report(duration_seconds=4000, filled=2, pe_ack=2, session_id="s1"),
        _minimal_soak_report(duration_seconds=4000, filled=2, pe_ack=2, session_id="s2"),
    ]
    for r in reports:
        r["startup_state_diagnostics"] = {
            "block_reason": "dirty_at_startup",
            "cleared": True,
            "blocked_count": 1,
            "cleared_count": 1,
            "uncleared_at_session_end": False,
        }
    assessment = build_promotion_assessment(
        reports,
        minimum_passing_sessions=2,
        minimum_total_duration_seconds=1000,
        minimum_total_fills=2,
        minimum_model_evaluations=1,
    )
    verdict_block = assessment.get("promotion_verdict") or {}
    reasons = verdict_block.get("reasons") or []
    rsd = assessment.get("reconcile_and_startup_diagnostics") or {}
    safety = assessment.get("safety_checks") or {}
    attr = assessment.get("reason_attribution") or {}
    assert rsd.get("startup_block_clear_rate") == 1.0
    assert rsd.get("sessions_with_uncleared_startup_state") == 0
    assert "unresolved_startup_state" not in reasons
    assert safety.get("startup_state_not_clearing") is False
    assert attr.get("sessions_causing_unresolved_startup_state") == []


def test_per_reason_session_attribution_in_json() -> None:
    """Per-reason session attribution appears in JSON assessment."""
    reports = [
        _minimal_soak_report(verdict="FAIL", session_id="fail_session"),
        _minimal_soak_report(filled=3, pe_submitted=2, pe_ack=2, session_id="fill_gap_session"),
    ]
    assessment = build_promotion_assessment(reports)
    attr = assessment.get("reason_attribution") or {}
    assert "sessions_causing_any_fail_soak_report" in attr
    assert "fail_session" in attr["sessions_causing_any_fail_soak_report"]
    assert "sessions_causing_fills_without_protective_exit_ack" in attr
    assert "fill_gap_session" in attr["sessions_causing_fills_without_protective_exit_ack"]
    reasons_by_session = assessment.get("reasons_by_session") or {}
    assert "fail_session" in reasons_by_session
    assert "any fail soak report" in reasons_by_session["fail_session"]
    assert "fill_gap_session" in reasons_by_session
    assert "fills without protective exit ack" in reasons_by_session["fill_gap_session"]


def test_fill_ack_gap_attribution_in_json() -> None:
    """Fill/ack gap attribution and session breakdown appear correctly."""
    reports = [
        _minimal_soak_report(filled=5, pe_ack=4, pe_submitted=5, session_id="gap_session"),
    ]
    assessment = build_promotion_assessment(reports)
    pe_attr = assessment.get("protective_exit_attribution") or {}
    assert "sessions_with_fill_ack_gap" in pe_attr
    assert "gap_session" in pe_attr["sessions_with_fill_ack_gap"]
    assert "session_breakdown" in pe_attr
    breakdown = pe_attr["session_breakdown"]
    assert len(breakdown) == 1
    row = breakdown[0]
    assert row["session_id"] == "gap_session"
    assert row["total_fills"] == 5
    assert row["protective_exit_submitted"] == 5
    assert row["protective_exit_ack"] == 4
    assert "protective_exit_failures" in row
    assert "protective_exit_ack_pending_count" in row
    assert "filled_entries_without_exit_ack" in row
    assert "fill_ack_gap_by_session" in pe_attr
    gap_rows = pe_attr["fill_ack_gap_by_session"]
    assert len(gap_rows) == 1
    assert gap_rows[0]["session_id"] == "gap_session"
    assert gap_rows[0]["total_fills"] == 5
    assert gap_rows[0]["protective_exit_ack"] == 4


def test_markdown_includes_reasons_by_session() -> None:
    """Markdown includes 'Reasons by Session' section for failing sessions."""
    reports = [
        _minimal_soak_report(verdict="FAIL", session_id="bad_session"),
    ]
    assessment = build_promotion_assessment(reports)
    md = build_promotion_markdown(assessment)
    assert "## Reasons by Session" in md
    assert "### bad_session" in md


def test_markdown_includes_fill_ack_gap_and_startup_sections() -> None:
    """Markdown includes 'Sessions with Fill/Ack Gap' and 'Sessions with Startup Issues' when present."""
    reports = [
        _minimal_soak_report(filled=3, pe_ack=2, pe_submitted=3, session_id="gap_sess"),
    ]
    reports[0]["startup_state_diagnostics"] = {"uncleared_at_session_end": True}
    assessment = build_promotion_assessment(reports)
    md = build_promotion_markdown(assessment)
    assert "## Sessions with Fill/Ack Gap" in md
    assert "gap_sess" in md
    assert "## Sessions with Startup Issues" in md
