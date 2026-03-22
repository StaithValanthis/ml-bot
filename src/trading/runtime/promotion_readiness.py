"""Promotion-readiness assessment: evaluates soak reports for environment progression."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading.util.logging import get_logger

# --- Verdict constants ---
VERDICT_NOT_READY = "NOT_READY"
VERDICT_CONTINUE_DEMO_SOAK = "CONTINUE_DEMO_SOAK"
VERDICT_READY_FOR_PAPER = "READY_FOR_PAPER"
VERDICT_READY_FOR_NEXT_PHASE = "READY_FOR_NEXT_PHASE"

# --- Configurable minimum thresholds ---
DEFAULT_MINIMUM_PASSING_SESSIONS = 3
DEFAULT_MINIMUM_TOTAL_DURATION_SECONDS = 7200
DEFAULT_MINIMUM_TOTAL_FILLS = 3
DEFAULT_MINIMUM_MODEL_EVALUATIONS = 10


def _g(d: dict[str, Any], key: str, default: Any = 0) -> Any:
    """Safe int get."""
    v = d.get(key, default)
    return int(v) if isinstance(v, (int, float)) else default


def _g_float(d: dict[str, Any], key: str, default: float | None = None) -> float | None:
    v = d.get(key, default)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _g_bool(d: dict[str, Any], key: str, default: bool = False) -> bool:
    v = d.get(key, default)
    return bool(v) if v is not None else default


def load_soak_reports(paths: list[Path]) -> list[dict[str, Any]]:
    """Load soak report JSON files. Returns list of parsed reports, skips invalid files."""
    reports: list[dict[str, Any]] = []
    for p in paths:
        if not p.exists() or not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and ("session_metadata" in data or "health_verdict" in data):
                reports.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            logging.getLogger(__name__).warning("promotion_skip_invalid_soak", path=str(p), error=str(exc))
    return reports


def collect_paths_from_input(input_specs: list[str] | None, dir_path: Path | None) -> list[Path]:
    """
    Collect soak report paths from --input (files/globs) or --dir.
    Returns deduplicated sorted list of Paths.
    """
    collected: set[Path] = set()
    if dir_path is not None and dir_path.is_dir():
        for p in dir_path.glob("soak_report_*.json"):
            collected.add(p.resolve())
    if input_specs:
        for spec in input_specs:
            p = Path(spec)
            if "*" in spec or "?" in spec:
                parent = p.parent if p.parent.exists() else Path(".")
                for m in parent.glob(p.name):
                    if m.is_file() and m.suffix == ".json":
                        collected.add(m.resolve())
            elif p.exists() and p.is_file():
                collected.add(p.resolve())
    return sorted(collected)


def aggregate_soak_reports(
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate metrics across soak reports.
    Returns structured assessment dict with session_coverage, execution_health, model_activity, safety_checks.
    """
    pass_count = 0
    pass_with_warnings_count = 0
    fail_count = 0
    total_duration_seconds = 0.0
    durations: list[float] = []

    total_submitted = 0
    total_fills = 0
    total_entry_fill_received = 0
    total_protective_exit_ack = 0
    total_protective_exit_failures = 0
    total_reconcile_mismatches = 0
    total_startup_state_blocks = 0
    total_orphan_blocks = 0
    total_position_add_blocks = 0
    total_working_entry_blocks = 0

    total_model_evaluations = 0
    total_model_allowed = 0
    total_model_blocked = 0
    prob_sum = 0.0
    prob_count = 0
    prob_max: float | None = None
    thresholds: list[float] = []

    fills_without_protective_exit_ack = False
    reconcile_abort_present = False
    session_abort_present = False
    startup_state_not_clearing = False
    orphan_block_present = False
    orphan_block_unresolved = False
    repeated_reconcile_issue_present = False
    repeated_position_add_attempts_present = False

    for r in reports:
        meta = r.get("session_metadata") or {}
        exec_s = r.get("execution_summary") or {}
        safety = r.get("safety_summary") or {}
        model_s = r.get("model_evaluation_summary") or {}
        verdict_block = r.get("health_verdict") or {}
        verdict = verdict_block.get("verdict", "")

        if verdict == "PASS":
            pass_count += 1
        elif verdict == "PASS_WITH_WARNINGS":
            pass_with_warnings_count += 1
        elif verdict == "FAIL":
            fail_count += 1

        dur = meta.get("duration_seconds")
        if dur is not None and isinstance(dur, (int, float)):
            total_duration_seconds += float(dur)
            durations.append(float(dur))

        total_submitted += _g(exec_s, "strategy_order_submitted_count")
        total_fills += _g(exec_s, "strategy_order_filled_count")
        total_entry_fill_received += _g(exec_s, "entry_fill_received_count")
        total_protective_exit_ack += _g(exec_s, "protective_exit_order_ack_received_count")
        total_protective_exit_failures += _g(exec_s, "protective_exit_placement_failed_count")

        total_reconcile_mismatches += _g(safety, "reconcile_mismatch_detected_count")
        total_startup_state_blocks += _g(safety, "startup_state_blocked_count")
        total_orphan_blocks += _g(safety, "orphan_position_blocked_count")
        total_position_add_blocks += _g(safety, "position_add_blocked_count")
        total_working_entry_blocks += _g(safety, "working_entry_blocked_count")

        total_model_evaluations += _g(model_s, "total_model_evaluations")
        total_model_allowed += _g(model_s, "model_allowed_count")
        total_model_blocked += _g(model_s, "model_blocked_count")
        t = _g_float(model_s, "threshold")
        if t is not None:
            thresholds.append(t)
        p_max = _g_float(model_s, "prob_max")
        if p_max is not None:
            prob_max = p_max if prob_max is None else max(prob_max, p_max)
        p_latest = _g_float(model_s, "prob_latest")
        if p_latest is not None:
            prob_sum += p_latest
            prob_count += 1

        if _g(exec_s, "strategy_order_filled_count") > _g(exec_s, "protective_exit_order_ack_received_count"):
            fills_without_protective_exit_ack = True
        if _g_bool(safety, "repeated_reconcile_mismatch_triggered"):
            repeated_reconcile_issue_present = True
        if not _g_bool(meta, "session_ended_cleanly", True) or (meta.get("abort_reasons") or []):
            session_abort_present = True
        if _g(safety, "startup_state_blocked_count", 0) > 0 and _g(safety, "startup_state_block_cleared_count", 0) == 0:
            startup_state_not_clearing = True
        ob = _g(safety, "orphan_position_blocked_count", 0)
        oc = _g(safety, "orphan_position_block_cleared_count", 0)
        if ob > 0:
            orphan_block_present = True
        if ob > 0 and oc == 0:
            orphan_block_unresolved = True
        if _g(safety, "position_add_blocked_count", 0) > 0 or _g(safety, "working_entry_blocked_count", 0) > 0:
            repeated_position_add_attempts_present = True

    sessions_with_uncleared_startup = 0
    sessions_with_reconcile_abort = 0
    sessions_with_startup_block = 0
    sessions_with_startup_cleared = 0
    reconcile_by_type: dict[str, int] = {}
    reconcile_by_symbol: dict[str, int] = {}
    reconcile_by_bucket: dict[str, int] = {}

    for r in reports:
        ssd = r.get("startup_state_diagnostics") or {}
        if _g_bool(ssd, "uncleared_at_session_end"):
            sessions_with_uncleared_startup += 1
        if _g(ssd, "blocked_count", 0) > 0:
            sessions_with_startup_block += 1
        if _g(ssd, "cleared_count", 0) > 0 or _g_bool(ssd, "cleared"):
            sessions_with_startup_cleared += 1
        if "repeated_reconcile_mismatch" in (r.get("session_metadata") or {}).get("abort_reasons", []):
            sessions_with_reconcile_abort += 1
        rd = r.get("reconcile_diagnostics") or {}
        for it, cnt in (rd.get("by_issue_type") or {}).items():
            reconcile_by_type[it] = reconcile_by_type.get(it, 0) + cnt
        for sym, cnt in (rd.get("by_symbol") or {}).items():
            reconcile_by_symbol[sym] = reconcile_by_symbol.get(sym, 0) + cnt
        for bkt, cnt in (rd.get("by_reason_bucket") or {}).items():
            reconcile_by_bucket[bkt] = reconcile_by_bucket.get(bkt, 0) + cnt

    def _top_key(d: dict[str, int]) -> str | None:
        return max(d, key=d.get) if d else None

    top_3_buckets = [
        {"reason_bucket": k, "count": v}
        for k, v in sorted(reconcile_by_bucket.items(), key=lambda x: -x[1])[:3]
    ]
    startup_block_clear_rate = (
        round(sessions_with_startup_cleared / sessions_with_startup_block, 3)
        if sessions_with_startup_block > 0 else None
    )

    threshold_consistency = len(set(thresholds)) <= 1 if thresholds else True

    average_duration_seconds = total_duration_seconds / len(durations) if durations else None
    average_probability = prob_sum / prob_count if prob_count > 0 else None
    threshold_used = thresholds[0] if thresholds else None

    return {
        "session_coverage": {
            "total_sessions": len(reports),
            "pass_count": pass_count,
            "pass_with_warnings_count": pass_with_warnings_count,
            "fail_count": fail_count,
            "total_duration_seconds": round(total_duration_seconds, 1),
            "average_duration_seconds": round(average_duration_seconds, 1) if average_duration_seconds is not None else None,
        },
        "execution_health": {
            "total_submitted": total_submitted,
            "total_fills": total_fills,
            "total_entry_fill_received": total_entry_fill_received,
            "total_protective_exit_ack": total_protective_exit_ack,
            "total_protective_exit_failures": total_protective_exit_failures,
            "total_reconcile_mismatches": total_reconcile_mismatches,
            "total_startup_state_blocks": total_startup_state_blocks,
            "total_orphan_blocks": total_orphan_blocks,
            "total_position_add_blocks": total_position_add_blocks,
            "total_working_entry_blocks": total_working_entry_blocks,
        },
        "model_activity": {
            "total_model_evaluations": total_model_evaluations,
            "total_model_allowed": total_model_allowed,
            "total_model_blocked": total_model_blocked,
            "average_probability": round(average_probability, 6) if average_probability is not None else None,
            "max_probability": prob_max,
            "threshold_used": threshold_used,
            "threshold_consistency": threshold_consistency,
        },
        "safety_checks": {
            "fills_without_protective_exit_ack": fills_without_protective_exit_ack,
            "reconcile_abort_present": repeated_reconcile_issue_present,
            "session_abort_present": session_abort_present,
            "startup_state_not_clearing": startup_state_not_clearing,
            "orphan_block_present": orphan_block_present,
            "orphan_block_unresolved": orphan_block_unresolved,
            "repeated_reconcile_issue_present": repeated_reconcile_issue_present,
            "repeated_position_add_attempts_present": repeated_position_add_attempts_present,
        },
        "reconcile_and_startup_diagnostics": {
            "dominant_reconcile_issue_type": _top_key(reconcile_by_type),
            "dominant_reconcile_symbol": _top_key(reconcile_by_symbol),
            "startup_block_clear_rate": startup_block_clear_rate,
            "sessions_with_uncleared_startup_state": sessions_with_uncleared_startup,
            "sessions_with_reconcile_abort": sessions_with_reconcile_abort,
            "top_3_reconcile_issue_buckets": top_3_buckets,
        },
        "session_ids": [r.get("session_metadata", {}).get("session_id", "unknown") for r in reports],
    }


def compute_promotion_verdict(
    aggregated: dict[str, Any],
    *,
    minimum_passing_sessions: int = DEFAULT_MINIMUM_PASSING_SESSIONS,
    minimum_total_duration_seconds: float = DEFAULT_MINIMUM_TOTAL_DURATION_SECONDS,
    minimum_total_fills: int = DEFAULT_MINIMUM_TOTAL_FILLS,
    minimum_model_evaluations: int = DEFAULT_MINIMUM_MODEL_EVALUATIONS,
) -> tuple[str, list[str]]:
    """
    Compute promotion verdict from aggregated soak data.
    Returns (verdict, reasons).
    """
    reasons: list[str] = []
    cov = aggregated.get("session_coverage") or {}
    exec_h = aggregated.get("execution_health") or {}
    model_a = aggregated.get("model_activity") or {}
    safety = aggregated.get("safety_checks") or {}

    fail_count = _g(cov, "fail_count")
    pass_count = _g(cov, "pass_count")
    pass_with_warnings_count = _g(cov, "pass_with_warnings_count")
    total_sessions = _g(cov, "total_sessions")
    total_duration = cov.get("total_duration_seconds") or 0
    total_fills = _g(exec_h, "total_fills")
    total_pe_failures = _g(exec_h, "total_protective_exit_failures")
    total_entry_fill = _g(exec_h, "total_entry_fill_received")
    total_pe_ack = _g(exec_h, "total_protective_exit_ack")
    total_model_evals = _g(model_a, "total_model_evaluations")

    # --- NOT_READY (hard failures) ---
    if fail_count > 0:
        reasons.append("any_fail_soak_report")
    if total_pe_failures > 0:
        reasons.append("protective_exit_placement_failed_count_gt_zero")
    if total_entry_fill > total_pe_ack:
        reasons.append("fills_without_protective_exit_ack")
    if _g_bool(safety, "session_abort_present"):
        reasons.append("session_aborted")
    if _g_bool(safety, "repeated_reconcile_issue_present"):
        reasons.append("repeated_reconcile_mismatch_abort_present")
    if _g_bool(safety, "startup_state_not_clearing"):
        reasons.append("unresolved_startup_state")
    if _g_bool(safety, "orphan_block_unresolved"):
        reasons.append("unresolved_orphan_block")
    if total_sessions == 0:
        reasons.append("no_valid_soak_reports")

    if reasons:
        return (VERDICT_NOT_READY, reasons)

    # --- CONTINUE_DEMO_SOAK (sample size too small) ---
    passing_sessions = pass_count + pass_with_warnings_count
    if passing_sessions < minimum_passing_sessions:
        reasons.append(f"passing_sessions_{passing_sessions}_lt_minimum_{minimum_passing_sessions}")
    if total_duration < minimum_total_duration_seconds:
        reasons.append(f"total_duration_{total_duration:.0f}s_lt_minimum_{minimum_total_duration_seconds:.0f}s")
    if total_fills < minimum_total_fills:
        reasons.append(f"total_fills_{total_fills}_lt_minimum_{minimum_total_fills}")
    if total_model_evals < minimum_model_evaluations:
        reasons.append(f"total_model_evaluations_{total_model_evals}_lt_minimum_{minimum_model_evaluations}")

    if reasons:
        return (VERDICT_CONTINUE_DEMO_SOAK, reasons)

    # --- READY_FOR_PAPER ---
    return (VERDICT_READY_FOR_PAPER, ["minimum_coverage_met", "protective_exit_clean", "reconcile_acceptable", "anti_stacking_guard_functioning"])


def build_promotion_assessment(
    reports: list[dict[str, Any]],
    *,
    minimum_passing_sessions: int = DEFAULT_MINIMUM_PASSING_SESSIONS,
    minimum_total_duration_seconds: float = DEFAULT_MINIMUM_TOTAL_DURATION_SECONDS,
    minimum_total_fills: int = DEFAULT_MINIMUM_TOTAL_FILLS,
    minimum_model_evaluations: int = DEFAULT_MINIMUM_MODEL_EVALUATIONS,
) -> dict[str, Any]:
    """
    Build full promotion readiness assessment from soak reports.
    """
    if not reports:
        verdict, reasons = VERDICT_NOT_READY, ["no_valid_soak_reports"]
        aggregated = {}
    else:
        aggregated = aggregate_soak_reports(reports)
        verdict, reasons = compute_promotion_verdict(
            aggregated,
            minimum_passing_sessions=minimum_passing_sessions,
            minimum_total_duration_seconds=minimum_total_duration_seconds,
            minimum_total_fills=minimum_total_fills,
            minimum_model_evaluations=minimum_model_evaluations,
        )

    assessment: dict[str, Any] = {
        "assessment_metadata": {
            "assessed_at": datetime.now(timezone.utc).isoformat(),
            "reports_evaluated": len(reports),
            "minimum_passing_sessions": minimum_passing_sessions,
            "minimum_total_duration_seconds": minimum_total_duration_seconds,
            "minimum_total_fills": minimum_total_fills,
            "minimum_model_evaluations": minimum_model_evaluations,
        },
        "promotion_verdict": {
            "verdict": verdict,
            "reasons": reasons,
        },
    }
    if aggregated:
        assessment["session_coverage"] = aggregated.get("session_coverage", {})
        assessment["execution_health"] = aggregated.get("execution_health", {})
        assessment["model_activity"] = aggregated.get("model_activity", {})
        assessment["safety_checks"] = aggregated.get("safety_checks", {})
        assessment["reconcile_and_startup_diagnostics"] = aggregated.get("reconcile_and_startup_diagnostics", {})
        assessment["session_ids"] = aggregated.get("session_ids", [])

    return assessment


def build_promotion_markdown(assessment: dict[str, Any]) -> str:
    """Build human-readable markdown promotion report."""
    lines: list[str] = [
        "# Promotion Readiness Assessment",
        "",
        "## Overview",
        "",
    ]
    meta = assessment.get("assessment_metadata") or {}
    verdict_block = assessment.get("promotion_verdict") or {}
    verdict = verdict_block.get("verdict", VERDICT_NOT_READY)
    reasons = verdict_block.get("reasons", [])

    lines.append(f"- **Assessed at:** {meta.get('assessed_at', '')}")
    lines.append(f"- **Reports evaluated:** {meta.get('reports_evaluated', 0)}")
    lines.append(f"- **Verdict:** **{verdict}**")
    lines.append("")

    lines.append("## Coverage Summary")
    lines.append("")
    cov = assessment.get("session_coverage") or {}
    if cov:
        for k, v in cov.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- (no sessions)")
    lines.append("")

    lines.append("## Execution Health")
    lines.append("")
    exec_h = assessment.get("execution_health") or {}
    if exec_h:
        for k, v in exec_h.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- (no data)")
    lines.append("")

    lines.append("## Model Activity")
    lines.append("")
    model_a = assessment.get("model_activity") or {}
    if model_a:
        for k, v in model_a.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- (no data)")
    lines.append("")

    lines.append("## Safety Findings")
    lines.append("")
    safety = assessment.get("safety_checks") or {}
    if safety:
        for k, v in safety.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- (no data)")
    lines.append("")

    rsd = assessment.get("reconcile_and_startup_diagnostics") or {}
    if rsd:
        lines.append("## Reconcile & Startup Diagnostics")
        lines.append("")
        lines.append(f"- Dominant reconcile issue type: {rsd.get('dominant_reconcile_issue_type') or '—'}")
        lines.append(f"- Dominant reconcile symbol: {rsd.get('dominant_reconcile_symbol') or '—'}")
        lines.append(f"- Startup block clear rate: {rsd.get('startup_block_clear_rate') or '—'}")
        lines.append(f"- Sessions with uncleared startup state: {rsd.get('sessions_with_uncleared_startup_state', 0)}")
        lines.append(f"- Sessions with reconcile abort: {rsd.get('sessions_with_reconcile_abort', 0)}")
        for b in rsd.get("top_3_reconcile_issue_buckets") or []:
            lines.append(f"- {b.get('reason_bucket', '')}: {b.get('count', 0)}")
        lines.append("")

    lines.append("## Final Recommendation")
    lines.append("")
    lines.append(f"**{verdict}**")
    if reasons:
        lines.append("")
        lines.append("### Reasons")
        for r in reasons:
            lines.append(f"- {r}")
    lines.append("")

    lines.append("## Recommended Next Action")
    lines.append("")
    if verdict == VERDICT_NOT_READY:
        lines.append("- Investigate failures before promotion. Fix protective exit flow, reconciliation, or startup/orphan issues.")
    elif verdict == VERDICT_CONTINUE_DEMO_SOAK:
        lines.append("- Continue demo soak. Sample size or activity too small. Run more sessions to meet minimum thresholds.")
    elif verdict == VERDICT_READY_FOR_PAPER:
        lines.append("- Proceed to paper trading. Minimum coverage met, protective exit and reconcile behavior acceptable.")
    else:
        lines.append("- Review assessment for next phase progression.")
    lines.append("")

    return "\n".join(lines)


def run_promotion_evaluation(
    input_specs: list[str] | None,
    dir_path: Path | None,
    output_dir: Path,
    *,
    minimum_passing_sessions: int = DEFAULT_MINIMUM_PASSING_SESSIONS,
    minimum_total_duration_seconds: float = DEFAULT_MINIMUM_TOTAL_DURATION_SECONDS,
    minimum_total_fills: int = DEFAULT_MINIMUM_TOTAL_FILLS,
    minimum_model_evaluations: int = DEFAULT_MINIMUM_MODEL_EVALUATIONS,
    logger: logging.Logger | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """
    Load soak reports, evaluate, write JSON and markdown. Returns (json_path, md_path, assessment).
    """
    log = logger or get_logger(__name__)
    paths = collect_paths_from_input(input_specs, dir_path)
    reports = load_soak_reports(paths)

    assessment = build_promotion_assessment(
        reports,
        minimum_passing_sessions=minimum_passing_sessions,
        minimum_total_duration_seconds=minimum_total_duration_seconds,
        minimum_total_fills=minimum_total_fills,
        minimum_model_evaluations=minimum_model_evaluations,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"promotion_readiness_{ts}.json"
    md_path = output_dir / f"promotion_readiness_{ts}.md"

    json_path.write_text(json.dumps(assessment, indent=2), encoding="utf-8")
    md_path.write_text(build_promotion_markdown(assessment), encoding="utf-8")

    verdict_block = assessment.get("promotion_verdict") or {}
    verdict = verdict_block.get("verdict", VERDICT_NOT_READY)
    reasons = verdict_block.get("reasons", [])

    log.info(
        "promotion_readiness_written",
        json_path=str(json_path),
        markdown_path=str(md_path),
        verdict=verdict,
        reports_evaluated=len(reports),
    )
    log.info(
        "promotion_readiness_verdict",
        verdict=verdict,
        reasons=reasons,
    )

    return (json_path, md_path, assessment)
