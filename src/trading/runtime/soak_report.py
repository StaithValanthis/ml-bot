"""Soak report: structured session health summary for demo/paper runs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trading.monitoring.metrics import MetricsSnapshot


# --- Verdict constants ---
VERDICT_PASS = "PASS"
VERDICT_PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
VERDICT_FAIL = "FAIL"

# --- Machine-readable reason codes ---
REASON_SESSION_ABORTED = "session_aborted"
REASON_FILLS_WITHOUT_PROTECTIVE_EXIT_ACK = "fills_without_protective_exit_ack"
REASON_PROTECTIVE_EXIT_FAILURES_PRESENT = "protective_exit_failures_present"
REASON_REPEATED_RECONCILE_MISMATCH_ABORT = "repeated_reconcile_mismatch_abort"
REASON_MODEL_ALLOWED_BUT_NO_SUBMISSIONS = "model_allowed_but_no_submissions"
REASON_SUBMITTED_GT_ACK = "submitted_gt_ack"
REASON_ENTRY_FILL_INCONSISTENT = "entry_fill_inconsistent"
REASON_NO_MODEL_EVALUATIONS = "no_model_evaluations"
REASON_NO_CANDIDATES_SEEN = "no_candidates_seen"
REASON_ALL_CANDIDATES_BLOCKED = "all_candidates_blocked"
REASON_NO_SUBMISSIONS = "no_submissions"
REASON_ORPHAN_POSITION_BLOCK_TRIGGERED = "orphan_position_block_triggered"
REASON_STARTUP_STATE_BLOCK_TRIGGERED = "startup_state_block_triggered"
REASON_RECONCILE_MISMATCHES_PRESENT = "reconcile_mismatches_present"


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


def _g_str(d: dict[str, Any], key: str, default: str = "") -> str:
    v = d.get(key, default)
    return str(v) if v is not None else default


def _g_bool(d: dict[str, Any], key: str, default: bool = False) -> bool:
    v = d.get(key, default)
    return bool(v) if v is not None else default


def build_soak_report(
    summary: dict[str, Any],
    metrics_snapshot: MetricsSnapshot | None = None,
) -> dict[str, Any]:
    """
    Build structured soak report from session summary and metrics.
    Does not modify inputs.
    """
    flow = summary.get("strategy_flow") or {}
    so = summary.get("strategy_order_outcomes") or {}
    mf = summary.get("model_filter") or {}
    counters: dict[str, float] = {}
    if metrics_snapshot is not None:
        counters = getattr(metrics_snapshot, "counters", counters) or {}

    bars_confirmed = _g(flow, "bars_confirmed")
    candidates = _g(flow, "candidates")
    regime_rejected = _g(flow, "regime_rejected")
    signal_rejected = _g(flow, "signal_rejected")
    sizing_rejected = _g(flow, "sizing_rejected")
    risk_rejected = _g(flow, "risk_rejected")
    model_filter_reached = _g(flow, "model_filter_reached")
    model_blocked = _g(flow, "model_blocked")
    submitted = _g(flow, "submitted")

    intents = _g(so, "intents")
    strategy_submitted = _g(so, "submissions")
    strategy_ack = _g(so, "acks")
    strategy_filled = _g(so, "filled")
    strategy_cancelled = _g(so, "cancelled")
    strategy_rejected = _g(so, "rejected")

    entry_fill_count = int(counters.get("entry_fill_received_count", strategy_filled))
    pe_plan = int(counters.get("protective_exit_plan_created_count", 0))
    pe_tracking = int(counters.get("protective_exit_tracking_registered_count", 0))
    pe_order_submitted = int(counters.get("protective_exit_order_submitted_count", 0))
    pe_ack = int(counters.get("protective_exit_order_ack_received_count", 0))
    pe_failed = int(counters.get("protective_exit_placement_failed_count", 0))

    model_allowed = _g(mf, "allowed")
    model_blocked_count = _g(mf, "blocked")
    total_model_evals = model_allowed + model_blocked_count
    if _g(mf, "prediction_unavailable", 0) > 0:
        total_model_evals += _g(mf, "prediction_unavailable")

    shadow_decisions = summary.get("model_shadow_decisions") or {}
    if shadow_decisions:
        total_model_evals = max(
            total_model_evals,
            _g(shadow_decisions, "total_model_evaluations"),
        )

    reconcile_cycles = _g(summary, "reconcile_mismatch_cycles")
    abort_reasons = summary.get("abort_reasons") or []

    session_id = _session_id_from_summary(summary)
    start_s = summary.get("session_start", "")
    end_s = summary.get("session_end", "")
    duration_seconds: float | None = None
    if start_s and end_s:
        try:
            from datetime import datetime
            st = datetime.fromisoformat(start_s.replace("Z", "+00:00"))
            et = datetime.fromisoformat(end_s.replace("Z", "+00:00"))
            duration_seconds = (et - st).total_seconds()
        except Exception:
            pass

    model_cal = summary.get("model_calibration") or {}
    rc = model_cal.get("runtime_calibration") or {}
    dist = rc.get("probability_distribution") or {}
    sug = rc.get("suggested_thresholds_when_above_max") or {}

    regime_passed = max(0, candidates - regime_rejected)
    signal_passed = max(0, regime_passed - signal_rejected)
    sizing_passed = max(0, signal_passed - sizing_rejected)
    risk_passed = max(0, sizing_passed - risk_rejected)

    report: dict[str, Any] = {
        "session_metadata": {
            "session_id": session_id,
            "mode": _g_str(summary, "mode"),
            "symbols": summary.get("symbols") or [],
            "started_at": start_s,
            "ended_at": end_s,
            "duration_seconds": round(duration_seconds, 1) if duration_seconds is not None else None,
            "session_ended_cleanly": _g_bool(summary, "session_ended_cleanly", True),
            "abort_reasons": abort_reasons,
        },
        "runtime_pipeline_totals": {
            "bars_confirmed": bars_confirmed,
            "candidates": candidates,
            "regime_rejected": regime_rejected,
            "signal_rejected": signal_rejected,
            "sizing_rejected": sizing_rejected,
            "risk_rejected": risk_rejected,
            "model_filter_reached": model_filter_reached,
            "model_blocked": model_blocked,
            "submitted": submitted,
        },
        "model_evaluation_summary": {
            "model_filter_mode": _g_str(mf, "mode", "hard_block"),
            "threshold": _g_float(mf, "threshold", 0.5),
            "total_model_evaluations": total_model_evals,
            "model_allowed_count": model_allowed,
            "model_blocked_count": model_blocked_count,
            "allow_ratio": round(model_allowed / total_model_evals, 6) if total_model_evals > 0 else None,
            "block_ratio": round(model_blocked_count / total_model_evals, 6) if total_model_evals > 0 else None,
            "prob_count": _g(mf, "prob_count"),
            "prob_min": _g_float(mf, "prob_min"),
            "prob_max": _g_float(mf, "prob_max"),
            "prob_latest": _g_float(mf, "prob_latest"),
            "p95": dist.get("p95"),
            "p99": dist.get("p99"),
            "suggested_thresholds": sug if sug else None,
        },
        "execution_summary": {
            "strategy_order_intent_created_count": intents,
            "strategy_order_submitted_count": strategy_submitted,
            "strategy_order_ack_received_count": strategy_ack,
            "strategy_order_filled_count": strategy_filled,
            "strategy_order_cancelled_count": strategy_cancelled,
            "strategy_order_rejected_count": strategy_rejected,
            "entry_fill_received_count": entry_fill_count,
            "protective_exit_plan_created_count": pe_plan,
            "protective_exit_tracking_registered_count": pe_tracking,
            "protective_exit_order_submitted_count": pe_order_submitted,
            "protective_exit_order_ack_received_count": pe_ack,
            "protective_exit_placement_failed_count": pe_failed,
        },
        "safety_summary": {
            "orphan_position_blocked_count": int(counters.get("orphan_position_blocked_count", 0)),
            "orphan_position_block_cleared_count": int(counters.get("orphan_position_block_cleared_count", 0)),
            "startup_state_blocked_count": int(counters.get("startup_state_blocked_count", 0)),
            "startup_state_block_cleared_count": int(counters.get("startup_state_block_cleared_count", 0)),
            "reconcile_mismatch_detected_count": reconcile_cycles,
            "repeated_reconcile_mismatch_triggered": "repeated_reconcile_mismatch" in abort_reasons,
            "last_orphan_position_details": summary.get("orphan_position_details") or None,
            "last_startup_state_details": summary.get("startup_state_details") or None,
        },
        "candidate_summary": {
            "final_blocking_stage": summary.get("blocking_stage") or "unknown",
            "blocking_stage_count": _blocking_stage_count(summary, flow),
            "candidate_pipeline_detail": {
                "raw_candidates": int(counters.get("strategy_raw_candidates_total", 0)),
                "relaxed_demo_candidates": int(counters.get("strategy_relaxed_demo_candidates_created", 0)),
                "regime_passed": regime_passed,
                "signal_passed": signal_passed,
                "sizing_passed": sizing_passed,
                "risk_passed": risk_passed,
                "model_reached": model_filter_reached,
                "model_blocked": model_blocked,
                "submitted": submitted,
            },
        },
    }

    verdict, failures, warnings = compute_verdict(report)
    report["health_verdict"] = {
        "verdict": verdict,
        "failures": failures,
        "warnings": warnings,
    }

    return report


def _session_id_from_summary(summary: dict[str, Any]) -> str:
    start = summary.get("session_start", "")
    if not start:
        return "unknown"
    ts = start[:19].replace("-", "").replace(":", "").replace("T", "_")
    return f"session_{ts}"


def _blocking_stage_count(summary: dict[str, Any], flow: dict[str, Any]) -> int | None:
    """Infer count of distinct blocking stages observed during run."""
    stages: set[str] = set()
    blocking = summary.get("blocking_stage")
    if blocking:
        stages.add(str(blocking))
    c = flow.get("candidates", 0) or 0
    if c == 0 and _g(flow, "bars_confirmed", 0) > 0:
        stages.add("no_candidates")
    regime = _g(flow, "regime_rejected", 0)
    if regime > 0 and regime >= c:
        stages.add("regime_rejected")
    if _g(flow, "signal_rejected", 0) > 0 or _g(flow, "sizing_rejected", 0) > 0:
        stages.add("signal_rejected")
    if _g(flow, "risk_rejected", 0) > 0:
        stages.add("risk_rejected")
    if _g(flow, "model_filter_reached", 0) > 0 and _g(flow, "model_blocked", 0) >= _g(flow, "model_filter_reached", 0):
        stages.add("model_blocked")
    if _g(flow, "submitted", 0) > 0:
        stages.add("submitted")
    return len(stages) if stages else None


def compute_verdict(report: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    """
    Compute health verdict from soak report.
    Returns (verdict, failures, warnings).
    """
    meta = report.get("session_metadata") or {}
    exec_s = report.get("execution_summary") or {}
    safety = report.get("safety_summary") or {}
    pipeline = report.get("runtime_pipeline_totals") or {}
    model_s = report.get("model_evaluation_summary") or {}
    abort_reasons = meta.get("abort_reasons") or []

    strategy_filled = _g(exec_s, "strategy_order_filled_count")
    pe_ack = _g(exec_s, "protective_exit_order_ack_received_count")
    pe_failed = _g(exec_s, "protective_exit_placement_failed_count")
    entry_fill = _g(exec_s, "entry_fill_received_count")
    submitted = _g(exec_s, "strategy_order_submitted_count")
    ack = _g(exec_s, "strategy_order_ack_received_count")

    model_filter_reached = _g(pipeline, "model_filter_reached")
    model_allowed = _g(model_s, "model_allowed_count")
    total_model_evals = _g(model_s, "total_model_evaluations")
    candidates = _g(pipeline, "candidates")

    session_clean = _g_bool(meta, "session_ended_cleanly", True)
    repeated_reconcile = safety.get("repeated_reconcile_mismatch_triggered") is True
    orphan_blocked = _g(safety, "orphan_position_blocked_count", 0) > 0
    startup_blocked = _g(safety, "startup_state_blocked_count", 0) > 0
    reconcile_count = _g(safety, "reconcile_mismatch_detected_count", 0)

    failures: list[str] = []
    warnings: list[str] = []

    # --- FAIL conditions ---
    if not session_clean or abort_reasons:
        failures.append(REASON_SESSION_ABORTED)

    if strategy_filled > pe_ack:
        failures.append(REASON_FILLS_WITHOUT_PROTECTIVE_EXIT_ACK)

    if pe_failed > 0:
        failures.append(REASON_PROTECTIVE_EXIT_FAILURES_PRESENT)

    if repeated_reconcile:
        failures.append(REASON_REPEATED_RECONCILE_MISMATCH_ABORT)

    if model_filter_reached > 0 and submitted == 0 and model_allowed > 0:
        failures.append(REASON_MODEL_ALLOWED_BUT_NO_SUBMISSIONS)

    if submitted > ack:
        failures.append(REASON_SUBMITTED_GT_ACK)

    if entry_fill > strategy_filled:
        failures.append(REASON_ENTRY_FILL_INCONSISTENT)

    # --- WARNINGS ---
    if total_model_evals == 0:
        warnings.append(REASON_NO_MODEL_EVALUATIONS)

    if candidates == 0:
        warnings.append(REASON_NO_CANDIDATES_SEEN)

    if candidates > 0 and _g(pipeline, "submitted", 0) == 0 and _g(pipeline, "model_blocked", 0) >= candidates:
        warnings.append(REASON_ALL_CANDIDATES_BLOCKED)

    if submitted == 0 and not failures:
        warnings.append(REASON_NO_SUBMISSIONS)

    if orphan_blocked:
        warnings.append(REASON_ORPHAN_POSITION_BLOCK_TRIGGERED)

    if startup_blocked:
        warnings.append(REASON_STARTUP_STATE_BLOCK_TRIGGERED)

    if reconcile_count > 0:
        warnings.append(REASON_RECONCILE_MISMATCHES_PRESENT)

    # --- Verdict ---
    if failures:
        return (VERDICT_FAIL, failures, warnings)
    if warnings:
        return (VERDICT_PASS_WITH_WARNINGS, failures, warnings)
    return (VERDICT_PASS, failures, warnings)


def build_soak_markdown(report: dict[str, Any]) -> str:
    """Build human-readable markdown soak report."""
    lines: list[str] = [
        "# Soak Report",
        "",
        "## Session Overview",
    ]
    meta = report.get("session_metadata") or {}
    lines.append(f"- **Session ID:** {meta.get('session_id', '')}")
    lines.append(f"- **Mode:** {meta.get('mode', '')}")
    lines.append(f"- **Symbols:** {', '.join(meta.get('symbols', []) or [])}")
    lines.append(f"- **Started:** {meta.get('started_at', '')}")
    lines.append(f"- **Ended:** {meta.get('ended_at', '')}")
    dur = meta.get("duration_seconds")
    lines.append(f"- **Duration:** {f'{dur:.1f}s' if isinstance(dur, (int, float)) else dur or '—'}")
    lines.append(f"- **Session ended cleanly:** {meta.get('session_ended_cleanly', True)}")
    if meta.get("abort_reasons"):
        lines.append(f"- **Abort reasons:** {', '.join(meta['abort_reasons'])}")
    lines.append("")

    lines.append("## Pipeline Totals")
    pipeline = report.get("runtime_pipeline_totals") or {}
    for k, v in pipeline.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## Model Summary")
    model_s = report.get("model_evaluation_summary") or {}
    lines.append(f"- Mode: {model_s.get('model_filter_mode', '')}")
    lines.append(f"- Threshold: {model_s.get('threshold', '')}")
    lines.append(f"- Total evaluations: {model_s.get('total_model_evaluations', 0)}")
    lines.append(f"- Allowed: {model_s.get('model_allowed_count', 0)} | Blocked: {model_s.get('model_blocked_count', 0)}")
    lines.append(f"- prob min/max/latest: {model_s.get('prob_min')} / {model_s.get('prob_max')} / {model_s.get('prob_latest')}")
    if model_s.get("p95") is not None or model_s.get("p99") is not None:
        lines.append(f"- p95: {model_s.get('p95')} | p99: {model_s.get('p99')}")
    lines.append("")

    lines.append("## Execution Summary")
    exec_s = report.get("execution_summary") or {}
    for k, v in exec_s.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## Safety Summary")
    safety = report.get("safety_summary") or {}
    for k, v in safety.items():
        if k in ("last_orphan_position_details", "last_startup_state_details") and v:
            lines.append(f"- {k}: (see JSON for details)")
        elif k in ("last_orphan_position_details", "last_startup_state_details"):
            lines.append(f"- {k}: —")
        else:
            lines.append(f"- {k}: {v}")
    lines.append("")

    verdict_block = report.get("health_verdict") or {}
    verdict = verdict_block.get("verdict", VERDICT_PASS)
    failures = verdict_block.get("failures") or []
    warnings = verdict_block.get("warnings") or []

    lines.append("## Final Verdict")
    lines.append(f"- **{verdict}**")
    if failures:
        lines.append("- Failures:")
        for f in failures:
            lines.append(f"  - {f}")
    if warnings:
        lines.append("- Warnings:")
        for w in warnings:
            lines.append(f"  - {w}")
    lines.append("")

    lines.append("## Recommended Next Action")
    if verdict == VERDICT_FAIL:
        lines.append("- Investigate failures before next run. Check protective exit flow and reconciliation.")
    elif verdict == VERDICT_PASS_WITH_WARNINGS:
        lines.append("- Run acceptable. Review warnings for operational awareness.")
    else:
        lines.append("- Run healthy. Proceed with confidence.")
    lines.append("")

    return "\n".join(lines)
