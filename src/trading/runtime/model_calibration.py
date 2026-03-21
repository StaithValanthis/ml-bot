"""ML model calibration and threshold-readiness analysis (reporting only, no behavior change)."""

from __future__ import annotations


def _get_probs(decisions: list[dict]) -> list[float]:
    return [float(d.get("model_probability", 0)) for d in decisions if "model_probability" in d]


def _percentile(sorted_arr: list[float], p: float) -> float | None:
    """Linear interpolation percentile. p in 0-100."""
    if not sorted_arr:
        return None
    n = len(sorted_arr)
    idx = p / 100.0 * (n - 1) if n > 1 else 0
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_arr[lo] * (1 - frac) + sorted_arr[hi] * frac


def _log_scale_buckets() -> list[tuple[str, float, float]]:
    """Log-scale probability buckets for runtime distribution visibility."""
    return [
        ("lt_1e_6", 0.0, 1e-6),
        ("1e_6_to_1e_5", 1e-6, 1e-5),
        ("1e_5_to_1e_4", 1e-5, 1e-4),
        ("1e_4_to_1e_3", 1e-4, 1e-3),
        ("1e_3_to_1e_2", 1e-3, 1e-2),
        ("1e_2_to_1e_1", 1e-2, 0.1),
        ("gte_1e_1", 0.1, 1.01),
    ]


def build_log_scale_buckets(probs: list[float]) -> list[dict]:
    """Count by log-scale probability buckets (<1e-6, 1e-6 to 1e-5, etc.)."""
    result: list[dict] = []
    for label, low, high in _log_scale_buckets():
        count = sum(1 for p in probs if low <= p < high)
        result.append({"bucket": label, "low": low, "high": high, "count": count})
    return result


def compute_retention_thresholds(probs: list[float]) -> dict[str, float | None]:
    """
    Empirical thresholds to retain X% of candidates.
    threshold_keep_90pct = p10, threshold_keep_75pct = p25, etc.
    """
    if not probs:
        return {
            "threshold_keep_90pct": None,
            "threshold_keep_75pct": None,
            "threshold_keep_50pct": None,
            "threshold_keep_25pct": None,
            "threshold_keep_10pct": None,
        }
    sorted_p = sorted(probs)
    return {
        "threshold_keep_90pct": round(_percentile(sorted_p, 10) or 0.0, 8),
        "threshold_keep_75pct": round(_percentile(sorted_p, 25) or 0.0, 8),
        "threshold_keep_50pct": round(_percentile(sorted_p, 50) or 0.0, 8),
        "threshold_keep_25pct": round(_percentile(sorted_p, 75) or 0.0, 8),
        "threshold_keep_10pct": round(_percentile(sorted_p, 90) or 0.0, 8),
    }


def build_runtime_calibration_stats(
    probs: list[float],
    *,
    current_threshold: float | None = None,
) -> dict:
    """
    Build runtime shadow calibration summary: distribution stats, percentiles,
    log-scale buckets, retention thresholds, and threshold viability flag.
    """
    if not probs:
        return {
            "total_shadow_evaluations": 0,
            "probability_distribution": {
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "p50": None,
                "p75": None,
                "p90": None,
                "p95": None,
                "p99": None,
            },
            "probability_buckets_log": [],
            "retention_thresholds": compute_retention_thresholds([]),
            "current_threshold_above_observed_max": None,
        }
    sorted_p = sorted(probs)
    n = len(probs)
    mean_p = sum(probs) / n
    median_p = _percentile(sorted_p, 50)
    dist = {
        "min": round(min(probs), 8),
        "max": round(max(probs), 8),
        "mean": round(mean_p, 8),
        "median": round(median_p, 8) if median_p is not None else None,
        "p50": round(_percentile(sorted_p, 50) or 0.0, 8),
        "p75": round(_percentile(sorted_p, 75) or 0.0, 8),
        "p90": round(_percentile(sorted_p, 90) or 0.0, 8),
        "p95": round(_percentile(sorted_p, 95) or 0.0, 8),
        "p99": round(_percentile(sorted_p, 99) or 0.0, 8),
    }
    obs_max = max(probs)
    threshold_above_max = (
        (current_threshold is not None and current_threshold > obs_max)
        if current_threshold is not None
        else None
    )
    retention = compute_retention_thresholds(probs)
    suggested: dict[str, float | None] | None = None
    if threshold_above_max and current_threshold is not None:
        pm = dist["max"]
        p95 = dist["p95"]
        p99 = dist["p99"]
        suggested = {
            "threshold_near_max": round(pm, 8) if pm is not None else None,
            "threshold_near_p99": p99,
            "threshold_near_p95": p95,
            "threshold_keep_50pct": retention.get("threshold_keep_50pct"),
            "threshold_keep_25pct": retention.get("threshold_keep_25pct"),
        }

    return {
        "total_shadow_evaluations": n,
        "probability_distribution": dist,
        "probability_buckets_log": build_log_scale_buckets(probs),
        "retention_thresholds": retention,
        "current_threshold_above_observed_max": threshold_above_max,
        "suggested_thresholds_when_above_max": suggested,
    }


def _sweep_for_threshold(decisions: list[dict], threshold: float) -> dict:
    blocks = sum(1 for d in decisions if float(d.get("model_probability", 0)) < threshold)
    allows = len(decisions) - blocks
    return {
        "threshold": threshold,
        "would_block_count": blocks,
        "would_allow_count": allows,
        "block_rate": round(blocks / len(decisions), 4) if decisions else 0,
    }


def build_probability_buckets(decisions: list[dict]) -> list[dict]:
    """Build histogram-style counts per probability bucket (0.2 width)."""
    buckets_def = [
        ("0.0-0.2", 0.0, 0.2),
        ("0.2-0.4", 0.2, 0.4),
        ("0.4-0.6", 0.4, 0.6),
        ("0.6-0.8", 0.6, 0.8),
        ("0.8-1.0", 0.8, 1.01),
    ]
    result: list[dict] = []
    for label, low, high in buckets_def:
        samples = [d for d in decisions if low <= float(d.get("model_probability", 0)) < high]
        blocks = sum(1 for s in samples if s.get("shadow_would_block"))
        allows = len(samples) - blocks
        result.append({
            "probability_bucket": label,
            "sample_count": len(samples),
            "shadow_block_count": blocks,
            "shadow_allow_count": allows,
        })
    return result


def build_threshold_sweep(
    decisions: list[dict],
    thresholds: tuple[float, ...] = (0.30, 0.40, 0.50, 0.60, 0.70),
) -> list[dict]:
    """For each hypothetical threshold, report would_block_count, would_allow_count, block_rate."""
    return [_sweep_for_threshold(decisions, t) for t in thresholds]


def build_model_calibration_summary(
    decisions: list[dict],
    *,
    threshold_configured: float | None,
    session_submitted: int = 0,
    session_filled: int = 0,
) -> dict:
    """
    Build session-level ML calibration summary from shadow decisions.
    Outcome linkage is session-level only (per-decision linkage not available).
    Includes runtime calibration stats (percentiles, log buckets, retention thresholds).
    """
    probs = _get_probs(decisions)
    total = len(decisions)
    blocks = sum(1 for d in decisions if d.get("shadow_would_block"))
    allows = total - blocks
    sorted_probs = sorted(probs) if probs else []
    mid = len(sorted_probs) // 2
    median_prob = (
        sorted_probs[mid] if len(sorted_probs) % 2 else (sorted_probs[mid - 1] + sorted_probs[mid]) / 2
    ) if sorted_probs else None
    mean_prob = round(sum(probs) / len(probs), 6) if probs else None

    base: dict = {
        "total_model_evaluations": total,
        "total_shadow_blocks": blocks,
        "total_shadow_allows": allows,
        "block_rate": round(blocks / total, 4) if total else 0,
        "mean_probability": mean_prob,
        "median_probability": round(median_prob, 6) if median_prob is not None else None,
        "min_probability": min(probs) if probs else None,
        "max_probability": max(probs) if probs else None,
        "probability_buckets": build_probability_buckets(decisions),
        "threshold_configured": threshold_configured,
        "threshold_sweep": build_threshold_sweep(decisions),
        "session_submitted_count": session_submitted,
        "session_filled_count": session_filled,
        "outcome_linkage_note": "per_decision_linkage_unavailable_session_aggregates_only",
    }

    if probs:
        base["runtime_calibration"] = build_runtime_calibration_stats(
            probs,
            current_threshold=threshold_configured,
        )

    return base


def _infer_promotion_verdict(
    *,
    current_threshold_above_max: bool | None,
    obs_max: float | None,
    obs_min: float | None,
    obs_mean: float | None,
) -> str:
    """
    Infer verdict from runtime calibration.
    remain_shadow | active_demo_ready | not_predictive_enough
    """
    if current_threshold_above_max is True:
        return "remain_shadow"
    if obs_max is None or obs_min is None:
        return "remain_shadow"
    spread = obs_max - obs_min
    if obs_mean is not None and obs_mean > 0:
        cv = spread / obs_mean if obs_mean > 0 else 0
        if spread < 1e-6 or cv < 0.01:
            return "not_predictive_enough"
    if current_threshold_above_max is False and obs_max > 1e-4:
        return "active_demo_ready"
    return "remain_shadow"


def build_promotion_recommendation(
    *,
    current_threshold: float | None,
    observed_max: float | None,
    observed_p95: float | None,
    observed_p99: float | None,
    observed_min: float | None = None,
    observed_mean: float | None = None,
    retention_thresholds: dict | None = None,
    verdict: str | None = None,
) -> dict:
    """
    Build promotion recommendation block for session/offline reports.
    Verdict: remain_shadow | active_demo_ready | not_predictive_enough
    """
    threshold_realistic = (
        (current_threshold is not None and observed_max is not None and current_threshold <= observed_max)
        if (current_threshold is not None and observed_max is not None)
        else None
    )
    threshold_above_max = (
        (current_threshold is not None and observed_max is not None and current_threshold > observed_max)
        if (current_threshold is not None and observed_max is not None)
        else None
    )
    if verdict is None:
        verdict = _infer_promotion_verdict(
            current_threshold_above_max=threshold_above_max,
            obs_max=observed_max,
            obs_min=observed_min,
            obs_mean=observed_mean,
        )
    return {
        "current_runtime_threshold": current_threshold,
        "observed_max_probability": observed_max,
        "observed_p95": observed_p95,
        "observed_p99": observed_p99,
        "current_threshold_realistic": threshold_realistic,
        "retention_thresholds": retention_thresholds or {},
        "suggested_threshold_shadow": (
            retention_thresholds.get("threshold_keep_75pct") if retention_thresholds else None
        ),
        "suggested_threshold_active_demo": (
            retention_thresholds.get("threshold_keep_50pct") if retention_thresholds else None
        ),
        "verdict": verdict,
    }
