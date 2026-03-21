"""ML model calibration and threshold-readiness analysis (reporting only, no behavior change)."""

from __future__ import annotations


def _get_probs(decisions: list[dict]) -> list[float]:
    return [float(d.get("model_probability", 0)) for d in decisions if "model_probability" in d]


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
    return {
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
