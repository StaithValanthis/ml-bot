"""
Offline threshold analysis for model filter.

Computes precision, recall, F1, support at each threshold.
Shadow-vs-baseline comparison for promotion readiness.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ThresholdMetrics:
    """Metrics at a single threshold."""

    threshold: float
    precision: float
    recall: float
    f1: float
    support: int
    retained_count: int
    filtered_count: int
    retain_ratio: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    win_rate_retained: float | None = None
    win_rate_baseline: float | None = None
    positive_rate_retained: float | None = None
    positive_rate_baseline: float | None = None


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return (round(prec, 6), round(rec, 6), round(f1, 6))


def compute_threshold_grid(
    probs: list[float],
    labels: list[int],
    thresholds: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7),
    profitable_fill: list[int] | None = None,
) -> list[ThresholdMetrics]:
    """
    For each threshold, compute precision, recall, F1, support, retained/filtered counts.
    profitable_fill: optional per-sample (1=profitable, 0=not, None=unknown) for win_rate.
    """
    n = len(probs)
    if n != len(labels):
        raise ValueError("probs and labels length mismatch")
    results: list[ThresholdMetrics] = []
    baseline_positive = sum(labels)
    baseline_total = n
    positive_rate_baseline = baseline_positive / baseline_total if baseline_total else 0.0
    win_rate_baseline = None
    if profitable_fill is not None and len(profitable_fill) == n:
        profitable_retained = sum(1 for i in range(n) if labels[i] == 1 and profitable_fill[i] == 1)
        filled_count = sum(labels)
        win_rate_baseline = profitable_retained / filled_count if filled_count else None
    for thresh in thresholds:
        retained_idx = [i for i in range(n) if probs[i] >= thresh]
        filtered_idx = [i for i in range(n) if probs[i] < thresh]
        retained_count = len(retained_idx)
        filtered_count = len(filtered_idx)
        retain_ratio = retained_count / n if n else 0.0
        tp = sum(1 for i in retained_idx if labels[i] == 1)
        fp = sum(1 for i in retained_idx if labels[i] == 0)
        fn = sum(1 for i in filtered_idx if labels[i] == 1)
        tn = sum(1 for i in filtered_idx if labels[i] == 0)
        prec, rec, f1 = _precision_recall_f1(tp, fp, fn)
        support = tp + fn
        win_rate_retained = None
        positive_rate_retained = tp / retained_count if retained_count else None
        if profitable_fill is not None and len(profitable_fill) == n and retained_count > 0:
            prof_ret = sum(1 for i in retained_idx if labels[i] == 1 and profitable_fill[i] == 1)
            filled_ret = sum(1 for i in retained_idx if labels[i] == 1)
            win_rate_retained = prof_ret / filled_ret if filled_ret else None
        results.append(
            ThresholdMetrics(
                threshold=thresh,
                precision=prec,
                recall=rec,
                f1=f1,
                support=support,
                retained_count=retained_count,
                filtered_count=filtered_count,
                retain_ratio=round(retain_ratio, 4),
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                true_negatives=tn,
                win_rate_retained=win_rate_retained,
                win_rate_baseline=win_rate_baseline,
                positive_rate_retained=positive_rate_retained,
                positive_rate_baseline=positive_rate_baseline,
            )
        )
    return results


@dataclass(slots=True)
class ShadowVsBaselineReport:
    """Comparison of baseline (no gating) vs model-gated by threshold."""

    total_candidates: int
    baseline_positive_count: int
    baseline_positive_rate: float
    baseline_win_rate: float | None
    per_threshold: list[dict]


def shadow_vs_baseline_report(
    probs: list[float],
    labels: list[int],
    thresholds: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7),
    profitable_fill: list[int] | None = None,
) -> ShadowVsBaselineReport:
    """
    Compare baseline (all candidates) vs model-gated retention.
    Returns total, baseline stats, and per-threshold: retained, filtered, retain_ratio,
    win_rate_before/after, uplift, false_negative_risk, false_positive_reduction.
    """
    grid = compute_threshold_grid(probs, labels, thresholds, profitable_fill)
    total = len(labels)
    baseline_positive = sum(labels)
    baseline_positive_rate = baseline_positive / total if total else 0.0
    win_baseline = None
    if profitable_fill and len(profitable_fill) == total:
        filled = sum(labels)
        prof = sum(1 for i in range(total) if labels[i] == 1 and profitable_fill[i] == 1)
        win_baseline = prof / filled if filled else None
    per_thresh: list[dict] = []
    for m in grid:
        fn_risk = m.false_negatives / (m.false_negatives + m.true_positives) if (m.false_negatives + m.true_positives) else 0
        fp_baseline = sum(1 for i in range(total) if labels[i] == 0)
        fp_retained = m.false_positives
        fp_reduction = 1 - (fp_retained / fp_baseline) if fp_baseline else 0
        uplift = None
        if m.positive_rate_retained is not None and baseline_positive_rate > 0:
            uplift = (m.positive_rate_retained - baseline_positive_rate) / baseline_positive_rate
        per_thresh.append({
            "threshold": m.threshold,
            "retained_count": m.retained_count,
            "filtered_count": m.filtered_count,
            "retain_ratio": m.retain_ratio,
            "positive_rate_retained": m.positive_rate_retained,
            "win_rate_retained": m.win_rate_retained,
            "false_negative_count": m.false_negatives,
            "false_negative_risk": round(fn_risk, 4),
            "false_positive_reduction": round(fp_reduction, 4),
            "uplift_positive_rate": round(uplift, 4) if uplift is not None else None,
        })
    return ShadowVsBaselineReport(
        total_candidates=total,
        baseline_positive_count=baseline_positive,
        baseline_positive_rate=round(baseline_positive_rate, 4),
        baseline_win_rate=win_baseline,
        per_threshold=per_thresh,
    )
