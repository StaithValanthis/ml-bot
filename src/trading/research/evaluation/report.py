"""Report and artifact generation for offline evaluation."""

from __future__ import annotations

import csv
from pathlib import Path

from trading.research.evaluation.offline_evaluator import OfflineEvalResult
from trading.util.json_util import dumps_json_safe


def write_eval_reports(result: OfflineEvalResult, output_dir: Path) -> dict[str, Path]:
    """
    Write JSON summary, CSV threshold table, CSV per-fold metrics, optional per-row CSV, markdown.
    Returns dict of written paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rid = result.run_id
    written: dict[str, Path] = {}

    summary = {
        "success": result.success,
        "run_id": rid,
        "dataset_path": result.dataset_path,
        "model_path": result.model_path,
        "total_rows": result.total_rows,
        "cv_config": result.cv_config,
        "n_folds": len(result.fold_results),
        "aggregated_threshold_metrics": result.aggregated_threshold_metrics,
        "shadow_vs_baseline": result.shadow_vs_baseline,
        "retention_recommendations": result.retention_recommendations,
        "error": result.error,
    }
    json_path = output_dir / f"eval_summary_{rid}.json"
    json_path.write_text(dumps_json_safe(summary, indent=2), encoding="utf-8")
    written["json"] = json_path

    if result.aggregated_threshold_metrics:
        csv_path = output_dir / f"threshold_table_{rid}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "threshold", "precision", "recall", "f1", "support",
                    "retained_count", "filtered_count", "retain_ratio",
                    "true_positives", "false_positives", "false_negatives", "true_negatives",
                    "win_rate_retained", "positive_rate_retained",
                ],
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in result.aggregated_threshold_metrics:
                writer.writerow(row)
        written["threshold_csv"] = csv_path

    total_preds = sum(len(fr.predictions) for fr in result.fold_results)
    if total_preds > 0 and total_preds <= 10000:
        pred_path = output_dir / f"predictions_{rid}.csv"
        with pred_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["fold_id", "model_probability", "label"])
            for fr in result.fold_results:
                for prob, label in fr.predictions:
                    writer.writerow([fr.fold_id, prob, label])
        written["predictions_csv"] = pred_path

    if result.fold_results:
        fold_path = output_dir / f"per_fold_metrics_{rid}.csv"
        with fold_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["fold_id", "val_size", "pred_count", "threshold", "precision", "recall", "f1", "retained_count", "filtered_count"])
            for fr in result.fold_results:
                for tm in fr.threshold_metrics:
                    writer.writerow([
                        fr.fold_id,
                        fr.val_size,
                        fr.pred_count,
                        tm.get("threshold"),
                        tm.get("precision"),
                        tm.get("recall"),
                        tm.get("f1"),
                        tm.get("retained_count"),
                        tm.get("filtered_count"),
                    ])
        written["per_fold_csv"] = fold_path

    md_path = output_dir / f"eval_report_{rid}.md"
    md_path.write_text(_build_markdown_report(result), encoding="utf-8")
    written["markdown"] = md_path

    return written


def _build_markdown_report(result: OfflineEvalResult) -> str:
    """Build human-readable markdown report."""
    lines = [
        "# Model Filter Offline Evaluation Report",
        "",
        f"**Run ID:** {result.run_id}",
        f"**Dataset:** {result.dataset_path}",
        f"**Model:** {result.model_path}",
        f"**Total rows:** {result.total_rows}",
        f"**Folds:** {len(result.fold_results)}",
        "",
        "## CV Configuration",
        f"- n_splits: {result.cv_config.get('n_splits')}",
        f"- embargo_seconds: {result.cv_config.get('embargo_seconds')}",
        f"- purge_seconds: {result.cv_config.get('purge_seconds')}",
        f"- expanding: {result.cv_config.get('expanding')}",
        "",
    ]
    if result.shadow_vs_baseline:
        sb = result.shadow_vs_baseline
        lines.extend([
            "## Shadow vs Baseline",
            f"- Total candidates: {sb.get('total_candidates')}",
            f"- Baseline positive count: {sb.get('baseline_positive_count')}",
            f"- Baseline positive rate: {sb.get('baseline_positive_rate')}",
            f"- Baseline win rate: {sb.get('baseline_win_rate')}",
            "",
            "### Per-Threshold (Shadow Gating)",
            "",
        ])
        for pt in sb.get("per_threshold", []):
            lines.append(f"- **thresh={pt.get('threshold')}**")
            lines.append(f"  - retained={pt.get('retained_count')} filtered={pt.get('filtered_count')} retain_ratio={pt.get('retain_ratio')}")
            lines.append(f"  - positive_rate_retained={pt.get('positive_rate_retained')} uplift={pt.get('uplift_positive_rate')}")
            lines.append(f"  - win_rate_retained={pt.get('win_rate_retained')}")
            lines.append(f"  - false_negative_count={pt.get('false_negative_count')} false_negative_risk={pt.get('false_negative_risk')}")
            lines.append(f"  - false_positive_reduction={pt.get('false_positive_reduction')}")
            lines.append("")
    lines.extend([
        "## Aggregated Threshold Metrics",
        "",
        "| threshold | precision | recall | f1 | retained | filtered | retain_ratio |",
        "|-----------|-----------|--------|-----|----------|----------|--------------|",
    ])
    for m in result.aggregated_threshold_metrics:
        lines.append(
            f"| {m.get('threshold')} | {m.get('precision')} | {m.get('recall')} | {m.get('f1')} | "
            f"{m.get('retained_count')} | {m.get('filtered_count')} | {m.get('retain_ratio')} |"
        )
    if result.retention_recommendations:
        lines.extend([
            "",
            "## Retention-Based Threshold Recommendations",
            "",
            "| profile | target_retain | threshold | retained | filtered | retain_ratio | positive_rate | uplift | fn_risk | precision | recall | f1 |",
            "|---------|---------------|-----------|----------|----------|--------------|---------------|-------|---------|-----------|--------|-----|",
        ])
        for profile, rec in result.retention_recommendations.items():
            lines.append(
                f"| {rec.get('profile')} | {rec.get('target_retain_pct')} | {rec.get('recommended_threshold')} | "
                f"{rec.get('retained_count')} | {rec.get('filtered_count')} | {rec.get('retain_ratio')} | "
                f"{rec.get('positive_rate_retained')} | {rec.get('uplift_vs_baseline')} | "
                f"{rec.get('false_negative_risk')} | {rec.get('precision')} | {rec.get('recall')} | {rec.get('f1')} |"
            )
        lines.append("")

    lines.extend([
        "",
        "## Promotion Recommendation",
        "",
    ])
    sb = result.shadow_vs_baseline or {}
    recs = result.retention_recommendations or {}
    cons = recs.get("conservative", {})
    bal = recs.get("balanced", {})
    agg = recs.get("aggressive", {})
    lines.append("- **Suggested threshold (shadow, ~75% retain):** " + str(agg.get("recommended_threshold", "N/A")))
    lines.append("- **Suggested threshold (active-demo, ~50% retain):** " + str(bal.get("recommended_threshold", "N/A")))
    lines.append("- **Suggested threshold (conservative, ~25% retain):** " + str(cons.get("recommended_threshold", "N/A")))
    lines.extend([
        "",
        "This offline evaluation supports the decision to promote from SHADOW to ACTIVE gating:",
        "- Review threshold metrics: precision/recall/F1 at each threshold.",
        "- Check shadow-vs-baseline: does gating improve positive rate and reduce false positives?",
        "- Consider false negative risk: how many good trades would be filtered at each threshold?",
        "- Compare runtime probability distribution (session summary) to offline predictions.",
        "- **Verdict:** remain_shadow | active_demo_ready | not_predictive_enough (derive from runtime + offline).",
        "",
    ])
    return "\n".join(lines)
