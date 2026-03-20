"""JSON + markdown reports for offline training/evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trading.research.training.baseline import BaselineExperimentResult, metrics_to_dict
from trading.research.training.evaluate import OfflineEvalResult
from trading.research.training.runner import OfflineTrainResult
from trading.util.json_util import dumps_json_safe


@dataclass(slots=True)
class OfflineTrainReport:
    """Structured report for offline training run, easy to compare over time."""

    run_id: str
    success: bool
    train_rows: int
    test_rows: int
    sample_counts: dict[str, int]
    label_counts: dict[str, int]
    split_metadata: dict[str, str]
    metrics: dict[str, float]
    model_type: str = "scaffold"
    baseline_always_zero_metrics: dict[str, float | int] | None = None
    baseline_majority_metrics: dict[str, float | int] | None = None
    model_metrics: dict[str, float | int] | None = None
    verdict: str = "baseline_only"
    model_beats_always_zero: bool = False
    model_beats_majority: bool = False
    feature_coverage: dict[str, float] | None = None
    label_trust: dict[str, str] | None = None
    class_imbalance_note: str | None = None
    error: str | None = None
    total_rows: int = 0
    model_training_skipped: bool = False
    model_training_skipped_reason: str | None = None


def _eval_to_report(
    eval_result: OfflineEvalResult | None,
    baseline_exp: BaselineExperimentResult | None,
) -> OfflineTrainReport | None:
    if eval_result is None:
        return None
    base_always = metrics_to_dict(baseline_exp.baseline_always_zero_metrics) if baseline_exp else None
    base_majority = metrics_to_dict(baseline_exp.baseline_majority_metrics) if baseline_exp else None
    model_m = metrics_to_dict(baseline_exp.model_metrics) if baseline_exp else None
    return OfflineTrainReport(
        run_id=eval_result.run_id,
        success=True,
        train_rows=eval_result.sample_counts.train_n,
        test_rows=eval_result.sample_counts.test_n,
        sample_counts={
            "train_n": eval_result.sample_counts.train_n,
            "test_n": eval_result.sample_counts.test_n,
        },
        label_counts=eval_result.label_counts,
        split_metadata={
            "train_start": eval_result.split_metadata.train_start,
            "train_end": eval_result.split_metadata.train_end,
            "test_start": eval_result.split_metadata.test_start,
            "test_end": eval_result.split_metadata.test_end,
        },
        metrics={
            "accuracy": eval_result.metrics.accuracy,
            "precision": eval_result.metrics.precision,
            "recall": eval_result.metrics.recall,
            "f1": eval_result.metrics.f1,
        },
        model_type=baseline_exp.model_type if baseline_exp else "scaffold",
        baseline_always_zero_metrics=base_always,
        baseline_majority_metrics=base_majority,
        model_metrics=model_m,
        verdict=baseline_exp.verdict.value if baseline_exp else "baseline_only",
        model_beats_always_zero=baseline_exp.model_beats_always_zero if baseline_exp else False,
        model_beats_majority=baseline_exp.model_beats_majority if baseline_exp else False,
    )


def build_offline_train_report(result: OfflineTrainResult) -> OfflineTrainReport:
    """Build report from OfflineTrainResult."""
    baseline_exp: BaselineExperimentResult | None = None
    if hasattr(result, "baseline_experiment") and isinstance(result.baseline_experiment, BaselineExperimentResult):
        baseline_exp = result.baseline_experiment
    extra = {
        "feature_coverage": getattr(result, "feature_coverage", None),
        "label_trust": getattr(result, "label_trust", None),
        "class_imbalance_note": getattr(result, "class_imbalance_note", None),
    }
    if result.eval_result is not None:
        r = _eval_to_report(result.eval_result, baseline_exp)
        if r is not None:
            return OfflineTrainReport(
                run_id=r.run_id,
                success=result.success,
                train_rows=r.train_rows,
                test_rows=r.test_rows,
                sample_counts=r.sample_counts,
                label_counts=r.label_counts,
                split_metadata=r.split_metadata,
                metrics=r.metrics,
                model_type=r.model_type,
                baseline_always_zero_metrics=r.baseline_always_zero_metrics,
                baseline_majority_metrics=r.baseline_majority_metrics,
                model_metrics=r.model_metrics,
                verdict=r.verdict,
                model_beats_always_zero=r.model_beats_always_zero,
                model_beats_majority=r.model_beats_majority,
                feature_coverage=extra["feature_coverage"],
                label_trust=extra["label_trust"],
                class_imbalance_note=extra["class_imbalance_note"],
                error=result.error,
                total_rows=getattr(result, "total_rows", 0),
                model_training_skipped=getattr(result, "model_training_skipped", False),
                model_training_skipped_reason=getattr(result, "model_training_skipped_reason", None),
            )
    return OfflineTrainReport(
        run_id=result.run_id,
        success=result.success,
        train_rows=result.train_rows,
        test_rows=result.test_rows,
        sample_counts={"train_n": result.train_rows, "test_n": result.test_rows},
        label_counts={},
        split_metadata={},
        metrics={},
        feature_coverage=extra["feature_coverage"],
        label_trust=extra["label_trust"],
        class_imbalance_note=extra["class_imbalance_note"],
        error=result.error,
        total_rows=getattr(result, "total_rows", 0),
        model_training_skipped=getattr(result, "model_training_skipped", False),
        model_training_skipped_reason=getattr(result, "model_training_skipped_reason", None),
    )


def report_to_dict(report: OfflineTrainReport) -> dict[str, object]:
    """Convert to JSON-serializable dict."""
    d: dict[str, object] = {
        "run_id": report.run_id,
        "success": report.success,
        "train_rows": report.train_rows,
        "test_rows": report.test_rows,
        "total_rows": report.total_rows,
        "sample_counts": report.sample_counts,
        "label_counts": report.label_counts,
        "split_metadata": report.split_metadata,
        "metrics": report.metrics,
        "model_type": report.model_type,
        "verdict": report.verdict,
        "model_beats_always_zero": report.model_beats_always_zero,
        "model_beats_majority": report.model_beats_majority,
        "model_training_skipped": report.model_training_skipped,
        "model_training_skipped_reason": report.model_training_skipped_reason,
        "error": report.error,
    }
    if report.baseline_always_zero_metrics:
        d["baseline_always_zero_metrics"] = report.baseline_always_zero_metrics
    if report.baseline_majority_metrics:
        d["baseline_majority_metrics"] = report.baseline_majority_metrics
    if report.model_metrics:
        d["model_metrics"] = report.model_metrics
    if report.feature_coverage:
        d["feature_coverage"] = report.feature_coverage
    if report.label_trust:
        d["label_trust"] = report.label_trust
    if report.class_imbalance_note:
        d["class_imbalance_note"] = report.class_imbalance_note
    return d


def build_offline_train_markdown(report: OfflineTrainReport) -> str:
    """Build markdown summary for offline training run."""
    lines = [
        "# Offline Training Report",
        "",
        f"**Run ID:** {report.run_id}",
        f"**Success:** {report.success}",
        f"**Model Type:** {report.model_type}",
        f"**Verdict:** {report.verdict}",
        "",
        "## Sample Counts",
        f"- Total rows: {report.total_rows}",
        f"- Train: {report.train_rows}",
        f"- Test: {report.test_rows}",
        "",
    ]
    if report.model_training_skipped:
        lines.extend([
            "## Model Training",
            f"- **Model training skipped:** yes",
            f"- **Reason:** {report.model_training_skipped_reason or 'unknown'}",
            "",
        ])
        if report.model_training_skipped_reason == "train_split_single_class":
            lines.extend([
                "## Recommendations",
                "- Generate more decision exports (run backtest with more bars or varied scenarios)",
                "- Increase scenario variety (e.g., different symbols, regimes)",
                "- Improve outcome diversity (ensure both filled and not-filled outcomes in exports)",
                "",
            ])
    lines.extend([
        "## Label Balance",
    ])
    for k, v in sorted(report.label_counts.items()):
        lines.append(f"- {k}: {v}")
    lines.extend([
        "",
        "## Split Metadata",
        f"- Train: {report.split_metadata.get('train_start', '')} to {report.split_metadata.get('train_end', '')}",
        f"- Test: {report.split_metadata.get('test_start', '')} to {report.split_metadata.get('test_end', '')}",
        "",
        "## Baseline Metrics (Always Zero)",
    ])
    if report.baseline_always_zero_metrics:
        m = report.baseline_always_zero_metrics
        lines.append(f"- Accuracy: {float(m.get('accuracy', 0)):.4f}")
        lines.append(f"- F1: {float(m.get('f1', 0)):.4f}")
    lines.extend([
        "",
        "## Baseline Metrics (Majority Class)",
    ])
    if report.baseline_majority_metrics:
        m = report.baseline_majority_metrics
        lines.append(f"- Accuracy: {float(m.get('accuracy', 0)):.4f}")
        lines.append(f"- F1: {float(m.get('f1', 0)):.4f}")
    lines.extend([
        "",
        "## Model Metrics",
        f"- Accuracy: {report.metrics.get('accuracy', 0):.4f}",
        f"- Precision: {report.metrics.get('precision', 0):.4f}",
        f"- Recall: {report.metrics.get('recall', 0):.4f}",
        f"- F1: {report.metrics.get('f1', 0):.4f}",
        "",
        "## Baseline Comparison",
        f"- Model beats always-zero: {report.model_beats_always_zero}",
        f"- Model beats majority: {report.model_beats_majority}",
        "",
    ])
    if report.class_imbalance_note:
        lines.extend(["## Class Imbalance", f"- {report.class_imbalance_note}", ""])
    if report.feature_coverage:
        lines.append("## Feature Coverage (non-missing fraction)")
        for k, v in sorted(report.feature_coverage.items()):
            lines.append(f"- {k}: {v:.2f}")
        lines.append("")
    if report.label_trust:
        lines.append("## Label Trust")
        for k, v in sorted(report.label_trust.items()):
            lines.append(f"- {k}: {v}")
        lines.append("")
    if report.error:
        lines.append(f"**Error:** {report.error}\n")
    return "\n".join(lines)


def write_offline_train_report(
    result: OfflineTrainResult,
    root_dir: Path | str,
) -> tuple[Path, Path]:
    """Write JSON and markdown reports to archive. Returns (json_path, md_path)."""
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    report_dir = root / "offline_train_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    run_id = result.run_id
    json_path = report_dir / f"offline_train_{run_id}.json"
    md_path = report_dir / f"offline_train_{run_id}.md"
    report = build_offline_train_report(result)
    json_path.write_text(dumps_json_safe(report_to_dict(report), indent=2), encoding="utf-8")
    md_path.write_text(build_offline_train_markdown(report), encoding="utf-8")
    return (json_path, md_path)
