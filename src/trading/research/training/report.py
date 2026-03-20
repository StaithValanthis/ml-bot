"""JSON + markdown reports for offline training/evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    error: str | None = None


def _eval_to_report(eval_result: OfflineEvalResult | None) -> OfflineTrainReport | None:
    if eval_result is None:
        return None
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
    )


def build_offline_train_report(result: OfflineTrainResult) -> OfflineTrainReport:
    """Build report from OfflineTrainResult."""
    if result.eval_result is not None:
        r = _eval_to_report(result.eval_result)
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
                error=result.error,
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
        error=result.error,
    )


def report_to_dict(report: OfflineTrainReport) -> dict[str, object]:
    """Convert to JSON-serializable dict."""
    return {
        "run_id": report.run_id,
        "success": report.success,
        "train_rows": report.train_rows,
        "test_rows": report.test_rows,
        "sample_counts": report.sample_counts,
        "label_counts": report.label_counts,
        "split_metadata": report.split_metadata,
        "metrics": report.metrics,
        "error": report.error,
    }


def build_offline_train_markdown(report: OfflineTrainReport) -> str:
    """Build markdown summary for offline training run."""
    lines = [
        "# Offline Training Report",
        "",
        f"**Run ID:** {report.run_id}",
        f"**Success:** {report.success}",
        "",
        "## Sample Counts",
        f"- Train: {report.train_rows}",
        f"- Test: {report.test_rows}",
        "",
        "## Label Counts (Class Balance)",
    ]
    for k, v in sorted(report.label_counts.items()):
        lines.append(f"- {k}: {v}")
    lines.extend([
        "",
        "## Split Metadata",
        f"- Train: {report.split_metadata.get('train_start', '')} to {report.split_metadata.get('train_end', '')}",
        f"- Test: {report.split_metadata.get('test_start', '')} to {report.split_metadata.get('test_end', '')}",
        "",
        "## Metrics",
        f"- Accuracy: {report.metrics.get('accuracy', 0):.4f}",
        f"- Precision: {report.metrics.get('precision', 0):.4f}",
        f"- Recall: {report.metrics.get('recall', 0):.4f}",
        f"- F1: {report.metrics.get('f1', 0):.4f}",
        "",
    ])
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
