"""Compare two offline experiment reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trading.util.json_util import dumps_json_safe


@dataclass(slots=True)
class ExperimentComparison:
    """Result of comparing two offline experiment reports."""

    run_id_a: str
    run_id_b: str
    metric_deltas: dict[str, float]
    verdict_a: str
    verdict_b: str
    verdict_changed: bool
    f1_delta: float
    accuracy_delta: float


def load_report_dict(path: Path) -> dict[str, object]:
    """Load report JSON as dict."""
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def compare_reports(
    report_a: dict[str, object],
    report_b: dict[str, object],
    run_id_a: str | None = None,
    run_id_b: str | None = None,
) -> ExperimentComparison:
    """
    Compare two offline experiment reports.

    Surfaces metric deltas (b - a) and verdict changes.
    Archive/report-side only.
    """
    rid_a = run_id_a or report_a.get("run_id", "a")
    rid_b = run_id_b or report_b.get("run_id", "b")
    metrics_a = report_a.get("metrics") or report_a.get("model_metrics") or {}
    metrics_b = report_b.get("metrics") or report_b.get("model_metrics") or {}
    if isinstance(metrics_a, dict):
        ma = metrics_a
    else:
        ma = {}
    if isinstance(metrics_b, dict):
        mb = metrics_b
    else:
        mb = {}
    metric_deltas: dict[str, float] = {}
    for key in ("accuracy", "precision", "recall", "f1"):
        va = float(ma.get(key, 0))
        vb = float(mb.get(key, 0))
        metric_deltas[key] = vb - va
    verdict_a = str(report_a.get("verdict", ""))
    verdict_b = str(report_b.get("verdict", ""))
    verdict_changed = verdict_a != verdict_b
    return ExperimentComparison(
        run_id_a=str(rid_a),
        run_id_b=str(rid_b),
        metric_deltas=metric_deltas,
        verdict_a=verdict_a,
        verdict_b=verdict_b,
        verdict_changed=verdict_changed,
        f1_delta=metric_deltas.get("f1", 0.0),
        accuracy_delta=metric_deltas.get("accuracy", 0.0),
    )


def write_comparison_report(
    comparison: ExperimentComparison,
    output_path: Path,
) -> None:
    """Write comparison to JSON for archive."""
    d = {
        "run_id_a": comparison.run_id_a,
        "run_id_b": comparison.run_id_b,
        "metric_deltas": comparison.metric_deltas,
        "verdict_a": comparison.verdict_a,
        "verdict_b": comparison.verdict_b,
        "verdict_changed": comparison.verdict_changed,
        "f1_delta": comparison.f1_delta,
        "accuracy_delta": comparison.accuracy_delta,
    }
    output_path.write_text(dumps_json_safe(d, indent=2), encoding="utf-8")
