"""Unit tests for experiment comparison."""

from __future__ import annotations

from pathlib import Path

import pytest

from trading.research.training.compare import (
    ExperimentComparison,
    compare_reports,
    load_report_dict,
    write_comparison_report,
)


def test_compare_reports_metric_deltas() -> None:
    """compare_reports surfaces metric deltas and verdict changes."""
    report_a = {
        "run_id": "run_a",
        "verdict": "model_trained_but_not_better",
        "metrics": {"accuracy": 0.5, "precision": 0.4, "recall": 0.5, "f1": 0.45},
    }
    report_b = {
        "run_id": "run_b",
        "verdict": "model_shows_promise",
        "metrics": {"accuracy": 0.6, "precision": 0.55, "recall": 0.6, "f1": 0.57},
    }
    comp = compare_reports(report_a, report_b)
    assert isinstance(comp, ExperimentComparison)
    assert comp.run_id_a == "run_a"
    assert comp.run_id_b == "run_b"
    assert comp.metric_deltas["accuracy"] == pytest.approx(0.1)
    assert comp.metric_deltas["f1"] == pytest.approx(0.12)
    assert comp.verdict_changed
    assert comp.verdict_a == "model_trained_but_not_better"
    assert comp.verdict_b == "model_shows_promise"


def test_compare_reports_verdict_unchanged() -> None:
    """compare_reports sets verdict_changed=False when same."""
    report_a = {"run_id": "a", "verdict": "baseline_only", "metrics": {}}
    report_b = {"run_id": "b", "verdict": "baseline_only", "metrics": {}}
    comp = compare_reports(report_a, report_b)
    assert not comp.verdict_changed


def test_write_comparison_report(tmp_path: Path) -> None:
    """write_comparison_report writes JSON artifact."""
    comp = ExperimentComparison(
        run_id_a="a",
        run_id_b="b",
        metric_deltas={"f1": 0.1},
        verdict_a="x",
        verdict_b="y",
        verdict_changed=True,
        f1_delta=0.1,
        accuracy_delta=0.05,
    )
    path = tmp_path / "compare.json"
    write_comparison_report(comp, path)
    assert path.exists()
    data = load_report_dict(path)
    assert data["run_id_a"] == "a"
    assert data["verdict_changed"] is True
    assert data["f1_delta"] == 0.1
