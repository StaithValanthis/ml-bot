"""Unit tests for offline report writing."""

from __future__ import annotations

from pathlib import Path

from trading.research.training.evaluate import (
    EvalMetrics,
    OfflineEvalResult,
    SampleCounts,
    SplitMetadata,
)
from trading.research.training.report import (
    OfflineTrainReport,
    build_offline_train_markdown,
    build_offline_train_report,
    report_to_dict,
    write_offline_train_report,
)
from trading.research.training.runner import OfflineTrainResult


def test_report_to_dict_json_serializable() -> None:
    """Report dict is JSON-serializable."""
    import json

    report = OfflineTrainReport(
        run_id="run_001",
        success=True,
        train_rows=70,
        test_rows=30,
        sample_counts={"train_n": 70, "test_n": 30},
        label_counts={"train_0": 40, "train_1": 30},
        split_metadata={"train_start": "2024-01-01", "train_end": "2024-01-15"},
        metrics={"accuracy": 0.5},
    )
    d = report_to_dict(report)
    json.dumps(d)


def test_build_offline_train_markdown_has_sections() -> None:
    """Markdown has Sample Counts, Label Balance, Baseline Comparison, Verdict."""
    report = OfflineTrainReport(
        run_id="run_001",
        success=True,
        train_rows=70,
        test_rows=30,
        sample_counts={"train_n": 70, "test_n": 30},
        label_counts={"train_0": 40, "train_1": 30},
        split_metadata={"train_start": "a", "train_end": "b", "test_start": "c", "test_end": "d"},
        metrics={"accuracy": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0},
        model_type="logistic_regression",
        verdict="model_shows_promise",
        model_beats_always_zero=True,
        model_beats_majority=False,
    )
    md = build_offline_train_markdown(report)
    assert "## Sample Counts" in md
    assert "## Label Balance" in md
    assert "## Split Metadata" in md
    assert "## Baseline Comparison" in md
    assert "run_001" in md
    assert "model_shows_promise" in md


def test_write_offline_train_report_creates_files(tmp_path: Path) -> None:
    """write_offline_train_report creates JSON and markdown in archive."""
    from trading.research.training.baseline import (
        BaselineExperimentResult,
        ComputedMetrics,
        Verdict,
    )

    eval_result = OfflineEvalResult(
        sample_counts=SampleCounts(train_n=50, test_n=20),
        label_counts={"train_0": 30, "train_1": 20},
        split_metadata=SplitMetadata("a", "b", "c", "d"),
        metrics=EvalMetrics(accuracy=0.6, precision=0.5, recall=0.5, f1=0.5),
        run_id="run_xyz",
    )
    baseline_exp = BaselineExperimentResult(
        model_type="logistic_regression",
        train_n=50,
        test_n=20,
        label_balance={"train_0": 30, "train_1": 20},
        baseline_always_zero_metrics=ComputedMetrics(0.6, 0.0, 0.0, 0.0),
        baseline_majority_metrics=ComputedMetrics(0.55, 0.5, 0.5, 0.5),
        model_metrics=ComputedMetrics(0.65, 0.6, 0.6, 0.6),
        verdict=Verdict.MODEL_SHOWS_PROMISE,
        model_beats_always_zero=True,
        model_beats_majority=True,
        model_predictions=[],
    )
    result = OfflineTrainResult(
        success=True,
        run_id="run_xyz",
        train_rows=50,
        test_rows=20,
        eval_result=eval_result,
        prepared_csv_path=None,
        baseline_experiment=baseline_exp,
    )
    json_path, md_path = write_offline_train_report(result, tmp_path)
    assert json_path.exists()
    assert md_path.exists()
    assert "offline_train_reports" in str(json_path)
    assert "offline_train_run_xyz" in json_path.name
    content = json_path.read_text(encoding="utf-8")
    assert "run_xyz" in content
    assert "sample_counts" in content
    assert "verdict" in content
    assert "model_type" in content


def test_report_includes_feature_coverage_and_label_trust(tmp_path: Path) -> None:
    """Report includes feature_coverage, label_trust, class_imbalance_note when present."""
    from trading.research.training.report import build_offline_train_report, report_to_dict

    from trading.research.training.runner import OfflineTrainResult

    result = OfflineTrainResult(
        success=True,
        run_id="r1",
        train_rows=10,
        test_rows=5,
        eval_result=None,
        prepared_csv_path=None,
        feature_coverage={"reference_price": 0.8, "confidence": 1.0},
        label_trust={"filled": "trustworthy", "profitable_fill": "scaffold_2_missing"},
        class_imbalance_note="filled=45%",
    )
    report = build_offline_train_report(result)
    d = report_to_dict(report)
    assert d.get("feature_coverage") == {"reference_price": 0.8, "confidence": 1.0}
    assert d.get("label_trust") == {"filled": "trustworthy", "profitable_fill": "scaffold_2_missing"}
    assert d.get("class_imbalance_note") == "filled=45%"
