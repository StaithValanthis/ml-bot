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
    """Markdown has Sample Counts, Label Counts, Split Metadata, Metrics."""
    report = OfflineTrainReport(
        run_id="run_001",
        success=True,
        train_rows=70,
        test_rows=30,
        sample_counts={"train_n": 70, "test_n": 30},
        label_counts={"train_0": 40, "train_1": 30},
        split_metadata={"train_start": "a", "train_end": "b", "test_start": "c", "test_end": "d"},
        metrics={"accuracy": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0},
    )
    md = build_offline_train_markdown(report)
    assert "## Sample Counts" in md
    assert "## Label Counts" in md
    assert "## Split Metadata" in md
    assert "## Metrics" in md
    assert "run_001" in md


def test_write_offline_train_report_creates_files(tmp_path: Path) -> None:
    """write_offline_train_report creates JSON and markdown in archive."""
    eval_result = OfflineEvalResult(
        sample_counts=SampleCounts(train_n=50, test_n=20),
        label_counts={"train_0": 30, "train_1": 20},
        split_metadata=SplitMetadata("a", "b", "c", "d"),
        metrics=EvalMetrics(),
        run_id="run_xyz",
    )
    result = OfflineTrainResult(
        success=True,
        run_id="run_xyz",
        train_rows=50,
        test_rows=20,
        eval_result=eval_result,
        prepared_csv_path=None,
    )
    json_path, md_path = write_offline_train_report(result, tmp_path)
    assert json_path.exists()
    assert md_path.exists()
    assert "offline_train_reports" in str(json_path)
    assert "offline_train_run_xyz" in json_path.name
    content = json_path.read_text(encoding="utf-8")
    assert "run_xyz" in content
    assert "sample_counts" in content
