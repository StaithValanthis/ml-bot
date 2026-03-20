"""Unit tests for evaluation result serialization."""

from __future__ import annotations

from pathlib import Path

from trading.research.training.evaluate import (
    EvalMetrics,
    OfflineEvalResult,
    SampleCounts,
    SplitMetadata,
)


def test_offline_eval_result_serializable() -> None:
    """OfflineEvalResult can be serialized to JSON."""
    import json

    result = OfflineEvalResult(
        sample_counts=SampleCounts(train_n=100, test_n=30),
        label_counts={"train_0": 60, "train_1": 40, "test_0": 18, "test_1": 12},
        split_metadata=SplitMetadata(
            train_start="2024-01-01T00:00:00",
            train_end="2024-01-15T00:00:00",
            test_start="2024-01-15T00:00:00",
            test_end="2024-01-20T00:00:00",
        ),
        metrics=EvalMetrics(accuracy=0.5, precision=0.0, recall=0.0, f1=0.0),
        run_id="run_001",
    )
    d = {
        "sample_counts": {"train_n": result.sample_counts.train_n, "test_n": result.sample_counts.test_n},
        "label_counts": result.label_counts,
        "split_metadata": {
            "train_start": result.split_metadata.train_start,
            "train_end": result.split_metadata.train_end,
            "test_start": result.split_metadata.test_start,
            "test_end": result.split_metadata.test_end,
        },
        "metrics": {
            "accuracy": result.metrics.accuracy,
            "precision": result.metrics.precision,
            "recall": result.metrics.recall,
            "f1": result.metrics.f1,
        },
        "run_id": result.run_id,
    }
    json.dumps(d)


def test_offline_eval_result_has_sample_counts_and_label_counts() -> None:
    """OfflineEvalResult includes sample counts and class balance."""
    result = OfflineEvalResult(
        sample_counts=SampleCounts(train_n=50, test_n=20),
        label_counts={"train_0": 30, "train_1": 20, "test_0": 12, "test_1": 8},
        split_metadata=SplitMetadata("", "", "", ""),
        metrics=EvalMetrics(),
        run_id="x",
    )
    assert result.sample_counts.train_n == 50
    assert result.sample_counts.test_n == 20
    assert result.label_counts["train_1"] == 20
    assert result.label_counts["test_1"] == 8
