"""Unit tests for baseline model, metrics, comparison, verdict."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from trading.research.datasets.prepare import ModelReadyRow
from trading.research.training.baseline import (
    BaselineExperimentResult,
    ComputedMetrics,
    Verdict,
    metrics_to_dict,
    run_baseline_experiment,
    write_test_predictions,
)


def _make_rows(n: int = 30, fill_ratio: float = 0.5) -> list[ModelReadyRow]:
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    rows: list[ModelReadyRow] = []
    for i in range(n):
        t = base + timedelta(minutes=i)
        label = 1 if (i / n) < fill_ratio else 0
        rows.append(
            ModelReadyRow(
                ts_utc=t,
                symbol="BTCUSDT",
                features={
                    "ts_ordinal": t.timestamp(),
                    "symbol_hash": 1.0,
                    "action_encoded": 1.0,
                    "qty": 0.001,
                    "risk_approved": 1.0,
                },
                label=label,
            )
        )
    return rows


def test_run_baseline_experiment_returns_result() -> None:
    """run_baseline_experiment returns BaselineExperimentResult with metrics."""
    train = _make_rows(50, 0.5)
    test = _make_rows(20, 0.4)
    result = run_baseline_experiment(train, test)
    assert isinstance(result, BaselineExperimentResult)
    assert result.model_type == "logistic_regression"
    assert result.train_n == 50
    assert result.test_n == 20
    assert 0 <= result.model_metrics.accuracy <= 1
    assert 0 <= result.model_metrics.f1 <= 1


def test_metrics_to_dict_serializable() -> None:
    """metrics_to_dict produces JSON-serializable dict."""
    import json

    m = ComputedMetrics(accuracy=0.8, precision=0.7, recall=0.9, f1=0.78, confusion_tn=10, confusion_fp=2, confusion_fn=1, confusion_tp=9)
    d = metrics_to_dict(m)
    json.dumps(d)
    assert d["accuracy"] == 0.8
    assert d["confusion_tp"] == 9


def test_baseline_comparison_model_vs_trivial() -> None:
    """Result includes model_beats_always_zero and model_beats_majority."""
    train = _make_rows(100, 0.6)
    test = _make_rows(30, 0.5)
    result = run_baseline_experiment(train, test)
    assert hasattr(result, "model_beats_always_zero")
    assert hasattr(result, "model_beats_majority")
    assert result.verdict in (Verdict.BASELINE_ONLY, Verdict.MODEL_NOT_BETTER, Verdict.MODEL_SHOWS_PROMISE)


def test_verdict_generation() -> None:
    """Verdict is one of baseline_only, model_not_better, model_shows_promise, model_training_skipped."""
    train = _make_rows(50, 0.5)
    test = _make_rows(15, 0.5)
    result = run_baseline_experiment(train, test)
    assert result.verdict.value in (
        "baseline_only",
        "model_trained_but_not_better",
        "model_shows_promise",
        "model_training_skipped",
    )


def test_run_baseline_single_class_train_fallback() -> None:
    """Single-class train split skips model training and uses scaffold fallback."""
    train = _make_rows(20, 1.0)
    test = _make_rows(10, 0.5)
    result = run_baseline_experiment(train, test)
    assert result.model_training_skipped
    assert result.model_training_skipped_reason == "train_split_single_class"
    assert result.verdict == Verdict.MODEL_TRAINING_SKIPPED
    assert result.model is None
    assert "single_class" in result.model_type


def test_run_baseline_tiny_dataset_fallback() -> None:
    """Tiny train split skips model training."""
    train = _make_rows(1, 0.5)
    test = _make_rows(5, 0.5)
    result = run_baseline_experiment(train, test)
    assert result.model_training_skipped
    assert result.model_training_skipped_reason == "dataset_too_small"
    assert result.model is None


def test_write_test_predictions(tmp_path: Path) -> None:
    """write_test_predictions writes typed JSON artifact."""
    from pathlib import Path

    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    rows = [
        ModelReadyRow(ts_utc=base, symbol="BTC", features={}, label=1),
        ModelReadyRow(ts_utc=base, symbol="ETH", features={}, label=0),
    ]
    preds = [1, 0]
    path = tmp_path / "preds.json"
    write_test_predictions(rows, preds, path)
    assert path.exists()
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    assert "predictions" in data
    assert data["count"] == 2
    assert data["predictions"][0]["label"] == 1
    assert data["predictions"][0]["pred"] == 1
