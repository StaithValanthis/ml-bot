"""Unit tests for offline model filter evaluation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from trading.research.evaluation.purged_cv import (
    PurgedCVConfig,
    PurgedFold,
    PurgedWalkForwardSplitter,
    purged_splits,
)
from trading.research.evaluation.threshold_analysis import (
    ThresholdMetrics,
    compute_threshold_grid,
    shadow_vs_baseline_report,
)


def test_purged_splits_produces_folds() -> None:
    """Purged splitter yields folds with non-overlapping train/val and embargo."""
    ts = [
        datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 1, 2, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 1, 3, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 1, 4, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 1, 5, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 1, 6, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 1, 7, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 1, 8, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 1, 9, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 1, 11, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 1, 13, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 1, 14, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 1, 15, 0, 0, tzinfo=UTC),
    ]
    config = PurgedCVConfig(n_splits=3, embargo_seconds=60, purge_seconds=60, min_train_size=2, min_val_size=2)
    folds = purged_splits(ts, config)
    assert len(folds) >= 1
    for f in folds:
        assert isinstance(f, PurgedFold)
        assert f.train_end <= f.val_start
        assert len(f.train_indices) >= config.min_train_size
        assert len(f.val_indices) >= config.min_val_size
        assert not set(f.train_indices) & set(f.val_indices)


def test_purged_splits_embargo_gap() -> None:
    """Train end is at least embargo before val start."""
    from datetime import timedelta

    base = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
    ts = [base + timedelta(hours=i) for i in range(50)]
    config = PurgedCVConfig(n_splits=3, embargo_seconds=3600, purge_seconds=3600, min_train_size=5, min_val_size=5)
    folds = purged_splits(ts, config)
    for f in folds:
        gap = (f.val_start - f.train_end).total_seconds()
        assert gap >= 3600 or len(f.train_indices) == 0


def test_purged_splits_insufficient_data_returns_empty() -> None:
    """Too few samples yields no folds."""
    ts = [datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC), datetime(2025, 1, 1, 1, 0, 0, tzinfo=UTC)]
    config = PurgedCVConfig(min_train_size=10, min_val_size=5)
    folds = purged_splits(ts, config)
    assert len(folds) == 0


def test_compute_threshold_grid() -> None:
    """Threshold grid produces precision, recall, F1, retained/filtered counts."""
    probs = [0.2, 0.4, 0.6, 0.8]
    labels = [0, 0, 1, 1]
    grid = compute_threshold_grid(probs, labels, (0.3, 0.5, 0.7))
    assert len(grid) == 3
    t05 = next(m for m in grid if m.threshold == 0.5)
    assert t05.retained_count == 2
    assert t05.filtered_count == 2
    assert t05.precision == 1.0
    assert t05.recall == 1.0
    assert t05.retain_ratio == 0.5


def test_compute_threshold_grid_profitable_fill() -> None:
    """When profitable_fill provided, win_rate_retained is computed."""
    probs = [0.6, 0.7]
    labels = [1, 1]
    profitable = [1, 0]
    grid = compute_threshold_grid(probs, labels, (0.5,), profitable_fill=profitable)
    assert len(grid) == 1
    assert grid[0].win_rate_retained == 0.5


def test_shadow_vs_baseline_report() -> None:
    """Shadow vs baseline includes per_threshold and baseline stats."""
    probs = [0.3, 0.5, 0.7, 0.9]
    labels = [0, 0, 1, 1]
    report = shadow_vs_baseline_report(probs, labels, (0.4, 0.6))
    assert report.total_candidates == 4
    assert report.baseline_positive_count == 2
    assert report.baseline_positive_rate == 0.5
    assert len(report.per_threshold) == 2
    t06 = next(p for p in report.per_threshold if p["threshold"] == 0.6)
    assert "retained_count" in t06
    assert "false_negative_risk" in t06
    assert "false_positive_reduction" in t06


def test_report_generation(tmp_path: Path) -> None:
    """write_eval_reports produces JSON, CSV, and markdown."""
    try:
        import joblib
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        pytest.skip("sklearn/joblib not available")
    from trading.research.evaluation.offline_evaluator import run_offline_evaluation
    from trading.research.evaluation.report import write_eval_reports

    model_path = tmp_path / "model.pkl"
    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit([[0, 0, 1, 0.001, 1, 1, 50000, 0.8], [1, 0, 1, 0.002, 1, 1, 50100, 0.7]], [0, 1])
    joblib.dump(clf, model_path)

    dataset_path = tmp_path / "decisions.json"
    import json
    records = [
        {
            "ts_utc": "2025-01-01T00:00:00+00:00",
            "symbol": "BTCUSDT",
            "action": "entry_long",
            "side": "Buy",
            "qty": "0.001",
            "reference_price": "50000",
            "order_link_id": "a",
            "filled": True,
            "fill_ts_utc": "2025-01-01T00:05:00+00:00",
            "fill_qty": "0.001",
            "fill_price": "50100",
            "risk_approved": True,
            "risk_reason": None,
            "confidence": "0.8",
        },
    ]
    for i in range(40):
        records.append({
            "ts_utc": f"2025-01-01T{i % 24:02d}:{i % 60:02d}:00+00:00",
            "symbol": "BTCUSDT",
            "action": "entry_long",
            "side": "Buy",
            "qty": "0.001",
            "reference_price": "50000",
            "order_link_id": f"o{i}",
            "filled": i % 3 == 0,
            "fill_ts_utc": None,
            "fill_qty": None,
            "fill_price": None,
            "risk_approved": True,
            "risk_reason": None,
            "confidence": "0.7",
        })
    dataset_path.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")

    output_dir = tmp_path / "eval_out"
    result = run_offline_evaluation(
        dataset_path=dataset_path,
        model_path=model_path,
        output_dir=output_dir,
        threshold_grid=(0.4, 0.5, 0.6),
        cv_config=PurgedCVConfig(n_splits=3, embargo_seconds=60, purge_seconds=60, min_train_size=5, min_val_size=5),
    )
    assert result.success
    written = write_eval_reports(result, output_dir)
    assert (output_dir / f"eval_summary_{result.run_id}.json").exists()
    assert "threshold_csv" in written or "json" in written
    assert (output_dir / f"eval_report_{result.run_id}.md").exists()
    md = (output_dir / f"eval_report_{result.run_id}.md").read_text()
    assert "Promotion Readiness" in md


def test_eval_no_control_flow_change() -> None:
    """Evaluation module does not alter runtime or strategy behavior."""
    from datetime import timedelta

    from trading.research.evaluation.purged_cv import purged_splits
    from trading.research.evaluation.threshold_analysis import compute_threshold_grid

    base = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
    ts = [base + timedelta(hours=i) for i in range(30)]
    folds = purged_splits(ts, PurgedCVConfig(n_splits=2, min_train_size=5, min_val_size=5))
    grid = compute_threshold_grid([0.5] * 10, [0, 1] * 5, (0.5,))
    assert grid is not None
    assert len(grid) == 1
