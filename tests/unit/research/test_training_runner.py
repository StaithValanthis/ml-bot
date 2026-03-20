"""Unit tests for training runner behavior in scaffold/baseline mode."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from trading.research.datasets.export import DecisionExportRecord, export_records_to_json
from trading.research.training.runner import OfflineTrainResult, run_offline_training
from trading.research.training.splits import SplitConfig


def _make_records(n: int = 20) -> list[DecisionExportRecord]:
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    records: list[DecisionExportRecord] = []
    for i in range(n):
        t = base + timedelta(minutes=i)
        records.append(
            DecisionExportRecord(
                ts_utc=t,
                symbol="BTCUSDT",
                action="entry_long",
                side="Buy",
                qty="0.001",
                reference_price=None,
                order_link_id=None,
                filled=(i % 2 == 0),
                fill_ts_utc=t if (i % 2 == 0) else None,
                fill_qty="0.001" if (i % 2 == 0) else None,
                fill_price="40000" if (i % 2 == 0) else None,
                risk_approved=True,
                risk_reason=None,
            )
        )
    return records


def test_run_offline_training_requires_records() -> None:
    """run_offline_training returns error when no records provided."""
    result = run_offline_training()
    assert not result.success
    assert "required" in (result.error or "")


def test_run_offline_training_with_records_succeeds() -> None:
    """run_offline_training succeeds with in-memory records."""
    records = _make_records(30)
    result = run_offline_training(records=records)
    assert result.success
    assert result.train_rows > 0
    assert result.test_rows >= 0
    assert result.eval_result is not None
    assert result.eval_result.sample_counts.train_n == result.train_rows
    assert result.eval_result.sample_counts.test_n == result.test_rows


def test_run_offline_training_with_json_path(tmp_path: Path) -> None:
    """run_offline_training loads from JSON path."""
    records = _make_records(25)
    json_path = tmp_path / "decisions.json"
    export_records_to_json(records, json_path)
    result = run_offline_training(records_path=json_path, output_dir=tmp_path)
    assert result.success
    assert result.prepared_csv_path is not None
    assert result.prepared_csv_path.exists()


def test_run_offline_training_empty_records_returns_error() -> None:
    """run_offline_training returns error when records produce no rows."""
    records: list[DecisionExportRecord] = []
    result = run_offline_training(records=records)
    assert not result.success
    assert "no rows" in (result.error or "")


def test_run_offline_training_respects_split_config() -> None:
    """run_offline_training uses SplitConfig for train/test split."""
    records = _make_records(100)
    cfg = SplitConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    result = run_offline_training(records=records, split_config=cfg)
    assert result.success
    total = result.train_rows + result.test_rows
    assert total <= 100
    assert result.train_rows > result.test_rows


def test_run_offline_training_single_class_no_model_artifact(tmp_path: Path) -> None:
    """When train is single-class, no model artifact is written."""
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    records: list[DecisionExportRecord] = []
    for i in range(15):
        t = base + timedelta(minutes=i)
        records.append(
            DecisionExportRecord(
                ts_utc=t,
                symbol="BTCUSDT",
                action="entry_long",
                side="Buy",
                qty="0.001",
                reference_price="40000",
                order_link_id=None,
                filled=True,
                fill_ts_utc=t,
                fill_qty="0.001",
                fill_price="40000",
                risk_approved=True,
                risk_reason=None,
            )
        )
    json_path = tmp_path / "decisions.json"
    export_records_to_json(records, json_path)
    result = run_offline_training(records_path=json_path, output_dir=tmp_path)
    assert result.success
    assert result.model_training_skipped
    assert result.model_training_skipped_reason == "train_split_single_class"
    model_files = list(tmp_path.glob("model_*.pkl"))
    assert len(model_files) == 0


def test_run_offline_main_no_export_exits_cleanly(tmp_path: Path) -> None:
    """run_offline main() exits without error when no decision export exists."""
    from unittest.mock import patch

    from trading.research.training.run_offline import main

    with patch.dict("os.environ", {"TRADING_ARCHIVE_DIR": str(tmp_path.resolve())}):
        main()
