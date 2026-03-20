"""Unit tests for dataset preparation row shape."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from trading.research.datasets.export import DecisionExportRecord
from trading.research.datasets.prepare import (
    FEATURE_NAMES,
    LABEL_NAME,
    ModelReadyRow,
    prepare_training_rows,
    write_training_rows_csv,
)


def test_model_ready_row_has_explicit_feature_label_boundaries() -> None:
    """ModelReadyRow has features dict and label."""
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    row = ModelReadyRow(
        ts_utc=base,
        symbol="BTCUSDT",
        features={"ts_ordinal": base.timestamp(), "symbol_hash": 1.0, "action_encoded": 1.0, "qty": 0.001, "risk_approved": 1.0},
        label=1,
    )
    assert row.label in (0, 1)
    assert set(row.features.keys()) == set(FEATURE_NAMES)


def test_prepare_training_rows_produces_correct_shape() -> None:
    """prepare_training_rows produces rows with all feature names and binary label."""
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    records = [
        DecisionExportRecord(
            ts_utc=base,
            symbol="BTCUSDT",
            action="entry_long",
            side="Buy",
            qty="0.001",
            reference_price=None,
            order_link_id=None,
            filled=True,
            fill_ts_utc=base,
            fill_qty="0.001",
            fill_price="40000",
            risk_approved=True,
            risk_reason=None,
        ),
        DecisionExportRecord(
            ts_utc=base,
            symbol="ETHUSDT",
            action="entry_short",
            side="Sell",
            qty="0.01",
            reference_price=None,
            order_link_id=None,
            filled=False,
            fill_ts_utc=None,
            fill_qty=None,
            fill_price=None,
            risk_approved=True,
            risk_reason=None,
        ),
    ]
    rows = prepare_training_rows(records)
    assert len(rows) == 2
    for row in rows:
        assert set(row.features.keys()) == set(FEATURE_NAMES)
        assert row.label in (0, 1)
    assert rows[0].label == 1
    assert rows[1].label == 0


def test_write_training_rows_csv(tmp_path: Path) -> None:
    """write_training_rows_csv produces valid CSV with feature and label columns."""
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    rows = [
        ModelReadyRow(
            ts_utc=base,
            symbol="BTCUSDT",
            features={
                "ts_ordinal": base.timestamp(),
                "symbol_hash": 1.0,
                "action_encoded": 1.0,
                "qty": 0.001,
                "risk_approved": 1.0,
            },
            label=1,
        ),
    ]
    path = tmp_path / "prepared.csv"
    write_training_rows_csv(rows, path)
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert LABEL_NAME in content
    for name in FEATURE_NAMES:
        assert name in content
