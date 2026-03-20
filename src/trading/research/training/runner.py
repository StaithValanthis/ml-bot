"""Offline training runner scaffold."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from trading.research.datasets.export import DecisionExportRecord
from trading.research.datasets.prepare import (
    ModelReadyRow,
    prepare_training_rows,
    write_training_rows_csv,
)
from trading.research.training.evaluate import (
    EvalMetrics,
    OfflineEvalResult,
    SampleCounts,
    SplitMetadata,
)
from trading.research.training.splits import DefaultTimeSeriesSplitter, SplitConfig, SplitResult


@dataclass(slots=True)
class OfflineTrainResult:
    """Result of offline training run."""

    success: bool
    run_id: str
    train_rows: int
    test_rows: int
    eval_result: OfflineEvalResult | None
    prepared_csv_path: Path | None
    error: str | None = None


def _load_records_from_json(path: Path) -> list[DecisionExportRecord]:
    """Load decision export records from JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    records_raw = data.get("records", [])
    records: list[DecisionExportRecord] = []
    for r in records_raw:
        ts = datetime.fromisoformat(r["ts_utc"]) if r.get("ts_utc") else datetime.min
        fill_ts = datetime.fromisoformat(r["fill_ts_utc"]) if r.get("fill_ts_utc") else None
        records.append(
            DecisionExportRecord(
                ts_utc=ts,
                symbol=r.get("symbol", ""),
                action=r.get("action", ""),
                side=r.get("side"),
                qty=r.get("qty", ""),
                reference_price=r.get("reference_price"),
                order_link_id=r.get("order_link_id"),
                filled=r.get("filled", False),
                fill_ts_utc=fill_ts,
                fill_qty=r.get("fill_qty"),
                fill_price=r.get("fill_price"),
                risk_approved=r.get("risk_approved", False),
                risk_reason=r.get("risk_reason"),
            )
        )
    return records


def _split_rows_by_time(
    rows: list[ModelReadyRow],
    split_result: SplitResult,
) -> tuple[list[ModelReadyRow], list[ModelReadyRow]]:
    """Split rows into train/test by time range. Uses train and test windows; val folded into test for simplicity."""
    train_rows: list[ModelReadyRow] = []
    test_rows: list[ModelReadyRow] = []
    for row in rows:
        if split_result.train_start <= row.ts_utc <= split_result.train_end:
            train_rows.append(row)
        elif split_result.test_start <= row.ts_utc <= split_result.test_end:
            test_rows.append(row)
    return (train_rows, test_rows)


def _compute_label_counts(rows: list[ModelReadyRow]) -> dict[str, int]:
    """Compute class balance / label counts."""
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.label)
        counts[key] = counts.get(key, 0) + 1
    return counts


def run_offline_training(
    records_path: Path | None = None,
    records: list[DecisionExportRecord] | None = None,
    output_dir: Path | None = None,
    split_config: SplitConfig | None = None,
    run_id: str | None = None,
) -> OfflineTrainResult:
    """
    Run offline training scaffold.

    Loads exported decision records, prepares model-ready rows, splits train/test,
    and runs minimal baseline path. Honest scaffold when no ML dependencies available.
    """
    if records_path is None and records is None:
        return OfflineTrainResult(
            success=False,
            run_id=run_id or "unknown",
            train_rows=0,
            test_rows=0,
            eval_result=None,
            prepared_csv_path=None,
            error="records_path or records required",
        )
    if records is None and records_path is not None:
        if not records_path.exists():
            return OfflineTrainResult(
                success=False,
                run_id=run_id or "unknown",
                train_rows=0,
                test_rows=0,
                eval_result=None,
                prepared_csv_path=None,
                error=f"records_path not found: {records_path}",
            )
        records = _load_records_from_json(records_path)
    assert records is not None

    rows = prepare_training_rows(records)
    if not rows:
        return OfflineTrainResult(
            success=False,
            run_id=run_id or "unknown",
            train_rows=0,
            test_rows=0,
            eval_result=None,
            prepared_csv_path=None,
            error="no rows prepared",
        )

    cfg = split_config or SplitConfig()
    splitter = DefaultTimeSeriesSplitter()
    ts_min = min(r.ts_utc for r in rows)
    ts_max = max(r.ts_utc for r in rows)
    split_result = splitter.split(ts_min, ts_max, cfg)
    train_rows, test_rows = _split_rows_by_time(rows, split_result)

    from datetime import timezone

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prepared_csv_path: Path | None = None
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prepared_csv_path = output_dir / f"prepared_{run_id}.csv"
        write_training_rows_csv(rows, prepared_csv_path)

    train_labels = _compute_label_counts(train_rows)
    test_labels = _compute_label_counts(test_rows)
    label_counts: dict[str, int] = {f"train_{k}": v for k, v in train_labels.items()}
    label_counts.update({f"test_{k}": v for k, v in test_labels.items()})
    split_meta = SplitMetadata(
        train_start=split_result.train_start.isoformat(),
        train_end=split_result.train_end.isoformat(),
        test_start=split_result.test_start.isoformat(),
        test_end=split_result.test_end.isoformat(),
    )
    metrics = EvalMetrics()
    eval_result = OfflineEvalResult(
        sample_counts=SampleCounts(train_n=len(train_rows), test_n=len(test_rows)),
        label_counts=label_counts,
        split_metadata=split_meta,
        metrics=metrics,
        run_id=run_id,
    )
    return OfflineTrainResult(
        success=True,
        run_id=run_id,
        train_rows=len(train_rows),
        test_rows=len(test_rows),
        eval_result=eval_result,
        prepared_csv_path=prepared_csv_path,
    )
