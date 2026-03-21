"""
Offline model filter evaluator: purged CV, threshold analysis, report generation.

Orchestrates: load dataset -> prepare rows -> load model -> purged splits ->
predict on each fold -> aggregate threshold metrics -> write artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from trading.models.filter_artifact import load_model_artifact
from trading.models.filter_predictor import RUNTIME_FEATURE_NAMES, predict_proba_fill
from trading.research.datasets.prepare import (
    MISSING_SENTINEL,
    LABEL_NAME,
    LABEL_PROFITABLE_FILL,
    ModelReadyRow,
    prepare_training_rows,
)
from trading.research.datasets.export import DecisionExportRecord
from trading.research.evaluation.purged_cv import PurgedCVConfig, PurgedWalkForwardSplitter
from trading.research.evaluation.threshold_analysis import (
    ThresholdMetrics,
    compute_threshold_grid,
    shadow_vs_baseline_report,
)
from trading.util.json_util import dumps_json_safe


def _load_records(path: Path) -> list[DecisionExportRecord]:
    """Load decision export records from JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows_raw = data.get("records", [])
    records: list[DecisionExportRecord] = []
    for r in rows_raw:
        ts = datetime.fromisoformat(r["ts_utc"]) if r.get("ts_utc") else datetime.min.replace(tzinfo=timezone.utc)
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
                confidence=r.get("confidence"),
            )
        )
    return records


def _predict_probs(model: object, rows: list[ModelReadyRow]) -> list[tuple[int, float, bool]]:
    """
    Predict prob_fill for each row. Returns (index, prob, available).
    Skips rows with MISSING_SENTINEL for reference_price or confidence.
    """
    results: list[tuple[int, float, bool]] = []
    for i, row in enumerate(rows):
        ref = row.features.get("reference_price", MISSING_SENTINEL)
        conf = row.features.get("confidence", MISSING_SENTINEL)
        if ref == MISSING_SENTINEL or conf == MISSING_SENTINEL:
            results.append((i, 0.0, False))
            continue
        feat = {k: row.features.get(k, MISSING_SENTINEL) for k in RUNTIME_FEATURE_NAMES}
        res = predict_proba_fill(model, feat)
        results.append((i, res.prob_fill, res.available))
    return results


@dataclass(slots=True)
class FoldResult:
    """Per-fold evaluation result."""

    fold_id: int
    val_size: int
    pred_count: int
    threshold_metrics: list[dict]
    threshold_grid: tuple[float, ...]
    predictions: list[tuple[float, int]] = field(default_factory=list)


@dataclass(slots=True)
class OfflineEvalResult:
    """Full offline evaluation result."""

    success: bool
    run_id: str
    dataset_path: str
    model_path: str
    total_rows: int
    cv_config: dict
    fold_results: list[FoldResult]
    aggregated_threshold_metrics: list[dict]
    shadow_vs_baseline: dict
    error: str | None = None


def run_offline_evaluation(
    *,
    dataset_path: Path,
    model_path: Path,
    output_dir: Path,
    threshold_grid: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7),
    cv_config: PurgedCVConfig | None = None,
    run_id: str | None = None,
) -> OfflineEvalResult:
    """
    Run purged CV evaluation with threshold analysis.

    Loads dataset, prepares rows, loads model, runs purged splits,
    predicts on each validation fold, aggregates metrics, produces reports.
    """
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if not dataset_path.exists():
        return OfflineEvalResult(
            success=False,
            run_id=run_id,
            dataset_path=str(dataset_path),
            model_path=str(model_path),
            total_rows=0,
            cv_config={},
            fold_results=[],
            aggregated_threshold_metrics=[],
            shadow_vs_baseline={},
            error=f"dataset_path not found: {dataset_path}",
        )
    load_result = load_model_artifact(model_path)
    if not load_result.loaded:
        return OfflineEvalResult(
            success=False,
            run_id=run_id,
            dataset_path=str(dataset_path),
            model_path=str(model_path),
            total_rows=0,
            cv_config={},
            fold_results=[],
            aggregated_threshold_metrics=[],
            shadow_vs_baseline={},
            error=f"model load failed: {load_result.error}",
        )
    model = load_result.model
    records = _load_records(dataset_path)
    rows = prepare_training_rows(records)
    if not rows:
        return OfflineEvalResult(
            success=False,
            run_id=run_id,
            dataset_path=str(dataset_path),
            model_path=str(model_path),
            total_rows=0,
            cv_config={},
            fold_results=[],
            aggregated_threshold_metrics=[],
            shadow_vs_baseline={},
            error="no rows prepared from dataset",
        )
    rows_sorted = sorted(rows, key=lambda r: r.ts_utc)
    timestamps = [r.ts_utc for r in rows_sorted]
    cfg = cv_config or PurgedCVConfig()
    splitter = PurgedWalkForwardSplitter(cfg)
    folds = list(splitter.split(timestamps))
    if not folds:
        return OfflineEvalResult(
            success=False,
            run_id=run_id,
            dataset_path=str(dataset_path),
            model_path=str(model_path),
            total_rows=len(rows),
            cv_config={
                "n_splits": cfg.n_splits,
                "embargo_seconds": cfg.embargo_seconds,
                "purge_seconds": cfg.purge_seconds,
                "expanding": cfg.expanding,
            },
            fold_results=[],
            aggregated_threshold_metrics=[],
            shadow_vs_baseline={},
            error="no folds produced (insufficient data or config)",
        )
    cv_config_dict = {
        "n_splits": cfg.n_splits,
        "embargo_seconds": cfg.embargo_seconds,
        "purge_seconds": cfg.purge_seconds,
        "min_train_size": cfg.min_train_size,
        "min_val_size": cfg.min_val_size,
        "expanding": cfg.expanding,
    }
    fold_results: list[FoldResult] = []
    all_probs: list[float] = []
    all_labels: list[int] = []
    all_profitable: list[int] = []

    for fold in folds:
        val_rows = [rows_sorted[i] for i in fold.val_indices]
        preds = _predict_probs(model, val_rows)
        probs = []
        labels = []
        profitable = []
        for (orig_idx, prob, avail), row in zip(preds, val_rows):
            idx = orig_idx
            if not avail:
                continue
            probs.append(prob)
            labels.append(row.label)
            if row.optional_labels and row.optional_labels.profitable_fill is not None:
                profitable.append(row.optional_labels.profitable_fill)
            else:
                profitable.append(-1)
        valid_profitable = [p for p in profitable if p >= 0]
        prof_for_report = valid_profitable if len(valid_profitable) == len(probs) else None
        if probs:
            grid = compute_threshold_grid(probs, labels, threshold_grid, prof_for_report)
            preds_tuples = [(p, l) for p, l in zip(probs, labels)]
            fold_results.append(
                FoldResult(
                    fold_id=fold.fold_id,
                    val_size=len(val_rows),
                    pred_count=len(probs),
                    threshold_metrics=[_metrics_to_dict(m) for m in grid],
                    threshold_grid=threshold_grid,
                    predictions=preds_tuples,
                )
            )
        for p, l, pr in zip(probs, labels, profitable):
            all_probs.append(p)
            all_labels.append(l)
            all_profitable.append(pr if pr >= 0 else -1)
    prof_use = all_profitable if all_profitable and all(p >= 0 for p in all_profitable) else None
    aggregated = compute_threshold_grid(all_probs, all_labels, threshold_grid, prof_use)
    shadow_report = shadow_vs_baseline_report(all_probs, all_labels, threshold_grid, prof_use)
    return OfflineEvalResult(
        success=True,
        run_id=run_id,
        dataset_path=str(dataset_path),
        model_path=str(model_path),
        total_rows=len(rows_sorted),
        cv_config=cv_config_dict,
        fold_results=fold_results,
        aggregated_threshold_metrics=[_metrics_to_dict(m) for m in aggregated],
        shadow_vs_baseline={
            "total_candidates": shadow_report.total_candidates,
            "baseline_positive_count": shadow_report.baseline_positive_count,
            "baseline_positive_rate": shadow_report.baseline_positive_rate,
            "baseline_win_rate": shadow_report.baseline_win_rate,
            "per_threshold": shadow_report.per_threshold,
        },
    )


def _metrics_to_dict(m: ThresholdMetrics) -> dict:
    d = {
        "threshold": m.threshold,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "support": m.support,
        "retained_count": m.retained_count,
        "filtered_count": m.filtered_count,
        "retain_ratio": m.retain_ratio,
        "true_positives": m.true_positives,
        "false_positives": m.false_positives,
        "false_negatives": m.false_negatives,
        "true_negatives": m.true_negatives,
    }
    if m.win_rate_retained is not None:
        d["win_rate_retained"] = m.win_rate_retained
    if m.positive_rate_retained is not None:
        d["positive_rate_retained"] = m.positive_rate_retained
    return d
