"""Baseline model training, prediction, and comparison."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from trading.research.datasets.prepare import FEATURE_NAMES, ModelReadyRow


MIN_TRAIN_ROWS_FOR_MODEL = 2
MIN_CLASSES_FOR_MODEL = 2


class Verdict(str, Enum):
    """Simple verdict on whether model adds value over trivial baseline."""

    BASELINE_ONLY = "baseline_only"
    MODEL_NOT_BETTER = "model_trained_but_not_better"
    MODEL_SHOWS_PROMISE = "model_shows_promise"
    MODEL_TRAINING_SKIPPED = "model_training_skipped"


@dataclass(frozen=True, slots=True)
class ComputedMetrics:
    """Explicit metrics for comparison across runs."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_tn: int = 0
    confusion_fp: int = 0
    confusion_fn: int = 0
    confusion_tp: int = 0


@dataclass(slots=True)
class BaselineExperimentResult:
    """Result of baseline experiment: model + trivial baselines."""

    model_type: str
    train_n: int
    test_n: int
    label_balance: dict[str, int]
    baseline_always_zero_metrics: ComputedMetrics
    baseline_majority_metrics: ComputedMetrics
    model_metrics: ComputedMetrics
    verdict: Verdict
    model_beats_always_zero: bool
    model_beats_majority: bool
    model_predictions: list[int] = ()
    model: object | None = None
    model_training_skipped: bool = False
    model_training_skipped_reason: str | None = None


def _rows_to_arrays(
    rows: list[ModelReadyRow],
) -> tuple[list[list[float]], list[int]]:
    """Convert rows to X (features) and y (labels) arrays."""
    X: list[list[float]] = []
    y: list[int] = []
    for row in rows:
        X.append([row.features.get(k, 0.0) for k in FEATURE_NAMES])
        y.append(row.label)
    return (X, y)


def _compute_metrics(y_true: list[int], y_pred: list[int]) -> ComputedMetrics:
    """Compute accuracy, precision, recall, f1, and confusion matrix counts."""
    n = len(y_true)
    if n == 0:
        return ComputedMetrics(0.0, 0.0, 0.0, 0.0)
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tn += 1
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return ComputedMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_tn=tn,
        confusion_fp=fp,
        confusion_fn=fn,
        confusion_tp=tp,
    )


def _predict_always_zero(n: int) -> list[int]:
    """Trivial baseline: always predict no fill."""
    return [0] * n


def _predict_majority(y_train: list[int], n_test: int) -> list[int]:
    """Majority class baseline: predict the most common train label."""
    ones = sum(1 for y in y_train if y == 1)
    pred = 1 if ones >= len(y_train) / 2 else 0
    return [pred] * n_test


def _should_skip_model_training(
    train_rows: list[ModelReadyRow],
    y_train: list[int],
) -> tuple[bool, str | None]:
    """Return (skip, reason) when model training should be skipped."""
    if len(train_rows) < MIN_TRAIN_ROWS_FOR_MODEL:
        return (True, "dataset_too_small")
    n_classes = len(set(y_train))
    if n_classes < MIN_CLASSES_FOR_MODEL:
        return (True, "train_split_single_class")
    return (False, None)


def run_baseline_experiment(
    train_rows: list[ModelReadyRow],
    test_rows: list[ModelReadyRow],
    model_type: str = "logistic_regression",
) -> BaselineExperimentResult:
    """
    Train model, run trivial baselines, compute metrics, compare.

    Uses logistic regression when sklearn available; otherwise scaffold path.
    Skips model training when train is single-class or too small; falls back to
    majority-class predictor. Does not crash on unsuitable datasets.
    """
    X_train, y_train = _rows_to_arrays(train_rows)
    X_test, y_test = _rows_to_arrays(test_rows)
    n_test = len(y_test)

    label_balance: dict[str, int] = {}
    for y in y_train:
        k = str(y)
        label_balance[f"train_{k}"] = label_balance.get(f"train_{k}", 0) + 1
    for y in y_test:
        k = str(y)
        label_balance[f"test_{k}"] = label_balance.get(f"test_{k}", 0) + 1

    baseline_always_zero = _compute_metrics(y_test, _predict_always_zero(n_test))
    baseline_majority = _compute_metrics(y_test, _predict_majority(y_train, n_test))

    model_pred: list[int]
    resolved_model_type = model_type
    trained_model: object | None = None
    model_training_skipped = False
    model_training_skipped_reason: str | None = None

    skip, skip_reason = _should_skip_model_training(train_rows, y_train)
    if skip:
        model_pred = _predict_majority(y_train, n_test)
        resolved_model_type = f"scaffold_{skip_reason}"
        model_training_skipped = True
        model_training_skipped_reason = skip_reason
    else:
        try:
            from sklearn.linear_model import LogisticRegression

            clf = LogisticRegression(max_iter=500, random_state=42)
            clf.fit(X_train, y_train)
            model_pred = [int(p) for p in clf.predict(X_test)]
            trained_model = clf
        except ImportError:
            model_pred = _predict_majority(y_train, n_test)
            resolved_model_type = "scaffold_majority_fallback"
        except ValueError as exc:
            model_pred = _predict_majority(y_train, n_test)
            resolved_model_type = "scaffold_fit_failed"
            model_training_skipped = True
            model_training_skipped_reason = str(exc)[:200]

    model_metrics = _compute_metrics(y_test, model_pred)

    model_beats_always_zero = model_metrics.f1 > baseline_always_zero.f1
    model_beats_majority = model_metrics.f1 > baseline_majority.f1

    if model_training_skipped:
        verdict = Verdict.MODEL_TRAINING_SKIPPED
    elif resolved_model_type == "scaffold_majority_fallback":
        verdict = Verdict.BASELINE_ONLY
    elif not model_beats_always_zero and not model_beats_majority:
        verdict = Verdict.MODEL_NOT_BETTER
    else:
        verdict = Verdict.MODEL_SHOWS_PROMISE

    return BaselineExperimentResult(
        model_type=resolved_model_type,
        train_n=len(train_rows),
        test_n=len(test_rows),
        label_balance=label_balance,
        baseline_always_zero_metrics=baseline_always_zero,
        baseline_majority_metrics=baseline_majority,
        model_metrics=model_metrics,
        verdict=verdict,
        model_beats_always_zero=model_beats_always_zero,
        model_beats_majority=model_beats_majority,
        model_predictions=model_pred,
        model=trained_model,
        model_training_skipped=model_training_skipped,
        model_training_skipped_reason=model_training_skipped_reason,
    )


def metrics_to_dict(m: ComputedMetrics) -> dict[str, float | int]:
    """Serialize metrics for JSON."""
    return {
        "accuracy": m.accuracy,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "confusion_tn": m.confusion_tn,
        "confusion_fp": m.confusion_fp,
        "confusion_fn": m.confusion_fn,
        "confusion_tp": m.confusion_tp,
    }


def save_baseline_model(model: object | None, path: Path) -> bool:
    """
    Save promoted baseline model artifact for runtime use.

    Returns True if saved, False if model is None or save fails.
    Expects sklearn-like classifier with predict/predict_proba.
    """
    if model is None:
        return False
    try:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        return True
    except ImportError:
        return False
    except Exception:
        return False


def write_test_predictions(
    rows: list[ModelReadyRow],
    predictions: list[int],
    path: Path,
) -> None:
    """Write test set predictions to archive. Simple typed format."""
    import json

    from trading.util.json_util import dumps_json_safe

    records = [
        {
            "ts_utc": row.ts_utc.isoformat(),
            "symbol": row.symbol,
            "label": row.label,
            "pred": pred,
        }
        for row, pred in zip(rows, predictions)
    ]
    path.write_text(dumps_json_safe({"predictions": records, "count": len(records)}, indent=2), encoding="utf-8")
