"""Typed evaluation metrics and result structures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EvalMetrics:
    """Evaluation metrics for a model."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    sharpe: float | None = None


@dataclass(slots=True)
class EvalResult:
    """Result of model evaluation."""

    metrics: EvalMetrics
    model_path: Path | None = None
    run_id: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class SampleCounts:
    """Sample counts for train/test splits."""

    train_n: int
    test_n: int


@dataclass(frozen=True, slots=True)
class SplitMetadata:
    """Train/test split metadata for reproducibility."""

    train_start: str
    train_end: str
    test_start: str
    test_end: str


@dataclass(frozen=True, slots=True)
class OfflineEvalResult:
    """
    Typed evaluation result for offline experiments.

    Includes sample counts, class balance, baseline metrics placeholders,
    and train/test split metadata. Does not fake strong ML results.
    """

    sample_counts: SampleCounts
    label_counts: dict[str, int]
    split_metadata: SplitMetadata
    metrics: EvalMetrics
    run_id: str
    error: str | None = None
