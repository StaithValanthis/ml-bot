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
