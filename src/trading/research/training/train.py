"""Typed training job/request/result structures."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class TrainRequest:
    """Request for a training job."""

    dataset_path: Path
    symbol: str
    train_start: datetime
    train_end: datetime
    model_output_dir: Path
    hyperparams: dict[str, object] | None = None


@dataclass(slots=True)
class TrainResult:
    """Result of a training job."""

    success: bool
    model_path: Path | None = None
    run_id: str | None = None
    metrics: dict[str, float] | None = None
    error: str | None = None


class TrainingJob(Protocol):
    """Protocol for training job execution."""

    def run(self, request: TrainRequest) -> TrainResult:
        """Execute the training job."""
        ...
