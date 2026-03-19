"""Typed interfaces for building datasets from market/funding/features inputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class IngestInputs:
    """Input boundaries for dataset ingestion."""

    symbol: str
    start_time: datetime
    end_time: datetime
    market_data_path: Path | None = None
    funding_data_path: Path | None = None
    features_path: Path | None = None


@dataclass(slots=True)
class IngestResult:
    """Result of dataset ingestion."""

    rows: int
    path: Path | None = None
    columns: tuple[str, ...] = ()
    error: str | None = None


class DatasetBuilder(Protocol):
    """Protocol for building datasets from market/funding/features inputs."""

    def build(self, inputs: IngestInputs) -> IngestResult:
        """Build a dataset from the given inputs."""
        ...
