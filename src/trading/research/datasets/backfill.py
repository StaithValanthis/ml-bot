"""Typed boundary for historical backfill jobs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Protocol


class BackfillStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class BackfillRequest:
    """Request for a historical backfill job."""

    symbol: str
    start_time: datetime
    end_time: datetime
    output_dir: Path
    include_funding: bool = True
    include_features: bool = False


@dataclass(slots=True)
class BackfillResult:
    """Result of a backfill job."""

    status: BackfillStatus
    rows_written: int = 0
    output_path: Path | None = None
    error: str | None = None


class BackfillJob(Protocol):
    """Protocol for historical backfill execution."""

    def run(self, request: BackfillRequest) -> BackfillResult:
        """Execute the backfill job."""
        ...
