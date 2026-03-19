"""Purged/embargo-style split scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Configuration for time-series splits with purging and embargo."""

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    purge_bars: int = 0
    embargo_bars: int = 0


@dataclass(frozen=True, slots=True)
class SplitResult:
    """Result of a split operation."""

    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    test_start: datetime
    test_end: datetime


class TimeSeriesSplitter(Protocol):
    """Protocol for purged/embargo time-series splitting."""

    def split(
        self,
        start: datetime,
        end: datetime,
        config: SplitConfig,
    ) -> SplitResult:
        """Compute train/val/test splits with purging and embargo."""
        ...
