"""
Purged walk-forward cross-validation for time-series candidate data.

Prevents leakage via purging (remove train samples near val boundary) and
embargo (gap between train_end and val_start). No random k-fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterator


@dataclass(frozen=True, slots=True)
class PurgedCVConfig:
    """Configuration for purged time-series CV."""

    n_splits: int = 5
    embargo_seconds: int = 300
    purge_seconds: int = 300
    min_train_size: int = 10
    min_val_size: int = 5
    expanding: bool = True


@dataclass(frozen=True, slots=True)
class PurgedFold:
    """Single fold with train and validation index slices."""

    fold_id: int
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    train_indices: list[int]
    val_indices: list[int]


class PurgedWalkForwardSplitter:
    """
    Walk-forward splits with purging and embargo.

    For each fold:
    - Train: samples before (val_start - embargo - purge)
    - Validation: samples in [val_start, val_end]
    - Embargo: gap between last train sample and first val sample
    - Purging: train samples within purge_seconds of val_start are excluded
    """

    def __init__(self, config: PurgedCVConfig) -> None:
        self._config = config

    def split(
        self,
        timestamps: list[datetime],
    ) -> Iterator[PurgedFold]:
        """
        Yield folds. timestamps must be sorted ascending.
        Train = samples before (val_start - embargo - purge). Val = samples in [val_start, val_end].
        """
        n = len(timestamps)
        if n < self._config.min_train_size + self._config.min_val_size:
            return
        embargo_delta = timedelta(seconds=self._config.embargo_seconds)
        purge_delta = timedelta(seconds=self._config.purge_seconds)
        gap = max(embargo_delta, purge_delta)
        t_min = timestamps[0]
        t_max = timestamps[-1]
        total_span = (t_max - t_min).total_seconds()
        if total_span <= 0:
            return
        segment_duration = total_span / (self._config.n_splits + 1)
        for fold_id in range(self._config.n_splits):
            val_start_ts = t_min.timestamp() + segment_duration * (fold_id + 1)
            val_end_ts = t_min.timestamp() + segment_duration * (fold_id + 2)
            val_start = datetime.fromtimestamp(val_start_ts, tz=timezone.utc)
            val_end = datetime.fromtimestamp(val_end_ts, tz=timezone.utc)
            train_cutoff = val_start - gap
            train_indices: list[int] = []
            val_indices: list[int] = []
            for i, ts in enumerate(timestamps):
                if ts < train_cutoff:
                    train_indices.append(i)
                elif val_start <= ts < val_end:
                    val_indices.append(i)
            if not self._config.expanding and fold_id > 0:
                prev_val_start = datetime.fromtimestamp(
                    t_min.timestamp() + val_duration * (fold_id - 1),
                    tz=timezone.utc,
                )
                rolling_train_start = prev_val_start - gap
                train_indices = [i for i in train_indices if timestamps[i] >= rolling_train_start]
            if len(train_indices) >= self._config.min_train_size and len(val_indices) >= self._config.min_val_size:
                train_start = timestamps[train_indices[0]] if train_indices else val_start
                train_end = timestamps[train_indices[-1]] if train_indices else val_start
                yield PurgedFold(
                    fold_id=fold_id,
                    train_start=train_start,
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                    train_indices=train_indices,
                    val_indices=val_indices,
                )


def purged_splits(
    timestamps: list[datetime],
    config: PurgedCVConfig | None = None,
) -> list[PurgedFold]:
    """Convenience: run splitter and return list of folds."""
    cfg = config or PurgedCVConfig()
    splitter = PurgedWalkForwardSplitter(cfg)
    return list(splitter.split(timestamps))
