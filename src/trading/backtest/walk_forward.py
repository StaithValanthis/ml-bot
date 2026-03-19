"""Walk-forward scaffold for segmented train/test or evaluation windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class WalkForwardSegment:
    segment_id: str
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    train_days: int
    test_days: int
    step_days: int


def generate_segments(
    config: WalkForwardConfig,
    data_start: datetime,
    data_end: datetime,
) -> list[WalkForwardSegment]:
    """
    Generate train/test window segments.

    Scaffold: produces non-overlapping segments. Does not validate
    that sufficient data exists for each window.
    """
    from datetime import timedelta

    segments: list[WalkForwardSegment] = []
    train_delta = timedelta(days=config.train_days)
    test_delta = timedelta(days=config.test_days)
    step_delta = timedelta(days=config.step_days)

    current = data_start
    idx = 0
    while current + train_delta + test_delta <= data_end:
        train_start = current
        train_end = train_start + train_delta
        test_start = train_end
        test_end = test_start + test_delta
        segments.append(
            WalkForwardSegment(
                segment_id=f"wf-{idx}",
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        idx += 1
        current += step_delta
    return segments
