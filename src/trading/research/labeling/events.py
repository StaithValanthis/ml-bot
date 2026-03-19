"""Event definitions for candidate labeling windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class LabelWindow:
    """Time window for label assignment."""

    symbol: str
    start_time: datetime
    end_time: datetime
    reference_price: Decimal


@dataclass(frozen=True, slots=True)
class LabelEvent:
    """Event with assigned label for training."""

    window: LabelWindow
    label: int
    upper_hit: bool
    lower_hit: bool
    horizon_bars: int
