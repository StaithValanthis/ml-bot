"""Typed triple-barrier label calculation scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from trading.research.labeling.events import LabelEvent, LabelWindow


@dataclass(frozen=True, slots=True)
class TripleBarrierConfig:
    """Configuration for triple-barrier labeling."""

    upper_pct: Decimal
    lower_pct: Decimal
    horizon_bars: int
    upper_label: int = 1
    lower_label: int = -1
    neutral_label: int = 0


@dataclass(frozen=True, slots=True)
class TripleBarrierLabels:
    """Result of triple-barrier label calculation."""

    events: tuple[LabelEvent, ...]
    config: TripleBarrierConfig


def compute_triple_barrier(
    window: LabelWindow,
    high_prices: tuple[Decimal, ...],
    low_prices: tuple[Decimal, ...],
    config: TripleBarrierConfig,
) -> TripleBarrierLabels:
    """
    Compute triple-barrier labels for a window.

    Scaffold: returns empty events until price series and barrier logic are implemented.
    """
    # Honest scaffold: no price series available, no barrier logic implemented.
    return TripleBarrierLabels(events=(), config=config)
