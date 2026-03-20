"""Strategy order outcome tracking for session summaries."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ModelFilterOutcomes:
    """DEMO-only model filter counts for operator visibility."""

    blocked: int = 0
    allowed: int = 0
    prediction_unavailable: int = 0


@dataclass(slots=True)
class StrategyOrderOutcomes:
    """Tracks strategy (non-drill) order lifecycle counts for operator visibility."""

    intents: int = 0
    submissions: int = 0
    acks: int = 0
    filled: int = 0
    cancelled: int = 0
    rejected: int = 0
    partially_filled: int = 0
    model_filter: ModelFilterOutcomes = field(default_factory=ModelFilterOutcomes)
    _seen_partially_filled: set[str] = field(default_factory=set)
    _seen_filled: set[str] = field(default_factory=set)
    _seen_cancelled: set[str] = field(default_factory=set)
    _seen_rejected: set[str] = field(default_factory=set)
