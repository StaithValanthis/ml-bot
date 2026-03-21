"""Strategy order outcome tracking for session summaries."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ModelFilterOutcomes:
    """DEMO-only model filter counts and calibration visibility for operator diagnostics."""

    blocked: int = 0
    allowed: int = 0
    prediction_unavailable: int = 0
    shadow_would_have_blocked: int = 0
    threshold: float = 0.5
    mode: str = "hard_block"
    prob_min: float | None = None
    prob_max: float | None = None
    prob_latest: float | None = None
    prob_count: int = 0
    latest_features: dict[str, float] | None = None


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
    _seen_new: set[str] = field(default_factory=set)
    _seen_partially_filled: set[str] = field(default_factory=set)
    _seen_filled: set[str] = field(default_factory=set)
    _seen_cancelled: set[str] = field(default_factory=set)
    _seen_rejected: set[str] = field(default_factory=set)
