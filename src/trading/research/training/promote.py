"""Typed model promotion decision scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from trading.research.training.evaluate import EvalMetrics, EvalResult


class PromoteDecision(str, Enum):
    PROMOTE = "promote"
    REJECT = "reject"
    DEFER = "defer"


@dataclass(frozen=True, slots=True)
class PromoteRequest:
    """Request for model promotion decision."""

    candidate_path: Path
    baseline_metrics: EvalMetrics | None = None
    candidate_metrics: EvalResult | None = None
    min_improvement_pct: float = 0.0


def decide_promotion(request: PromoteRequest) -> PromoteDecision:
    """
    Decide whether to promote a candidate model.

    Scaffold: returns DEFER until promotion criteria are implemented.
    """
    if request.candidate_metrics is None or request.candidate_metrics.error:
        return PromoteDecision.REJECT
    return PromoteDecision.DEFER
