"""Predictor interface for live/backtest inference."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import AsyncIterator, Protocol

from trading.models.model_bundle import ModelBundle


@dataclass(frozen=True, slots=True)
class PredictionInput:
    """Input for a single prediction."""

    symbol: str
    features: dict[str, float]


@dataclass(frozen=True, slots=True)
class PredictionOutput:
    """Output of a single prediction."""

    label: int
    confidence: Decimal
    raw_logits: tuple[float, ...] | None = None


class Predictor(Protocol):
    """
    Predictor interface for live/backtest inference.

    Usable as a plug-in point for strategy/signal layers.
    """

    def __init__(self, bundle: ModelBundle) -> None:
        ...

    def predict(self, inp: PredictionInput) -> PredictionOutput:
        """Single prediction."""
        ...

    def predict_batch(
        self,
        inputs: AsyncIterator[PredictionInput],
    ) -> AsyncIterator[PredictionOutput]:
        """Batch prediction. Scaffold: yields nothing until implemented."""
        ...
