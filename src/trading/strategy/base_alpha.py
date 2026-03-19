from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any


class CandidateType(str, Enum):
    BREAKOUT_LONG = "breakout_long"
    BREAKOUT_SHORT = "breakout_short"
    TREND_CONTINUATION_LONG = "trend_continuation_long"
    TREND_CONTINUATION_SHORT = "trend_continuation_short"


@dataclass(slots=True, frozen=True)
class AlphaCandidate:
    symbol: str
    candidate_type: CandidateType
    confidence: Decimal
    reference_price: Decimal
    stop_price: Decimal
    timeframe: str
    signal_time: datetime
    metadata: dict[str, Any]


class BaseAlpha(ABC):
    """Interface for deterministic candidate generation."""

    @abstractmethod
    def on_closed_candle(self, symbol: str, bars_5m: list[object]) -> list[AlphaCandidate]:
        """Evaluate closed 5m bars and produce trade candidates."""
