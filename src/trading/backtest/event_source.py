"""Event source boundaries for backtest: synthetic and file-based (scaffolded)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import AsyncIterator

from trading.backtest.engine import CandleEvent
from trading.util.types import OHLCVBar


async def synthetic_candle_events(
    symbols: list[str],
    bars: int,
    timeframe: str = "5",
) -> AsyncIterator[CandleEvent]:
    """
    Generate synthetic candle events for backtest.

    Uses deterministic price progression. Suitable for smoke tests and
    scaffold runs when no historical data is available.
    """
    base = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    for i in range(bars):
        t = base + timedelta(minutes=int(timeframe) * (i + 1))
        for symbol in symbols:
            base_price = Decimal("40000") if "BTC" in symbol else Decimal("2500")
            bump = Decimal((i % 7) * 15)
            close = base_price + Decimal(i) * Decimal("5") + bump
            yield CandleEvent(
                symbol=symbol,
                bar=OHLCVBar(
                    symbol=symbol,
                    timeframe=timeframe,
                    open_time=t,
                    close_time=t,
                    open=close,
                    high=close + Decimal("5"),
                    low=close - Decimal("5"),
                    close=close,
                    volume=Decimal("10"),
                    turnover=close * Decimal("10"),
                    confirmed=True,
                ),
            )


async def candle_events_from_path(path: Path) -> AsyncIterator[CandleEvent]:
    """
    Load candle events from a file.

    Scaffold: raises NotImplementedError. Add JSON Lines or Parquet loader
    when historical data format is defined.
    """
    raise NotImplementedError(
        "File-based event loading not yet implemented. "
        "Use synthetic events (default) or implement candle_events_from_path for your data format."
    ) from None
