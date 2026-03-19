"""Unit tests for backtest engine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading.backtest.engine import BacktestEngine, BacktestResult, CandleEvent
from trading.util.types import OHLCVBar


async def _event_stream(n: int):
    base = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    for i in range(n):
        t = base + timedelta(minutes=5 * (i + 1))
        close = Decimal("40000") + Decimal(i) * Decimal("5")
        yield CandleEvent(
            symbol="BTCUSDT",
            bar=OHLCVBar(
                symbol="BTCUSDT",
                timeframe="5",
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


@pytest.mark.asyncio
async def test_backtest_engine_runs() -> None:
    engine = BacktestEngine()
    result = await engine.run(_event_stream(350))
    assert isinstance(result, BacktestResult)
    assert result.initial_equity_usdt == Decimal("10000")
    assert result.final_equity_usdt >= Decimal("0")
    assert result.start_time is not None
    assert result.end_time is not None


@pytest.mark.asyncio
async def test_backtest_engine_records_events() -> None:
    engine = BacktestEngine()
    result = await engine.run(_event_stream(350))
    assert isinstance(result.events, list)


@pytest.mark.asyncio
async def test_backtest_engine_pnl_records() -> None:
    engine = BacktestEngine()
    result = await engine.run(_event_stream(350))
    assert isinstance(result.pnl_records, list)
    assert len(result.pnl_records) > 0
