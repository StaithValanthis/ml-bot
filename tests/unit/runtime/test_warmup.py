"""Unit tests for warmup kline preload."""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from trading.exchange.schemas import KlineItem
from trading.runtime.warmup import preload_warmup_klines, WarmupResult
from trading.util.types import OHLCVBar


def _make_kline_item(
    start_ms: int,
    open_price: Decimal = Decimal("40000"),
    high: Decimal | None = None,
    low: Decimal | None = None,
    close: Decimal | None = None,
) -> KlineItem:
    h = high or open_price + Decimal("10")
    l = low or open_price - Decimal("10")
    c = close or open_price
    return KlineItem(
        start_time_ms=start_ms,
        open_price=open_price,
        high_price=h,
        low_price=l,
        close_price=c,
        volume=Decimal("100"),
        turnover=open_price * Decimal("100"),
    )


@pytest.mark.asyncio
async def test_preload_populates_5m_and_1h_history() -> None:
    """Preload populates bar_history with 5m and 1h bars."""
    base_ms = int(datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC).timestamp() * 1000)
    klines_5m = [
        _make_kline_item(base_ms + i * 5 * 60 * 1000)
        for i in range(25)
    ]
    klines_1h = [
        _make_kline_item(base_ms + i * 60 * 60 * 1000)
        for i in range(25)
    ]

    mock_rest = MagicMock()
    async def get_kline_side_effect(*, category: str, symbol: str, interval: str, **kwargs: object) -> list[KlineItem]:
        if interval == "5":
            return klines_5m
        return klines_1h
    mock_rest.get_kline = AsyncMock(side_effect=get_kline_side_effect)

    bar_history: dict[str, dict[str, deque[OHLCVBar]]] = {
        "BTCUSDT": {"5": deque(maxlen=800), "60": deque(maxlen=800)},
    }
    results = await preload_warmup_klines(
        mock_rest,
        bar_history,
        symbols=["BTCUSDT"],
        category="linear",
        candle_timeframe="5",
        regime_timeframe="60",
        min_5m_bars=22,
        min_1h_bars=24,
    )
    assert len(results) == 2
    assert bar_history["BTCUSDT"]["5"].__len__() == 25
    assert bar_history["BTCUSDT"]["60"].__len__() == 25
    bars_5m = list(bar_history["BTCUSDT"]["5"])
    assert all(b.confirmed for b in bars_5m)
    assert bars_5m[0].timeframe == "5"
    assert bars_5m[-1].timeframe == "5"


@pytest.mark.asyncio
async def test_preload_satisfied_when_enough_bars() -> None:
    """Warmup results show satisfied=True when enough bars loaded."""
    base_ms = int(datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC).timestamp() * 1000)
    mock_rest = MagicMock()
    mock_rest.get_kline = AsyncMock(
        return_value=[
            _make_kline_item(base_ms + i * 5 * 60 * 1000)
            for i in range(30)
        ]
    )
    bar_history: dict[str, dict[str, deque[OHLCVBar]]] = {
        "BTCUSDT": {"5": deque(maxlen=800), "60": deque(maxlen=800)},
    }
    results = await preload_warmup_klines(
        mock_rest,
        bar_history,
        symbols=["BTCUSDT"],
        category="linear",
        candle_timeframe="5",
        regime_timeframe="60",
        min_5m_bars=22,
        min_1h_bars=24,
    )
    r_5m = next(r for r in results if r.timeframe == "5")
    r_1h = next(r for r in results if r.timeframe == "60")
    assert r_5m.satisfied is True
    assert r_5m.bars_loaded >= 22
    assert r_1h.satisfied is True
    assert r_1h.bars_loaded >= 24


@pytest.mark.asyncio
async def test_preload_readiness_after_preload() -> None:
    """After preload, candidate readiness shows enough history."""
    from trading.strategy.candidates import get_candidate_readiness

    base_ms = int(datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC).timestamp() * 1000)
    mock_rest = MagicMock()
    async def get_kline_side_effect(*, interval: str, **kwargs: object) -> list[KlineItem]:
        if interval == "5":
            return [_make_kline_item(base_ms + i * 5 * 60 * 1000) for i in range(25)]
        return [_make_kline_item(base_ms + i * 60 * 60 * 1000) for i in range(25)]
    mock_rest.get_kline = AsyncMock(side_effect=get_kline_side_effect)

    bar_history: dict[str, dict[str, deque[OHLCVBar]]] = {
        "BTCUSDT": {"5": deque(maxlen=800), "60": deque(maxlen=800)},
    }
    await preload_warmup_klines(
        mock_rest,
        bar_history,
        symbols=["BTCUSDT"],
        category="linear",
        candle_timeframe="5",
        regime_timeframe="60",
    )
    bars_5m = list(bar_history["BTCUSDT"]["5"])
    bars_1h = list(bar_history["BTCUSDT"]["60"])
    readiness = get_candidate_readiness("BTCUSDT", bars_5m, bars_1h)
    assert readiness["has_enough_5m"] is True
    assert readiness["has_enough_1h"] is True
    assert readiness["reason"] == "ready"


@pytest.mark.asyncio
async def test_preload_logs_start_and_complete() -> None:
    """Preload logs warmup_preload_start and warmup_preload_complete."""
    from unittest.mock import patch

    base_ms = int(datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC).timestamp() * 1000)
    mock_rest = MagicMock()
    mock_rest.get_kline = AsyncMock(
        return_value=[_make_kline_item(base_ms + i * 5 * 60 * 1000) for i in range(25)]
    )
    bar_history: dict[str, dict[str, deque[OHLCVBar]]] = {
        "BTCUSDT": {"5": deque(maxlen=800), "60": deque(maxlen=800)},
    }
    log_calls: list[tuple[str, dict]] = []

    def capture_info(msg: str, **kwargs: object) -> None:
        log_calls.append((msg, dict(kwargs)))

    with patch("trading.runtime.warmup.get_logger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_logger.info = capture_info
        mock_logger.warning = MagicMock()
        mock_get_logger.return_value = mock_logger
        await preload_warmup_klines(
            mock_rest,
            bar_history,
            symbols=["BTCUSDT"],
            category="linear",
            candle_timeframe="5",
            regime_timeframe="60",
        )
    assert any(msg == "warmup_preload_start" for msg, _ in log_calls)
    assert any(msg == "warmup_preload_complete" for msg, _ in log_calls)
    complete_kw = next(kw for msg, kw in log_calls if msg == "warmup_preload_complete")
    assert "all_5m_satisfied" in complete_kw
    assert "all_1h_satisfied" in complete_kw


@pytest.mark.asyncio
async def test_preload_handles_failure_gracefully() -> None:
    """Preload returns WarmupResult with satisfied=False when REST fails."""
    mock_rest = MagicMock()
    mock_rest.get_kline = AsyncMock(side_effect=Exception("network error"))
    bar_history: dict[str, dict[str, deque[OHLCVBar]]] = {
        "BTCUSDT": {"5": deque(maxlen=800), "60": deque(maxlen=800)},
    }
    results = await preload_warmup_klines(
        mock_rest,
        bar_history,
        symbols=["BTCUSDT"],
        category="linear",
        candle_timeframe="5",
        regime_timeframe="60",
    )
    assert len(results) == 2
    assert all(r.bars_loaded == 0 for r in results)
    assert all(r.satisfied is False for r in results)
