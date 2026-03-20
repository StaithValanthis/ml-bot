"""Startup warmup: preload historical klines via REST for candidate/regime readiness."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from trading.exchange.bybit_rest import BybitRestClient
from trading.util.logging import get_logger
from trading.util.types import OHLCVBar


@dataclass(frozen=True, slots=True)
class WarmupResult:
    """Result of warmup preload for operator visibility."""

    symbol: str
    timeframe: str
    bars_loaded: int
    min_required: int
    satisfied: bool


def _kline_item_to_ohlcv_bar(
    item: Any,
    symbol: str,
    timeframe: str,
) -> OHLCVBar:
    """Convert REST KlineItem to OHLCVBar. Historical klines are confirmed."""
    from datetime import timedelta

    open_time = datetime.fromtimestamp(item.start_time_ms / 1000, tz=UTC)
    interval_minutes = int(timeframe) if timeframe.isdigit() else 60
    close_time = open_time + timedelta(minutes=interval_minutes)
    return OHLCVBar(
        symbol=symbol,
        timeframe=timeframe,
        open_time=open_time,
        close_time=close_time,
        open=item.open_price,
        high=item.high_price,
        low=item.low_price,
        close=item.close_price,
        volume=item.volume,
        turnover=item.turnover,
        confirmed=True,
    )


async def preload_warmup_klines(
    rest_client: BybitRestClient,
    bar_history: dict[str, dict[str, deque[OHLCVBar]]],
    *,
    symbols: list[str],
    category: str,
    candle_timeframe: str,
    regime_timeframe: str,
    min_5m_bars: int = 22,
    min_1h_bars: int = 24,
    fetch_5m_limit: int = 30,
    fetch_1h_limit: int = 30,
) -> list[WarmupResult]:
    """
    Preload historical klines via REST into bar_history for warmup.

    Fetches confirmed 5m and 1h bars for each symbol, converts to OHLCVBar,
    and appends to bar_history. Bars are sorted ascending by time.
    Returns WarmupResult per symbol/timeframe for operator visibility.
    """
    logger = get_logger("trading.runtime.warmup")
    logger.info(
        "warmup_preload_start",
        symbols=symbols,
        candle_timeframe=candle_timeframe,
        regime_timeframe=regime_timeframe,
        min_5m_bars=min_5m_bars,
        min_1h_bars=min_1h_bars,
    )
    results: list[WarmupResult] = []

    for symbol in symbols:
        for tf, min_req, limit in [
            (candle_timeframe, min_5m_bars, fetch_5m_limit),
            (regime_timeframe, min_1h_bars, fetch_1h_limit),
        ]:
            history = bar_history[symbol][tf]
            try:
                items = await rest_client.get_kline(
                    category=category,
                    symbol=symbol,
                    interval=tf,
                    limit=limit,
                )
            except Exception as exc:
                logger.warning(
                    "warmup_preload_failed",
                    symbol=symbol,
                    timeframe=tf,
                    error=str(exc),
                )
                results.append(
                    WarmupResult(
                        symbol=symbol,
                        timeframe=tf,
                        bars_loaded=0,
                        min_required=min_req,
                        satisfied=False,
                    )
                )
                continue

            sorted_items = sorted(items, key=lambda x: x.start_time_ms)
            bars: list[OHLCVBar] = []
            for item in sorted_items:
                bar = _kline_item_to_ohlcv_bar(item, symbol, tf)
                bars.append(bar)
            for bar in bars:
                history.append(bar)
            loaded = len(bars)
            satisfied = loaded >= min_req
            results.append(
                WarmupResult(
                    symbol=symbol,
                    timeframe=tf,
                    bars_loaded=loaded,
                    min_required=min_req,
                    satisfied=satisfied,
                )
            )
            logger.info(
                "warmup_preload_symbol_timeframe",
                symbol=symbol,
                timeframe=tf,
                bars_loaded=loaded,
                min_required=min_req,
                satisfied=satisfied,
            )

    all_5m = all(
        r.satisfied for r in results if r.timeframe == candle_timeframe
    )
    all_1h = all(
        r.satisfied for r in results if r.timeframe == regime_timeframe
    )
    logger.info(
        "warmup_preload_complete",
        results=[{"symbol": r.symbol, "tf": r.timeframe, "loaded": r.bars_loaded, "satisfied": r.satisfied} for r in results],
        all_5m_satisfied=all_5m,
        all_1h_satisfied=all_1h,
    )
    return results
