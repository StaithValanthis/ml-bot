from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from trading.marketdata.normalizers import NormalizedKline, NormalizedTrade
from trading.util.types import OHLCVBar


@dataclass(slots=True)
class _WorkingCandle:
    symbol: str
    timeframe_minutes: int
    start: datetime
    end: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    turnover: Decimal


class CandleBuilder:
    """
    Deterministic candle builder for trade-based fallback aggregation.

    In live mode, exchange-confirmed kline close should be preferred. This
    builder remains useful for replay, diagnostics, and fallback data paths.
    """

    def __init__(self, *, timeframe_minutes: int) -> None:
        if timeframe_minutes <= 0:
            raise ValueError("timeframe_minutes must be > 0")
        self._timeframe_minutes = timeframe_minutes
        self._working: dict[str, _WorkingCandle] = {}

    def on_trade(self, trade: NormalizedTrade) -> OHLCVBar | None:
        bucket_start = self._bucket_start(trade.ts_event_utc, self._timeframe_minutes)
        bucket_end = bucket_start + timedelta(minutes=self._timeframe_minutes)
        existing = self._working.get(trade.symbol)

        if existing is None or bucket_start > existing.start:
            closed = self._finalize(existing, confirmed=True) if existing is not None else None
            self._working[trade.symbol] = _WorkingCandle(
                symbol=trade.symbol,
                timeframe_minutes=self._timeframe_minutes,
                start=bucket_start,
                end=bucket_end,
                open=trade.price,
                high=trade.price,
                low=trade.price,
                close=trade.price,
                volume=trade.size,
                turnover=trade.price * trade.size,
            )
            return closed

        existing.high = max(existing.high, trade.price)
        existing.low = min(existing.low, trade.price)
        existing.close = trade.price
        existing.volume += trade.size
        existing.turnover += trade.price * trade.size
        return None

    def on_confirmed_kline(self, kline: NormalizedKline) -> OHLCVBar | None:
        if not kline.confirmed:
            return None
        open_time = datetime.fromtimestamp(kline.start_ms / 1000, tz=UTC)
        close_time = datetime.fromtimestamp(kline.end_ms / 1000, tz=UTC)
        return OHLCVBar(
            symbol=kline.symbol,
            timeframe=kline.interval,
            open_time=open_time,
            close_time=close_time,
            open=kline.open,
            high=kline.high,
            low=kline.low,
            close=kline.close,
            volume=kline.volume,
            turnover=kline.turnover,
            confirmed=True,
        )

    def flush_open_candle(self, symbol: str) -> OHLCVBar | None:
        """
        Flush current building candle as non-confirmed bar.

        Useful during graceful shutdown for diagnostics only.
        """
        working = self._working.pop(symbol, None)
        if working is None:
            return None
        return self._finalize(working, confirmed=False)

    def _finalize(self, candle: _WorkingCandle | None, *, confirmed: bool) -> OHLCVBar | None:
        if candle is None:
            return None
        return OHLCVBar(
            symbol=candle.symbol,
            timeframe=str(candle.timeframe_minutes),
            open_time=candle.start,
            close_time=candle.end,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
            turnover=candle.turnover,
            confirmed=confirmed,
        )

    @staticmethod
    def _bucket_start(ts: datetime, timeframe_minutes: int) -> datetime:
        ts_utc = ts.astimezone(UTC)
        minute = (ts_utc.minute // timeframe_minutes) * timeframe_minutes
        return ts_utc.replace(second=0, microsecond=0, minute=minute)
