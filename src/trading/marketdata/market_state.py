from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field

from trading.marketdata.normalizers import (
    NormalizedEvent,
    NormalizedExecution,
    NormalizedKline,
    NormalizedOrderUpdate,
    NormalizedPositionUpdate,
    NormalizedTicker,
    NormalizedTrade,
    NormalizedWalletUpdate,
)


@dataclass(slots=True)
class MarketStateSnapshot:
    tickers: dict[str, NormalizedTicker] = field(default_factory=dict)
    last_trade: dict[str, NormalizedTrade] = field(default_factory=dict)
    confirmed_klines: dict[tuple[str, str], NormalizedKline] = field(default_factory=dict)
    order_updates: dict[str, NormalizedOrderUpdate] = field(default_factory=dict)
    latest_executions: dict[str, NormalizedExecution] = field(default_factory=dict)
    wallet: NormalizedWalletUpdate | None = None
    positions: dict[str, NormalizedPositionUpdate] = field(default_factory=dict)


class MarketState:
    """
    Thread-safe (async) in-memory state container.

    The goal here is deterministic state transitions from normalized events and
    replay-friendly behavior, while tolerating partial private stream payloads.
    """

    def __init__(self, *, max_trade_buffer_per_symbol: int = 2000) -> None:
        self._lock = asyncio.Lock()
        self._snapshot = MarketStateSnapshot()
        self._trade_buffers: dict[str, deque[NormalizedTrade]] = defaultdict(
            lambda: deque(maxlen=max_trade_buffer_per_symbol)
        )
        self._confirmed_kline_queue: asyncio.Queue[NormalizedKline] = asyncio.Queue()

    async def apply_event(self, event: NormalizedEvent) -> None:
        async with self._lock:
            if isinstance(event, NormalizedTicker):
                self._snapshot.tickers[event.symbol] = event
                return
            if isinstance(event, NormalizedTrade):
                self._snapshot.last_trade[event.symbol] = event
                self._trade_buffers[event.symbol].append(event)
                return
            if isinstance(event, NormalizedKline):
                if event.confirmed:
                    self._snapshot.confirmed_klines[(event.symbol, event.interval)] = event
                    self._confirmed_kline_queue.put_nowait(event)
                return
            if isinstance(event, NormalizedOrderUpdate):
                if event.order_id:
                    previous = self._snapshot.order_updates.get(event.order_id)
                    self._snapshot.order_updates[event.order_id] = _merge_order_update(previous, event)
                return
            if isinstance(event, NormalizedExecution):
                if event.exec_id:
                    self._snapshot.latest_executions[event.exec_id] = event
                return
            if isinstance(event, NormalizedWalletUpdate):
                previous_wallet = self._snapshot.wallet
                self._snapshot.wallet = _merge_wallet_update(previous_wallet, event)
                return
            if isinstance(event, NormalizedPositionUpdate):
                if event.symbol:
                    previous_position = self._snapshot.positions.get(event.symbol)
                    self._snapshot.positions[event.symbol] = _merge_position_update(previous_position, event)

    async def apply_events(self, events: list[NormalizedEvent]) -> None:
        for event in events:
            await self.apply_event(event)

    async def snapshot(self) -> MarketStateSnapshot:
        async with self._lock:
            return MarketStateSnapshot(
                tickers=dict(self._snapshot.tickers),
                last_trade=dict(self._snapshot.last_trade),
                confirmed_klines=dict(self._snapshot.confirmed_klines),
                order_updates=dict(self._snapshot.order_updates),
                latest_executions=dict(self._snapshot.latest_executions),
                wallet=self._snapshot.wallet,
                positions=dict(self._snapshot.positions),
            )

    async def next_confirmed_kline(self) -> NormalizedKline:
        return await self._confirmed_kline_queue.get()

    async def recent_trades(self, symbol: str) -> list[NormalizedTrade]:
        async with self._lock:
            return list(self._trade_buffers.get(symbol, deque()))


def _merge_order_update(
    previous: NormalizedOrderUpdate | None,
    update: NormalizedOrderUpdate,
) -> NormalizedOrderUpdate:
    if previous is None:
        return update
    merged_data = previous.model_dump()
    for key, value in update.model_dump().items():
        if value is not None:
            merged_data[key] = value
    return NormalizedOrderUpdate.model_validate(merged_data)


def _merge_wallet_update(
    previous: NormalizedWalletUpdate | None,
    update: NormalizedWalletUpdate,
) -> NormalizedWalletUpdate:
    if previous is None:
        return update
    merged_data = previous.model_dump()
    for key, value in update.model_dump().items():
        if key == "raw":
            continue
        if value is not None:
            merged_data[key] = value
    merged_data["raw"] = update.raw
    return NormalizedWalletUpdate.model_validate(merged_data)


def _merge_position_update(
    previous: NormalizedPositionUpdate | None,
    update: NormalizedPositionUpdate,
) -> NormalizedPositionUpdate:
    if previous is None:
        return update
    merged_data = previous.model_dump()
    for key, value in update.model_dump().items():
        if key == "raw":
            continue
        if value is not None:
            merged_data[key] = value
    merged_data["raw"] = update.raw
    return NormalizedPositionUpdate.model_validate(merged_data)
