from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from trading.util.time import utc_now


@dataclass(frozen=True, slots=True)
class LedgerEvent:
    event_type: str
    timestamp: datetime
    payload: dict[str, Any]


class LedgerSink(Protocol):
    async def write_event(self, event: LedgerEvent) -> None: ...


class RuntimeLedger:
    """Explicit structured ledger for decisions/execution/reconciliation/pnl events."""

    def __init__(self, sinks: list[LedgerSink] | None = None) -> None:
        self._events: list[LedgerEvent] = []
        self._sinks = sinks or []
        self._lock = asyncio.Lock()

    async def record(self, event_type: str, payload: dict[str, Any]) -> LedgerEvent:
        event = LedgerEvent(event_type=event_type, timestamp=utc_now(), payload=payload)
        async with self._lock:
            self._events.append(event)
        for sink in self._sinks:
            await sink.write_event(event)
        return event

    async def events(self) -> list[LedgerEvent]:
        async with self._lock:
            return list(self._events)
