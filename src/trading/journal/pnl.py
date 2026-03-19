from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from trading.util.time import utc_now


@dataclass(frozen=True, slots=True)
class PnLRecord:
    timestamp: datetime
    equity_usdt: Decimal
    available_usdt: Decimal
    realized_pnl_usdt: Decimal
    unrealized_pnl_usdt: Decimal


class PnLTracker:
    """Minimal runtime PnL snapshot tracker."""

    def __init__(self) -> None:
        self._records: list[PnLRecord] = []
        self._lock = asyncio.Lock()

    async def add_snapshot(
        self,
        *,
        equity_usdt: Decimal,
        available_usdt: Decimal,
        realized_pnl_usdt: Decimal,
        unrealized_pnl_usdt: Decimal,
    ) -> PnLRecord:
        record = PnLRecord(
            timestamp=utc_now(),
            equity_usdt=equity_usdt,
            available_usdt=available_usdt,
            realized_pnl_usdt=realized_pnl_usdt,
            unrealized_pnl_usdt=unrealized_pnl_usdt,
        )
        async with self._lock:
            self._records.append(record)
        return record

    async def latest(self) -> PnLRecord | None:
        async with self._lock:
            if not self._records:
                return None
            return self._records[-1]

    async def all(self) -> list[PnLRecord]:
        async with self._lock:
            return list(self._records)
