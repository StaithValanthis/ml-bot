from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import asyncpg

from trading.journal.ledger import LedgerEvent


class PostgresJournalStore:
    """
    Minimal typed Postgres adapter for ledger durability.

    This is an operational scaffold: it manages a single append table and
    intentionally avoids full migration/schema framework in this phase.
    """

    def __init__(self, dsn: str | None) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        if self._dsn is None:
            return
        self._pool = await asyncpg.create_pool(dsn=self._dsn, min_size=1, max_size=3)
        await self._ensure_schema()

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def write_event(self, event: LedgerEvent) -> None:
        if self._pool is None:
            return
        payload_json = json.dumps(event.payload, separators=(",", ":"), default=_json_default)
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO trading_ledger_events (event_type, ts_utc, payload_json)
                VALUES ($1, $2, $3::jsonb)
                """,
                event.event_type,
                event.timestamp,
                payload_json,
            )

    async def _ensure_schema(self) -> None:
        if self._pool is None:
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trading_ledger_events (
                    id BIGSERIAL PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    ts_utc TIMESTAMPTZ NOT NULL,
                    payload_json JSONB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_trading_ledger_events_ts ON trading_ledger_events (ts_utc);
                """
            )


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
