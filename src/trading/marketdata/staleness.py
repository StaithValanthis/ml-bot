from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta

from trading.util.time import utc_now


@dataclass(frozen=True, slots=True)
class FeedStalenessLimit:
    max_age: timedelta


class FeedStalenessWatchdog:
    """
    Tracks last seen event times and identifies stale channels quickly.

    Channels are namespaced as '<stream>:<symbol>', e.g. 'public:BTCUSDT'.
    Expected channels are registered so we can detect feeds that never emit.
    """

    def __init__(self, *, limits: dict[str, FeedStalenessLimit], default_limit: FeedStalenessLimit) -> None:
        self._limits = limits
        self._default_limit = default_limit
        self._last_seen: dict[str, datetime] = {}
        self._expected_channels: set[str] = set()
        self._lock = asyncio.Lock()

    def set_expected_channels(self, channels: set[str]) -> None:
        """Register channels we expect to receive data from; used to detect never-emitting feeds."""
        self._expected_channels = set(channels)
        now = utc_now()
        for ch in channels:
            if ch not in self._last_seen:
                self._last_seen[ch] = now

    async def mark_seen(self, channel: str) -> None:
        async with self._lock:
            self._last_seen[channel] = utc_now()

    async def stale_channels(self) -> list[str]:
        now = utc_now()
        async with self._lock:
            stale: list[str] = []
            channels_to_check = set(self._last_seen.keys()) | self._expected_channels
            for channel in channels_to_check:
                seen_at = self._last_seen.get(channel, now)
                stream_name = channel.split(":", maxsplit=1)[0]
                limit = self._limits.get(stream_name, self._default_limit)
                if now - seen_at > limit.max_age:
                    stale.append(channel)
            return stale

    async def assert_healthy(self) -> None:
        stale = await self.stale_channels()
        if stale:
            raise RuntimeError(f"Feed staleness detected: {stale}")


def build_default_watchdog() -> FeedStalenessWatchdog:
    return FeedStalenessWatchdog(
        limits={
            "public": FeedStalenessLimit(max_age=timedelta(seconds=15)),
            "private": FeedStalenessLimit(max_age=timedelta(seconds=30)),
        },
        default_limit=FeedStalenessLimit(max_age=timedelta(seconds=20)),
    )
