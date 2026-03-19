from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from time import monotonic


@dataclass(frozen=True, slots=True)
class EndpointLimit:
    max_requests: int
    window_seconds: float


class EndpointRateLimiter:
    """
    Sliding-window async rate limiter keyed by endpoint group.

    This is intentionally exchange-aware at the endpoint-group level so we can
    tune limits per Bybit route category without coupling limiter internals to
    REST client implementation details.
    """

    def __init__(self, limits: dict[str, EndpointLimit], default_limit: EndpointLimit) -> None:
        self._limits = limits
        self._default_limit = default_limit
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def acquire(self, endpoint_group: str) -> None:
        limit = self._limits.get(endpoint_group, self._default_limit)
        lock = self._locks[endpoint_group]
        async with lock:
            while True:
                now = monotonic()
                events = self._events[endpoint_group]
                window_start = now - limit.window_seconds

                while events and events[0] < window_start:
                    events.popleft()

                if len(events) < limit.max_requests:
                    events.append(now)
                    return

                sleep_for = max(0.001, limit.window_seconds - (now - events[0]))
                await asyncio.sleep(sleep_for)


def build_default_bybit_limiter() -> EndpointRateLimiter:
    """
    Conservative defaults for v5 unified account endpoints.

    TODO(phase-6): Move endpoint limits into config for environment-specific tuning.
    """
    limits = {
        "market": EndpointLimit(max_requests=20, window_seconds=1.0),
        "trade": EndpointLimit(max_requests=10, window_seconds=1.0),
        "position": EndpointLimit(max_requests=10, window_seconds=1.0),
        "account": EndpointLimit(max_requests=10, window_seconds=1.0),
    }
    return EndpointRateLimiter(
        limits=limits,
        default_limit=EndpointLimit(max_requests=8, window_seconds=1.0),
    )
