from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable


async def run_periodic(
    *,
    name: str,
    interval_seconds: float,
    stop_event: asyncio.Event,
    task_fn: Callable[[], Awaitable[None]],
) -> None:
    """
    Run coroutine periodically until stop_event is set.

    Exceptions are propagated to caller so orchestrator can fail fast and
    trigger controlled shutdown.
    """
    if interval_seconds <= 0:
        raise ValueError(f"interval_seconds for periodic task '{name}' must be > 0")

    while not stop_event.is_set():
        await task_fn()
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except TimeoutError:
            continue
