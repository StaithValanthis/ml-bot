from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    RetryError,
    Retrying,
    stop_after_attempt,
    wait_exponential,
)


def _before_sleep_log(
    state: RetryCallState,
    logger: Callable[[str], None],
) -> None:
    exception = state.outcome.exception() if state.outcome else None
    logger(
        f"Retrying after attempt={state.attempt_number}, "
        f"error={type(exception).__name__ if exception else 'unknown'}"
    )


def build_retrying(
    *,
    max_attempts: int,
    base_seconds: float,
    max_seconds: float,
    logger: Callable[[str], None],
) -> Retrying:
    return Retrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=base_seconds, max=max_seconds),
        reraise=True,
        before_sleep=lambda s: _before_sleep_log(s, logger),
    )


def build_async_retrying(
    *,
    max_attempts: int,
    base_seconds: float,
    max_seconds: float,
    logger: Callable[[str], None],
) -> AsyncRetrying:
    return AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=base_seconds, max=max_seconds),
        reraise=True,
        before_sleep=lambda s: _before_sleep_log(s, logger),
    )


def run_with_retry(
    fn: Callable[..., Any],
    *args: Any,
    retrying: Retrying,
    **kwargs: Any,
) -> Any:
    try:
        for attempt in retrying:
            with attempt:
                return fn(*args, **kwargs)
    except RetryError as exc:
        raise RuntimeError("Retry attempts exhausted.") from exc
    raise RuntimeError("Retry execution failed without exception details.")
