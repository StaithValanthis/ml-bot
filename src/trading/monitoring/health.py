from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from threading import Lock

from trading.util.time import utc_now


@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    ws_public_connected: bool
    ws_private_connected: bool
    stale_channels: list[str]
    circuit_breaker_tripped: bool
    last_decision_at: datetime | None
    last_reconcile_at: datetime | None
    updated_at: datetime
    private_stream_error: str | None


class HealthState:
    """Thread-safe health status container for runtime control-plane visibility."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._ws_public_connected = False
        self._ws_private_connected = False
        self._stale_channels: list[str] = []
        self._circuit_breaker_tripped = False
        self._last_decision_at: datetime | None = None
        self._last_reconcile_at: datetime | None = None
        self._updated_at = utc_now()
        self._private_stream_error: str | None = None

    def set_ws_public(self, connected: bool) -> None:
        with self._lock:
            self._ws_public_connected = connected
            self._updated_at = utc_now()

    def set_ws_private(self, connected: bool) -> None:
        with self._lock:
            self._ws_private_connected = connected
            if connected:
                self._private_stream_error = None
            self._updated_at = utc_now()

    def set_private_stream_error(self, error: str | None) -> None:
        with self._lock:
            self._private_stream_error = error
            self._updated_at = utc_now()

    def set_stale_channels(self, channels: list[str]) -> None:
        with self._lock:
            self._stale_channels = list(channels)
            self._updated_at = utc_now()

    def set_circuit_breaker(self, tripped: bool) -> None:
        with self._lock:
            self._circuit_breaker_tripped = tripped
            self._updated_at = utc_now()

    def mark_decision(self) -> None:
        with self._lock:
            self._last_decision_at = utc_now()
            self._updated_at = utc_now()

    def mark_reconcile(self) -> None:
        with self._lock:
            self._last_reconcile_at = utc_now()
            self._updated_at = utc_now()

    def snapshot(self) -> HealthSnapshot:
        with self._lock:
            return HealthSnapshot(
                ws_public_connected=self._ws_public_connected,
                ws_private_connected=self._ws_private_connected,
                stale_channels=list(self._stale_channels),
                circuit_breaker_tripped=self._circuit_breaker_tripped,
                last_decision_at=self._last_decision_at,
                last_reconcile_at=self._last_reconcile_at,
                updated_at=self._updated_at,
                private_stream_error=self._private_stream_error,
            )
