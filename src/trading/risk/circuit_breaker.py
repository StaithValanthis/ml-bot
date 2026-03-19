from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

from trading.util.time import utc_now


@dataclass(slots=True, frozen=True)
class CircuitBreakerConfig:
    max_consecutive_rejections: int = 5
    max_consecutive_losses: int = 4
    max_intraday_drawdown_usdt: Decimal = Decimal("1500")
    cooldown_seconds: int = 120


class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        self._cfg = config or CircuitBreakerConfig()
        self._consecutive_rejections = 0
        self._consecutive_losses = 0
        self._intraday_pnl = Decimal("0")
        self._tripped_until = None

    def record_order_rejection(self) -> None:
        self._consecutive_rejections += 1
        self._trip_if_needed()

    def record_fill_pnl(self, pnl_delta: Decimal) -> None:
        self._intraday_pnl += pnl_delta
        if pnl_delta < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        self._trip_if_needed()

    def reset_intraday(self) -> None:
        self._consecutive_rejections = 0
        self._consecutive_losses = 0
        self._intraday_pnl = Decimal("0")
        self._tripped_until = None

    def trip(self, *, reason: str | None = None) -> None:
        # reason is currently informational only; telemetry integration arrives in monitoring phase.
        _ = reason
        self._tripped_until = utc_now() + timedelta(seconds=self._cfg.cooldown_seconds)

    def is_tripped(self) -> bool:
        if self._tripped_until is None:
            return False
        return utc_now() <= self._tripped_until

    def status_reason(self) -> str | None:
        if not self.is_tripped():
            return None
        return "circuit_breaker_tripped"

    def _trip_if_needed(self) -> None:
        should_trip = (
            self._consecutive_rejections >= self._cfg.max_consecutive_rejections
            or self._consecutive_losses >= self._cfg.max_consecutive_losses
            or self._intraday_pnl <= -self._cfg.max_intraday_drawdown_usdt
        )
        if should_trip:
            self._tripped_until = utc_now() + timedelta(seconds=self._cfg.cooldown_seconds)
