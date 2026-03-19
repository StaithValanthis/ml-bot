from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from trading.util.logging import get_logger


class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class AlertEvent:
    level: AlertLevel
    code: str
    message: str
    context: dict[str, str]


class AlertSink(Protocol):
    def emit(self, event: AlertEvent) -> None: ...


class LogAlertSink:
    """Default alert sink for baseline runtime operations."""

    def __init__(self) -> None:
        self._logger = get_logger("trading.monitoring.alerts")

    def emit(self, event: AlertEvent) -> None:
        if event.level == AlertLevel.CRITICAL:
            self._logger.error("alert", code=event.code, message=event.message, context=event.context)
        elif event.level == AlertLevel.WARNING:
            self._logger.warning("alert", code=event.code, message=event.message, context=event.context)
        else:
            self._logger.info("alert", code=event.code, message=event.message, context=event.context)
