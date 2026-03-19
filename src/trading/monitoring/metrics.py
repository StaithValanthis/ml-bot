from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from threading import Lock
from typing import DefaultDict


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    counters: dict[str, float]
    gauges: dict[str, float]
    histograms: dict[str, list[float]]


class MetricsRegistry:
    """
    Lightweight metrics abstraction.

    This baseline keeps in-memory counters/gauges/histograms and can later map
    to Prometheus exporters without changing runtime call sites.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: DefaultDict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: DefaultDict[str, list[float]] = defaultdict(list)

    def inc(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._histograms[name].append(value)

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            return MetricsSnapshot(
                counters=dict(self._counters),
                gauges=dict(self._gauges),
                histograms={k: list(v) for k, v in self._histograms.items()},
            )
