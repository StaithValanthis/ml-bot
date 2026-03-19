"""Optimizer scaffold for parameter search and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol


@dataclass(slots=True)
class OptimizerResult:
    params: dict[str, Any]
    total_pnl_usdt: Decimal
    sharpe_ratio: Decimal | None
    max_drawdown_bps: Decimal | None
    fills: int
    decisions: int
    completed_at: datetime


class BacktestRunner(Protocol):
    """Protocol for runnable backtest; allows optimizer to invoke without coupling."""

    async def run(self, params: dict[str, Any]) -> OptimizerResult: ...


@dataclass(frozen=True, slots=True)
class OptimizerScaffold:
    """
    Scaffold for parameter search.

    Does not implement search logic; provides typed interfaces for
    parameter grids and result aggregation. Callers supply the
    evaluation function.
    """

    param_grid: dict[str, list[Any]] = field(default_factory=dict)

    def expand_grid(self) -> list[dict[str, Any]]:
        """Expand param_grid into list of param dicts. Scaffold: simple product."""
        if not self.param_grid:
            return [{}]
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        result: list[dict[str, Any]] = []
        self._product(keys, values, 0, {}, result)
        return result

    def _product(
        self,
        keys: list[str],
        values: list[list[Any]],
        idx: int,
        current: dict[str, Any],
        out: list[dict[str, Any]],
    ) -> None:
        if idx == len(keys):
            out.append(dict(current))
            return
        for v in values[idx]:
            current[keys[idx]] = v
            self._product(keys, values, idx + 1, current, out)
