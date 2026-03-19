"""Backtest foundation: event-driven engine, cost/funding models, walk-forward and optimizer scaffolds."""

from trading.backtest.cost_model import CostBreakdown, CostModel, CostModelConfig
from trading.backtest.engine import BacktestEngine, BacktestResult, CandleEvent
from trading.backtest.funding_model import FundingAccrual, FundingModel, FundingModelConfig
from trading.backtest.optimizer import OptimizerResult, OptimizerScaffold
from trading.backtest.walk_forward import WalkForwardConfig, WalkForwardSegment, generate_segments

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "CandleEvent",
    "CostBreakdown",
    "CostModel",
    "CostModelConfig",
    "FundingAccrual",
    "FundingModel",
    "FundingModelConfig",
    "OptimizerResult",
    "OptimizerScaffold",
    "WalkForwardConfig",
    "WalkForwardSegment",
    "generate_segments",
]
