"""Training: splits, train, evaluate, promote workflow boundaries."""

from trading.research.training.evaluate import EvalMetrics, EvalResult
from trading.research.training.promote import PromoteDecision, PromoteRequest, decide_promotion
from trading.research.training.splits import SplitConfig, SplitResult, TimeSeriesSplitter
from trading.research.training.train import TrainRequest, TrainResult, TrainingJob

__all__ = [
    "EvalMetrics",
    "EvalResult",
    "PromoteDecision",
    "PromoteRequest",
    "decide_promotion",
    "SplitConfig",
    "SplitResult",
    "TimeSeriesSplitter",
    "TrainRequest",
    "TrainResult",
    "TrainingJob",
]
