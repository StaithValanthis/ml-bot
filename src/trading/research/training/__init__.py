"""Training: splits, train, evaluate, promote, offline runner workflow boundaries."""

from trading.research.training.baseline import (
    BaselineExperimentResult,
    ComputedMetrics,
    Verdict,
    metrics_to_dict,
    run_baseline_experiment,
    write_test_predictions,
)
from trading.research.training.evaluate import (
    EvalMetrics,
    EvalResult,
    OfflineEvalResult,
    SampleCounts,
    SplitMetadata,
)
from trading.research.training.promote import PromoteDecision, PromoteRequest, decide_promotion
from trading.research.training.report import (
    OfflineTrainReport,
    build_offline_train_report,
    write_offline_train_report,
)
from trading.research.training.runner import OfflineTrainResult, run_offline_training
from trading.research.training.splits import (
    DefaultTimeSeriesSplitter,
    SplitConfig,
    SplitResult,
    TimeSeriesSplitter,
)
from trading.research.training.train import TrainRequest, TrainResult, TrainingJob

__all__ = [
    "BaselineExperimentResult",
    "ComputedMetrics",
    "DefaultTimeSeriesSplitter",
    "EvalMetrics",
    "EvalResult",
    "OfflineEvalResult",
    "OfflineTrainReport",
    "OfflineTrainResult",
    "PromoteDecision",
    "PromoteRequest",
    "SampleCounts",
    "SplitConfig",
    "SplitMetadata",
    "SplitResult",
    "TimeSeriesSplitter",
    "TrainRequest",
    "TrainResult",
    "TrainingJob",
    "Verdict",
    "build_offline_train_report",
    "decide_promotion",
    "metrics_to_dict",
    "run_baseline_experiment",
    "run_offline_training",
    "write_offline_train_report",
    "write_test_predictions",
]
