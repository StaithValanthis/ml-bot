"""Offline ML model filter evaluation: purged CV, threshold analysis, promotion-readiness."""

from trading.research.evaluation.purged_cv import (
    PurgedCVConfig,
    PurgedFold,
    PurgedWalkForwardSplitter,
    purged_splits,
)
from trading.research.evaluation.threshold_analysis import (
    ShadowVsBaselineReport,
    ThresholdMetrics,
    compute_retention_based_recommendations,
    compute_threshold_grid,
    shadow_vs_baseline_report,
)

__all__ = [
    "PurgedCVConfig",
    "PurgedFold",
    "PurgedWalkForwardSplitter",
    "purged_splits",
    "ShadowVsBaselineReport",
    "ThresholdMetrics",
    "compute_retention_based_recommendations",
    "compute_threshold_grid",
    "shadow_vs_baseline_report",
]
