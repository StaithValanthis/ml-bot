"""Dataset ingestion, backfill, export, and preparation boundaries."""

from trading.research.datasets.backfill import BackfillJob, BackfillRequest, BackfillResult
from trading.research.datasets.export import (
    DecisionExportRecord,
    extract_decision_records,
    export_records_to_json,
)
from trading.research.datasets.ingest import DatasetBuilder, IngestInputs, IngestResult
from trading.research.datasets.prepare import (
    FEATURE_NAMES,
    ModelReadyRow,
    OptionalLabels,
    compute_feature_coverage,
    compute_label_trust,
    prepare_training_rows,
    write_training_rows_csv,
)

__all__ = [
    "FEATURE_NAMES",
    "BackfillJob",
    "BackfillRequest",
    "BackfillResult",
    "DatasetBuilder",
    "DecisionExportRecord",
    "IngestInputs",
    "IngestResult",
    "ModelReadyRow",
    "OptionalLabels",
    "compute_feature_coverage",
    "compute_label_trust",
    "extract_decision_records",
    "export_records_to_json",
    "prepare_training_rows",
    "write_training_rows_csv",
]
