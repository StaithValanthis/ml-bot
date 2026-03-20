"""Labeling: event definitions, triple-barrier, and decision-outcome scaffold."""

from trading.research.labeling.events import LabelEvent, LabelWindow
from trading.research.labeling.scaffold import (
    DecisionOutcomeRecord,
    from_export_record,
    to_label_window,
)
from trading.research.labeling.triple_barrier import (
    TripleBarrierConfig,
    TripleBarrierLabels,
    compute_triple_barrier,
)

__all__ = [
    "DecisionOutcomeRecord",
    "LabelEvent",
    "LabelWindow",
    "TripleBarrierConfig",
    "TripleBarrierLabels",
    "compute_triple_barrier",
    "from_export_record",
    "to_label_window",
]
