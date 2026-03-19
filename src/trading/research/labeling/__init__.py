"""Labeling: event definitions and triple-barrier label calculation."""

from trading.research.labeling.events import LabelEvent, LabelWindow
from trading.research.labeling.triple_barrier import (
    TripleBarrierConfig,
    TripleBarrierLabels,
    compute_triple_barrier,
)

__all__ = [
    "LabelEvent",
    "LabelWindow",
    "TripleBarrierConfig",
    "TripleBarrierLabels",
    "compute_triple_barrier",
]
