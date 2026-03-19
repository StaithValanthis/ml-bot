"""Registry lookup/promotion boundary for MLflow or similar."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from trading.models.model_bundle import ModelBundle


@dataclass(frozen=True, slots=True)
class RegistryLookup:
    """Request for registry lookup."""

    experiment_name: str
    stage: str
    model_name: str | None = None
    run_id: str | None = None


class RegistryClient(Protocol):
    """
    Typed stub/adaptor boundary for MLflow or similar.

    Do not overbuild: minimal interface for lookup and promotion.
    """

    def get_model(self, lookup: RegistryLookup) -> ModelBundle | None:
        """Get a model bundle by experiment/stage/name."""
        ...

    def promote(self, run_id: str, stage: str) -> bool:
        """Promote a run to a stage. Returns True on success."""
        ...

    def download_artifact(self, run_id: str, path: str) -> Path:
        """Download artifact from registry."""
        ...
