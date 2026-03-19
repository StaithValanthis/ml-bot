"""Model bundle contract for inference artifacts and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Metadata for a trained model."""

    run_id: str
    symbol: str
    trained_at: datetime
    version: str = "1.0"
    extra: dict[str, Any] | None = None


@dataclass(slots=True)
class ModelBundle:
    """
    Model bundle for live inference: artifacts + metadata.

    Usable as a plug-in point for strategy/signal layers.
    """

    metadata: ModelMetadata
    artifact_path: Path
    _loaded: object | None = None

    def load(self) -> object:
        """Load the model artifact. Scaffold: returns None until loader is implemented."""
        return self._loaded
