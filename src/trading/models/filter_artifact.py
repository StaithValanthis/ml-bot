"""Typed path to load promoted offline baseline model artifact."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelArtifactLoadResult:
    """Result of attempting to load model artifact."""

    model: Any
    path: Path
    loaded: bool
    error: str | None = None


def load_model_artifact(path: Path | str | None) -> ModelArtifactLoadResult:
    """
    Load promoted offline baseline model artifact if one exists.

    Optional and explicit. Returns loaded=False when path is None, missing, or load fails.
    Expects joblib-serialized sklearn classifier (e.g. LogisticRegression).
    """
    if path is None:
        return ModelArtifactLoadResult(model=None, path=Path(), loaded=False, error="no_path")
    p = Path(path)
    if not p.exists():
        return ModelArtifactLoadResult(model=None, path=p, loaded=False, error="path_not_found")
    try:
        import joblib

        model = joblib.load(p)
        return ModelArtifactLoadResult(model=model, path=p, loaded=True)
    except ImportError:
        return ModelArtifactLoadResult(model=None, path=p, loaded=False, error="joblib_not_available")
    except Exception as exc:
        return ModelArtifactLoadResult(model=None, path=p, loaded=False, error=str(exc))
