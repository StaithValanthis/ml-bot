"""Pytest: pin config directory so load_settings() always uses repo configs/symbols.yaml."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REPO_CONFIGS = _REPO_ROOT / "configs"


def pytest_configure(config) -> None:
    if (_REPO_CONFIGS / "base.yaml").is_file() and (_REPO_CONFIGS / "symbols.yaml").is_file():
        os.environ["TRADING_CONFIG_DIR"] = str(_REPO_CONFIGS.resolve())
