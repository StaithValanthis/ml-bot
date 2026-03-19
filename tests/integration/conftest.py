"""Integration test fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from trading.settings import load_settings


@pytest.fixture(autouse=True)
def _integration_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure integration tests use paper mode and testnet config."""
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.setenv("TRADING_ENV", "bybit_testnet")
    monkeypatch.delenv("TRADING_POSTGRES_DSN", raising=False)
    monkeypatch.setenv("TRADING_ARCHIVE_DIR", str(Path.cwd() / "tmp_integration_archive"))


@pytest.fixture
def settings():
    """Load settings for integration tests."""
    return load_settings()
