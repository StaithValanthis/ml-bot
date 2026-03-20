"""Unit tests for backtest decision export generation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from trading.backtest.run import run_backtest
from trading.research.datasets.export import extract_decision_records
from trading.settings import load_settings


@pytest.mark.asyncio
async def test_backtest_writes_decision_export_when_decisions_exist(tmp_path: Path) -> None:
    """Backtest writes decisions_<timestamp>.json when decisions exist."""
    settings = load_settings()
    with patch.dict("os.environ", {"TRADING_ARCHIVE_DIR": str(tmp_path.resolve()), "TRADING_MODE": "backtest"}):
        await run_backtest(settings)

    export_dir = tmp_path / "decision_exports"
    assert export_dir.exists()
    json_files = list(export_dir.glob("decisions_*.json"))
    assert len(json_files) >= 1
    path = json_files[0]
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "records" in data
    assert data["count"] >= 1
    assert len(data["records"]) >= 1


@pytest.mark.asyncio
async def test_backtest_zero_decisions_skips_export_file(tmp_path: Path) -> None:
    """When backtest produces zero decisions, no export file is written (but export_dir is created)."""
    settings = load_settings()
    with (
        patch.dict("os.environ", {"TRADING_ARCHIVE_DIR": str(tmp_path.resolve()), "TRADING_MODE": "backtest"}),
        patch.object(settings.runtime, "backtest_bars", 50),
    ):
        await run_backtest(settings)

    export_dir = tmp_path / "decision_exports"
    assert export_dir.exists()
    json_files = list(export_dir.glob("decisions_*.json"))
    assert len(json_files) == 0
