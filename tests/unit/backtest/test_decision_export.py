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


@pytest.mark.asyncio
async def test_backtest_produces_mixed_filled_and_not_filled(tmp_path: Path) -> None:
    """Backtest with fill_probability produces both filled and not-filled outcomes in export."""
    from trading.backtest.engine import BacktestConfig
    from trading.backtest.event_source import synthetic_candle_events
    from trading.research.datasets.export import extract_decision_records
    from trading.settings import backtest_config_from_settings

    settings = load_settings()
    cfg = backtest_config_from_settings(settings)
    cfg = BacktestConfig(
        initial_equity_usdt=cfg.initial_equity_usdt,
        candle_timeframe=cfg.candle_timeframe,
        regime_timeframe=cfg.regime_timeframe,
        max_total_notional_usdt=cfg.max_total_notional_usdt,
        max_leverage=cfg.max_leverage,
        daily_loss_limit_usdt=cfg.daily_loss_limit_usdt,
        liquidation_buffer_bps=cfg.liquidation_buffer_bps,
        symbol_specs=cfg.symbol_specs,
        per_symbol_limits=cfg.per_symbol_limits,
        fill_probability=0.5,
        fill_seed=42,
    )
    event_source = synthetic_candle_events(
        symbols=settings.trading.symbols,
        bars=350,
        timeframe=settings.trading.candle_timeframe,
    )
    from trading.backtest.engine import BacktestEngine

    engine = BacktestEngine(config=cfg)
    result = await engine.run(event_source)
    records = extract_decision_records(result.events)
    filled = [r for r in records if r.filled]
    not_filled = [r for r in records if not r.filled]
    assert len(records) >= 1, "backtest should produce at least one decision"
    assert len(filled) >= 1, "should have at least one filled outcome"
    assert len(not_filled) >= 1, "should have at least one not-filled outcome"


@pytest.mark.asyncio
async def test_backtest_export_includes_dataset_diagnostics(tmp_path: Path) -> None:
    """Exported JSON includes dataset_diagnostics (filled_count, not_filled_count, fill_rate, has_both_classes)."""
    settings = load_settings()
    with patch.dict("os.environ", {"TRADING_ARCHIVE_DIR": str(tmp_path.resolve()), "TRADING_MODE": "backtest"}):
        await run_backtest(settings)

    export_dir = tmp_path / "decision_exports"
    json_files = list(export_dir.glob("decisions_*.json"))
    if not json_files:
        pytest.skip("backtest produced no decisions (bars/settings may yield zero)")
    data = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert "dataset_diagnostics" in data
    diag = data["dataset_diagnostics"]
    assert "total_decisions" in diag
    assert "filled_count" in diag
    assert "not_filled_count" in diag
    assert "fill_rate" in diag
    assert "has_both_classes" in diag
    assert "train_split_single_class_risk" in diag


@pytest.mark.asyncio
async def test_scaled_backtest_produces_enough_for_training(tmp_path: Path) -> None:
    """Default scaled backtest (1200 bars, 0.55 fill_prob) produces enough rows and mixed labels for training."""
    from trading.research.training.runner import run_offline_training

    settings = load_settings()
    with patch.dict("os.environ", {"TRADING_ARCHIVE_DIR": str(tmp_path.resolve()), "TRADING_MODE": "backtest"}):
        await run_backtest(settings)

    export_dir = tmp_path / "decision_exports"
    json_files = list(export_dir.glob("decisions_*.json"))
    assert len(json_files) >= 1, "backtest should produce an export"
    data = json.loads(json_files[0].read_text(encoding="utf-8"))
    records_data = data.get("records", [])
    assert len(records_data) >= 20, "scaled backtest should produce at least 20 decisions"
    filled = sum(1 for r in records_data if r.get("filled"))
    not_filled = len(records_data) - filled
    assert filled >= 2, "need at least 2 filled for train split diversity"
    assert not_filled >= 2, "need at least 2 not-filled for train split diversity"

    result = run_offline_training(records_path=json_files[0], output_dir=tmp_path)
    assert result.success
    assert result.total_rows >= 20
    assert not result.model_training_skipped, (
        f"training should proceed with mixed labels; reason={result.model_training_skipped_reason}"
    )
