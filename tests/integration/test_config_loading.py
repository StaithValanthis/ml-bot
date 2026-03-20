"""Integration tests for runtime config loading."""

from __future__ import annotations

import os
from decimal import Decimal

import pytest

from trading.settings import load_settings
from trading.util.types import RuntimeMode


def test_load_settings_returns_valid_structure() -> None:
    """Config loading produces valid AppSettings with all required sections."""
    settings = load_settings()
    assert settings.runtime is not None
    assert settings.exchange is not None
    assert settings.trading is not None
    assert settings.risk is not None
    assert settings.logging is not None
    assert settings.trading.symbols
    assert settings.get_symbol_specs()


def test_load_settings_paper_mode_default() -> None:
    """Default mode is paper when TRADING_MODE=paper."""
    settings = load_settings()
    assert settings.runtime.mode == RuntimeMode.PAPER


def test_load_settings_backtest_mode_when_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config respects TRADING_MODE=backtest."""
    monkeypatch.setenv("TRADING_MODE", "backtest")
    settings = load_settings()
    assert settings.runtime.mode == RuntimeMode.BACKTEST
    assert settings.runtime.backtest_bars >= 1


def test_load_settings_symbols_validated() -> None:
    """Symbols config is validated and matches trading.symbols."""
    settings = load_settings()
    for sym in settings.trading.symbols:
        assert sym in settings.symbols
        spec = settings.symbols[sym]
        assert spec.qty_step > 0
        assert spec.min_qty > 0
        assert spec.price_tick > 0
        assert spec.max_leverage > 0


def test_load_settings_risk_per_symbol_optional() -> None:
    """Risk per_symbol can be empty or populated."""
    settings = load_settings()
    assert hasattr(settings.risk, "per_symbol")
    # Config may or may not have per_symbol entries
    for sym, limit in settings.risk.per_symbol.items():
        assert limit.max_notional_usdt > 0
        assert limit.max_position_abs > 0


def test_load_settings_demo_drill_max_notional_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    """TRADING_DEMO_DRILL_MAX_NOTIONAL_USDT overrides drill max notional cap."""
    monkeypatch.setenv("TRADING_DEMO_DRILL_MAX_NOTIONAL_USDT", "500")
    settings = load_settings()
    assert settings.runtime.demo_drill.max_notional_usdt == Decimal("500")
