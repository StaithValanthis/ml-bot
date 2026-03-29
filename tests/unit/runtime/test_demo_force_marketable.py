"""Unit tests for demo_force_marketable_entries validation mode."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import patch

from trading.execution.execution_engine import ExecutionEngine
from trading.settings import load_settings
from trading.strategy.signal_engine import SignalAction, SignalDecision
from trading.util.types import OrderSide, OrderType, TimeInForce


def _sym() -> str:
    return load_settings().trading.symbols[0]


def _signal(side: OrderSide = OrderSide.BUY) -> SignalDecision:
    return SignalDecision(
        symbol=_sym(),
        action=SignalAction.ENTER_LONG,
        side=side,
        reference_price=Decimal("50000"),
        confidence=Decimal("0.6"),
        stop_price=Decimal("49000"),
        reason="test",
        generated_at=datetime.now(UTC),
        metadata={},
    )


def test_build_entry_intent_default_is_limit_post_only() -> None:
    """Flag false: entry remains LIMIT POST_ONLY (existing behavior)."""
    engine = ExecutionEngine(strategy_id="v1alpha")
    intent = engine.build_entry_intent(
        signal=_signal(),
        qty=Decimal("0.01"),
        reference_price=Decimal("50000"),
        now=datetime.now(UTC),
        force_marketable=False,
    )
    assert intent is not None
    assert intent.order_type == OrderType.LIMIT
    assert intent.time_in_force == TimeInForce.POST_ONLY
    assert intent.price is not None
    assert intent.metadata.get("post_only") is True


def test_build_entry_intent_force_marketable_is_market_ioc() -> None:
    """Flag true: entry becomes MARKET IOC (immediately fillable)."""
    engine = ExecutionEngine(strategy_id="v1alpha")
    intent = engine.build_entry_intent(
        signal=_signal(),
        qty=Decimal("0.01"),
        reference_price=Decimal("50000"),
        now=datetime.now(UTC),
        force_marketable=True,
    )
    assert intent is not None
    assert intent.order_type == OrderType.MARKET
    assert intent.time_in_force == TimeInForce.IOC
    assert intent.price is None
    assert intent.metadata.get("demo_force_marketable") is True


def test_orchestrator_settings_produce_marketable_intent_when_demo_and_flag_enabled() -> None:
    """With DEMO + demo_force_marketable_entries, orchestrator logic yields force_marketable=True -> MARKET intent."""
    from trading.settings import load_settings

def _sym() -> str:
    return load_settings().trading.symbols[0]

    from trading.util.types import RuntimeMode

    settings = load_settings()
    settings.runtime.mode = RuntimeMode.DEMO
    settings.runtime.demo_force_marketable_entries = True

    force_marketable = (
        settings.runtime.mode == RuntimeMode.DEMO
        and settings.runtime.demo_force_marketable_entries
    )
    assert force_marketable is True

    engine = ExecutionEngine(strategy_id="v1alpha")
    intent = engine.build_entry_intent(
        signal=_signal(),
        qty=Decimal("0.01"),
        reference_price=Decimal("50000"),
        now=datetime.now(UTC),
        force_marketable=force_marketable,
    )
    assert intent is not None
    assert intent.order_type == OrderType.MARKET
    assert intent.time_in_force == TimeInForce.IOC


def test_demo_force_marketable_entries_env_parsing() -> None:
    """TRADING_DEMO_FORCE_MARKETABLE_ENTRIES env var loads into settings correctly."""
    from trading.settings import load_settings

    for val, expected in [("true", True), ("1", True), ("yes", True), ("false", False), ("0", False)]:
        with patch.dict(os.environ, {"TRADING_DEMO_FORCE_MARKETABLE_ENTRIES": val}):
            settings = load_settings()
            assert settings.runtime.demo_force_marketable_entries is expected, val
