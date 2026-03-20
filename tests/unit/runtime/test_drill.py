"""Unit tests for demo execution drill."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import patch

import pytest

from trading.runtime.drill import (
    DrillConfig,
    DrillMode,
    DrillOutcome,
    build_drill_intent,
    generate_drill_order_link_id,
    validate_drill,
)
from trading.util.types import OrderSide, PositionSide, RuntimeMode


def test_validate_drill_refused_outside_demo() -> None:
    """Drill is refused when mode is not DEMO."""
    assert validate_drill(
        mode=RuntimeMode.PAPER,
        dry_run=False,
        symbol="BTCUSDT",
        qty=Decimal("0.001"),
        configured_symbols=["BTCUSDT"],
        symbol_spec=None,
    ) == "drill_refused_mode_not_demo"
    assert validate_drill(
        mode=RuntimeMode.LIVE,
        dry_run=False,
        symbol="BTCUSDT",
        qty=Decimal("0.001"),
        configured_symbols=["BTCUSDT"],
        symbol_spec=None,
    ) == "drill_refused_mode_not_demo"


def test_validate_drill_refused_when_dry_run() -> None:
    """Drill is refused when dry_run is True."""
    assert validate_drill(
        mode=RuntimeMode.DEMO,
        dry_run=True,
        symbol="BTCUSDT",
        qty=Decimal("0.001"),
        configured_symbols=["BTCUSDT"],
        symbol_spec=None,
    ) == "drill_refused_dry_run"


def test_validate_drill_refused_when_symbol_not_configured() -> None:
    """Drill is refused when symbol is not in configured trading symbols."""
    assert validate_drill(
        mode=RuntimeMode.DEMO,
        dry_run=False,
        symbol="ETHUSDT",
        qty=Decimal("0.001"),
        configured_symbols=["BTCUSDT"],
        symbol_spec=None,
    ) == "drill_refused_symbol_not_configured"


def test_validate_drill_refused_when_qty_below_min() -> None:
    """Drill is refused when qty is below symbol min_qty."""
    class FakeSpec:
        min_qty = Decimal("0.01")

    assert validate_drill(
        mode=RuntimeMode.DEMO,
        dry_run=False,
        symbol="BTCUSDT",
        qty=Decimal("0.001"),
        configured_symbols=["BTCUSDT"],
        symbol_spec=FakeSpec(),
    ) == "drill_refused_qty_below_min_0.01"


def test_validate_drill_refused_when_notional_exceeds_cap() -> None:
    """Drill is refused when notional exceeds safe cap."""
    assert validate_drill(
        mode=RuntimeMode.DEMO,
        dry_run=False,
        symbol="BTCUSDT",
        qty=Decimal("1"),
        configured_symbols=["BTCUSDT"],
        symbol_spec=None,
        reference_price=Decimal("100000"),
        max_drill_notional_usdt=Decimal("10"),
    ) == "drill_refused_notional_exceeds_cap_10"


def test_validate_drill_allowed() -> None:
    """Drill is allowed when all constraints pass."""
    class FakeSpec:
        min_qty = Decimal("0.001")

    assert (
        validate_drill(
            mode=RuntimeMode.DEMO,
            dry_run=False,
            symbol="BTCUSDT",
            qty=Decimal("0.001"),
            configured_symbols=["BTCUSDT"],
            symbol_spec=FakeSpec(),
            reference_price=Decimal("50000"),
            max_drill_notional_usdt=Decimal("100"),
        )
        is None
    )


def test_build_drill_intent_post_only() -> None:
    """build_drill_intent produces post-only limit order near bid/ask."""
    config = DrillConfig(symbol="BTCUSDT", side=OrderSide.BUY, qty=Decimal("0.001"), mode=DrillMode.POST_ONLY_LIMIT)
    link_id = generate_drill_order_link_id("BTCUSDT")
    now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    intent = build_drill_intent(config=config, reference_price=Decimal("50000"), order_link_id=link_id, now=now)
    assert intent.symbol == "BTCUSDT"
    assert intent.side == OrderSide.BUY
    assert intent.qty == Decimal("0.001")
    assert intent.reduce_only is False
    assert intent.price is not None
    assert intent.price < Decimal("50000")
    assert intent.metadata.get("drill") is True
    assert link_id.startswith("drill-")


def test_generate_drill_order_link_id() -> None:
    """Drill order link id has drill prefix and fits Bybit length."""
    link_id = generate_drill_order_link_id("BTCUSDT")
    assert link_id.startswith("drill-")
    assert "btcusdt" in link_id.lower() or "btcu" in link_id.lower()
    assert len(link_id) <= 36


def test_drill_outcome_defaults() -> None:
    """DrillOutcome has correct defaults."""
    o = DrillOutcome()
    assert o.enabled is False
    assert o.attempted is False
    assert o.ack_received is False
    assert o.completed is False
    assert o.aborted is False
