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
from trading.settings import load_settings
from trading.util.types import OrderSide, PositionSide, RuntimeMode


def _sym() -> str:
    return load_settings().trading.symbols[0]


def test_validate_drill_refused_outside_demo() -> None:
    """Drill is refused when mode is not DEMO."""
    r = validate_drill(
        mode=RuntimeMode.PAPER,
        dry_run=False,
        symbol=_sym(),
        qty=Decimal("0.001"),
        configured_symbols=[_sym()],
        symbol_spec=None,
    )
    assert r is not None and r.reason == "drill_refused_mode_not_demo"
    r = validate_drill(
        mode=RuntimeMode.LIVE,
        dry_run=False,
        symbol=_sym(),
        qty=Decimal("0.001"),
        configured_symbols=[_sym()],
        symbol_spec=None,
    )
    assert r is not None and r.reason == "drill_refused_mode_not_demo"


def test_validate_drill_refused_when_dry_run() -> None:
    """Drill is refused when dry_run is True."""
    r = validate_drill(
        mode=RuntimeMode.DEMO,
        dry_run=True,
        symbol=_sym(),
        qty=Decimal("0.001"),
        configured_symbols=[_sym()],
        symbol_spec=None,
    )
    assert r is not None and r.reason == "drill_refused_dry_run"


def test_validate_drill_refused_when_symbol_not_configured() -> None:
    """Drill is refused when symbol is not in configured trading symbols."""
    not_configured = next(s for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT") if s != _sym())
    r = validate_drill(
        mode=RuntimeMode.DEMO,
        dry_run=False,
        symbol=not_configured,
        qty=Decimal("0.001"),
        configured_symbols=[_sym()],
        symbol_spec=None,
    )
    assert r is not None and r.reason == "drill_refused_symbol_not_configured"


def test_validate_drill_refused_when_qty_below_min() -> None:
    """Drill is refused when qty is below symbol min_qty."""
    class FakeSpec:
        min_qty = Decimal("0.01")

    r = validate_drill(
        mode=RuntimeMode.DEMO,
        dry_run=False,
        symbol=_sym(),
        qty=Decimal("0.001"),
        configured_symbols=[_sym()],
        symbol_spec=FakeSpec(),
    )
    assert r is not None and r.reason == "drill_refused_qty_below_min_0.01"


def test_validate_drill_refused_when_notional_exceeds_cap() -> None:
    """Drill is refused when notional exceeds safe cap."""
    r = validate_drill(
        mode=RuntimeMode.DEMO,
        dry_run=False,
        symbol=_sym(),
        qty=Decimal("1"),
        configured_symbols=[_sym()],
        symbol_spec=None,
        reference_price=Decimal("100000"),
        max_drill_notional_usdt=Decimal("10"),
    )
    assert r is not None and r.reason == "drill_refused_notional_exceeds_cap_10"


def test_validate_drill_allowed() -> None:
    """Drill is allowed when all constraints pass."""
    class FakeSpec:
        min_qty = Decimal("0.001")

    assert (
        validate_drill(
            mode=RuntimeMode.DEMO,
            dry_run=False,
            symbol=_sym(),
            qty=Decimal("0.001"),
            configured_symbols=[_sym()],
            symbol_spec=FakeSpec(),
            reference_price=Decimal("50000"),
            max_drill_notional_usdt=Decimal("100"),
        )
        is None
    )


def test_validate_drill_configurable_max_notional() -> None:
    """Drill max notional is configurable; higher cap allows larger orders."""
    class FakeSpec:
        min_qty = Decimal("0.001")

    # Cap 10: 0.001 BTC @ 50000 = 50 USDT → refused
    r = validate_drill(
        mode=RuntimeMode.DEMO,
        dry_run=False,
        symbol=_sym(),
        qty=Decimal("0.001"),
        configured_symbols=[_sym()],
        symbol_spec=FakeSpec(),
        reference_price=Decimal("50000"),
        max_drill_notional_usdt=Decimal("10"),
    )
    assert r is not None and r.reason == "drill_refused_notional_exceeds_cap_10"

    # Cap 100: 0.001 BTC @ 50000 = 50 USDT → allowed
    assert (
        validate_drill(
            mode=RuntimeMode.DEMO,
            dry_run=False,
            symbol=_sym(),
            qty=Decimal("0.001"),
            configured_symbols=[_sym()],
            symbol_spec=FakeSpec(),
            reference_price=Decimal("50000"),
            max_drill_notional_usdt=Decimal("100"),
        )
        is None
    )


def test_validate_drill_conflicting_min_qty_vs_cap() -> None:
    """BTCUSDT: min_qty 0.001, cap 10, ref 50000 → min notional 50 > cap, no valid qty."""
    class FakeSpec:
        min_qty = Decimal("0.001")

    # 0.001 BTC (min) → notional 50 > cap 10 → refused
    r = validate_drill(
        mode=RuntimeMode.DEMO,
        dry_run=False,
        symbol=_sym(),
        qty=Decimal("0.001"),
        configured_symbols=[_sym()],
        symbol_spec=FakeSpec(),
        reference_price=Decimal("50000"),
        max_drill_notional_usdt=Decimal("10"),
    )
    assert r is not None
    assert r.reason == "drill_refused_notional_exceeds_cap_10"
    assert r.details["symbol"] == _sym()
    assert r.details["qty"] == "0.001"
    assert r.details["min_qty"] == "0.001"
    assert r.details["estimated_notional_usdt"] == "50"
    assert r.details["max_notional_usdt"] == "10"

    # 0.0001 BTC → below min_qty → refused
    r2 = validate_drill(
        mode=RuntimeMode.DEMO,
        dry_run=False,
        symbol=_sym(),
        qty=Decimal("0.0001"),
        configured_symbols=[_sym()],
        symbol_spec=FakeSpec(),
        reference_price=Decimal("50000"),
        max_drill_notional_usdt=Decimal("10"),
    )
    assert r2 is not None
    assert r2.reason == "drill_refused_qty_below_min_0.001"


def test_validate_drill_refusal_details_qty_below_min() -> None:
    """Refusal for qty below min includes structured details."""
    class FakeSpec:
        min_qty = Decimal("0.001")

    r = validate_drill(
        mode=RuntimeMode.DEMO,
        dry_run=False,
        symbol=_sym(),
        qty=Decimal("0.0001"),
        configured_symbols=[_sym()],
        symbol_spec=FakeSpec(),
        reference_price=Decimal("50000"),
        max_drill_notional_usdt=Decimal("100"),
    )
    assert r is not None
    assert r.reason == "drill_refused_qty_below_min_0.001"
    assert r.details["symbol"] == _sym()
    assert r.details["qty"] == "0.0001"
    assert r.details["min_qty"] == "0.001"
    assert r.details["max_notional_usdt"] == "100"
    assert r.details["estimated_notional_usdt"] == "5"


def test_validate_drill_refusal_details_notional_exceeds_cap() -> None:
    """Refusal for notional exceeds cap includes structured details."""
    r = validate_drill(
        mode=RuntimeMode.DEMO,
        dry_run=False,
        symbol=_sym(),
        qty=Decimal("0.01"),
        configured_symbols=[_sym()],
        symbol_spec=None,
        reference_price=Decimal("100000"),
        max_drill_notional_usdt=Decimal("500"),
    )
    assert r is not None
    assert r.reason == "drill_refused_notional_exceeds_cap_500"
    assert r.details["symbol"] == _sym()
    assert r.details["qty"] == "0.01"
    assert r.details["estimated_notional_usdt"] == "1000"
    assert r.details["max_notional_usdt"] == "500"


def test_build_drill_intent_post_only() -> None:
    """build_drill_intent produces post-only limit order near bid/ask."""
    config = DrillConfig(symbol=_sym(), side=OrderSide.BUY, qty=Decimal("0.001"), mode=DrillMode.POST_ONLY_LIMIT)
    link_id = generate_drill_order_link_id(_sym())
    now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    intent = build_drill_intent(config=config, reference_price=Decimal("50000"), order_link_id=link_id, now=now)
    assert intent.symbol == _sym()
    assert intent.side == OrderSide.BUY
    assert intent.qty == Decimal("0.001")
    assert intent.reduce_only is False
    assert intent.price is not None
    assert intent.price < Decimal("50000")
    assert intent.metadata.get("drill") is True
    assert link_id.startswith("drill-")


def test_generate_drill_order_link_id() -> None:
    """Drill order link id has drill prefix and fits Bybit length."""
    sym = _sym()
    link_id = generate_drill_order_link_id(sym)
    assert link_id.startswith("drill-")
    compact = sym.replace("USDT", "U").lower()[:7]
    assert compact in link_id.lower()
    assert len(link_id) <= 36


def test_drill_outcome_defaults() -> None:
    """DrillOutcome has correct defaults."""
    o = DrillOutcome()
    assert o.enabled is False
    assert o.attempted is False
    assert o.ack_received is False
    assert o.completed is False
    assert o.aborted is False
    assert o.abort_details is None


def test_drill_outcome_abort_details() -> None:
    """DrillOutcome stores abort_details for structured timeout reporting."""
    o = DrillOutcome()
    o.abort_details = {
        "waited_seconds": 25.0,
        "symbol": _sym(),
        "ws_public_connected": False,
        "ticker_seen": False,
        "trade_seen": False,
    }
    assert o.abort_details["waited_seconds"] == 25.0
    assert o.abort_details["symbol"] == _sym()


def test_drill_post_ack_status_classification() -> None:
    """_drill_post_ack_status classifies outcome for operator visibility."""
    from trading.runtime.orchestrator import _drill_post_ack_status

    assert _drill_post_ack_status(DrillOutcome(ack_received=False)) == "no_ack"
    assert (
        _drill_post_ack_status(
            DrillOutcome(ack_received=True, final_status=None, completed=False)
        )
        == "ack_only_no_transition"
    )
    assert (
        _drill_post_ack_status(
            DrillOutcome(ack_received=True, final_status="New", completed=False)
        )
        == "resting_open"
    )
    assert (
        _drill_post_ack_status(
            DrillOutcome(ack_received=True, final_status="PartiallyFilled", completed=False)
        )
        == "resting_open"
    )
    assert (
        _drill_post_ack_status(
            DrillOutcome(ack_received=True, final_status="Filled", completed=True)
        )
        == "filled"
    )
    assert (
        _drill_post_ack_status(
            DrillOutcome(ack_received=True, final_status="Cancelled", completed=True)
        )
        == "cancelled"
    )
    assert (
        _drill_post_ack_status(
            DrillOutcome(ack_received=True, final_status="Rejected", completed=True)
        )
        == "rejected"
    )
    # Terminal status without completed (e.g. first update is Filled, prev_status was None)
    assert (
        _drill_post_ack_status(
            DrillOutcome(ack_received=True, final_status="Filled", completed=False)
        )
        == "filled"
    )
