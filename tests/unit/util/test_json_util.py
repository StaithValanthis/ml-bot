"""Unit tests for JSON-safe serialization."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading.util.json_util import _json_default, dumps_json_safe, json_safe


def test_json_default_decimal() -> None:
    """_json_default converts Decimal to string."""
    assert _json_default(Decimal("0.001")) == "0.001"
    assert _json_default(Decimal("100")) == "100"


def test_json_default_datetime() -> None:
    """_json_default converts datetime to isoformat."""
    dt = datetime(2026, 3, 20, 10, 0, 0, tzinfo=UTC)
    assert _json_default(dt) == "2026-03-20T10:00:00+00:00"


def test_dumps_json_safe_with_decimal_payload() -> None:
    """dumps_json_safe serializes payloads containing Decimal without crash."""
    payload = {
        "symbol": "BTCUSDT",
        "qty": Decimal("0.001"),
        "price": Decimal("50000"),
        "notional": Decimal("50"),
    }
    result = dumps_json_safe(payload)
    parsed = json.loads(result)
    assert parsed["symbol"] == "BTCUSDT"
    assert parsed["qty"] == "0.001"
    assert parsed["price"] == "50000"
    assert parsed["notional"] == "50"


def test_dumps_json_safe_with_nested_decimal() -> None:
    """dumps_json_safe handles nested Decimal in dict/list."""
    payload = {"drill_abort_details": {"qty": Decimal("0.001"), "min_qty": Decimal("0.001")}}
    result = dumps_json_safe(payload)
    parsed = json.loads(result)
    assert parsed["drill_abort_details"]["qty"] == "0.001"
    assert parsed["drill_abort_details"]["min_qty"] == "0.001"


def test_json_safe_recursive() -> None:
    """json_safe recursively normalizes dict/list."""
    obj = {"a": Decimal("1.5"), "b": [Decimal("2"), {"c": Decimal("3")}]}
    result = json_safe(obj)
    assert result == {"a": "1.5", "b": ["2", {"c": "3"}]}


def test_drill_payload_with_decimal_serializes() -> None:
    """Simulated drill ledger payload with Decimal serializes without error."""
    payload = {
        "symbol": "BTCUSDT",
        "side": "Buy",
        "qty": Decimal("0.001"),
        "order_link_id": "drill-btcu-240320100000",
        "estimated_notional_usdt": Decimal("50"),
        "max_notional_usdt": Decimal("100"),
    }
    result = dumps_json_safe(payload)
    assert "Object of type Decimal is not JSON serializable" not in result
    parsed = json.loads(result)
    assert parsed["qty"] == "0.001"
    assert parsed["estimated_notional_usdt"] == "50"
