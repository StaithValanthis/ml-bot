"""Unit tests for PositionItem schema parsing."""

from __future__ import annotations

from decimal import Decimal

import pytest

from trading.exchange.schemas import PositionItem


def test_position_item_parses_empty_strings_as_zero() -> None:
    """PositionItem tolerates empty-string payloads from Bybit for flat/empty positions."""
    raw = {
        "symbol": "BTCUSDT",
        "side": "",
        "size": "",
        "avgPrice": "",
        "markPrice": "",
        "positionValue": "",
        "leverage": "",
        "liqPrice": "",
        "unrealisedPnl": "",
        "updatedTime": "1710000000000",
    }
    item = PositionItem.model_validate(raw)
    assert item.symbol == "BTCUSDT"
    assert item.side == ""
    assert item.size == Decimal("0")
    assert item.avg_price == Decimal("0")
    assert item.mark_price == Decimal("0")
    assert item.position_value == Decimal("0")
    assert item.leverage == Decimal("0")
    assert item.liq_price == ""
    assert item.unrealised_pnl == Decimal("0")


def test_position_item_parses_mixed_empty_and_valid() -> None:
    """PositionItem parses when some fields are empty and others valid."""
    raw = {
        "symbol": "ETHUSDT",
        "side": "Buy",
        "size": "0",
        "avgPrice": "",
        "markPrice": "3500",
        "positionValue": "",
        "leverage": "10",
        "liqPrice": "",
        "unrealisedPnl": "",
        "updatedTime": "1710000000000",
    }
    item = PositionItem.model_validate(raw)
    assert item.symbol == "ETHUSDT"
    assert item.side == "Buy"
    assert item.size == Decimal("0")
    assert item.avg_price == Decimal("0")
    assert item.mark_price == Decimal("3500")
    assert item.position_value == Decimal("0")
    assert item.leverage == Decimal("10")
    assert item.unrealised_pnl == Decimal("0")


def test_position_item_parses_valid_position() -> None:
    """PositionItem parses normal non-empty position correctly."""
    raw = {
        "symbol": "BTCUSDT",
        "side": "Buy",
        "size": "0.1",
        "avgPrice": "60000",
        "markPrice": "60100",
        "positionValue": "6010",
        "leverage": "10",
        "liqPrice": "54000",
        "unrealisedPnl": "10",
        "updatedTime": "1710000000000",
    }
    item = PositionItem.model_validate(raw)
    assert item.symbol == "BTCUSDT"
    assert item.size == Decimal("0.1")
    assert item.avg_price == Decimal("60000")
    assert item.mark_price == Decimal("60100")
    assert item.position_value == Decimal("6010")
    assert item.leverage == Decimal("10")
    assert item.liq_price == "54000"
    assert item.unrealised_pnl == Decimal("10")
