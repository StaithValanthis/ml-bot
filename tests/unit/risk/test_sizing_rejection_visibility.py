"""Unit tests for sizing rejection visibility."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from trading.risk.sizing import SizingInputs, VolatilityAwareSizer
from trading.util.types import MarketSymbol


def _market_symbol(
    qty_step: Decimal = Decimal("0.001"),
    min_qty: Decimal = Decimal("0.001"),
    price_tick: Decimal = Decimal("0.10"),
    max_leverage: Decimal = Decimal("100"),
) -> MarketSymbol:
    return MarketSymbol(
        symbol="BTCUSDT",
        qty_step=qty_step,
        min_qty=min_qty,
        price_tick=price_tick,
        max_leverage=max_leverage,
    )


def test_reject_reason_returns_none_when_qty_positive() -> None:
    """reject_reason returns None when size_qty would return positive qty."""
    sizer = VolatilityAwareSizer()
    inputs = SizingInputs(
        equity_usdt=Decimal("10000"),
        confidence=Decimal("0.5"),
        volatility_bps=Decimal("100"),
        reference_price=Decimal("50000"),
        max_leverage=Decimal("3"),
    )
    symbol_info = _market_symbol()
    assert sizer.reject_reason(inputs, symbol_info) is None
    assert sizer.size_qty(inputs, symbol_info) > 0


def test_reject_reason_reference_price_zero() -> None:
    """reject_reason returns reference_price_or_equity_zero when reference_price <= 0."""
    sizer = VolatilityAwareSizer()
    inputs = SizingInputs(
        equity_usdt=Decimal("10000"),
        confidence=Decimal("0.5"),
        volatility_bps=Decimal("100"),
        reference_price=Decimal("0"),
        max_leverage=Decimal("3"),
    )
    symbol_info = _market_symbol()
    assert sizer.reject_reason(inputs, symbol_info) == "reference_price_or_equity_zero"
    assert sizer.size_qty(inputs, symbol_info) == 0


def test_reject_reason_equity_zero() -> None:
    """reject_reason returns reference_price_or_equity_zero when equity_usdt <= 0."""
    sizer = VolatilityAwareSizer()
    inputs = SizingInputs(
        equity_usdt=Decimal("0"),
        confidence=Decimal("0.5"),
        volatility_bps=Decimal("100"),
        reference_price=Decimal("50000"),
        max_leverage=Decimal("3"),
    )
    symbol_info = _market_symbol()
    assert sizer.reject_reason(inputs, symbol_info) == "reference_price_or_equity_zero"


def test_reject_reason_confidence_below_min() -> None:
    """reject_reason returns confidence_below_min when confidence < 0.2."""
    sizer = VolatilityAwareSizer()
    inputs = SizingInputs(
        equity_usdt=Decimal("10000"),
        confidence=Decimal("0.1"),
        volatility_bps=Decimal("100"),
        reference_price=Decimal("50000"),
        max_leverage=Decimal("3"),
    )
    symbol_info = _market_symbol()
    assert sizer.reject_reason(inputs, symbol_info) == "confidence_below_min"
    assert sizer.size_qty(inputs, symbol_info) == 0


def test_reject_reason_qty_below_min_after_rounding() -> None:
    """reject_reason returns qty_below_min_after_rounding when stepped_qty < min_qty."""
    sizer = VolatilityAwareSizer()
    inputs = SizingInputs(
        equity_usdt=Decimal("1"),
        confidence=Decimal("0.5"),
        volatility_bps=Decimal("10000"),
        reference_price=Decimal("100000"),
        max_leverage=Decimal("3"),
    )
    symbol_info = _market_symbol(min_qty=Decimal("0.001"), qty_step=Decimal("0.001"))
    assert sizer.reject_reason(inputs, symbol_info) == "qty_below_min_after_rounding"
    assert sizer.size_qty(inputs, symbol_info) == 0


@pytest.mark.asyncio
async def test_session_summary_includes_last_sizing_rejection() -> None:
    """Session summary includes last_sizing_rejection when sizing rejected a candidate."""
    from trading.runtime.orchestrator import RuntimeOrchestrator
    from trading.settings import load_settings

    settings = load_settings()
    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)
        orch._last_sizing_rejection = {
            "symbol": "BTCUSDT",
            "reason": "qty_below_min_after_rounding",
            "equity_usdt": 100.0,
            "confidence": 0.55,
            "volatility_bps": 150.0,
            "reference_price": 95000.0,
            "min_qty": 0.001,
            "qty_step": 0.001,
        }
        summary = await orch._build_session_summary()

    assert "last_sizing_rejection" in summary
    lsr = summary["last_sizing_rejection"]
    assert lsr["symbol"] == "BTCUSDT"
    assert lsr["reason"] == "qty_below_min_after_rounding"
    assert lsr["equity_usdt"] == 100.0
    assert lsr["reference_price"] == 95000.0


@pytest.mark.asyncio
async def test_markdown_summary_includes_sizing_rejection_section() -> None:
    """Markdown summary includes Last Sizing Rejection section when present."""
    from trading.runtime.orchestrator import RuntimeOrchestrator
    from trading.settings import load_settings

    settings = load_settings()
    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)
        orch._last_sizing_rejection = {
            "symbol": "ETHUSDT",
            "reason": "confidence_below_min",
            "equity_usdt": 5000.0,
            "confidence": 0.15,
            "volatility_bps": 80.0,
            "reference_price": 3500.0,
            "min_qty": 0.01,
            "qty_step": 0.01,
        }
        summary = await orch._build_session_summary()
        md = orch._build_markdown_summary(summary)

    assert "## Last Sizing Rejection" in md
    assert "ETHUSDT" in md
    assert "confidence_below_min" in md
