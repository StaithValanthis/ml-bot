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


# --- DEMO min-notional floor tests ---


def test_demo_floor_applied_when_qty_below_min() -> None:
    """With demo_min_notional_floor_usdt, floor is applied when stepped_qty < min_qty."""
    sizer = VolatilityAwareSizer(demo_min_notional_floor_usdt=Decimal("75"))
    inputs = SizingInputs(
        equity_usdt=Decimal("1000"),
        confidence=Decimal("0.5"),
        volatility_bps=Decimal("150"),
        reference_price=Decimal("69850"),
        max_leverage=Decimal("3"),
    )
    symbol_info = _market_symbol(min_qty=Decimal("0.001"), qty_step=Decimal("0.001"))
    qty = sizer.size_qty(inputs, symbol_info)
    assert qty >= symbol_info.min_qty
    assert qty == Decimal("0.001")
    assert sizer._last_floor_applied is True
    assert sizer._last_floor_details is not None
    assert sizer._last_floor_details["effective_notional"] >= 75
    assert sizer._last_floor_details["qty"] == 0.001


def test_no_floor_when_sizer_has_no_demo_floor() -> None:
    """Without demo_min_notional_floor_usdt, qty below min is rejected (PAPER/LIVE behavior)."""
    sizer = VolatilityAwareSizer()
    inputs = SizingInputs(
        equity_usdt=Decimal("1000"),
        confidence=Decimal("0.5"),
        volatility_bps=Decimal("150"),
        reference_price=Decimal("69850"),
        max_leverage=Decimal("3"),
    )
    symbol_info = _market_symbol(min_qty=Decimal("0.001"), qty_step=Decimal("0.001"))
    qty = sizer.size_qty(inputs, symbol_info)
    assert qty == 0
    assert sizer.reject_reason(inputs, symbol_info) == "qty_below_min_after_rounding"
    assert sizer._last_floor_applied is False


def test_floor_not_applied_when_computed_qty_already_meets_min() -> None:
    """Floor is not applied when computed qty already >= min_qty."""
    sizer = VolatilityAwareSizer(demo_min_notional_floor_usdt=Decimal("75"))
    inputs = SizingInputs(
        equity_usdt=Decimal("10000"),
        confidence=Decimal("0.5"),
        volatility_bps=Decimal("100"),
        reference_price=Decimal("50000"),
        max_leverage=Decimal("3"),
    )
    symbol_info = _market_symbol(min_qty=Decimal("0.001"), qty_step=Decimal("0.001"))
    qty = sizer.size_qty(inputs, symbol_info)
    assert qty > 0
    assert qty >= symbol_info.min_qty
    assert sizer._last_floor_applied is False
    assert sizer._last_floor_details is None


def test_demo_floor_capped_by_equity_fraction() -> None:
    """Floor is capped at max_equity_fraction_for_floor of equity."""
    sizer = VolatilityAwareSizer(
        demo_min_notional_floor_usdt=Decimal("500"),
        max_equity_fraction_for_floor=Decimal("0.1"),
    )
    inputs = SizingInputs(
        equity_usdt=Decimal("1000"),
        confidence=Decimal("0.5"),
        volatility_bps=Decimal("200"),
        reference_price=Decimal("50000"),
        max_leverage=Decimal("3"),
    )
    symbol_info = _market_symbol(min_qty=Decimal("0.001"), qty_step=Decimal("0.001"))
    qty = sizer.size_qty(inputs, symbol_info)
    assert qty >= symbol_info.min_qty
    assert sizer._last_floor_applied is True
    effective = sizer._last_floor_details["effective_notional"]
    assert effective <= 100
    assert effective == 100


@pytest.mark.asyncio
async def test_session_summary_includes_last_sizing_floor_applied() -> None:
    """Session summary includes last_sizing_floor_applied when floor was applied."""
    from trading.runtime.orchestrator import RuntimeOrchestrator
    from trading.settings import load_settings

    settings = load_settings()
    with (
        patch("trading.runtime.orchestrator.BybitRestClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", MagicMock()),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", MagicMock()),
    ):
        orch = RuntimeOrchestrator(settings)
        orch._last_sizing_floor_applied = {
            "symbol": "BTCUSDT",
            "original_notional": 8.5,
            "effective_notional": 75.0,
            "qty": 0.001,
        }
        summary = await orch._build_session_summary()
        md = orch._build_markdown_summary(summary)

    assert "last_sizing_floor_applied" in summary
    lsf = summary["last_sizing_floor_applied"]
    assert lsf["symbol"] == "BTCUSDT"
    assert lsf["effective_notional"] == 75.0
    assert lsf["qty"] == 0.001
    assert "## Last Sizing Floor Applied" in md
    assert "BTCUSDT" in md


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
