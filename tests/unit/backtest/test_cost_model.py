"""Unit tests for backtest cost model."""

from decimal import Decimal

from trading.backtest.cost_model import CostBreakdown, CostModel, CostModelConfig


def test_cost_model_compute_maker() -> None:
    cfg = CostModelConfig(maker_fee_bps=Decimal("2"), taker_fee_bps=Decimal("6"), spread_bps=Decimal("2"), slippage_bps=Decimal("5"))
    model = CostModel(cfg)
    result = model.compute(
        notional_usdt=Decimal("10000"),
        is_maker=True,
        qty=Decimal("0.1"),
        price=Decimal("100000"),
    )
    assert isinstance(result, CostBreakdown)
    assert result.is_maker is True
    assert result.fee_usdt == Decimal("2")
    assert result.total_usdt > 0


def test_cost_model_compute_taker() -> None:
    cfg = CostModelConfig(maker_fee_bps=Decimal("2"), taker_fee_bps=Decimal("6"))
    model = CostModel(cfg)
    result = model.compute(
        notional_usdt=Decimal("10000"),
        is_maker=False,
        qty=Decimal("0.1"),
        price=Decimal("100000"),
    )
    assert result.is_maker is False
    assert result.fee_usdt == Decimal("6")


def test_cost_model_fee_only() -> None:
    model = CostModel()
    fee = model.compute_fee_only(Decimal("10000"), is_maker=True)
    assert fee == Decimal("2")
