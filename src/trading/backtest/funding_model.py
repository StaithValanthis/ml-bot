"""Funding accrual model for simulated perpetual positions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from trading.util.types import PositionSide


@dataclass(frozen=True, slots=True)
class FundingModelConfig:
    """Configuration for funding accrual."""

    default_rate_bps: Decimal = Decimal("1")


@dataclass(frozen=True, slots=True)
class FundingAccrual:
    symbol: str
    side: PositionSide
    notional_usdt: Decimal
    rate_bps: Decimal
    accrual_usdt: Decimal
    accrued_at: datetime


class FundingModel:
    """
    Simple funding accrual model for backtest.

    Accrual = notional * rate_bps / 10000, with sign by position side.
    Applied per event; no time-weighting or interval scaling.

    Limitations:
    - Uses fixed or passed funding rate; no historical rate lookup.
    - No cross-asset or term-structure modeling.
    """

    def __init__(self, config: FundingModelConfig | None = None) -> None:
        self._cfg = config or FundingModelConfig()

    def compute(
        self,
        *,
        symbol: str,
        side: PositionSide,
        notional_usdt: Decimal,
        rate_bps: Decimal | None = None,
        accrued_at: datetime,
    ) -> FundingAccrual:
        rate = rate_bps if rate_bps is not None else self._cfg.default_rate_bps
        accrual_raw = (notional_usdt * rate) / Decimal("10000")
        if side == PositionSide.SHORT:
            accrual_raw = -accrual_raw
        return FundingAccrual(
            symbol=symbol,
            side=side,
            notional_usdt=notional_usdt,
            rate_bps=rate,
            accrual_usdt=accrual_raw,
            accrued_at=accrued_at,
        )
