from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from trading.execution.order_intent import IntentPurpose, OrderIntent
from trading.strategy.signal_engine import SignalAction, SignalDecision
from trading.util.ids import generate_order_link_id
from trading.util.types import OrderSide, OrderType, TimeInForce, PositionSide


@dataclass(slots=True, frozen=True)
class ExecutionPolicyConfig:
    entry_post_only_offset_bps: Decimal = Decimal("2")


class ExecutionEngine:
    """
    Converts approved signals into deterministic order intents.

    Policy:
    - entries: post-only limits with slight passive offset
    - exits: reduce-only orders (market by default for deterministic risk-off)
    """

    def __init__(self, *, strategy_id: str, config: ExecutionPolicyConfig | None = None) -> None:
        self._strategy_id = strategy_id
        self._cfg = config or ExecutionPolicyConfig()

    def build_entry_intent(
        self,
        *,
        signal: SignalDecision,
        qty: Decimal,
        reference_price: Decimal,
        now: datetime,
    ) -> OrderIntent | None:
        if signal.action not in {SignalAction.ENTER_LONG, SignalAction.ENTER_SHORT}:
            return None
        if signal.side is None or qty <= 0:
            return None
        price = self._entry_price(side=signal.side, reference=reference_price)
        return OrderIntent(
            symbol=signal.symbol,
            side=signal.side,
            qty=qty,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.POST_ONLY,
            reduce_only=False,
            price=price,
            order_link_id=generate_order_link_id(strategy_id=self._strategy_id, symbol=signal.symbol),
            purpose=IntentPurpose.ENTRY,
            created_at=now,
            metadata={
                "signal_action": signal.action.value,
                "reason": signal.reason,
                "post_only": True,
            },
        )

    def build_exit_intent(
        self,
        *,
        symbol: str,
        side_to_close: PositionSide,
        qty: Decimal,
        now: datetime,
        reason: str,
    ) -> OrderIntent | None:
        if qty <= 0 or side_to_close == PositionSide.FLAT:
            return None
        # Close long via sell, close short via buy.
        close_side = OrderSide.SELL if side_to_close == PositionSide.LONG else OrderSide.BUY
        return OrderIntent(
            symbol=symbol,
            side=close_side,
            qty=qty,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.IOC,
            reduce_only=True,
            price=None,
            order_link_id=generate_order_link_id(strategy_id=self._strategy_id, symbol=symbol),
            purpose=IntentPurpose.EXIT,
            created_at=now,
            metadata={"reason": reason, "reduce_only": True},
        )

    def build_stop_intent(
        self,
        *,
        symbol: str,
        side_to_close: PositionSide,
        qty: Decimal,
        stop_price: Decimal,
        now: datetime,
        metadata: dict[str, Any] | None = None,
    ) -> OrderIntent | None:
        """
        Optional stop intent constructor for future order manager policies.

        TODO(phase-5): route this as conditional stop trigger via exchange adapter.
        """
        if qty <= 0 or stop_price <= 0 or side_to_close == PositionSide.FLAT:
            return None
        close_side = OrderSide.SELL if side_to_close == PositionSide.LONG else OrderSide.BUY
        return OrderIntent(
            symbol=symbol,
            side=close_side,
            qty=qty,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.IOC,
            reduce_only=True,
            price=None,
            order_link_id=generate_order_link_id(strategy_id=self._strategy_id, symbol=symbol),
            purpose=IntentPurpose.EXIT,
            created_at=now,
            metadata={"reason": "protective_stop", "stop_price": stop_price, **(metadata or {})},
        )

    def _entry_price(self, *, side: OrderSide, reference: Decimal) -> Decimal:
        offset = (reference * self._cfg.entry_post_only_offset_bps) / Decimal("10000")
        if side == OrderSide.BUY:
            return max(Decimal("0"), reference - offset)
        return reference + offset
