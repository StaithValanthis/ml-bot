"""Demo execution drill: safe, explicit validation of order lifecycle on DEMO only."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum

from trading.execution.order_intent import IntentPurpose, OrderIntent
from trading.util.types import OrderSide, OrderType, PositionSide, RuntimeMode, TimeInForce


class DrillMode(str, Enum):
    """Price mode for demo drill order."""

    POST_ONLY_LIMIT = "post_only"
    REDUCE_ONLY = "reduce_only"


@dataclass(frozen=True, slots=True)
class DrillConfig:
    """Configuration for a single demo drill execution."""

    symbol: str
    side: OrderSide
    qty: Decimal
    mode: DrillMode


@dataclass(slots=True)
class DrillOutcome:
    """Tracks drill execution outcome for session summary."""

    enabled: bool = False
    attempted: bool = False
    refused_reason: str | None = None
    symbol: str | None = None
    side: str | None = None
    qty: str | None = None
    order_link_id: str | None = None
    ack_received: bool = False
    order_id: str | None = None
    reconcile_mismatch: bool = False
    final_status: str | None = None
    completed: bool = False
    aborted: bool = False


def generate_drill_order_link_id(symbol: str) -> str:
    """Generate order_link_id for drill orders; max 36 chars for Bybit."""
    ts = datetime.now(UTC).strftime("%y%m%d%H%M%S")
    symbol_compact = symbol.replace("USDT", "U").lower()[:7]
    return f"drill-{symbol_compact}-{ts}"[:36]


def validate_drill(
    *,
    mode: RuntimeMode,
    dry_run: bool,
    symbol: str,
    qty: Decimal,
    configured_symbols: list[str],
    symbol_spec: object | None,
    reference_price: Decimal | None = None,
    max_drill_notional_usdt: Decimal = Decimal("10"),
) -> str | None:
    """
    Validate drill parameters. Returns refusal reason or None if allowed.
    """
    if mode != RuntimeMode.DEMO:
        return "drill_refused_mode_not_demo"
    if dry_run:
        return "drill_refused_dry_run"
    if symbol not in configured_symbols:
        return "drill_refused_symbol_not_configured"
    if qty <= 0:
        return "drill_refused_qty_invalid"

    if symbol_spec is not None:
        min_qty = getattr(symbol_spec, "min_qty", None)
        if min_qty is not None and qty < min_qty:
            return f"drill_refused_qty_below_min_{min_qty}"

    if reference_price is not None and reference_price > 0:
        notional = qty * reference_price
        if notional > max_drill_notional_usdt:
            return f"drill_refused_notional_exceeds_cap_{max_drill_notional_usdt}"

    return None


def build_drill_intent(
    *,
    config: DrillConfig,
    reference_price: Decimal | None,
    order_link_id: str,
    now: datetime,
    position_side: PositionSide | None = None,
) -> OrderIntent:
    """Build OrderIntent for drill: post-only limit near bid/ask or reduce-only exit."""
    if config.mode == DrillMode.POST_ONLY_LIMIT:
        if reference_price is None or reference_price <= 0:
            raise ValueError("reference_price required for post_only drill")
        offset_bps = Decimal("2")
        offset = (reference_price * offset_bps) / Decimal("10000")
        price = reference_price - offset if config.side == OrderSide.BUY else reference_price + offset
        return OrderIntent(
            symbol=config.symbol,
            side=config.side,
            qty=config.qty,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.POST_ONLY,
            reduce_only=False,
            price=price,
            order_link_id=order_link_id,
            purpose=IntentPurpose.ENTRY,
            created_at=now,
            metadata={"drill": True, "mode": config.mode.value},
        )
    if config.mode == DrillMode.REDUCE_ONLY:
        if position_side is None or position_side == PositionSide.FLAT:
            raise ValueError("position_side required for reduce_only drill")
        close_side = OrderSide.SELL if position_side == PositionSide.LONG else OrderSide.BUY
        return OrderIntent(
            symbol=config.symbol,
            side=close_side,
            qty=config.qty,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.IOC,
            reduce_only=True,
            price=None,
            order_link_id=order_link_id,
            purpose=IntentPurpose.EXIT,
            created_at=now,
            metadata={"drill": True, "mode": config.mode.value},
        )
    raise ValueError(f"Unsupported drill mode: {config.mode}")
