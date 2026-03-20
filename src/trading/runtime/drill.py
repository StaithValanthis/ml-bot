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
    abort_details: dict[str, object] | None = None
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


def _fmt_decimal(d: Decimal) -> str:
    """Format Decimal for refusal details; avoid scientific notation, strip trailing zeros."""
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s else "0"


@dataclass(frozen=True, slots=True)
class DrillRefusal:
    """Structured refusal result from validate_drill."""

    reason: str
    details: dict[str, object]


def validate_drill(
    *,
    mode: RuntimeMode,
    dry_run: bool,
    symbol: str,
    qty: Decimal,
    configured_symbols: list[str],
    symbol_spec: object | None,
    reference_price: Decimal | None = None,
    max_drill_notional_usdt: Decimal = Decimal("100"),
) -> DrillRefusal | None:
    """
    Validate drill parameters. Returns DrillRefusal with reason and details if refused, None if allowed.
    """
    if mode != RuntimeMode.DEMO:
        return DrillRefusal("drill_refused_mode_not_demo", {"symbol": symbol, "qty": str(qty)})
    if dry_run:
        return DrillRefusal("drill_refused_dry_run", {"symbol": symbol, "qty": str(qty)})
    if symbol not in configured_symbols:
        return DrillRefusal("drill_refused_symbol_not_configured", {"symbol": symbol, "qty": str(qty)})
    if qty <= 0:
        return DrillRefusal("drill_refused_qty_invalid", {"symbol": symbol, "qty": str(qty)})

    min_qty: Decimal | None = getattr(symbol_spec, "min_qty", None) if symbol_spec is not None else None
    estimated_notional_usdt: Decimal | None = None
    if reference_price is not None and reference_price > 0:
        estimated_notional_usdt = qty * reference_price

    if min_qty is not None and qty < min_qty:
        details: dict[str, object] = {
            "symbol": symbol,
            "qty": _fmt_decimal(qty),
            "min_qty": _fmt_decimal(min_qty),
            "max_notional_usdt": _fmt_decimal(max_drill_notional_usdt),
        }
        if estimated_notional_usdt is not None:
            details["estimated_notional_usdt"] = _fmt_decimal(estimated_notional_usdt)
        return DrillRefusal(f"drill_refused_qty_below_min_{min_qty}", details)

    if estimated_notional_usdt is not None and estimated_notional_usdt > max_drill_notional_usdt:
        details = {
            "symbol": symbol,
            "qty": _fmt_decimal(qty),
            "min_qty": _fmt_decimal(min_qty) if min_qty is not None else None,
            "estimated_notional_usdt": _fmt_decimal(estimated_notional_usdt),
            "max_notional_usdt": _fmt_decimal(max_drill_notional_usdt),
        }
        return DrillRefusal(
            f"drill_refused_notional_exceeds_cap_{max_drill_notional_usdt}",
            details,
        )

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
