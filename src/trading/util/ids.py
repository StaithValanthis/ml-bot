from __future__ import annotations

import secrets
from datetime import UTC, datetime


def generate_order_link_id(*, strategy_id: str, symbol: str) -> str:
    """
    Generate deterministic-format idempotency key for exchange order placement.

    Bybit orderLinkId max length is 36. Keep this compact and traceable.
    """
    ts = datetime.now(UTC).strftime("%y%m%d%H%M%S")
    token = secrets.token_hex(4)
    strategy = strategy_id[:6].lower()
    symbol_compact = symbol.replace("USDT", "U").lower()[:7]
    return f"{strategy}-{symbol_compact}-{ts}-{token}"[:36]
