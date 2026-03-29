"""Unit tests for entry creation guard (position add / pyramiding prevention)."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from trading.execution.order_intent import IntentPurpose, OrderIntent
from trading.runtime.orchestrator import RuntimeOrchestrator
from trading.settings import load_settings
from trading.util.types import OrderSide, OrderType, PositionSide, TimeInForce


def _sym() -> str:
    return load_settings().trading.symbols[0]


def _make_orchestrator() -> RuntimeOrchestrator:
    mock_rest = MagicMock()
    mock_ws_public = MagicMock()
    mock_ws_public.subscribe = MagicMock()
    mock_ws_public.run_forever = MagicMock()
    mock_ws_public.close = MagicMock()
    mock_ws_private = MagicMock()
    mock_ws_private.subscribe = MagicMock()
    mock_ws_private.run_forever = MagicMock()
    mock_ws_private.close = MagicMock()
    with (
        patch("trading.runtime.orchestrator.BybitRestClient", return_value=mock_rest),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", return_value=mock_ws_public),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", return_value=mock_ws_private),
    ):
        return RuntimeOrchestrator(load_settings())


def _mock_order(symbol: str | None = None, reduce_only: bool = False, drill: bool = False, order_link_id: str = "link-1"):
    from trading.execution.order_manager import ManagedOrder
    from datetime import datetime, timezone
    from trading.util.types import OrderStatus

    sym = _sym() if symbol is None else symbol
    return ManagedOrder(
        order_id="ord-1",
        order_link_id=order_link_id,
        symbol=sym,
        status=OrderStatus.NEW,
        qty=Decimal("0.001"),
        filled_qty=Decimal("0"),
        avg_price=None,
        reduce_only=reduce_only,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        metadata={"drill": drill} if drill else {},
    )


def test_blocks_entry_when_symbol_has_non_flat_position() -> None:
    """Block new entry when symbol has non-flat position."""
    orch = _make_orchestrator()
    orch._settings.runtime.allow_position_adds = False
    orch._settings.runtime.max_concurrent_entries_per_symbol = 1
    from trading.risk.portfolio_state import PositionRiskView

    orch._portfolio.positions[_sym()] = PositionRiskView(
        symbol=_sym(),
        side=PositionSide.LONG,
        qty=Decimal("0.01"),
        entry_price=Decimal("60000"),
        mark_price=Decimal("60100"),
        leverage=Decimal("1"),
        liquidation_price=None,
    )

    result = orch._should_block_new_entry(_sym(), [])
    assert result is not None
    blocked, reason, payload = result
    assert blocked is True
    assert reason == "existing_position"
    assert payload["symbol"] == _sym()
    assert payload["current_position_size"] == 0.01
    assert payload["allow_position_adds"] is False


def test_blocks_entry_when_working_entry_order_exists() -> None:
    """Block new entry when working non-reduce-only entry order exists."""
    orch = _make_orchestrator()
    orch._settings.runtime.allow_position_adds = False
    orch._settings.runtime.max_concurrent_entries_per_symbol = 1

    open_orders = [_mock_order(symbol=_sym(), reduce_only=False, drill=False)]
    result = orch._should_block_new_entry(_sym(), open_orders)
    assert result is not None
    blocked, reason, payload = result
    assert blocked is True
    assert reason == "existing_working_entry"
    assert payload["open_entry_order_count"] == 1


def test_allows_entry_when_flat_and_no_working_entry() -> None:
    """Allow entry when flat and no working entry exists."""
    orch = _make_orchestrator()
    orch._settings.runtime.allow_position_adds = False
    orch._settings.runtime.max_concurrent_entries_per_symbol = 1

    blocked, block_reason, payload = orch._should_block_new_entry(_sym(), [])
    assert blocked is False
    assert block_reason is None
    assert payload["symbol"] == _sym()


def test_allows_only_one_concurrent_entry_by_default() -> None:
    """With allow_position_adds=False, one working entry blocks further entries."""
    orch = _make_orchestrator()
    orch._settings.runtime.allow_position_adds = False

    open_orders = [_mock_order(symbol=_sym(), reduce_only=False)]
    result = orch._should_block_new_entry(_sym(), open_orders)
    assert result is not None
    blocked, reason, _ = result
    assert blocked is True
    assert reason in ("existing_working_entry", "existing_position")


def test_flat_with_stale_reduce_only_allows_entry() -> None:
    """When position is flat, stale local reduce-only does NOT block (avoids permanent block)."""
    orch = _make_orchestrator()
    orch._settings.runtime.allow_position_adds = False
    orch._portfolio.positions = {}

    open_orders = [_mock_order(symbol=_sym(), reduce_only=True)]
    blocked, block_reason, payload = orch._should_block_new_entry(_sym(), open_orders)
    assert blocked is False
    assert block_reason is None
    assert payload["current_position_size"] == 0.0
    assert payload["open_reduce_only_order_count"] == 1


def test_respects_allow_position_adds_true_plus_max_concurrent() -> None:
    """When allow_position_adds=True, enforce max_concurrent_entries_per_symbol."""
    orch = _make_orchestrator()
    orch._settings.runtime.allow_position_adds = True
    orch._settings.runtime.max_concurrent_entries_per_symbol = 2

    open_orders = [
        _mock_order(symbol=_sym(), reduce_only=False),
        _mock_order(symbol=_sym(), reduce_only=False, drill=False),
    ]
    open_orders[1].order_link_id = "link-2"
    open_orders[1].order_id = "ord-2"

    result = orch._should_block_new_entry(_sym(), open_orders)
    assert result is not None
    blocked, reason, payload = result
    assert blocked is True
    assert reason == "max_concurrent_entries"
    assert payload["open_entry_order_count"] == 2
    assert payload["max_concurrent_entries_per_symbol"] == 2


def test_allow_position_adds_true_allows_when_under_limit() -> None:
    """When allow_position_adds=True and under limit, allow entry."""
    orch = _make_orchestrator()
    orch._settings.runtime.allow_position_adds = True
    orch._settings.runtime.max_concurrent_entries_per_symbol = 2

    open_orders = [_mock_order(symbol=_sym(), reduce_only=False)]
    blocked, block_reason, payload = orch._should_block_new_entry(_sym(), open_orders)
    assert blocked is False
    assert block_reason is None
    assert payload["open_entry_order_count"] == 1


def test_drill_orders_excluded_from_guard() -> None:
    """Drill orders are not counted as blocking entry orders."""
    orch = _make_orchestrator()
    orch._settings.runtime.allow_position_adds = False

    open_orders = [_mock_order(symbol=_sym(), reduce_only=False, drill=True)]
    blocked, block_reason, payload = orch._should_block_new_entry(_sym(), open_orders)
    assert blocked is False
    assert block_reason is None
    assert payload["open_entry_order_count"] == 0


def test_non_flat_position_with_reduce_only_still_blocks() -> None:
    """Real non-flat position still blocks even when reduce-only exists (managed exit)."""
    orch = _make_orchestrator()
    orch._settings.runtime.allow_position_adds = False
    from trading.risk.portfolio_state import PositionRiskView

    orch._portfolio.positions[_sym()] = PositionRiskView(
        symbol=_sym(),
        side=PositionSide.LONG,
        qty=Decimal("0.01"),
        entry_price=Decimal("60000"),
        mark_price=Decimal("60100"),
        leverage=Decimal("1"),
        liquidation_price=None,
    )

    open_orders = [_mock_order(symbol=_sym(), reduce_only=True)]
    blocked, block_reason, payload = orch._should_block_new_entry(_sym(), open_orders)
    assert blocked is True
    assert block_reason == "existing_position"
    assert payload["current_position_size"] == 0.01
    assert payload["open_reduce_only_order_count"] == 1


def test_entry_blocked_existing_position_log_merge_no_duplicate_keywords() -> None:
    """Regression: guard_payload includes block_reason_bucket; merging once avoids structlog TypeError."""
    orch = _make_orchestrator()
    orch._settings.runtime.allow_position_adds = False
    from trading.risk.portfolio_state import PositionRiskView

    orch._portfolio.positions[_sym()] = PositionRiskView(
        symbol=_sym(),
        side=PositionSide.LONG,
        qty=Decimal("0.01"),
        entry_price=Decimal("60000"),
        mark_price=Decimal("60100"),
        leverage=Decimal("1"),
        liquidation_price=None,
    )
    blocked, block_reason, guard_payload = orch._should_block_new_entry(_sym(), [])
    assert blocked is True
    assert block_reason == "existing_position"
    assert guard_payload.get("block_reason_bucket") == "existing_position"
    block_bucket = guard_payload.get("block_reason_bucket", block_reason or "unknown")
    # Exact shape used by _decision_loop after fix (single ** expansion).
    _guard_block_log = {**guard_payload, "block_reason_bucket": block_bucket}
    orch._logger.info("entry_blocked_existing_position", **_guard_block_log)
    # Proves the pre-fix call form is invalid Python / structlog duplicate-kwarg territory.
    with pytest.raises(TypeError, match="multiple values"):
        orch._logger.info(
            "entry_blocked_existing_position",
            **guard_payload,
            block_reason_bucket=block_bucket,
        )


def test_entry_blocked_working_entry_and_max_concurrent_log_merge_no_duplicate_keywords() -> None:
    """Same structlog hazard for entry_blocked_existing_working_entry / max_concurrent_entries."""
    orch = _make_orchestrator()
    orch._settings.runtime.allow_position_adds = False
    orch._settings.runtime.max_concurrent_entries_per_symbol = 1
    open_working = [
        _mock_order(symbol=_sym(), reduce_only=False),
    ]
    blocked_w, reason_w, payload_w = orch._should_block_new_entry(_sym(), open_working)
    assert blocked_w and reason_w == "existing_working_entry"
    bucket_w = payload_w.get("block_reason_bucket", reason_w or "unknown")
    log_w = {**payload_w, "block_reason_bucket": bucket_w}
    orch._logger.info("entry_blocked_existing_working_entry", **log_w)

    orch2 = _make_orchestrator()
    orch2._settings.runtime.allow_position_adds = True
    orch2._settings.runtime.max_concurrent_entries_per_symbol = 1
    two_entries = [
        _mock_order(symbol=_sym(), reduce_only=False, order_link_id="link-a"),
        _mock_order(symbol=_sym(), reduce_only=False, order_link_id="link-b"),
    ]
    blocked_m, reason_m, payload_m = orch2._should_block_new_entry(_sym(), two_entries)
    assert blocked_m and reason_m == "max_concurrent_entries"
    bucket_m = payload_m.get("block_reason_bucket", reason_m or "unknown")
    log_m = {**payload_m, "block_reason_bucket": bucket_m}
    orch2._logger.info("entry_blocked_max_concurrent_entries", **log_m)


def test_effectively_flat_dust_position_does_not_block_existing_position() -> None:
    """Dust positions below qty-step threshold are treated as flat for guard checks."""
    orch = _make_orchestrator()
    orch._settings.runtime.allow_position_adds = False
    from trading.risk.portfolio_state import PositionRiskView

    orch._portfolio.positions[_sym()] = PositionRiskView(
        symbol=_sym(),
        side=PositionSide.LONG,
        qty=Decimal("0.0004"),
        entry_price=Decimal("60000"),
        mark_price=Decimal("60100"),
        leverage=Decimal("1"),
        liquidation_price=None,
    )

    blocked, block_reason, payload = orch._should_block_new_entry(_sym(), [])
    assert blocked is False
    assert block_reason is None
    assert payload["position_effectively_flat"] is True
    step = orch._settings.symbols[_sym()].qty_step
    assert payload["effective_flat_threshold_qty"] == float(step / Decimal("2"))


def test_defaults_without_env_vars_block_repeated_adds(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no TRADING_ALLOW_POSITION_ADDS or TRADING_MAX_CONCURRENT_ENTRIES_PER_SYMBOL, defaults are conservative."""
    monkeypatch.delenv("TRADING_ALLOW_POSITION_ADDS", raising=False)
    monkeypatch.delenv("TRADING_MAX_CONCURRENT_ENTRIES_PER_SYMBOL", raising=False)
    settings = load_settings()
    assert settings.runtime.allow_position_adds is False
    assert settings.runtime.max_concurrent_entries_per_symbol == 1


def test_guard_diagnostic_payload_has_required_fields() -> None:
    """Guard payload includes symbol, allow_position_adds, open_entry_count, blocked, block_reason."""
    orch = _make_orchestrator()
    orch._settings.runtime.allow_position_adds = False
    orch._settings.runtime.max_concurrent_entries_per_symbol = 1

    blocked, block_reason, payload = orch._should_block_new_entry(_sym(), [])
    assert "symbol" in payload
    assert payload["symbol"] == _sym()
    assert "allow_position_adds" in payload
    assert "max_concurrent_entries_per_symbol" in payload
    assert "current_position_size" in payload
    assert "current_position_side" in payload
    assert "open_entry_order_count" in payload
    assert "open_reduce_only_order_count" in payload
    assert blocked is False
    assert block_reason is None

    open_orders = [_mock_order(symbol=_sym(), reduce_only=False)]
    blocked2, block_reason2, payload2 = orch._should_block_new_entry(_sym(), open_orders)
    assert blocked2 is True
    assert block_reason2 == "existing_working_entry"
    assert payload2["open_entry_order_count"] == 1


@pytest.mark.asyncio
async def test_session_summary_includes_entry_guard_counts() -> None:
    """Session summary includes position_add_blocked_count and working_entry_blocked_count when non-zero."""
    orch = _make_orchestrator()
    orch._metrics.inc("position_add_blocked_count", 3)
    orch._metrics.inc("working_entry_blocked_count", 2)

    summary = await orch._build_session_summary()
    assert summary.get("position_add_blocked_count") == 3
    assert summary.get("working_entry_blocked_count") == 2


@pytest.mark.asyncio
async def test_session_summary_includes_entry_guard_config_and_by_symbol() -> None:
    """Session summary includes entry_guard and entry_guard_by_symbol for visibility."""
    orch = _make_orchestrator()

    summary = await orch._build_session_summary()
    entry_guard = summary.get("entry_guard")
    assert entry_guard is not None
    assert "allow_position_adds" in entry_guard
    assert "max_concurrent_entries_per_symbol" in entry_guard
    assert entry_guard.get("entry_guard_enabled") is True
    by_symbol = summary.get("entry_guard_by_symbol")
    assert by_symbol is not None
    for sym in orch._settings.trading.symbols:
        assert sym in by_symbol
        sym_info = by_symbol[sym]
        assert "open_entry_count" in sym_info
        assert "open_reduce_only_count" in sym_info
        assert "position_size" in sym_info
        assert "position_side" in sym_info


def test_guard_payload_has_local_tracked_order_count() -> None:
    """Guard payload includes local_tracked_order_count for diagnostics."""
    orch = _make_orchestrator()
    orch._settings.runtime.allow_position_adds = False

    _, _, payload = orch._should_block_new_entry(_sym(), [])
    assert "local_tracked_order_count" in payload
    assert payload["local_tracked_order_count"] == 0

    open_orders = [_mock_order(symbol=_sym(), reduce_only=True)]
    _, _, payload2 = orch._should_block_new_entry(_sym(), open_orders)
    assert payload2["local_tracked_order_count"] == 1


@pytest.mark.asyncio
async def test_session_summary_includes_entry_guard_block_reasons() -> None:
    """Session summary includes entry_guard_block_reasons aggregation for soak diagnostics."""
    orch = _make_orchestrator()
    syms = orch._settings.trading.symbols
    s0, s1 = syms[0], syms[1] if len(syms) > 1 else syms[0]
    orch._entry_guard_block_reasons = [
        {"symbol": s0, "block_reason": "existing_position", "block_reason_bucket": "existing_position"},
        {"symbol": s0, "block_reason": "existing_position", "block_reason_bucket": "existing_position"},
        {"symbol": s1, "block_reason": "existing_working_entry", "block_reason_bucket": "existing_working_entry"},
    ]

    summary = await orch._build_session_summary()
    block_reasons = summary.get("entry_guard_block_reasons")
    assert block_reasons is not None
    assert block_reasons.get("by_type", {}).get("existing_position") == 2
    assert block_reasons.get("by_type", {}).get("existing_working_entry") == 1
    by_sym = block_reasons.get("by_symbol") or {}
    assert by_sym.get(s0, {}).get("existing_position") == 2
    assert by_sym.get(s1, {}).get("existing_working_entry") == 1
    assert len(block_reasons.get("recent_context", [])) == 3
