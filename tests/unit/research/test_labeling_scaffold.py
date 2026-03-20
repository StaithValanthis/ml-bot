"""Unit tests for labeling scaffold (DecisionOutcomeRecord, to_label_window)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from trading.research.datasets.export import DecisionExportRecord
from trading.research.labeling.scaffold import (
    DecisionOutcomeRecord,
    from_export_record,
    to_label_window,
)


def test_from_export_record_builds_outcome() -> None:
    """from_export_record converts DecisionExportRecord to DecisionOutcomeRecord."""
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    r = DecisionExportRecord(
        ts_utc=base,
        symbol="BTCUSDT",
        action="entry_long",
        side="Buy",
        qty="0.001",
        reference_price="40000",
        order_link_id=None,
        filled=True,
        fill_ts_utc=base,
        fill_qty="0.001",
        fill_price="40100",
        risk_approved=True,
        risk_reason=None,
    )
    out = from_export_record(r)
    assert out is not None
    assert isinstance(out, DecisionOutcomeRecord)
    assert out.symbol == "BTCUSDT"
    assert out.reference_price == Decimal("40000")
    assert out.fill_price == Decimal("40100")
    assert out.filled


def test_to_label_window_returns_none_when_not_filled() -> None:
    """to_label_window returns None for unfilled records."""
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    rec = DecisionOutcomeRecord(
        decision_ts=base,
        symbol="BTCUSDT",
        action="entry_long",
        reference_price=Decimal("40000"),
        filled=False,
        fill_ts=None,
        fill_price=None,
        fill_qty=None,
    )
    assert to_label_window(rec) is None


def test_to_label_window_produces_label_window() -> None:
    """to_label_window produces LabelWindow for filled records with price."""
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    fill_ts = datetime(2024, 1, 1, 12, 30, 0, tzinfo=UTC)
    rec = DecisionOutcomeRecord(
        decision_ts=base,
        symbol="BTCUSDT",
        action="entry_long",
        reference_price=Decimal("40000"),
        filled=True,
        fill_ts=fill_ts,
        fill_price=Decimal("40100"),
        fill_qty=Decimal("0.001"),
    )
    w = to_label_window(rec)
    assert w is not None
    assert w.symbol == "BTCUSDT"
    assert w.start_time == base
    assert w.end_time == fill_ts
    assert w.reference_price == Decimal("40000")


def test_to_label_window_uses_fill_price_fallback() -> None:
    """to_label_window uses fill_price when reference_price is None."""
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    rec = DecisionOutcomeRecord(
        decision_ts=base,
        symbol="BTCUSDT",
        action="entry_long",
        reference_price=None,
        filled=True,
        fill_ts=base,
        fill_price=Decimal("40100"),
        fill_qty=Decimal("0.001"),
    )
    w = to_label_window(rec)
    assert w is not None
    assert w.reference_price == Decimal("40100")
