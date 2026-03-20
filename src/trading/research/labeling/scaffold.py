"""Scaffold for turning decision/outcome data into training-ready records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from trading.research.datasets.export import DecisionExportRecord
from trading.research.labeling.events import LabelWindow


@dataclass(frozen=True, slots=True)
class DecisionOutcomeRecord:
    """
    Decision linked to execution outcome for labeling.

    Use from_export_record to build from DecisionExportRecord.
    """

    decision_ts: datetime
    symbol: str
    action: str
    reference_price: Decimal | None
    filled: bool
    fill_ts: datetime | None
    fill_price: Decimal | None
    fill_qty: Decimal | None


def from_export_record(r: DecisionExportRecord) -> DecisionOutcomeRecord | None:
    """Build DecisionOutcomeRecord from DecisionExportRecord. Returns None if invalid."""
    def _dec(s: str | None) -> Decimal | None:
        if not s or not s.strip():
            return None
        try:
            return Decimal(s)
        except Exception:
            return None

    ref = _dec(r.reference_price)
    fill_price = _dec(r.fill_price)
    fill_qty = _dec(r.fill_qty)
    return DecisionOutcomeRecord(
        decision_ts=r.ts_utc,
        symbol=r.symbol,
        action=r.action,
        reference_price=ref,
        filled=r.filled,
        fill_ts=r.fill_ts_utc,
        fill_price=fill_price,
        fill_qty=fill_qty,
    )


def to_label_window(
    record: DecisionOutcomeRecord,
    horizon_seconds: int = 3600,
) -> LabelWindow | None:
    """
    Convert a filled decision outcome into a LabelWindow for barrier labeling.

    Scaffold: uses decision_ts as start, fill_ts or decision_ts+horizon as end.
    Uses reference_price when available, else fill_price as fallback.
    Returns None if record has no fill or no usable price.
    """
    if not record.filled or record.fill_ts is None:
        return None
    ref = record.reference_price or record.fill_price
    if ref is None:
        return None
    from datetime import timedelta

    end = record.fill_ts
    start = record.decision_ts
    if end <= start:
        end = start + timedelta(seconds=horizon_seconds)
    return LabelWindow(
        symbol=record.symbol,
        start_time=start,
        end_time=end,
        reference_price=ref,
    )
