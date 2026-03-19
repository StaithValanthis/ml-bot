from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from trading.journal.ledger import LedgerEvent
from trading.journal.pnl import PnLRecord


@dataclass(frozen=True, slots=True)
class DailyRuntimeReport:
    report_date: date
    decisions: int
    intents: int
    acks: int
    fills: int
    reconcile_issues: int
    latest_equity_usdt: Decimal | None
    latest_realized_pnl_usdt: Decimal | None


def build_daily_report(*, report_date: date, events: list[LedgerEvent], pnl_records: list[PnLRecord]) -> DailyRuntimeReport:
    decisions = sum(1 for event in events if event.event_type == "decision")
    intents = sum(1 for event in events if event.event_type == "order_intent")
    acks = sum(1 for event in events if event.event_type == "order_ack")
    fills = sum(1 for event in events if event.event_type == "fill")
    reconcile_issues = sum(1 for event in events if event.event_type == "reconcile_issues")

    latest_equity = pnl_records[-1].equity_usdt if pnl_records else None
    latest_realized = pnl_records[-1].realized_pnl_usdt if pnl_records else None
    return DailyRuntimeReport(
        report_date=report_date,
        decisions=decisions,
        intents=intents,
        acks=acks,
        fills=fills,
        reconcile_issues=reconcile_issues,
        latest_equity_usdt=latest_equity,
        latest_realized_pnl_usdt=latest_realized,
    )
