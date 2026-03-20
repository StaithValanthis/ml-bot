"""Typed export of decision/outcome data for research and ML preparation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from trading.journal.ledger import LedgerEvent


@dataclass(frozen=True, slots=True)
class DecisionExportRecord:
    """
    Single decision with candidate/signal/risk/execution outcome for research export.

    Extracted from ledger events; use extract_decision_records to build from events.
    """

    ts_utc: datetime
    symbol: str
    action: str
    side: str | None
    qty: str
    reference_price: str | None
    order_link_id: str | None
    filled: bool
    fill_ts_utc: datetime | None
    fill_qty: str | None
    fill_price: str | None
    risk_approved: bool
    risk_reason: str | None


def extract_decision_records(events: list[LedgerEvent]) -> list[DecisionExportRecord]:
    """
    Extract typed decision records from ledger events for research export.

    Pairs decision + order_intent + fill where available.
    - Runtime: links via order_link_id when present.
    - Backtest: pairs by sequential order (decision -> intent -> fill per symbol).
    Honest scaffold: risk_approved inferred from presence of order_intent after decision.
    """
    decisions: list[tuple[LedgerEvent, dict]] = []
    intents: list[tuple[LedgerEvent, dict]] = []
    fills: list[tuple[LedgerEvent, dict]] = []

    for evt in events:
        if evt.event_type == "decision":
            decisions.append((evt, evt.payload))
        elif evt.event_type == "order_intent":
            intents.append((evt, evt.payload))
        elif evt.event_type == "fill":
            fills.append((evt, evt.payload))

    fills_by_link: dict[str, tuple[LedgerEvent, dict]] = {}
    for evt, p in fills:
        link = p.get("order_link_id") or ""
        if link:
            fills_by_link[link] = (evt, p)

    used_intent_idxs: set[int] = set()
    used_fill_idxs: set[int] = set()

    records: list[DecisionExportRecord] = []
    for devt, dp in decisions:
        symbol = dp.get("symbol", "")
        action = dp.get("action", "")

        intent_evt, intent_p = None, None
        for idx, (ievt, ip) in enumerate(intents):
            if idx in used_intent_idxs:
                continue
            if ip.get("symbol") != symbol:
                continue
            if ievt.timestamp < devt.timestamp:
                continue
            intent_evt, intent_p = ievt, ip
            used_intent_idxs.add(idx)
            break
        if intent_evt is None:
            for idx, (ievt, ip) in enumerate(intents):
                if idx in used_intent_idxs:
                    continue
                if ip.get("symbol") == symbol:
                    intent_evt, intent_p = ievt, ip
                    used_intent_idxs.add(idx)
                    break

        fill_evt, fill_p = None, None
        if intent_p:
            link = intent_p.get("order_link_id")
            if link and link in fills_by_link:
                fill_evt, fill_p = fills_by_link[link]
            elif not link:
                for idx, (fevt, fp) in enumerate(fills):
                    if idx in used_fill_idxs:
                        continue
                    if fp.get("symbol") != symbol:
                        continue
                    min_ts = intent_evt.timestamp if intent_evt else devt.timestamp
                    if fevt.timestamp < min_ts:
                        continue
                    fill_evt, fill_p = fevt, fp
                    used_fill_idxs.add(idx)
                    break

        risk_approved = intent_p is not None
        risk_reason = None if risk_approved else "no_intent_recorded"
        fill_ts = fill_evt.timestamp if fill_evt else None
        fill_qty = str(fill_p.get("exec_qty", fill_p.get("qty", ""))) if fill_p else None
        fill_price = str(fill_p.get("exec_price", "")) if fill_p else None

        records.append(
            DecisionExportRecord(
                ts_utc=devt.timestamp,
                symbol=symbol,
                action=action,
                side=intent_p.get("side") if intent_p else None,
                qty=intent_p.get("qty", "") if intent_p else "",
                reference_price=None,
                order_link_id=intent_p.get("order_link_id") if intent_p else None,
                filled=fill_p is not None,
                fill_ts_utc=fill_ts,
                fill_qty=fill_qty,
                fill_price=fill_price,
                risk_approved=risk_approved,
                risk_reason=risk_reason,
            )
        )

    return records


def export_records_to_json(records: list[DecisionExportRecord], path: Path) -> None:
    """Write decision export records to JSON for research consumption."""
    import json

    from trading.util.json_util import dumps_json_safe

    rows = [
        {
            "ts_utc": r.ts_utc.isoformat(),
            "symbol": r.symbol,
            "action": r.action,
            "side": r.side,
            "qty": r.qty,
            "reference_price": r.reference_price,
            "order_link_id": r.order_link_id,
            "filled": r.filled,
            "fill_ts_utc": r.fill_ts_utc.isoformat() if r.fill_ts_utc else None,
            "fill_qty": r.fill_qty,
            "fill_price": r.fill_price,
            "risk_approved": r.risk_approved,
            "risk_reason": r.risk_reason,
        }
        for r in records
    ]
    path.write_text(dumps_json_safe({"records": rows, "count": len(rows)}, indent=2), encoding="utf-8")
