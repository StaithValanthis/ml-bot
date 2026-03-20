"""Unit tests for dataset export structures."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from trading.journal.ledger import LedgerEvent
from trading.research.datasets.export import (
    DecisionExportRecord,
    extract_decision_records,
    export_records_to_json,
)


def _ledger_events_backtest_style() -> list[LedgerEvent]:
    """Backtest-style events: decision, intent, fill in sequence, no order_link_id."""
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    return [
        LedgerEvent("decision", base, {"symbol": "BTCUSDT", "action": "entry_long"}),
        LedgerEvent("order_intent", base, {"symbol": "BTCUSDT", "side": "Buy", "qty": "0.001"}),
        LedgerEvent("fill", base, {"symbol": "BTCUSDT", "qty": "0.001"}),
    ]


def _ledger_events_runtime_style() -> list[LedgerEvent]:
    """Runtime-style events with order_link_id linking intent and fill."""
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    link = "drill_abc123"
    return [
        LedgerEvent("decision", base, {"symbol": "ETHUSDT", "action": "entry_short"}),
        LedgerEvent(
            "order_intent",
            base,
            {"symbol": "ETHUSDT", "side": "Sell", "qty": "0.01", "order_link_id": link},
        ),
        LedgerEvent(
            "fill",
            base,
            {
                "order_link_id": link,
                "symbol": "ETHUSDT",
                "exec_qty": "0.01",
                "exec_price": "2500.50",
            },
        ),
    ]


def test_decision_export_record_structure() -> None:
    """DecisionExportRecord has required fields for research."""
    r = DecisionExportRecord(
        ts_utc=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
        symbol="BTCUSDT",
        action="entry_long",
        side="Buy",
        qty="0.001",
        reference_price=None,
        order_link_id=None,
        filled=True,
        fill_ts_utc=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
        fill_qty="0.001",
        fill_price="40000",
        risk_approved=True,
        risk_reason=None,
    )
    assert r.symbol == "BTCUSDT"
    assert r.filled
    assert r.fill_price == "40000"


def test_extract_decision_records_backtest_style() -> None:
    """Extract pairs decision+intent+fill for backtest-style events."""
    events = _ledger_events_backtest_style()
    records = extract_decision_records(events)
    assert len(records) == 1
    r = records[0]
    assert r.symbol == "BTCUSDT"
    assert r.action == "entry_long"
    assert r.side == "Buy"
    assert r.qty == "0.001"
    assert r.risk_approved
    assert r.filled
    assert r.fill_qty == "0.001"


def test_extract_decision_records_runtime_style() -> None:
    """Extract pairs decision+intent+fill via order_link_id for runtime-style events."""
    events = _ledger_events_runtime_style()
    records = extract_decision_records(events)
    assert len(records) == 1
    r = records[0]
    assert r.symbol == "ETHUSDT"
    assert r.action == "entry_short"
    assert r.order_link_id == "drill_abc123"
    assert r.filled
    assert r.fill_price == "2500.50"
    assert r.fill_qty == "0.01"


def test_extract_decision_records_no_intent() -> None:
    """Decision without intent has risk_approved=False."""
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    events = [
        LedgerEvent("decision", base, {"symbol": "BTCUSDT", "action": "entry_long"}),
    ]
    records = extract_decision_records(events)
    assert len(records) == 1
    assert records[0].risk_approved is False
    assert records[0].risk_reason == "no_intent_recorded"
    assert not records[0].filled


def test_export_records_to_json(tmp_path: Path) -> None:
    """export_records_to_json writes valid JSON with records array."""
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    records = [
        DecisionExportRecord(
            ts_utc=base,
            symbol="BTCUSDT",
            action="entry_long",
            side="Buy",
            qty="0.001",
            reference_price=None,
            order_link_id=None,
            filled=True,
            fill_ts_utc=base,
            fill_qty="0.001",
            fill_price="40000",
            risk_approved=True,
            risk_reason=None,
        ),
    ]
    path = tmp_path / "export.json"
    export_records_to_json(records, path)
    assert path.exists()
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    assert "records" in data
    assert data["count"] == 1
    row = data["records"][0]
    assert row["symbol"] == "BTCUSDT"
    assert row["filled"] is True
