"""Unit tests for backtest report structure and usability."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from trading.backtest.engine import BacktestResult
from trading.backtest.report import (
    BacktestReport,
    build_backtest_markdown,
    build_backtest_report,
    report_to_dict,
    write_backtest_report,
)
from trading.journal.ledger import LedgerEvent


def _make_result(
    decisions: int = 5,
    intents: int = 3,
    fills: int = 3,
) -> BacktestResult:
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    events: list[LedgerEvent] = []
    for i in range(decisions):
        events.append(
            LedgerEvent(
                event_type="decision",
                timestamp=base,
                payload={"symbol": "BTCUSDT", "action": "entry_long"},
            )
        )
    for i in range(intents):
        events.append(
            LedgerEvent(
                event_type="order_intent",
                timestamp=base,
                payload={"symbol": "BTCUSDT", "side": "Buy", "qty": "0.001"},
            )
        )
    for i in range(fills):
        events.append(
            LedgerEvent(
                event_type="fill",
                timestamp=base,
                payload={"symbol": "BTCUSDT", "qty": "0.001"},
            )
        )
    return BacktestResult(
        start_time=base,
        end_time=base,
        initial_equity_usdt=Decimal("10000"),
        final_equity_usdt=Decimal("10100"),
        total_pnl_usdt=Decimal("100"),
        total_costs_usdt=Decimal("5"),
        total_funding_usdt=Decimal("0"),
        decisions=decisions,
        fills=fills,
        events=events,
        pnl_records=[],
    )


def test_build_backtest_report_structure() -> None:
    """Report has parity fields for DEMO comparison."""
    result = _make_result(decisions=4, intents=3, fills=2)
    report = build_backtest_report(result, symbols=["BTCUSDT"])
    assert report.mode == "backtest"
    assert report.decisions_total == 4
    assert report.order_intents_total == 3
    assert report.fills_total == 2
    assert report.initial_equity_usdt == "10000"
    assert report.final_equity_usdt == "10100"
    assert report.total_pnl_usdt == "100"
    assert report.strategy_order_outcomes is not None
    assert report.strategy_order_outcomes["intents"] == 3
    assert report.strategy_order_outcomes["filled"] == 2


def test_report_to_dict_json_serializable() -> None:
    """Report dict is JSON-serializable."""
    result = _make_result()
    report = build_backtest_report(result, symbols=["BTCUSDT", "ETHUSDT"])
    d = report_to_dict(report)
    assert "mode" in d
    assert "decisions_total" in d
    assert "strategy_order_outcomes" in d
    import json

    json.dumps(d)


def test_build_backtest_markdown_demo_aligned() -> None:
    """Markdown has Counts and Strategy Order Outcomes sections like DEMO."""
    result = _make_result(decisions=2, intents=2, fills=1)
    report = build_backtest_report(result, symbols=["BTCUSDT"])
    md = build_backtest_markdown(report)
    assert "## Counts" in md
    assert "Decisions: 2" in md
    assert "Intents: 2" in md
    assert "Fills: 1" in md
    assert "## Strategy Order Outcomes" in md
    assert "## PnL" in md


def test_write_backtest_report_creates_files(tmp_path: Path) -> None:
    """write_backtest_report creates JSON and markdown files."""
    result = _make_result()
    report = build_backtest_report(result, symbols=["BTCUSDT"])
    json_path, md_path = write_backtest_report(report, tmp_path)
    assert json_path.exists()
    assert md_path.exists()
    assert json_path.suffix == ".json"
    assert md_path.suffix == ".md"
    assert "backtest_reports" in str(json_path)
    content = json_path.read_text(encoding="utf-8")
    assert "decisions_total" in content
    md_content = md_path.read_text(encoding="utf-8")
    assert "Backtest Report" in md_content
