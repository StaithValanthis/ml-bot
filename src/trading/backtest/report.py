"""Backtest report building for parity with DEMO session summaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from trading.backtest.engine import BacktestResult
from trading.journal.ledger import LedgerEvent
from trading.util.json_util import dumps_json_safe


@dataclass(slots=True)
class BacktestReport:
    """Structured backtest report for comparison with DEMO runtime summaries."""

    mode: str = "backtest"
    session_start: str = ""
    session_end: str = ""
    symbols: list[str] = ()
    decisions_total: int = 0
    order_intents_total: int = 0
    order_submissions_total: int = 0
    order_acks_total: int = 0
    fills_total: int = 0
    initial_equity_usdt: str = "0"
    final_equity_usdt: str = "0"
    total_pnl_usdt: str = "0"
    total_costs_usdt: str = "0"
    total_funding_usdt: str = "0"
    strategy_order_outcomes: dict[str, int] | None = None


def build_backtest_report(result: BacktestResult, symbols: list[str]) -> BacktestReport:
    """Build report from BacktestResult with parity fields for DEMO comparison."""
    intents = sum(1 for e in result.events if e.event_type == "order_intent")
    fills = sum(1 for e in result.events if e.event_type == "fill")
    start_iso = result.start_time.isoformat() if result.start_time else ""
    end_iso = result.end_time.isoformat() if result.end_time else ""
    return BacktestReport(
        mode="backtest",
        session_start=start_iso,
        session_end=end_iso,
        symbols=symbols,
        decisions_total=result.decisions,
        order_intents_total=intents,
        order_submissions_total=intents,
        order_acks_total=intents,
        fills_total=fills,
        initial_equity_usdt=str(result.initial_equity_usdt),
        final_equity_usdt=str(result.final_equity_usdt),
        total_pnl_usdt=str(result.total_pnl_usdt),
        total_costs_usdt=str(result.total_costs_usdt),
        total_funding_usdt=str(result.total_funding_usdt),
        strategy_order_outcomes={
            "intents": intents,
            "submissions": intents,
            "acks": intents,
            "resting_opens": 0,
            "partially_filled": 0,
            "filled": fills,
            "cancelled": 0,
            "rejected": 0,
        },
    )


def report_to_dict(report: BacktestReport) -> dict[str, object]:
    """Convert BacktestReport to JSON-serializable dict."""
    d: dict[str, object] = {
        "mode": report.mode,
        "session_start": report.session_start,
        "session_end": report.session_end,
        "symbols": report.symbols,
        "decisions_total": report.decisions_total,
        "order_intents_total": report.order_intents_total,
        "order_submissions_total": report.order_submissions_total,
        "order_acks_total": report.order_acks_total,
        "fills_total": report.fills_total,
        "initial_equity_usdt": report.initial_equity_usdt,
        "final_equity_usdt": report.final_equity_usdt,
        "total_pnl_usdt": report.total_pnl_usdt,
        "total_costs_usdt": report.total_costs_usdt,
        "total_funding_usdt": report.total_funding_usdt,
    }
    if report.strategy_order_outcomes:
        d["strategy_order_outcomes"] = report.strategy_order_outcomes
    return d


def build_backtest_markdown(report: BacktestReport) -> str:
    """Build markdown summary aligned with DEMO session summary format."""
    lines = [
        "# Backtest Report",
        "",
        f"**Mode:** {report.mode}",
        f"**Symbols:** {', '.join(report.symbols)}",
        f"**Started:** {report.session_start}",
        f"**Ended:** {report.session_end}",
        "",
        "## Counts",
        f"- Decisions: {report.decisions_total}",
        f"- Intents: {report.order_intents_total}",
        f"- Submissions: {report.order_submissions_total}",
        f"- Acks: {report.order_acks_total}",
        f"- Fills: {report.fills_total}",
        "",
        "## PnL",
        f"- Initial equity: {report.initial_equity_usdt}",
        f"- Final equity: {report.final_equity_usdt}",
        f"- Total PnL: {report.total_pnl_usdt}",
        f"- Total costs: {report.total_costs_usdt}",
        f"- Total funding: {report.total_funding_usdt}",
        "",
    ]
    if report.strategy_order_outcomes:
        lines.append("## Strategy Order Outcomes")
        o = report.strategy_order_outcomes
        lines.append(f"- Intents: {o.get('intents', 0)}")
        lines.append(f"- Submissions: {o.get('submissions', 0)}")
        lines.append(f"- Acks: {o.get('acks', 0)}")
        lines.append(f"- Filled: {o.get('filled', 0)}")
        lines.append("")
    return "\n".join(lines)


def write_backtest_report(
    report: BacktestReport,
    root_dir: Path | str,
    *,
    ts: str | None = None,
) -> tuple[Path, Path]:
    """Write JSON and markdown reports; returns (json_path, md_path)."""
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    report_dir = root / "backtest_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    if ts is None:
        ts = (
            report.session_start[:19].replace("-", "").replace(":", "").replace("T", "_")
            if report.session_start
            else "unknown"
        )
    json_path = report_dir / f"backtest_{ts}.json"
    md_path = report_dir / f"backtest_{ts}.md"
    json_path.write_text(dumps_json_safe(report_to_dict(report), indent=2), encoding="utf-8")
    md_path.write_text(build_backtest_markdown(report), encoding="utf-8")
    return (json_path, md_path)
