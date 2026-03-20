"""Minimal backtest CLI entrypoint."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from trading.backtest.engine import BacktestEngine
from trading.backtest.event_source import synthetic_candle_events
from trading.backtest.report import build_backtest_report, write_backtest_report
from trading.research.datasets.export import extract_decision_records, export_records_to_json
from trading.settings import AppSettings, backtest_config_from_settings
from trading.util.logging import get_logger


async def run_backtest(settings: AppSettings) -> None:
    """
    Run backtest with config-backed settings and synthetic event source.

    Uses TRADING_BACKTEST_BARS (default 1200) for synthetic bar count.
    Writes JSON and markdown reports to archive for parity with DEMO summaries.
    """
    logger = get_logger("trading.backtest.run")
    cfg = backtest_config_from_settings(settings)
    bars = settings.runtime.backtest_bars
    symbols = settings.trading.symbols
    timeframe = settings.trading.candle_timeframe

    logger.info(
        "backtest_starting",
        symbols=symbols,
        bars=bars,
        timeframe=timeframe,
        initial_equity=str(cfg.initial_equity_usdt),
    )

    event_source = synthetic_candle_events(
        symbols=symbols,
        bars=bars,
        timeframe=timeframe,
    )
    engine = BacktestEngine(config=cfg)
    result = await engine.run(event_source)

    report = build_backtest_report(result, symbols)
    archive_dir = Path(os.getenv("TRADING_ARCHIVE_DIR", "data/archive"))
    export_dir = archive_dir / "decision_exports"
    ts = report.session_start[:19].replace("-", "").replace(":", "").replace("T", "_") if report.session_start else "unknown"
    export_path = export_dir / f"decisions_{ts}.json"

    try:
        json_path, md_path = write_backtest_report(report, archive_dir)
        logger.info(
            "backtest_report_written",
            json_path=str(json_path),
            md_path=str(md_path),
        )
        records = extract_decision_records(result.events)
        export_dir.mkdir(parents=True, exist_ok=True)
        filled_count = sum(1 for r in records if r.filled)
        not_filled_count = len(records) - filled_count
        fill_rate = filled_count / len(records) if records else 0.0
        has_both_classes = filled_count >= 1 and not_filled_count >= 1
        train_split_single_class_risk = min(filled_count, not_filled_count) < 2
        diagnostics = {
            "total_decisions": len(records),
            "filled_count": filled_count,
            "not_filled_count": not_filled_count,
            "fill_rate": round(fill_rate, 4),
            "has_both_classes": has_both_classes,
            "train_split_single_class_risk": train_split_single_class_risk,
        }
        if records:
            export_records_to_json(records, export_path, dataset_diagnostics=diagnostics)
            logger.info(
                "backtest_decision_export_written",
                path=str(export_path.resolve()),
                count=len(records),
                filled_count=filled_count,
                not_filled_count=not_filled_count,
                fill_rate=round(fill_rate, 4),
                has_both_classes=has_both_classes,
                train_split_single_class_risk=train_split_single_class_risk,
                export_dir=str(export_dir.resolve()),
            )
        else:
            logger.info(
                "backtest_decision_export_skipped",
                path=str(export_path.resolve()),
                count=0,
                reason="no_decisions_to_export",
                export_dir=str(export_dir.resolve()),
            )
    except OSError as exc:
        logger.warning(
            "backtest_report_write_failed",
            path=str(archive_dir),
            error=str(exc),
        )

    logger.info(
        "backtest_complete",
        start_time=str(result.start_time),
        end_time=str(result.end_time),
        initial_equity=str(result.initial_equity_usdt),
        final_equity=str(result.final_equity_usdt),
        total_pnl=str(result.total_pnl_usdt),
        total_costs=str(result.total_costs_usdt),
        total_funding=str(result.total_funding_usdt),
        decisions=result.decisions,
        fills=result.fills,
    )


def main() -> None:
    """Run backtest entrypoint; loads settings and executes."""
    import os

    from trading.settings import load_settings
    from trading.util.logging import configure_logging

    os.environ.setdefault("TRADING_MODE", "backtest")
    settings = load_settings()
    configure_logging(
        level=settings.logging.level,
        json_output=settings.logging.json_output,
        logger_name=settings.logging.logger_name,
    )
    asyncio.run(run_backtest(settings))
