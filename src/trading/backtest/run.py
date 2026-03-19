"""Minimal backtest CLI entrypoint."""

from __future__ import annotations

import asyncio

from trading.backtest.engine import BacktestEngine
from trading.backtest.event_source import synthetic_candle_events
from trading.settings import AppSettings, backtest_config_from_settings
from trading.util.logging import get_logger


async def run_backtest(settings: AppSettings) -> None:
    """
    Run backtest with config-backed settings and synthetic event source.

    Uses TRADING_BACKTEST_BARS (default 350) for synthetic bar count.
    File-based loading is scaffolded; use synthetic_candle_events for now.
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
