from __future__ import annotations

import asyncio
import sys

from trading.runtime.mode import is_backtest_mode, is_streaming_mode
from trading.settings import load_settings
from trading.util.logging import configure_logging, get_logger


async def _run_streaming(settings: object) -> None:
    from trading.runtime.orchestrator import RuntimeOrchestrator

    orchestrator = RuntimeOrchestrator(settings)
    try:
        await orchestrator.run()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise exc


async def _run_backtest(settings: object) -> None:
    from trading.backtest.run import run_backtest

    await run_backtest(settings)


async def run() -> None:
    settings = load_settings()
    configure_logging(
        level=settings.logging.level,
        json_output=settings.logging.json_output,
        logger_name=settings.logging.logger_name,
    )
    logger = get_logger("trading.main")
    logger.info(
        "startup_complete",
        mode=settings.runtime.mode.value,
        symbols=settings.trading.symbols,
        testnet=settings.exchange.testnet,
        dry_run=settings.runtime.dry_run,
    )

    try:
        if is_backtest_mode(settings.runtime.mode):
            await _run_backtest(settings)
        elif is_streaming_mode(settings.runtime.mode):
            await _run_streaming(settings)
        else:
            logger.error(
                "unsupported_mode",
                mode=settings.runtime.mode.value,
                supported="backtest, paper, demo, live",
            )
            sys.exit(1)
    except asyncio.CancelledError:
        logger.warning("runtime_cancelled")
        raise
    except Exception as exc:
        logger.exception("runtime_failed", error=str(exc))
        raise
    finally:
        logger.info("shutdown_complete")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
