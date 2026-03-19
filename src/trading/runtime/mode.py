from __future__ import annotations

from trading.util.types import RuntimeMode


def is_live_execution_mode(mode: RuntimeMode) -> bool:
    return mode in {RuntimeMode.DEMO, RuntimeMode.LIVE}


def is_streaming_mode(mode: RuntimeMode) -> bool:
    return mode in {RuntimeMode.PAPER, RuntimeMode.DEMO, RuntimeMode.LIVE}


def is_backtest_mode(mode: RuntimeMode) -> bool:
    return mode == RuntimeMode.BACKTEST
