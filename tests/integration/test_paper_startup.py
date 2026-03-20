"""Integration tests for paper-mode startup and minimal runtime cycle."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading.exchange.schemas import ServerTimeResult
from trading.journal.ledger import LedgerEvent, LedgerSink
from trading.runtime.orchestrator import RuntimeOrchestrator
from trading.settings import load_settings


class _CaptureSink(LedgerSink):
    """In-memory sink that captures all ledger events."""

    def __init__(self) -> None:
        self.events: list[LedgerEvent] = []

    async def write_event(self, event: LedgerEvent) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_paper_startup_completes_with_mocked_exchange() -> None:
    """Paper-mode orchestrator starts and shuts down cleanly with mocked REST/WS."""
    settings = load_settings()
    capture = _CaptureSink()

    mock_server_time = ServerTimeResult(time_second="1700000000", time_nano="0")
    mock_rest = MagicMock()
    mock_rest.get_server_time = AsyncMock(return_value=mock_server_time)
    mock_rest.get_wallet = AsyncMock(return_value=[])
    mock_rest.get_positions = AsyncMock(return_value=[])
    mock_rest.get_open_orders = AsyncMock(return_value=[])
    mock_rest.close = AsyncMock()

    async def mock_ws_run_forever() -> None:
        await asyncio.sleep(0.05)

    mock_ws_public = MagicMock()
    mock_ws_public.subscribe = AsyncMock()
    mock_ws_public.run_forever = AsyncMock(side_effect=mock_ws_run_forever)
    mock_ws_public.close = AsyncMock()

    mock_ws_private = MagicMock()
    mock_ws_private.subscribe = AsyncMock()
    mock_ws_private.run_forever = AsyncMock(side_effect=mock_ws_run_forever)
    mock_ws_private.close = AsyncMock()

    with (
        patch(
            "trading.runtime.orchestrator.BybitRestClient",
            return_value=mock_rest,
        ),
        patch(
            "trading.runtime.orchestrator.BybitWsPublicClient",
            return_value=mock_ws_public,
        ),
        patch(
            "trading.runtime.orchestrator.BybitWsPrivateClient",
            return_value=mock_ws_private,
        ),
    ):
        orch = RuntimeOrchestrator(settings)
        orch._ledger._sinks.insert(0, capture)

        async def run_and_stop() -> None:
            await asyncio.sleep(0.3)
            await orch.stop()

        task = asyncio.create_task(orch.run())
        stop_task = asyncio.create_task(run_and_stop())
        await asyncio.gather(task, stop_task)

    event_types = [e.event_type for e in capture.events]
    assert "runtime_start" in event_types
    assert "runtime_stop" in event_types
    mock_rest.get_server_time.assert_called_once()
    mock_ws_public.subscribe.assert_called_once()
    mock_ws_public.close.assert_called_once()


@pytest.mark.asyncio
async def test_paper_startup_logs_capabilities_and_durable_sinks() -> None:
    """Startup logs capability summary, durable sinks, and order state recovery status."""
    settings = load_settings()
    log_calls: list[tuple[str, dict]] = []

    def capture_info(msg: str, **kwargs: object) -> None:
        log_calls.append((msg, dict(kwargs)))

    mock_rest = MagicMock()
    mock_rest.get_server_time = AsyncMock(return_value=ServerTimeResult(time_second="1700000000", time_nano="0"))
    mock_rest.get_wallet = AsyncMock(return_value=[])
    mock_rest.get_positions = AsyncMock(return_value=[])
    mock_rest.get_open_orders = AsyncMock(return_value=[])
    mock_rest.close = AsyncMock()

    mock_ws_public = MagicMock()
    mock_ws_public.subscribe = AsyncMock()
    mock_ws_public.run_forever = AsyncMock(side_effect=lambda: asyncio.sleep(0.05))
    mock_ws_public.close = AsyncMock()

    mock_ws_private = MagicMock()
    mock_ws_private.subscribe = AsyncMock()
    mock_ws_private.run_forever = AsyncMock(return_value=None)
    mock_ws_private.close = AsyncMock()

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", return_value=mock_rest),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", return_value=mock_ws_public),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", return_value=mock_ws_private),
        patch("trading.runtime.orchestrator.get_logger") as mock_get_logger,
    ):
        mock_logger = MagicMock()
        mock_logger.info = capture_info
        mock_logger.warning = capture_info
        mock_logger.exception = MagicMock()
        mock_get_logger.return_value = mock_logger

        orch = RuntimeOrchestrator(settings)
        task = asyncio.create_task(orch.run())
        await asyncio.sleep(0.1)
        await orch.stop()
        await task

    assert any(msg == "runtime_capabilities" for msg, _ in log_calls)
    assert any(msg == "runtime_durable_sinks" for msg, _ in log_calls)
    assert any(msg == "runtime_order_state" for msg, _ in log_calls)
    order_state = next((kw for msg, kw in log_calls if msg == "runtime_order_state"), {})
    assert order_state.get("state") == "starting_fresh"
    assert order_state.get("recovery_implemented") is False


def _make_mock_kline_items(interval_min: int, count: int) -> list:
    """Create mock KlineItems for warmup tests."""
    from datetime import UTC, datetime
    from decimal import Decimal

    from trading.exchange.schemas import KlineItem

    base_ms = int(datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC).timestamp() * 1000)
    return [
        KlineItem(
            start_time_ms=base_ms + i * interval_min * 60 * 1000,
            open_price=Decimal("40000"),
            high_price=Decimal("40010"),
            low_price=Decimal("39990"),
            close_price=Decimal("40005"),
            volume=Decimal("100"),
            turnover=Decimal("4000500"),
        )
        for i in range(count)
    ]


@pytest.mark.asyncio
async def test_session_summary_written_for_paper_run() -> None:
    """Paper run writes session summary to archive/session_summaries/."""
    settings = load_settings()
    archive_dir = Path.cwd() / "tmp_integration_archive"
    summaries_dir = archive_dir / "session_summaries"

    mock_rest = MagicMock()
    mock_rest.get_server_time = AsyncMock(return_value=ServerTimeResult(time_second="1700000000", time_nano="0"))
    mock_rest.get_wallet = AsyncMock(return_value=[])
    mock_rest.get_positions = AsyncMock(return_value=[])
    mock_rest.get_open_orders = AsyncMock(return_value=[])
    mock_rest.close = AsyncMock()

    async def mock_get_kline(*, category: str, symbol: str, interval: str, **kwargs: object) -> list:
        if interval == "5":
            return _make_mock_kline_items(5, 30)
        return _make_mock_kline_items(60, 30)

    mock_rest.get_kline = AsyncMock(side_effect=mock_get_kline)

    mock_ws_public = MagicMock()
    mock_ws_public.subscribe = AsyncMock()
    mock_ws_public.run_forever = AsyncMock(side_effect=lambda: asyncio.sleep(0.05))
    mock_ws_public.close = AsyncMock()

    mock_ws_private = MagicMock()
    mock_ws_private.subscribe = AsyncMock()
    mock_ws_private.run_forever = AsyncMock(return_value=None)
    mock_ws_private.close = AsyncMock()

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", return_value=mock_rest),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", return_value=mock_ws_public),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", return_value=mock_ws_private),
    ):
        orch = RuntimeOrchestrator(settings)
        task = asyncio.create_task(orch.run())
        await asyncio.sleep(0.15)
        await orch.stop()
        await task

    files = list(summaries_dir.glob("session_*.json"))
    assert len(files) >= 1
    summary_path = files[-1]
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "session_start" in data
    assert "session_end" in data
    assert data.get("mode") == "paper"
    assert "symbols" in data
    assert "decisions_total" in data
    assert "order_intents_total" in data
    assert "order_submissions_total" in data
    assert "order_acks_total" in data
    assert "reconcile_mismatch_cycles" in data
    assert "staleness_incidents_total" in data
    assert "circuit_breaker_trips_total" in data
    assert "strategy_flow" in data
    assert "blocking_stage" in data
    flow = data.get("strategy_flow", {})
    assert "bars_confirmed" in flow
    assert "candidates" in flow
    assert "regime_rejected" in flow
    assert "candidate_readiness" in data
    assert "warmup_results" in data
    warmup = data.get("warmup_results", [])
    assert len(warmup) >= 2


@pytest.mark.asyncio
async def test_controlled_shutdown_emits_shutdown_logs() -> None:
    """Controlled shutdown emits runtime_stopping, session_summary_written, runtime_stopped."""
    settings = load_settings()
    log_calls: list[tuple[str, dict]] = []

    def capture_info(msg: str, **kwargs: object) -> None:
        log_calls.append((msg, dict(kwargs)))

    mock_rest = MagicMock()
    mock_rest.get_server_time = AsyncMock(return_value=ServerTimeResult(time_second="1700000000", time_nano="0"))
    mock_rest.get_wallet = AsyncMock(return_value=[])
    mock_rest.get_positions = AsyncMock(return_value=[])
    mock_rest.get_open_orders = AsyncMock(return_value=[])
    mock_rest.close = AsyncMock()

    mock_ws_public = MagicMock()
    mock_ws_public.subscribe = AsyncMock()
    mock_ws_public.run_forever = AsyncMock(side_effect=lambda: asyncio.sleep(0.05))
    mock_ws_public.close = AsyncMock()

    mock_ws_private = MagicMock()
    mock_ws_private.subscribe = AsyncMock()
    mock_ws_private.run_forever = AsyncMock(return_value=None)
    mock_ws_private.close = AsyncMock()

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", return_value=mock_rest),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", return_value=mock_ws_public),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", return_value=mock_ws_private),
        patch("trading.runtime.orchestrator.get_logger") as mock_get_logger,
    ):
        mock_logger = MagicMock()
        mock_logger.info = capture_info
        mock_logger.warning = MagicMock()
        mock_logger.exception = MagicMock()
        mock_get_logger.return_value = mock_logger

        orch = RuntimeOrchestrator(settings)
        task = asyncio.create_task(orch.run())
        await asyncio.sleep(0.1)
        await orch.stop()
        await task

    assert any(msg == "runtime_stopping" for msg, _ in log_calls)
    assert any(msg == "session_summary_written" for msg, _ in log_calls)
    assert any(msg == "runtime_stopped" for msg, _ in log_calls)


@pytest.mark.asyncio
async def test_full_run_shutdown_emits_shutdown_complete() -> None:
    """Full run() entry point emits shutdown_complete when stopped gracefully."""
    from trading.main import run

    settings = load_settings()
    main_log_calls: list[tuple[str, dict]] = []

    def capture_main_info(msg: str, **kwargs: object) -> None:
        main_log_calls.append((msg, dict(kwargs)))

    mock_rest = MagicMock()
    mock_rest.get_server_time = AsyncMock(return_value=ServerTimeResult(time_second="1700000000", time_nano="0"))
    mock_rest.get_wallet = AsyncMock(return_value=[])
    mock_rest.get_positions = AsyncMock(return_value=[])
    mock_rest.get_open_orders = AsyncMock(return_value=[])
    mock_rest.close = AsyncMock()

    mock_ws_public = MagicMock()
    mock_ws_public.subscribe = AsyncMock()
    mock_ws_public.run_forever = AsyncMock(side_effect=lambda: asyncio.sleep(0.05))
    mock_ws_public.close = AsyncMock()

    mock_ws_private = MagicMock()
    mock_ws_private.subscribe = AsyncMock()
    mock_ws_private.run_forever = AsyncMock(return_value=None)
    mock_ws_private.close = AsyncMock()

    orch_ref: list = []

    async def run_and_stop() -> None:
        await asyncio.sleep(0.15)
        if orch_ref:
            await orch_ref[0].stop()

    async def patched_run_streaming(s: object) -> None:
        from trading.runtime.orchestrator import RuntimeOrchestrator

        orch = RuntimeOrchestrator(s)
        orch_ref.append(orch)
        __import__("trading.main", fromlist=["_install_shutdown_handlers"])._install_shutdown_handlers(orch)
        await orch.run()

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", return_value=mock_rest),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", return_value=mock_ws_public),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", return_value=mock_ws_private),
        patch("trading.main._run_streaming", side_effect=patched_run_streaming),
        patch("trading.main.get_logger") as mock_get_logger,
    ):
        mock_logger = MagicMock()
        mock_logger.info = capture_main_info
        mock_logger.error = MagicMock()
        mock_logger.warning = MagicMock()
        mock_logger.exception = MagicMock()
        mock_get_logger.return_value = mock_logger

        run_task = asyncio.create_task(run())
        stop_task = asyncio.create_task(run_and_stop())
        await asyncio.gather(run_task, stop_task)

    assert any(msg == "shutdown_complete" for msg, _ in main_log_calls)


@pytest.mark.asyncio
async def test_controlled_shutdown_writes_session_summary() -> None:
    """Controlled shutdown writes both JSON and markdown session summaries."""
    settings = load_settings()
    archive_dir = Path.cwd() / "tmp_integration_archive"
    summaries_dir = archive_dir / "session_summaries"

    mock_rest = MagicMock()
    mock_rest.get_server_time = AsyncMock(return_value=ServerTimeResult(time_second="1700000000", time_nano="0"))
    mock_rest.get_wallet = AsyncMock(return_value=[])
    mock_rest.get_positions = AsyncMock(return_value=[])
    mock_rest.get_open_orders = AsyncMock(return_value=[])
    mock_rest.close = AsyncMock()

    mock_ws_public = MagicMock()
    mock_ws_public.subscribe = AsyncMock()
    mock_ws_public.run_forever = AsyncMock(side_effect=lambda: asyncio.sleep(0.05))
    mock_ws_public.close = AsyncMock()

    mock_ws_private = MagicMock()
    mock_ws_private.subscribe = AsyncMock()
    mock_ws_private.run_forever = AsyncMock(return_value=None)
    mock_ws_private.close = AsyncMock()

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", return_value=mock_rest),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", return_value=mock_ws_public),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", return_value=mock_ws_private),
    ):
        orch = RuntimeOrchestrator(settings)
        task = asyncio.create_task(orch.run())
        await asyncio.sleep(0.1)
        await orch.stop()
        await task

    json_files = list(summaries_dir.glob("session_*.json"))
    md_files = list(summaries_dir.glob("session_*.md"))
    assert len(json_files) >= 1
    assert len(md_files) >= 1
    data = json.loads(json_files[-1].read_text(encoding="utf-8"))
    assert "session_start" in data
    assert "session_end" in data


@pytest.mark.asyncio
async def test_strategy_order_outcomes_in_session_summary() -> None:
    """Session summary includes strategy_order_outcomes section with intents, acks, resting_opens, etc."""
    settings = load_settings()
    archive_dir = Path.cwd() / "tmp_integration_archive"
    summaries_dir = archive_dir / "session_summaries"

    mock_rest = MagicMock()
    mock_rest.get_server_time = AsyncMock(return_value=ServerTimeResult(time_second="1700000000", time_nano="0"))
    mock_rest.get_wallet = AsyncMock(return_value=[])
    mock_rest.get_positions = AsyncMock(return_value=[])
    mock_rest.get_open_orders = AsyncMock(return_value=[])
    mock_rest.close = AsyncMock()

    mock_ws_public = MagicMock()
    mock_ws_public.subscribe = AsyncMock()
    mock_ws_public.run_forever = AsyncMock(side_effect=lambda: asyncio.sleep(0.05))
    mock_ws_public.close = AsyncMock()

    mock_ws_private = MagicMock()
    mock_ws_private.subscribe = AsyncMock()
    mock_ws_private.run_forever = AsyncMock(return_value=None)
    mock_ws_private.close = AsyncMock()

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", return_value=mock_rest),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", return_value=mock_ws_public),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", return_value=mock_ws_private),
    ):
        orch = RuntimeOrchestrator(settings)
        orch._strategy_order_outcomes.intents = 2
        orch._strategy_order_outcomes.submissions = 1
        orch._strategy_order_outcomes.acks = 1
        orch._strategy_order_outcomes.filled = 0
        orch._strategy_order_outcomes.cancelled = 0
        orch._strategy_order_outcomes.rejected = 0
        orch._strategy_order_outcomes.partially_filled = 0

        summary = await orch._build_session_summary()

    outcomes = summary.get("strategy_order_outcomes")
    assert outcomes is not None
    assert outcomes.get("intents") == 2
    assert outcomes.get("submissions") == 1
    assert outcomes.get("acks") == 1
    assert "resting_opens" in outcomes
    assert "filled" in outcomes
    assert "cancelled" in outcomes
    assert "rejected" in outcomes

    md = orch._build_markdown_summary(summary)
    assert "## Strategy Order Outcomes" in md
    assert "Intents: 2" in md
    assert "Submissions: 1" in md
