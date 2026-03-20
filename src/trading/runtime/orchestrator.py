from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict, deque
from datetime import date, datetime
from pathlib import Path

from decimal import Decimal

from trading.exchange.bybit_rest import (
    BybitAPIError,
    BybitAuthenticationError,
    BybitRestClient,
    BybitRestError,
)
from trading.exchange.bybit_ws_private import BybitWsPrivateClient
from trading.exchange.bybit_ws_public import BybitWsPublicClient
from trading.exchange.schemas import PlaceOrderRequest
from trading.execution.execution_engine import ExecutionEngine
from trading.execution.order_manager import OrderManager
from trading.execution.reconciler import Reconciler
from trading.journal.ledger import RuntimeLedger
from trading.journal.pnl import PnLTracker
from trading.marketdata.candle_builder import CandleBuilder
from trading.marketdata.market_state import MarketState
from trading.marketdata.normalizers import NormalizedEvent, NormalizedExecution, NormalizedOrderUpdate
from trading.marketdata.staleness import build_default_watchdog
from trading.monitoring.alerts import AlertEvent, AlertLevel, AlertSink, LogAlertSink
from trading.monitoring.health import HealthState
from trading.monitoring.metrics import MetricsRegistry
from trading.risk.circuit_breaker import CircuitBreaker
from trading.risk.portfolio_state import PortfolioState, PositionRiskView
from trading.risk.risk_engine import PerSymbolLimit, RiskEngine
from trading.risk.sizing import SizingInputs, VolatilityAwareSizer
from trading.settings import AppSettings
from trading.strategy.candidates import BreakoutTrendCandidateGenerator
from trading.strategy.regime_filter import RegimeFilter
from trading.strategy.signal_engine import SignalEngine
from trading.util.logging import get_logger
from trading.util.time import utc_now
from trading.util.types import MarketSymbol, OrderSide, PositionSide, RuntimeMode
from trading.storage.parquet_store import ParquetArchiveStore
from trading.runtime.drill import (
    DrillConfig,
    DrillMode,
    DrillOutcome,
    build_drill_intent,
    generate_drill_order_link_id,
    validate_drill,
)
from trading.storage.postgres import PostgresJournalStore
from trading.runtime.mode import is_live_execution_mode, is_streaming_mode
from trading.runtime.scheduler import run_periodic


class RuntimeOrchestrator:
    """
    Minimal live runtime wiring for market-data-driven decision execution.

    This orchestrator intentionally implements a strict baseline:
    - decisions only on confirmed closed candles
    - explicit async task lifecycle and shutdown
    - clear module boundaries across strategy/risk/execution/reconciliation
    """

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._logger = get_logger("trading.runtime.orchestrator")
        self._stop_event = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []
        self._shutdown_started = False
        self._session_start_time: datetime | None = None
        self._ws_public_ever_connected = False
        self._ws_private_ever_connected = False
        self._session_ended_cleanly = True
        self._abort_reasons: list[str] = []
        self._consecutive_reconcile_mismatches = 0
        self._startup_auth_disabled = False
        self._drill_outcome = DrillOutcome()

        self._rest = BybitRestClient(settings.exchange)
        self._metrics = MetricsRegistry()
        self._health = HealthState()
        self._alerts: AlertSink = LogAlertSink()
        self._staleness = build_default_watchdog()
        self._pnl = PnLTracker()

        postgres_dsn = os.getenv("TRADING_POSTGRES_DSN")
        archive_dir = os.getenv("TRADING_ARCHIVE_DIR", "data/archive")
        self._postgres_store = PostgresJournalStore(postgres_dsn)
        self._parquet_store = ParquetArchiveStore(archive_dir)
        self._ledger = RuntimeLedger(sinks=[self._parquet_store, self._postgres_store])

        self._market_state = MarketState()
        self._candle_builder = CandleBuilder(timeframe_minutes=int(settings.trading.candle_timeframe))
        self._order_manager = OrderManager()
        self._reconciler = Reconciler(
            rest_client=self._rest,
            order_manager=self._order_manager,
            category=settings.trading.category,
            symbols=settings.trading.symbols,
        )

        self._circuit_breaker = CircuitBreaker()
        per_symbol = {
            s: PerSymbolLimit(
                max_notional_usdt=p.max_notional_usdt,
                max_position_abs=p.max_position_abs,
            )
            for s, p in settings.risk.per_symbol.items()
        }
        self._risk_engine = RiskEngine(
            max_total_notional_usdt=settings.risk.max_total_notional_usdt,
            max_leverage=settings.risk.max_leverage,
            daily_loss_limit_usdt=settings.risk.daily_loss_limit_usdt,
            liquidation_buffer_bps=settings.risk.liquidation_buffer_bps,
            circuit_breaker=self._circuit_breaker,
            per_symbol_limits=per_symbol,
        )
        self._sizer = VolatilityAwareSizer()
        self._candidate_generator = BreakoutTrendCandidateGenerator()
        self._regime_filter = RegimeFilter()
        self._signal_engine = SignalEngine()
        self._execution_engine = ExecutionEngine(strategy_id="v1alpha")

        self._bar_history: dict[str, dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=800)))
        self._portfolio = PortfolioState(
            equity_usdt=Decimal("0"),
            available_balance_usdt=Decimal("0"),
            safe_mode=settings.risk.safe_mode,
            pnl_date=date.today(),
        )
        self._symbol_specs = settings.get_symbol_specs()

        self._ws_public = BybitWsPublicClient(
            settings.exchange,
            message_handler=self._on_public_events,
            connection_state_handler=self._health.set_ws_public,
        )
        self._ws_private = BybitWsPrivateClient(
            settings.exchange,
            message_handler=self._on_private_events,
            connection_state_handler=self._health.set_ws_private,
        )

    async def run(self) -> None:
        if not is_streaming_mode(self._settings.runtime.mode):
            raise RuntimeError(
                f"Runtime mode '{self._settings.runtime.mode.value}' is not supported by live orchestrator."
            )
        await self._startup()
        try:
            await self._stop_event.wait()
        finally:
            await self._shutdown()

    async def stop(self) -> None:
        self._stop_event.set()

    async def _startup(self) -> None:
        self._logger.info(
            "runtime_starting",
            mode=self._settings.runtime.mode.value,
            symbols=self._settings.trading.symbols,
            dry_run=self._settings.runtime.dry_run,
        )
        self._log_startup_capabilities()
        self._log_durable_sinks()
        self._log_order_state_recovery()
        self._log_recovery_gaps()
        self._log_execution_mode_warning()
        try:
            await self._postgres_store.connect()
            server_time = await self._rest.get_server_time()
            self._logger.info("exchange_time_synced", time_second=server_time.time_second)
            self._session_start_time = utc_now()
            await self._ledger.record("runtime_start", {"mode": self._settings.runtime.mode.value})

            if self._has_auth_credentials():
                await self._refresh_portfolio_snapshot()
                self._tasks.append(
                    asyncio.create_task(
                        run_periodic(
                            name="portfolio-refresh",
                            interval_seconds=20.0,
                            stop_event=self._stop_event,
                            task_fn=self._refresh_portfolio_snapshot,
                        ),
                        name="runtime-portfolio-refresh",
                    )
                )
                self._tasks.append(
                    asyncio.create_task(
                        run_periodic(
                            name="reconcile",
                            interval_seconds=30.0,
                            stop_event=self._stop_event,
                            task_fn=self._reconcile_cycle,
                        ),
                        name="runtime-reconcile",
                    )
                )
            else:
                self._startup_auth_disabled = True
                self._logger.info("runtime_auth_features_disabled", reason="missing_api_credentials")

            public_topics = _public_topics(
                symbols=self._settings.trading.symbols,
                candle_timeframe=self._settings.trading.candle_timeframe,
                regime_timeframe=self._settings.trading.regime_timeframe,
            )
            await self._ws_public.subscribe(public_topics)
            expected_public = {f"public:{s}" for s in self._settings.trading.symbols}
            self._staleness.set_expected_channels(expected_public)

            self._tasks.append(asyncio.create_task(self._ws_public.run_forever(), name="runtime-ws-public"))
            self._tasks.append(asyncio.create_task(self._decision_loop(), name="runtime-decision"))
            self._tasks.append(
                asyncio.create_task(
                    run_periodic(
                        name="watchdog",
                        interval_seconds=5.0,
                        stop_event=self._stop_event,
                        task_fn=self._watchdog_cycle,
                    ),
                    name="runtime-watchdog",
                )
            )

            if self._settings.runtime.mode in {RuntimeMode.PAPER, RuntimeMode.DEMO}:
                self._tasks.append(
                    asyncio.create_task(
                        run_periodic(
                            name="runtime-summary",
                            interval_seconds=60.0,
                            stop_event=self._stop_event,
                            task_fn=self._runtime_summary_cycle,
                        ),
                        name="runtime-summary",
                    )
                )

            if self._can_use_private_stream():
                await self._ws_private.subscribe(["order", "execution", "position", "wallet"])
                self._tasks.append(asyncio.create_task(self._ws_private.run_forever(), name="runtime-ws-private"))

            drill_cfg = self._settings.runtime.demo_drill
            if drill_cfg.enabled:
                self._drill_outcome.enabled = True
                self._logger.warning(
                    "demo_drill_enabled",
                    symbol=drill_cfg.symbol,
                    side=drill_cfg.side,
                    qty=str(drill_cfg.qty),
                    mode=drill_cfg.mode,
                    message="Demo execution drill is enabled; will submit one test order.",
                )
                if (
                    self._settings.runtime.mode == RuntimeMode.DEMO
                    and not self._settings.runtime.dry_run
                    and self._can_place_exchange_orders()
                ):
                    self._tasks.append(
                        asyncio.create_task(self._demo_drill_cycle(), name="runtime-demo-drill"),
                    )

            self._tasks.append(asyncio.create_task(self._task_supervisor(), name="runtime-supervisor"))
            self._logger.info("runtime_started")
        except Exception as exc:
            self._session_ended_cleanly = False
            self._abort_reasons.append("startup_failed")
            self._logger.exception("runtime_startup_failed", error=str(exc))
            await self._shutdown()
            raise

    async def _shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True

        self._logger.info("runtime_stopping")
        self._stop_event.set()
        await self._ws_public.close()
        await self._ws_private.close()

        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        health_snapshot = self._health.snapshot()
        await self._ledger.record(
            "runtime_stop",
            {
                "circuit_breaker_tripped": self._circuit_breaker.is_tripped(),
                "health": {
                    "ws_public_connected": str(health_snapshot.ws_public_connected),
                    "ws_private_connected": str(health_snapshot.ws_private_connected),
                    "stale_channels": ",".join(health_snapshot.stale_channels),
                    "circuit_breaker_tripped": str(health_snapshot.circuit_breaker_tripped),
                },
            },
        )
        await self._write_session_summary()
        self._log_recovery_gaps()
        await self._rest.close()
        await self._postgres_store.close()
        self._logger.info("runtime_stopped")

    async def _task_supervisor(self) -> None:
        """
        Fail-fast supervisor: stop runtime when any critical task exits unexpectedly.
        """
        while not self._stop_event.is_set():
            done = [task for task in self._tasks if task.done() and not task.cancelled()]
            for task in done:
                if task.get_name() == "runtime-supervisor":
                    continue
                exc = task.exception()
                if exc is not None:
                    self._session_ended_cleanly = False
                    reason = f"task_failed:{task.get_name()}"
                    self._abort_reasons.append(reason)
                    if task.get_name() == "runtime-ws-private":
                        self._health.set_private_stream_error(str(exc))
                    self._logger.exception(
                        "runtime_task_failed",
                        task=task.get_name(),
                        error=str(exc),
                        abort_reason=reason,
                    )
                    self._stop_event.set()
                    return
            await asyncio.sleep(0.5)

    async def _wait_for_drill_reference_price(
        self,
        *,
        symbol: str,
        side: OrderSide,
        timeout_seconds: float = 25.0,
        poll_interval_seconds: float = 2.0,
    ) -> tuple[Decimal | None, dict[str, object]]:
        """
        Wait for usable reference price from WS market state, with REST fallback.
        Returns (ref_price, abort_details). abort_details populated only on timeout.
        """
        waited = 0.0
        logged_waiting = False
        rest_fallback_attempted = False

        while waited < timeout_seconds:
            if self._stop_event.is_set():
                health = self._health.snapshot()
                return (
                    None,
                    {
                        "waited_seconds": round(waited, 1),
                        "symbol": symbol,
                        "ws_public_connected": health.ws_public_connected,
                        "ticker_seen": False,
                        "trade_seen": False,
                        "reason": "stop_requested",
                    },
                )

            snap = await self._market_state.snapshot()
            ticker = snap.tickers.get(symbol)
            last_trade = snap.last_trade.get(symbol)
            health = self._health.snapshot()

            if ticker and ticker.bid_price and ticker.ask_price and ticker.bid_price > 0 and ticker.ask_price > 0:
                ref = ticker.bid_price if side == OrderSide.BUY else ticker.ask_price
                self._logger.info(
                    "drill_market_data_ready",
                    symbol=symbol,
                    source="ticker",
                    waited_seconds=round(waited, 1),
                )
                await self._ledger.record(
                    "drill_market_data_ready",
                    {"symbol": symbol, "source": "ticker", "waited_seconds": round(waited, 1)},
                )
                return (ref, {})

            if last_trade and last_trade.price and last_trade.price > 0:
                self._logger.info(
                    "drill_market_data_ready",
                    symbol=symbol,
                    source="last_trade",
                    waited_seconds=round(waited, 1),
                )
                await self._ledger.record(
                    "drill_market_data_ready",
                    {"symbol": symbol, "source": "last_trade", "waited_seconds": round(waited, 1)},
                )
                return (last_trade.price, {})

            if not logged_waiting:
                self._logger.info(
                    "drill_waiting_for_market_data",
                    symbol=symbol,
                    ws_public_connected=health.ws_public_connected,
                    ticker_seen=ticker is not None,
                    trade_seen=last_trade is not None,
                )
                await self._ledger.record(
                    "drill_waiting_for_market_data",
                    {
                        "symbol": symbol,
                        "ws_public_connected": health.ws_public_connected,
                        "ticker_seen": ticker is not None,
                        "trade_seen": last_trade is not None,
                    },
                )
                logged_waiting = True

            remaining = timeout_seconds - waited
            if remaining <= 5.0 and not rest_fallback_attempted:
                rest_fallback_attempted = True
                try:
                    ticker_item = await self._rest.get_ticker(
                        category=self._settings.trading.category,
                        symbol=symbol,
                    )
                    if ticker_item:
                        bid_val = Decimal(ticker_item.bid1_price) if ticker_item.bid1_price else Decimal("0")
                        ask_val = Decimal(ticker_item.ask1_price) if ticker_item.ask1_price else Decimal("0")
                        last_val = Decimal(ticker_item.last_price) if ticker_item.last_price else Decimal("0")
                        ref = bid_val if side == OrderSide.BUY and bid_val > 0 else ask_val if ask_val > 0 else last_val
                        if ref > 0:
                            self._logger.info(
                                "drill_market_data_ready",
                                symbol=symbol,
                                source="rest_fallback",
                                waited_seconds=round(waited, 1),
                            )
                            await self._ledger.record(
                                "drill_market_data_ready",
                                {"symbol": symbol, "source": "rest_fallback", "waited_seconds": round(waited, 1)},
                            )
                            return (ref, {})
                except BybitRestError:
                    pass

            await asyncio.sleep(min(poll_interval_seconds, remaining))
            waited += poll_interval_seconds

        health = self._health.snapshot()
        snap = await self._market_state.snapshot()
        ticker = snap.tickers.get(symbol)
        last_trade = snap.last_trade.get(symbol)
        abort_details = {
            "waited_seconds": round(waited, 1),
            "symbol": symbol,
            "ws_public_connected": health.ws_public_connected,
            "ticker_seen": ticker is not None,
            "trade_seen": last_trade is not None,
            "rest_fallback_attempted": rest_fallback_attempted,
            "reason": "timeout",
        }
        self._logger.warning("drill_market_data_timeout", **abort_details)
        await self._ledger.record("drill_market_data_timeout", abort_details)
        return (None, abort_details)

    async def _demo_drill_cycle(self) -> None:
        """Run one demo drill order; DEMO-only, gated by config."""
        await asyncio.sleep(15.0)
        if self._stop_event.is_set():
            return
        drill_cfg = self._settings.runtime.demo_drill
        if not drill_cfg.enabled or self._settings.runtime.mode != RuntimeMode.DEMO:
            return
        side = OrderSide.BUY if drill_cfg.side == "Buy" else OrderSide.SELL
        mode = DrillMode.POST_ONLY_LIMIT if drill_cfg.mode == "post_only" else DrillMode.REDUCE_ONLY
        config = DrillConfig(symbol=drill_cfg.symbol, side=side, qty=drill_cfg.qty, mode=mode)

        symbol_spec = self._symbol_specs.get(config.symbol)
        ref_price: Decimal | None = None
        if mode == DrillMode.POST_ONLY_LIMIT:
            ref_price, abort_details = await self._wait_for_drill_reference_price(
                symbol=config.symbol,
                side=config.side,
                timeout_seconds=25.0,
                poll_interval_seconds=2.0,
            )
            if ref_price is None:
                self._drill_outcome.attempted = True
                self._drill_outcome.aborted = True
                self._drill_outcome.refused_reason = "drill_refused_market_data_timeout"
                self._drill_outcome.abort_details = abort_details
                await self._ledger.record("drill_aborted", {"reason": "market_data_timeout", "details": abort_details})
                self._logger.warning("demo_drill_aborted", reason="market_data_timeout", **abort_details)
                return

        refuse = validate_drill(
            mode=self._settings.runtime.mode,
            dry_run=self._settings.runtime.dry_run,
            symbol=config.symbol,
            qty=config.qty,
            configured_symbols=self._settings.trading.symbols,
            symbol_spec=symbol_spec,
            reference_price=ref_price,
        )
        if refuse:
            self._drill_outcome.attempted = True
            self._drill_outcome.aborted = True
            self._drill_outcome.refused_reason = refuse
            await self._ledger.record("drill_aborted", {"reason": refuse})
            self._logger.warning("demo_drill_aborted", reason=refuse)
            return

        self._drill_outcome.attempted = True
        self._drill_outcome.symbol = config.symbol
        self._drill_outcome.side = config.side.value
        self._drill_outcome.qty = str(config.qty)
        order_link_id = generate_drill_order_link_id(config.symbol)
        self._drill_outcome.order_link_id = order_link_id

        await self._ledger.record(
            "drill_requested",
            {"symbol": config.symbol, "side": config.side.value, "qty": str(config.qty), "order_link_id": order_link_id},
        )
        self._logger.info(
            "drill_requested",
            symbol=config.symbol,
            side=config.side.value,
            qty=str(config.qty),
            order_link_id=order_link_id,
        )

        position_side: PositionSide | None = None
        if mode == DrillMode.REDUCE_ONLY:
            pos = self._portfolio.positions.get(config.symbol)
            if pos is None or pos.qty <= 0:
                self._drill_outcome.aborted = True
                self._drill_outcome.refused_reason = "drill_refused_no_position"
                await self._ledger.record("drill_aborted", {"reason": "no_position"})
                return
            position_side = pos.side

        try:
            intent = build_drill_intent(
                config=config,
                reference_price=ref_price,
                order_link_id=order_link_id,
                now=utc_now(),
                position_side=position_side,
            )
        except ValueError as exc:
            self._drill_outcome.aborted = True
            self._drill_outcome.refused_reason = str(exc)
            await self._ledger.record("drill_aborted", {"reason": str(exc)})
            return

        await self._order_manager.register_intent(intent)
        await self._ledger.record(
            "drill_submitted",
            {"order_link_id": order_link_id, "symbol": config.symbol, "side": config.side.value, "qty": str(config.qty)},
        )
        self._logger.info("drill_submitted", order_link_id=order_link_id, symbol=config.symbol)

        await self._submit_intent(intent, is_drill=True)

    async def _on_public_events(self, events: list[NormalizedEvent]) -> None:
        await self._market_state.apply_events(events)
        for event in events:
            symbol = getattr(event, "symbol", None)
            if isinstance(symbol, str) and symbol:
                await self._staleness.mark_seen(f"public:{symbol}")

    async def _on_private_events(self, events: list[NormalizedEvent]) -> None:
        await self._market_state.apply_events(events)
        for event in events:
            if isinstance(event, NormalizedOrderUpdate):
                link_id = event.order_link_id or ""
                prev = await self._order_manager.get_by_link_id(link_id) if link_id else None
                prev_status = prev.status.value if prev and prev.status else None
                await self._order_manager.apply_order_update(
                    order_id=event.order_id,
                    order_link_id=event.order_link_id,
                    status=event.status,
                    filled_qty=event.qty,
                    avg_price=event.avg_price,
                    updated_at=event.ts_event_utc,
                )
                new_status = event.status.value if event.status is not None else ""
                await self._ledger.record(
                    "order_update",
                    {
                        "order_id": event.order_id or "",
                        "order_link_id": event.order_link_id or "",
                        "symbol": event.symbol or "",
                        "status": new_status,
                        "qty": str(event.qty) if event.qty is not None else "",
                    },
                )
                if prev_status and prev_status != new_status:
                    self._metrics.inc("order_state_transitions_total")
                    await self._ledger.record(
                        "order_state_transition",
                        {
                            "order_link_id": event.order_link_id or "",
                            "from_status": prev_status,
                            "to_status": new_status,
                        },
                    )
                    if link_id and self._drill_outcome.order_link_id == link_id:
                        self._drill_outcome.final_status = new_status
                        await self._ledger.record(
                            "drill_state_transition",
                            {"order_link_id": link_id, "from_status": prev_status, "to_status": new_status},
                        )
                        if new_status in ("Filled", "Cancelled", "Rejected"):
                            self._drill_outcome.completed = True
                            await self._ledger.record(
                                "drill_completed",
                                {"order_link_id": link_id, "final_status": new_status},
                            )
                    self._logger.info(
                        "order_state_transition",
                        order_link_id=event.order_link_id or "",
                        from_status=prev_status,
                        to_status=new_status,
                    )
            if isinstance(event, NormalizedExecution):
                await self._ledger.record(
                    "fill",
                    {
                        "exec_id": event.exec_id or "",
                        "order_id": event.order_id or "",
                        "order_link_id": event.order_link_id or "",
                        "symbol": event.symbol or "",
                        "exec_qty": str(event.exec_qty) if event.exec_qty is not None else "",
                        "exec_price": str(event.exec_price) if event.exec_price is not None else "",
                        "exec_fee": str(event.exec_fee) if event.exec_fee is not None else "",
                    },
                )
                self._metrics.inc("fills_total")

    async def _decision_loop(self) -> None:
        while not self._stop_event.is_set():
            kline = await self._market_state.next_confirmed_kline()
            bar = self._candle_builder.on_confirmed_kline(kline)
            if bar is None or not bar.confirmed:
                continue

            history = self._bar_history[bar.symbol][bar.timeframe]
            history.append(bar)
            if bar.timeframe != self._settings.trading.candle_timeframe:
                continue

            bars_5m = list(self._bar_history[bar.symbol][self._settings.trading.candle_timeframe])
            bars_1h = list(self._bar_history[bar.symbol][self._settings.trading.regime_timeframe])
            candidates = self._candidate_generator.on_closed_candle(bar.symbol, bars_5m)

            for candidate in candidates:
                regime = self._regime_filter.evaluate(candidate=candidate, bars_1h=bars_1h)
                signal = self._signal_engine.evaluate(candidate, regime)
                if signal.side is None or signal.reference_price is None:
                    continue
                await self._ledger.record(
                    "decision",
                    {
                        "symbol": signal.symbol,
                        "action": signal.action.value,
                        "reason": signal.reason,
                        "confidence": str(signal.confidence),
                    },
                )
                self._metrics.inc("decisions_total")
                self._health.mark_decision()

                symbol_spec = self._symbol_specs.get(signal.symbol)
                if symbol_spec is None:
                    self._logger.warning("missing_symbol_spec", symbol=signal.symbol)
                    continue

                qty = self._sizer.size_qty(
                    SizingInputs(
                        equity_usdt=max(self._portfolio.equity_usdt, Decimal("1")),
                        confidence=signal.confidence,
                        volatility_bps=regime.volatility_bps,
                        reference_price=signal.reference_price,
                        max_leverage=self._settings.risk.max_leverage,
                    ),
                    symbol_spec,
                )
                if qty <= 0:
                    continue

                risk = self._risk_engine.evaluate(
                    signal=signal,
                    portfolio=self._portfolio,
                    expected_order_notional=qty * signal.reference_price,
                )
                if not risk.approved:
                    self._logger.info(
                        "signal_blocked_by_risk",
                        symbol=signal.symbol,
                        reason=risk.reason,
                        metadata=risk.metadata,
                    )
                    continue

                intent = self._execution_engine.build_entry_intent(
                    signal=signal,
                    qty=qty,
                    reference_price=signal.reference_price,
                    now=utc_now(),
                )
                if intent is None:
                    continue

                await self._order_manager.register_intent(intent)
                self._metrics.inc("order_intents_total")
                await self._ledger.record(
                    "order_intent",
                    {
                        "symbol": intent.symbol,
                        "side": intent.side.value,
                        "qty": str(intent.qty),
                        "order_link_id": intent.order_link_id,
                        "reduce_only": str(intent.reduce_only),
                    },
                )
                self._logger.info(
                    "order_intent_created",
                    symbol=intent.symbol,
                    side=intent.side.value,
                    qty=str(intent.qty),
                    order_link_id=intent.order_link_id,
                    dry_run=self._settings.runtime.dry_run,
                )

                if self._can_place_exchange_orders():
                    await self._submit_intent(intent, is_drill=False)

    async def _submit_intent(self, intent: object, *, is_drill: bool = False) -> None:
        from trading.execution.order_intent import OrderIntent

        if not isinstance(intent, OrderIntent):
            return
        self._metrics.inc("order_submissions_total")
        await self._ledger.record(
            "order_submission_attempt",
            {
                "order_link_id": intent.order_link_id,
                "symbol": intent.symbol,
                "side": intent.side.value,
                "qty": str(intent.qty),
                "reduce_only": str(intent.reduce_only),
            },
        )
        self._logger.info(
            "order_submission_attempt",
            order_link_id=intent.order_link_id,
            symbol=intent.symbol,
            side=intent.side.value,
            qty=str(intent.qty),
        )
        try:
            ack = await self._rest.place_order(
                PlaceOrderRequest(
                    category=self._settings.trading.category,
                    symbol=intent.symbol,
                    side=intent.side,
                    order_type=intent.order_type,
                    qty=intent.qty,
                    price=intent.price,
                    time_in_force=intent.time_in_force,
                    order_link_id=intent.order_link_id,
                    reduce_only=intent.reduce_only,
                )
            )
            await self._order_manager.ack_exchange_order(
                order_link_id=ack.order_link_id,
                order_id=ack.order_id,
                updated_at=utc_now(),
            )
            await self._ledger.record(
                "order_ack",
                {"order_link_id": ack.order_link_id, "order_id": ack.order_id},
            )
            if is_drill and self._drill_outcome.order_link_id == ack.order_link_id:
                self._drill_outcome.ack_received = True
                self._drill_outcome.order_id = ack.order_id
                await self._ledger.record(
                    "drill_ack_received",
                    {"order_link_id": ack.order_link_id, "order_id": ack.order_id},
                )
                self._logger.info("drill_ack_received", order_link_id=ack.order_link_id, order_id=ack.order_id)
            self._logger.info(
                "order_ack_received",
                order_link_id=ack.order_link_id,
                order_id=ack.order_id,
            )
            self._metrics.inc("order_acks_total")
        except BybitRestError as exc:
            if is_drill and isinstance(intent, OrderIntent) and self._drill_outcome.order_link_id == intent.order_link_id:
                self._drill_outcome.aborted = True
                self._drill_outcome.refused_reason = str(exc)
                await self._ledger.record("drill_aborted", {"reason": str(exc)})
            log_ctx: dict[str, object] = {"order_link_id": intent.order_link_id, "error": str(exc)}
            if isinstance(exc, BybitAPIError):
                log_ctx["ret_code"] = exc.ret_code
                log_ctx["ret_msg"] = exc.ret_msg
                log_ctx["operation"] = exc.operation
                log_ctx["scope"] = exc.scope
            self._logger.exception("order_submit_failed", **log_ctx)
            was_tripped = self._circuit_breaker.is_tripped()
            self._circuit_breaker.record_order_rejection()
            now_tripped = self._circuit_breaker.is_tripped()
            self._health.set_circuit_breaker(now_tripped)
            if now_tripped and not was_tripped:
                self._metrics.inc("circuit_breaker_trips_total")
                self._logger.warning(
                    "circuit_breaker_tripped",
                    reason="order_rejection_threshold",
                    order_link_id=intent.order_link_id,
                )
                await self._ledger.record(
                    "circuit_breaker_trip",
                    {"reason": "order_rejection_threshold", "order_link_id": intent.order_link_id},
                )
            self._alerts.emit(
                AlertEvent(
                    level=AlertLevel.WARNING,
                    code="order_submit_failed",
                    message="Order submission failed; rejection recorded.",
                    context={"order_link_id": intent.order_link_id},
                )
            )

    async def _refresh_portfolio_snapshot(self) -> None:
        if not self._has_auth_credentials():
            return
        try:
            wallets = await self._rest.get_wallet(account_type="UNIFIED")
            positions = []
            for sym in self._settings.trading.symbols:
                pos_list = await self._rest.get_positions(
                    category=self._settings.trading.category, symbol=sym
                )
                positions.extend(pos_list)
        except BybitRestError as exc:
            if isinstance(exc, BybitAPIError):
                self._logger.warning(
                    "portfolio_refresh_failed",
                    error=str(exc),
                    ret_code=exc.ret_code,
                    ret_msg=exc.ret_msg,
                    operation=exc.operation,
                    scope=exc.scope,
                )
            else:
                self._logger.warning("portfolio_refresh_failed", error=str(exc))
            return

        equity = Decimal("0")
        available = Decimal("0")
        if wallets:
            top = wallets[0]
            equity = top.total_equity
            available = top.total_available_balance

        position_map: dict[str, PositionRiskView] = {}
        for p in positions:
            if p.size <= 0:
                continue
            side = _to_position_side(p.side)
            liq_price = _decimal_or_none(p.liq_price)
            position_map[p.symbol] = PositionRiskView(
                symbol=p.symbol,
                side=side,
                qty=p.size,
                entry_price=p.avg_price,
                mark_price=p.mark_price,
                leverage=p.leverage,
                liquidation_price=liq_price,
            )

        self._portfolio = PortfolioState(
            equity_usdt=equity,
            available_balance_usdt=available,
            positions=position_map,
            realized_pnl_today_usdt=self._portfolio.realized_pnl_today_usdt,
            pnl_date=self._portfolio.pnl_date or date.today(),
            safe_mode=self._settings.risk.safe_mode,
        )
        unrealized = sum(
            (
                _to_decimal_or_zero(getattr(position, "unrealised_pnl", Decimal("0")))
                for position in positions
            ),
            start=Decimal("0"),
        )
        pnl_record = await self._pnl.add_snapshot(
            equity_usdt=self._portfolio.equity_usdt,
            available_usdt=self._portfolio.available_balance_usdt,
            realized_pnl_usdt=self._portfolio.realized_pnl_today_usdt,
            unrealized_pnl_usdt=unrealized,
        )
        await self._ledger.record(
            "portfolio_snapshot",
            {
                "equity_usdt": str(pnl_record.equity_usdt),
                "available_usdt": str(pnl_record.available_usdt),
                "realized_pnl_usdt": str(pnl_record.realized_pnl_usdt),
                "unrealized_pnl_usdt": str(pnl_record.unrealized_pnl_usdt),
            },
        )
        self._metrics.set_gauge("equity_usdt", float(self._portfolio.equity_usdt))
        self._metrics.set_gauge("available_usdt", float(self._portfolio.available_balance_usdt))

    async def _reconcile_cycle(self) -> None:
        if not self._has_auth_credentials():
            return
        try:
            report_orders = await self._reconciler.reconcile_orders()
            report_positions = await self._reconciler.reconcile_positions()
        except BybitAuthenticationError as exc:
            self._logger.warning("reconcile_skipped_auth", error=str(exc))
            return
        except BybitRestError as exc:
            if isinstance(exc, BybitAPIError):
                self._logger.warning(
                    "reconcile_failed",
                    error=str(exc),
                    ret_code=exc.ret_code,
                    ret_msg=exc.ret_msg,
                    operation=exc.operation,
                    scope=exc.scope,
                )
            else:
                self._logger.warning("reconcile_failed", error=str(exc))
            return

        if not report_orders.ok or not report_positions.ok:
            self._metrics.inc("reconcile_mismatch_cycles")
            self._consecutive_reconcile_mismatches += 1
            if self._consecutive_reconcile_mismatches >= 3 and "repeated_reconcile_mismatch" not in self._abort_reasons:
                self._abort_reasons.append("repeated_reconcile_mismatch")
                self._logger.warning(
                    "abort_condition_repeated_reconcile_mismatch",
                    consecutive_cycles=self._consecutive_reconcile_mismatches,
                )
            order_issues = report_orders.issues
            position_issues = report_positions.issues
            issue_types: dict[str, int] = {}
            affected_link_ids: list[str] = []
            affected_order_ids: list[str] = []
            for i in order_issues + position_issues:
                issue_types[i.issue_type] = issue_types.get(i.issue_type, 0) + 1
                if i.order_link_id:
                    affected_link_ids.append(i.order_link_id)
                if i.order_id:
                    affected_order_ids.append(i.order_id)
            if self._drill_outcome.order_link_id and self._drill_outcome.order_link_id in affected_link_ids:
                self._drill_outcome.reconcile_mismatch = True
                await self._ledger.record(
                    "drill_reconcile_result",
                    {"order_link_id": self._drill_outcome.order_link_id, "mismatch": True},
                )
            payload: dict[str, object] = {
                "order_issues": [i.details for i in order_issues],
                "position_issues": [i.details for i in position_issues],
                "issue_types": issue_types,
            }
            if affected_link_ids:
                payload["affected_order_link_ids"] = affected_link_ids
            if affected_order_ids:
                payload["affected_order_ids"] = affected_order_ids
            await self._ledger.record("reconcile_mismatch_detected", payload)
            self._logger.warning(
                "reconcile_mismatch_detected",
                order_issues=[i.details for i in order_issues],
                position_issues=[i.details for i in position_issues],
                issue_types=issue_types,
                affected_order_link_ids=affected_link_ids if affected_link_ids else None,
                affected_order_ids=affected_order_ids if affected_order_ids else None,
            )
            recovery_note = (
                "local_synced_from_exchange_for_missing_locally_qty_mismatch; "
                "no_auto_cancel_for_missing_on_exchange; no_auto_place_implemented"
            )
            await self._ledger.record(
                "reconcile_recovery_action",
                {
                    "action": "synced_and_reported",
                    "note": recovery_note,
                    "auto_cancel_implemented": False,
                    "auto_place_implemented": False,
                },
            )
            self._logger.info(
                "reconcile_recovery_action",
                action="synced_and_reported",
                note=recovery_note,
            )
            self._metrics.inc("reconcile_issues_total", float(len(order_issues) + len(position_issues)))
        else:
            self._consecutive_reconcile_mismatches = 0
            await self._ledger.record(
                "reconcile_ok",
                {"order_issues": 0, "position_issues": 0},
            )
            if self._drill_outcome.order_link_id and self._drill_outcome.attempted:
                await self._ledger.record(
                    "drill_reconcile_result",
                    {"order_link_id": self._drill_outcome.order_link_id, "mismatch": False},
                )
        self._health.mark_reconcile()

    async def _watchdog_cycle(self) -> None:
        stale = await self._staleness.stale_channels(trigger_streams={"public"})
        self._health.set_stale_channels(stale)
        self._health.set_circuit_breaker(self._circuit_breaker.is_tripped())
        self._metrics.set_gauge("stale_channel_count", float(len(stale)))
        self._metrics.set_gauge("circuit_breaker_tripped", 1.0 if self._circuit_breaker.is_tripped() else 0.0)
        health_snapshot = self._health.snapshot()
        self._metrics.set_gauge("ws_public_connected", 1.0 if health_snapshot.ws_public_connected else 0.0)
        self._metrics.set_gauge("ws_private_connected", 1.0 if health_snapshot.ws_private_connected else 0.0)
        if health_snapshot.ws_public_connected:
            self._ws_public_ever_connected = True
        if health_snapshot.ws_private_connected:
            self._ws_private_ever_connected = True

        if stale:
            was_tripped = self._circuit_breaker.is_tripped()
            self._circuit_breaker.trip(reason="feed_stale")
            self._health.set_circuit_breaker(True)
            prev_safe = self._portfolio.safe_mode
            self._portfolio.safe_mode = True
            self._logger.warning(
                "feed_stale_safe_mode",
                channels=stale,
                circuit_breaker_tripped=True,
                safe_mode_enabled=True,
            )
            self._metrics.inc("staleness_incidents_total")
            if "stale_feed" not in self._abort_reasons:
                self._abort_reasons.append("stale_feed")
            self._logger.warning("abort_condition_stale_feed", channels=stale)
            await self._ledger.record("staleness_violation", {"channels": stale})
            if not was_tripped:
                self._metrics.inc("circuit_breaker_trips_total")
                await self._ledger.record(
                    "circuit_breaker_trip",
                    {"reason": "feed_stale", "channels": stale},
                )
            if not prev_safe:
                await self._ledger.record("safe_mode_transition", {"enabled": True, "reason": "feed_stale"})
            self._alerts.emit(
                AlertEvent(
                    level=AlertLevel.CRITICAL,
                    code="feed_stale",
                    message="Feed staleness detected, circuit breaker tripped and safe mode enabled.",
                    context={"channels": ",".join(stale)},
                )
            )

    def _has_auth_credentials(self) -> bool:
        return (
            self._settings.exchange.bybit_api_key is not None
            and self._settings.exchange.bybit_api_secret is not None
        )

    def _can_use_private_stream(self) -> bool:
        return self._has_auth_credentials() and self._settings.runtime.mode in {RuntimeMode.DEMO, RuntimeMode.LIVE}

    def _can_place_exchange_orders(self) -> bool:
        return (
            self._has_auth_credentials()
            and is_live_execution_mode(self._settings.runtime.mode)
            and not self._settings.runtime.dry_run
        )

    def _log_startup_capabilities(self) -> None:
        """Log a concise startup capability summary (enabled/disabled)."""
        private_stream = self._can_use_private_stream()
        persistence_postgres = self._postgres_store._dsn is not None
        persistence_parquet = True
        place_orders = self._can_place_exchange_orders()
        self._logger.info(
            "runtime_capabilities",
            private_stream=private_stream,
            persistence_postgres=persistence_postgres,
            persistence_parquet=persistence_parquet,
            place_exchange_orders=place_orders,
            dry_run=self._settings.runtime.dry_run,
            safe_mode=self._settings.risk.safe_mode,
        )

    def _log_durable_sinks(self) -> None:
        """Log configured durable sinks at startup."""
        postgres_status = "configured" if self._postgres_store._dsn else "skipped"
        parquet_path = str(self._parquet_store._root_dir)
        self._logger.info(
            "runtime_durable_sinks",
            postgres=postgres_status,
            parquet_path=parquet_path,
        )

    def _log_order_state_recovery(self) -> None:
        """Log order state origin; explicit that full recovery is not implemented."""
        self._logger.info(
            "runtime_order_state",
            state="starting_fresh",
            recovery_implemented=False,
            note="Order state is in-memory only; no restore from durable store.",
        )

    def _log_recovery_gaps(self) -> None:
        """Log which recovery features are NOT implemented; visible in demo/live-capable runs."""
        if self._settings.runtime.mode not in {RuntimeMode.DEMO, RuntimeMode.LIVE}:
            return
        gaps = [
            "order_state_restore",
            "auto_cancel_stray_orders",
            "auto_place_missing_orders",
            "ws_reconnect_order_merge",
        ]
        self._logger.info(
            "recovery_gaps",
            not_implemented=gaps,
            note="Supervised runs only; no unattended production readiness.",
        )

    def _log_execution_mode_warning(self) -> None:
        """Log prominent warning when demo/live with order placement enabled."""
        if not self._can_place_exchange_orders():
            return
        mode = self._settings.runtime.mode.value
        dry_run = self._settings.runtime.dry_run
        symbols = self._settings.trading.symbols
        reduce_only_in_path = False
        self._logger.warning(
            "execution_mode_warning",
            alert="ORDER_PLACEMENT_ENABLED",
            mode=mode,
            dry_run=dry_run,
            symbols=symbols,
            reduce_only_exits_in_decision_path=reduce_only_in_path,
            note="Supervised demo run: verify acks, status transitions, reconciliation.",
        )
        self._logger.warning(
            "execution_mode_banner",
            banner="*** EXCHANGE ORDER PLACEMENT ENABLED ***",
            mode=mode,
            dry_run=dry_run,
        )

    async def _runtime_summary_cycle(self) -> None:
        """Concise periodic runtime summary for paper/demo modes."""
        health_snap = self._health.snapshot()
        metrics_snap = self._metrics.snapshot()
        equity = float(self._portfolio.equity_usdt)
        decisions = metrics_snap.counters.get("decisions_total", 0)
        self._logger.info(
            "runtime_summary",
            mode=self._settings.runtime.mode.value,
            equity_usdt=equity,
            ws_public=health_snap.ws_public_connected,
            ws_private=health_snap.ws_private_connected,
            private_stream_error=health_snap.private_stream_error,
            circuit_breaker=health_snap.circuit_breaker_tripped,
            stale_count=len(health_snap.stale_channels),
            decisions_total=decisions,
        )

    def _build_session_summary(self) -> dict[str, object]:
        """Build concise session summary from metrics and session state."""
        metrics = self._metrics.snapshot()
        start = self._session_start_time or utc_now()
        end = utc_now()
        summary: dict[str, object] = {
            "session_start": start.isoformat(),
            "session_end": end.isoformat(),
            "mode": self._settings.runtime.mode.value,
            "symbols": self._settings.trading.symbols,
            "startup_capabilities": {
                "private_stream": self._can_use_private_stream(),
                "persistence_postgres": self._postgres_store._dsn is not None,
                "placement_enabled": self._can_place_exchange_orders(),
            },
            "ws_public_ever_connected": self._ws_public_ever_connected,
            "ws_private_ever_connected": self._ws_private_ever_connected,
            "order_placement_enabled": self._can_place_exchange_orders(),
            "session_ended_cleanly": self._session_ended_cleanly,
            "decisions_total": int(metrics.counters.get("decisions_total", 0)),
            "order_intents_total": int(metrics.counters.get("order_intents_total", 0)),
            "order_submissions_total": int(metrics.counters.get("order_submissions_total", 0)),
            "order_acks_total": int(metrics.counters.get("order_acks_total", 0)),
            "order_state_transitions_total": int(metrics.counters.get("order_state_transitions_total", 0)),
            "reconcile_mismatch_cycles": int(metrics.counters.get("reconcile_mismatch_cycles", 0)),
            "reconcile_issues_total": int(metrics.counters.get("reconcile_issues_total", 0)),
            "staleness_incidents_total": int(metrics.counters.get("staleness_incidents_total", 0)),
            "circuit_breaker_trips_total": int(metrics.counters.get("circuit_breaker_trips_total", 0)),
        }
        if self._health.snapshot().private_stream_error is not None:
            summary["private_stream_error"] = self._health.snapshot().private_stream_error
        if self._drill_outcome.enabled:
            summary["drill_enabled"] = True
            summary["drill_attempted"] = self._drill_outcome.attempted
            if self._drill_outcome.symbol:
                summary["drill_symbol"] = self._drill_outcome.symbol
            if self._drill_outcome.side:
                summary["drill_side"] = self._drill_outcome.side
            if self._drill_outcome.qty:
                summary["drill_qty"] = self._drill_outcome.qty
            summary["drill_ack_received"] = self._drill_outcome.ack_received
            summary["drill_reconcile_mismatch"] = self._drill_outcome.reconcile_mismatch
            if self._drill_outcome.completed:
                summary["drill_outcome"] = "completed"
            elif self._drill_outcome.aborted:
                summary["drill_outcome"] = "aborted"
                if self._drill_outcome.refused_reason:
                    summary["drill_refused_reason"] = self._drill_outcome.refused_reason
                if self._drill_outcome.abort_details:
                    summary["drill_abort_details"] = self._drill_outcome.abort_details
            else:
                summary["drill_outcome"] = "pending"
        if self._abort_reasons:
            summary["abort_reasons"] = self._abort_reasons
        if self._startup_auth_disabled:
            summary["startup_auth_disabled"] = True
        return summary

    def _build_markdown_summary(self, summary: dict[str, object]) -> str:
        """Build short human-readable markdown summary."""
        lines: list[str] = [
            "# Session Summary",
            "",
            f"**Mode:** {summary.get('mode', '')}",
            f"**Symbols:** {', '.join(summary.get('symbols', []))}",
            f"**Started:** {summary.get('session_start', '')}",
            f"**Ended:** {summary.get('session_end', '')}",
            "",
            "## Capabilities",
            f"- Private stream: {summary.get('ws_private_ever_connected', False)}",
            f"- Public stream: {summary.get('ws_public_ever_connected', False)}",
            f"- Order placement enabled: {summary.get('order_placement_enabled', False)}",
            f"- Session ended cleanly: {summary.get('session_ended_cleanly', True)}",
            "",
            "## Counts",
            f"- Decisions: {summary.get('decisions_total', 0)}",
            f"- Submissions: {summary.get('order_submissions_total', 0)}",
            f"- Acks: {summary.get('order_acks_total', 0)}",
            f"- State transitions: {summary.get('order_state_transitions_total', 0)}",
            f"- Reconcile mismatch cycles: {summary.get('reconcile_mismatch_cycles', 0)}",
            f"- Reconcile issues: {summary.get('reconcile_issues_total', 0)}",
            "",
        ]
        if summary.get("drill_enabled"):
            lines.append("## Demo Drill")
            lines.append(f"- Enabled: {summary.get('drill_enabled', False)}")
            lines.append(f"- Attempted: {summary.get('drill_attempted', False)}")
            if summary.get("drill_symbol"):
                lines.append(f"- Symbol/Side/Qty: {summary.get('drill_symbol')} {summary.get('drill_side', '')} {summary.get('drill_qty', '')}")
            lines.append(f"- Ack received: {summary.get('drill_ack_received', False)}")
            lines.append(f"- Reconcile mismatch: {summary.get('drill_reconcile_mismatch', False)}")
            lines.append(f"- Outcome: {summary.get('drill_outcome', 'pending')}")
            if summary.get("drill_refused_reason"):
                lines.append(f"- Refused reason: {summary.get('drill_refused_reason')}")
            if details := summary.get("drill_abort_details"):
                lines.append("- Abort details:")
                for k, v in details.items():
                    lines.append(f"  - {k}: {v}")
            lines.append("")
        if summary.get("abort_reasons"):
            lines.append("## Abort Reasons")
            for r in summary["abort_reasons"]:
                lines.append(f"- {r}")
            lines.append("")
        return "\n".join(lines)

    async def _write_session_summary(self) -> None:
        """Write session summary to report path for demo/paper runs."""
        if self._settings.runtime.mode not in {RuntimeMode.PAPER, RuntimeMode.DEMO}:
            return
        summary = self._build_session_summary()
        root = Path(self._parquet_store._root_dir)
        root.mkdir(parents=True, exist_ok=True)
        start = self._session_start_time or utc_now()
        ts = start.strftime("%Y%m%d_%H%M%S")
        report_dir = root / "session_summaries"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"session_{ts}.json"
        md_path = report_dir / f"session_{ts}.md"
        try:
            json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            md_path.write_text(self._build_markdown_summary(summary), encoding="utf-8")
            self._logger.info(
                "session_summary_written",
                path=str(json_path),
                session_ended_cleanly=summary.get("session_ended_cleanly"),
                abort_reasons=summary.get("abort_reasons"),
            )
        except OSError as exc:
            self._logger.warning("session_summary_write_failed", path=str(json_path), error=str(exc))


def _public_topics(*, symbols: list[str], candle_timeframe: str, regime_timeframe: str) -> list[str]:
    topics: list[str] = []
    for symbol in symbols:
        topics.append(f"tickers.{symbol}")
        topics.append(f"publicTrade.{symbol}")
        topics.append(f"kline.{candle_timeframe}.{symbol}")
        if regime_timeframe != candle_timeframe:
            topics.append(f"kline.{regime_timeframe}.{symbol}")
    return topics


def _to_position_side(value: str) -> PositionSide:
    if value == "Buy":
        return PositionSide.LONG
    if value == "Sell":
        return PositionSide.SHORT
    return PositionSide.FLAT


def _decimal_or_none(value: str) -> Decimal | None:
    if value in {"", "0", "0.0", None}:
        return None
    try:
        return Decimal(value)
    except Exception:
        return None


def _to_decimal_or_zero(value: object) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal("0")
