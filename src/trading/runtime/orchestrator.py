from __future__ import annotations

import asyncio
import csv
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
from trading.risk.risk_engine import PerSymbolLimit, RiskDecisionReport, RiskEngine, RiskEvaluationContext
from trading.risk.sizing import SizingInputs, VolatilityAwareSizer
from trading.settings import AppSettings
from trading.strategy.candidates import (
    BreakoutTrendCandidateGenerator,
    CandidateGeneratorConfig,
)
from trading.strategy.regime_filter import RegimeFilter, RegimeFilterReport
from trading.strategy.signal_engine import SignalEngine
from trading.util.json_util import dumps_json_safe
from trading.util.logging import get_logger
from trading.util.time import utc_now
from trading.util.types import MarketSymbol, ModelFilterMode, OHLCVBar, OrderSide, PositionSide, RuntimeMode
from trading.storage.parquet_store import ParquetArchiveStore
from trading.runtime.drill import (
    DrillConfig,
    DrillMode,
    DrillOutcome,
    build_drill_intent,
    generate_drill_order_link_id,
    validate_drill,
)
from trading.runtime.model_calibration import (
    build_model_calibration_summary,
    build_promotion_recommendation,
)
from trading.runtime.soak_report import (
    build_soak_markdown,
    build_soak_report,
)
from trading.runtime.strategy_orders import ModelFilterOutcomes, StrategyOrderOutcomes
from trading.runtime.warmup import preload_warmup_klines, WarmupResult
from trading.storage.postgres import PostgresJournalStore
from trading.runtime.mode import is_live_execution_mode, is_streaming_mode
from trading.runtime.scheduler import run_periodic


def _drill_post_ack_status(outcome: DrillOutcome) -> str:
    """Classify post-ack state for operator visibility."""
    if not outcome.ack_received:
        return "no_ack"
    if outcome.final_status:
        if outcome.final_status == "Filled":
            return "filled"
        if outcome.final_status == "Cancelled":
            return "cancelled"
        if outcome.final_status == "Rejected":
            return "rejected"
        if outcome.final_status in ("New", "PartiallyFilled"):
            return "resting_open"
    return "ack_only_no_transition"


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
        self._strategy_order_outcomes = StrategyOrderOutcomes()
        self._model_filter_model: object | None = None
        self._model_filter_active: bool = False
        self._model_filter_mode: ModelFilterMode = ModelFilterMode.HARD_BLOCK
        self._model_filter_threshold: float | None = None

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
        demo_min_notional = None
        if (
            self._settings.runtime.mode == RuntimeMode.DEMO
            and self._settings.runtime.demo_sizing_min_notional_usdt is not None
        ):
            demo_min_notional = self._settings.runtime.demo_sizing_min_notional_usdt
        self._sizer = VolatilityAwareSizer(demo_min_notional_floor_usdt=demo_min_notional)
        self._candidate_generator = self._build_candidate_generator()
        self._regime_filter = RegimeFilter()
        self._signal_engine = SignalEngine()
        self._execution_engine = ExecutionEngine(strategy_id="v1alpha")

        self._bar_history: dict[str, dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=800)))
        self._last_candidate_readiness: dict[str, dict[str, object]] = {}
        self._last_sizing_rejection: dict[str, object] | None = None
        self._last_sizing_floor_applied: dict[str, object] | None = None
        self._last_regime_rejection: dict[str, object] | None = None
        self._last_risk_rejection: dict[str, object] | None = None
        self._orphan_position_blocked: bool = False
        self._orphan_position_details: list[dict[str, object]] = []
        self._startup_state_blocked: bool = False
        self._startup_state_details: list[dict[str, object]] = []
        self._model_shadow_decisions: list[dict[str, object]] = []
        self._model_shadow_decisions_max: int = 100
        self._warmup_results: list[WarmupResult] = []
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

    def _build_candidate_generator(self) -> BreakoutTrendCandidateGenerator:
        """Build candidate generator; apply DEMO-only overrides when mode is DEMO."""
        from dataclasses import replace

        cfg = CandidateGeneratorConfig()
        overrides = self._settings.runtime.demo_candidate_overrides
        more_opportunities = (
            self._settings.runtime.mode == RuntimeMode.DEMO
            and self._settings.runtime.demo_more_opportunities_enabled
        )
        if (
            self._settings.runtime.mode == RuntimeMode.DEMO
            and overrides is not None
            and any(
                getattr(overrides, k) is not None
                for k in ("min_breakout_bps", "min_trend_bps", "min_volume_multiplier")
            )
            and not more_opportunities
        ):
            updates: dict[str, object] = {}
            if overrides.min_breakout_bps is not None:
                updates["min_breakout_bps"] = Decimal(str(overrides.min_breakout_bps))
            if overrides.min_trend_bps is not None:
                updates["min_trend_bps"] = Decimal(str(overrides.min_trend_bps))
            if overrides.min_volume_multiplier is not None:
                updates["min_volume_multiplier"] = Decimal(str(overrides.min_volume_multiplier))
            cfg = replace(cfg, **updates)
            self._logger.info(
                "demo_candidate_overrides_applied",
                min_breakout_bps=float(cfg.min_breakout_bps),
                min_trend_bps=float(cfg.min_trend_bps),
                min_volume_multiplier=float(cfg.min_volume_multiplier),
            )
        if more_opportunities:
            profile = {
                "min_breakout_bps": Decimal("1"),
                "min_trend_bps": Decimal("2"),
                "min_volume_multiplier": Decimal("1.0"),
                "lookback_bars": 15,
            }
            cfg = replace(cfg, **profile)
            self._logger.info(
                "demo_more_opportunities_profile_applied",
                min_breakout_bps=1,
                min_trend_bps=2,
                min_volume_multiplier=1.0,
                lookback_bars=15,
            )
        return BreakoutTrendCandidateGenerator(config=cfg)

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
                await self._inspect_startup_exchange_state()
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

            self._logger.info(
                "warmup_starting",
                symbols=self._settings.trading.symbols,
                candle_tf=self._settings.trading.candle_timeframe,
                regime_tf=self._settings.trading.regime_timeframe,
            )
            warmup_results = await preload_warmup_klines(
                self._rest,
                self._bar_history,
                symbols=self._settings.trading.symbols,
                category=self._settings.trading.category,
                candle_timeframe=self._settings.trading.candle_timeframe,
                regime_timeframe=self._settings.trading.regime_timeframe,
                min_5m_bars=22,
                min_1h_bars=24,
            )
            self._warmup_results = warmup_results

            self._logger.info(
                "warmup_executed",
                symbols=self._settings.trading.symbols,
                candle_timeframe=self._settings.trading.candle_timeframe,
                regime_timeframe=self._settings.trading.regime_timeframe,
                results=[{"symbol": r.symbol, "tf": r.timeframe, "bars": r.bars_loaded, "satisfied": r.satisfied} for r in warmup_results],
            )
            for sym in self._settings.trading.symbols:
                counts: dict[str, int] = {}
                for tf in [self._settings.trading.candle_timeframe, self._settings.trading.regime_timeframe]:
                    counts[tf] = len(self._bar_history[sym][tf])
                self._logger.info("warmup_post_snapshot", symbol=sym, bar_counts=counts)

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

            self._init_model_filter()

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
            max_drill_notional_usdt=drill_cfg.max_notional_usdt,
        )
        if refuse:
            self._drill_outcome.attempted = True
            self._drill_outcome.aborted = True
            self._drill_outcome.refused_reason = refuse.reason
            self._drill_outcome.abort_details = refuse.details
            await self._ledger.record("drill_aborted", {"reason": refuse.reason, "details": refuse.details})
            self._logger.warning("demo_drill_aborted", reason=refuse.reason, **refuse.details)
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
                if link_id and self._drill_outcome.order_link_id == link_id and new_status:
                    self._drill_outcome.final_status = new_status
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
                    elif prev and not prev.metadata.get("drill"):
                        so = self._strategy_order_outcomes
                        if new_status == "New" and link_id not in so._seen_new:
                            so._seen_new.add(link_id)
                            self._logger.info(
                                "strategy_order_new",
                                order_link_id=link_id,
                                symbol=event.symbol or "",
                            )
                        elif new_status == "PartiallyFilled" and link_id not in so._seen_partially_filled:
                            so._seen_partially_filled.add(link_id)
                            so.partially_filled += 1
                            self._logger.info(
                                "strategy_order_partially_filled",
                                order_link_id=link_id,
                                symbol=event.symbol or "",
                            )
                        elif new_status == "Filled" and link_id not in so._seen_filled:
                            so._seen_filled.add(link_id)
                            so.filled += 1
                            self._logger.info(
                                "strategy_order_filled",
                                order_link_id=link_id,
                                symbol=event.symbol or "",
                            )
                            self._metrics.inc("entry_fill_received_count")
                            await self._ledger.record(
                                "entry_fill_received",
                                {
                                    "order_link_id": link_id,
                                    "symbol": event.symbol or "",
                                    "filled_qty": str(event.qty) if event.qty is not None else "",
                                    "avg_price": str(event.avg_price) if event.avg_price is not None else "",
                                },
                            )
                            self._logger.info(
                                "entry_fill_received",
                                order_link_id=link_id,
                                symbol=event.symbol or "",
                                filled_qty=str(event.qty) if event.qty else "",
                                avg_price=str(event.avg_price) if event.avg_price else "",
                            )
                            await self._place_protective_exit_after_fill(link_id=link_id, event=event, prev=prev)
                        elif new_status == "Cancelled" and link_id not in so._seen_cancelled:
                            so._seen_cancelled.add(link_id)
                            so.cancelled += 1
                            self._logger.info(
                                "strategy_order_cancelled",
                                order_link_id=link_id,
                                symbol=event.symbol or "",
                            )
                        elif new_status == "Rejected" and link_id not in so._seen_rejected:
                            so._seen_rejected.add(link_id)
                            so.rejected += 1
                            self._logger.info(
                                "strategy_order_rejected",
                                order_link_id=link_id,
                                symbol=event.symbol or "",
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
            self._logger.debug(
                "decision_loop_kline_consumed",
                symbol=kline.symbol,
                interval=kline.interval,
                confirmed=kline.confirmed,
            )
            bar = self._candle_builder.on_confirmed_kline(kline)
            if bar is None or not bar.confirmed:
                continue

            configured = self._settings.trading.symbols
            effective_symbol = bar.symbol or (
                configured[0] if len(configured) == 1 else ""
            )
            if not effective_symbol:
                self._logger.warning(
                    "decision_loop_empty_symbol_skip",
                    kline_symbol=kline.symbol,
                    bar_symbol=bar.symbol,
                    configured_symbols=configured,
                )
                continue

            bar_to_append: OHLCVBar = (
                bar
                if bar.symbol == effective_symbol
                else OHLCVBar(
                    symbol=effective_symbol,
                    timeframe=bar.timeframe,
                    open_time=bar.open_time,
                    close_time=bar.close_time,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    turnover=bar.turnover,
                    confirmed=bar.confirmed,
                )
            )
            if not bar.symbol:
                self._logger.info(
                    "decision_loop_symbol_fallback_applied",
                    kline_symbol=kline.symbol,
                    effective_symbol=effective_symbol,
                    timeframe=bar_to_append.timeframe,
                )

            history = self._bar_history[effective_symbol][bar_to_append.timeframe]
            history.append(bar_to_append)
            if bar_to_append.timeframe != self._settings.trading.candle_timeframe:
                continue

            bars_5m = list(self._bar_history[effective_symbol][self._settings.trading.candle_timeframe])
            bars_1h = list(self._bar_history[effective_symbol][self._settings.trading.regime_timeframe])
            readiness = self._candidate_generator.get_readiness(effective_symbol, bars_5m, bars_1h)
            candidates = self._candidate_generator.on_closed_candle(effective_symbol, bars_5m)
            raw_candidates_count = len(candidates)
            for _ in range(raw_candidates_count):
                self._metrics.inc("strategy_raw_candidates_total")
            readiness = dict(readiness)
            readiness["candidate_count"] = len(candidates)
            if readiness["reason"] == "ready" and len(candidates) == 0:
                readiness["reason"] = "no_pattern_match"

            if readiness["reason"] == "no_pattern_match":
                precondition = self._candidate_generator.get_precondition_report(
                    effective_symbol, bars_5m
                )
                if precondition is not None:
                    readiness["breakout_precondition"] = precondition.to_log_dict()
                    self._logger.info(
                        "breakout_precondition_no_match",
                        **precondition.to_log_dict(),
                    )
                    if (
                        self._settings.runtime.mode == RuntimeMode.DEMO
                        and self._settings.runtime.demo_relaxed_candidate_validation
                        and len(candidates) == 0
                        and precondition.failed_conditions
                    ):
                        relaxed = self._candidate_generator.create_relaxed_validation_candidates(
                            effective_symbol, precondition, bars_5m
                        )
                        for cand in relaxed:
                            meta = cand.metadata or {}
                            self._metrics.inc("strategy_relaxed_demo_candidates_created")
                            self._logger.info(
                                "demo_candidate_validation_relaxed",
                                symbol=cand.symbol,
                                candidate_type=cand.candidate_type.value,
                                side="Buy" if "long" in cand.candidate_type.value else "Sell",
                                original_failed_conditions=list(meta.get("original_failed_conditions", [])),
                                relaxed_reason=meta.get("relaxed_reason", "near_miss"),
                            )
                        candidates = relaxed

            self._last_candidate_readiness[effective_symbol] = readiness
            if len(candidates) == 0:
                self._logger.info(
                    "candidate_readiness",
                    symbol=effective_symbol,
                    bars_5m=readiness["bars_5m"],
                    bars_1h=readiness["bars_1h"],
                    has_enough_5m=readiness["has_enough_5m"],
                    has_enough_1h=readiness["has_enough_1h"],
                    unconfirmed_in_5m_window=readiness["unconfirmed_in_5m_window"],
                    reason=readiness["reason"],
                    candidate_count=len(candidates),
                    failed_conditions=readiness.get("breakout_precondition", {}).get(
                        "failed_conditions", []
                    ),
                )

            self._metrics.inc("strategy_bars_confirmed")
            open_orders = await self._order_manager.get_open_orders(None)
            for candidate in candidates:
                self._metrics.inc("strategy_candidates_total")
                regime, regime_report = self._regime_filter.evaluate_with_report(
                    candidate=candidate, bars_1h=bars_1h
                )
                if not regime.allow:
                    self._metrics.inc("strategy_regime_rejected")
                    self._last_regime_rejection = regime_report.to_log_dict()
                    readiness_with_regime = dict(self._last_candidate_readiness.get(effective_symbol, {}))
                    readiness_with_regime["regime_rejection"] = regime_report.to_log_dict()
                    self._last_candidate_readiness[effective_symbol] = readiness_with_regime
                    self._logger.info(
                        "regime_rejected_detail",
                        **regime_report.to_log_dict(),
                    )
                    continue
                signal = self._signal_engine.evaluate(candidate, regime)
                if signal.side is None or signal.reference_price is None:
                    self._metrics.inc("strategy_signal_rejected")
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

                sizing_inputs = SizingInputs(
                    equity_usdt=max(self._portfolio.equity_usdt, Decimal("1")),
                    confidence=signal.confidence,
                    volatility_bps=regime.volatility_bps,
                    reference_price=signal.reference_price,
                    max_leverage=self._settings.risk.max_leverage,
                )
                qty = self._sizer.size_qty(sizing_inputs, symbol_spec)
                if self._sizer._last_floor_applied and self._sizer._last_floor_details:
                    self._last_sizing_floor_applied = {
                        "symbol": signal.symbol,
                        **self._sizer._last_floor_details,
                    }
                    self._logger.info(
                        "sizing_floor_applied",
                        symbol=signal.symbol,
                        **self._sizer._last_floor_details,
                    )
                if qty <= 0:
                    self._metrics.inc("strategy_sizing_rejected")
                    reason = self._sizer.reject_reason(sizing_inputs, symbol_spec)
                    self._last_sizing_rejection = {
                        "symbol": signal.symbol,
                        "reason": reason or "qty_zero",
                        "equity_usdt": float(sizing_inputs.equity_usdt),
                        "confidence": float(sizing_inputs.confidence),
                        "volatility_bps": float(sizing_inputs.volatility_bps),
                        "reference_price": float(sizing_inputs.reference_price),
                        "min_qty": float(symbol_spec.min_qty),
                        "qty_step": float(symbol_spec.qty_step),
                    }
                    self._logger.info(
                        "sizing_rejected",
                        symbol=signal.symbol,
                        reason=reason or "qty_zero",
                        equity_usdt=float(sizing_inputs.equity_usdt),
                        confidence=float(sizing_inputs.confidence),
                        volatility_bps=float(sizing_inputs.volatility_bps),
                        reference_price=float(sizing_inputs.reference_price),
                        min_qty=float(symbol_spec.min_qty),
                    )
                    continue

                risk_ctx = RiskEvaluationContext(
                    candidate_type=candidate.candidate_type.value,
                    orphan_position_blocked=self._orphan_position_blocked,
                    current_open_orders_count=sum(1 for o in open_orders if o.symbol == signal.symbol),
                )
                risk, risk_report = self._risk_engine.evaluate_with_report(
                    signal=signal,
                    portfolio=self._portfolio,
                    expected_order_notional=qty * signal.reference_price,
                    context=risk_ctx,
                )
                if not risk.approved:
                    self._metrics.inc("strategy_risk_rejected")
                    report_dict = risk_report.to_log_dict()
                    self._last_risk_rejection = report_dict
                    readiness_with_risk = dict(self._last_candidate_readiness.get(effective_symbol, {}))
                    readiness_with_risk["risk_rejection"] = report_dict
                    self._last_candidate_readiness[effective_symbol] = readiness_with_risk
                    self._logger.info("risk_rejected_detail", **report_dict)
                    continue

                if self._model_filter_active:
                    from trading.models.filter_predictor import score_for_filter

                    mf_mode = self._model_filter_mode
                    model_filter_threshold = (
                        self._model_filter_threshold
                        if self._model_filter_threshold is not None
                        else 0.5
                    )
                    pred_result, allow = score_for_filter(
                        self._model_filter_model,
                        symbol=signal.symbol,
                        action=signal.action.value,
                        side=signal.side.value if signal.side else None,
                        qty=float(qty),
                        risk_approved=True,
                        reference_price=signal.reference_price,
                        confidence=signal.confidence,
                        ts_utc=utc_now(),
                        threshold=model_filter_threshold,
                    )
                    mf = self._strategy_order_outcomes.model_filter
                    mf.threshold = model_filter_threshold
                    mf.mode = mf_mode.value
                    self._metrics.inc("strategy_model_filter_reached")
                    if not pred_result.available:
                        mf.prediction_unavailable += 1
                        self._metrics.inc("strategy_model_blocked")
                        self._logger.info(
                            "model_filter_prediction_unavailable",
                            symbol=signal.symbol,
                            note=pred_result.feature_missing_note,
                        )
                        continue
                    prob = pred_result.prob_fill
                    mf.prob_count += 1
                    mf.prob_latest = prob
                    if mf.prob_min is None or prob < mf.prob_min:
                        mf.prob_min = prob
                    if mf.prob_max is None or prob > mf.prob_max:
                        mf.prob_max = prob
                    if pred_result.features_used is not None:
                        mf.latest_features = dict(pred_result.features_used)
                    would_block = not allow
                    bar_close_time = bars_5m[-1].close_time if bars_5m else None
                    self._record_model_decision(
                        symbol=signal.symbol,
                        candidate_type=candidate.candidate_type.value,
                        side=signal.side.value if signal.side else None,
                        bar_close_time=bar_close_time,
                        model_probability=float(prob),
                        threshold=model_filter_threshold,
                        shadow_would_block=would_block,
                        allow=allow if mf_mode != ModelFilterMode.SHADOW else None,
                        reference_price=signal.reference_price,
                        confidence=signal.confidence,
                        qty=qty,
                    )
                    if mf_mode == ModelFilterMode.SHADOW:
                        if would_block:
                            mf.shadow_would_have_blocked += 1
                            self._logger.info(
                                "model_filter_shadow_would_have_blocked",
                                symbol=signal.symbol,
                                prob_fill=prob,
                                threshold=model_filter_threshold,
                            )
                        else:
                            mf.allowed += 1
                            self._logger.info(
                                "model_filter_allowed",
                                symbol=signal.symbol,
                                prob_fill=prob,
                                threshold=model_filter_threshold,
                            )
                    elif would_block:
                        mf.blocked += 1
                        self._metrics.inc("strategy_model_blocked")
                        self._logger.info(
                            "model_filter_blocked",
                            symbol=signal.symbol,
                            prob_fill=prob,
                            threshold=model_filter_threshold,
                        )
                        continue
                    else:
                        mf.allowed += 1
                        self._logger.info(
                            "model_filter_allowed",
                            symbol=signal.symbol,
                            prob_fill=prob,
                            threshold=model_filter_threshold,
                        )

                if self._startup_state_blocked:
                    self._logger.info(
                        "startup_state_skipped_trade",
                        symbol=signal.symbol,
                        reason="startup_state_blocked",
                    )
                    continue
                if self._orphan_position_blocked:
                    self._logger.info(
                        "orphan_position_skipped_trade",
                        symbol=signal.symbol,
                        reason="orphan_position_blocked",
                    )
                    continue

                force_marketable = (
                    self._settings.runtime.mode == RuntimeMode.DEMO
                    and self._settings.runtime.demo_force_marketable_entries
                )
                intent = self._execution_engine.build_entry_intent(
                    signal=signal,
                    qty=qty,
                    reference_price=signal.reference_price,
                    now=utc_now(),
                    force_marketable=force_marketable,
                )
                if intent is None:
                    continue

                await self._order_manager.register_intent(intent)
                self._metrics.inc("order_intents_total")
                self._strategy_order_outcomes.intents += 1
                await self._ledger.record(
                    "order_intent",
                    {
                        "symbol": intent.symbol,
                        "side": intent.side.value,
                        "qty": str(intent.qty),
                        "order_link_id": intent.order_link_id,
                        "reduce_only": str(intent.reduce_only),
                        "reference_price": str(signal.reference_price),
                    },
                )
                self._logger.info(
                    "strategy_order_intent_created",
                    symbol=intent.symbol,
                    side=intent.side.value,
                    qty=str(intent.qty),
                    order_link_id=intent.order_link_id,
                    dry_run=self._settings.runtime.dry_run,
                )

                if self._can_place_exchange_orders():
                    await self._submit_intent(intent, is_drill=False)

    async def _place_protective_exit_after_fill(
        self,
        *,
        link_id: str,
        event: NormalizedOrderUpdate,
        prev: object,
    ) -> None:
        """On entry fill: create, register, and submit reduce-only protective exit. Orphan block remains if placement fails."""
        from trading.execution.order_manager import ManagedOrder

        if not isinstance(prev, ManagedOrder):
            return
        if prev.reduce_only:
            return
        signal_action = prev.metadata.get("signal_action")
        if signal_action == "enter_long":
            side_to_close = PositionSide.LONG
        elif signal_action == "enter_short":
            side_to_close = PositionSide.SHORT
        else:
            self._logger.warning(
                "protective_exit_placement_skipped",
                order_link_id=link_id,
                reason="unknown_signal_action",
                signal_action=signal_action,
            )
            return
        symbol = prev.symbol or event.symbol or ""
        if not symbol:
            self._logger.warning("protective_exit_placement_skipped", order_link_id=link_id, reason="missing_symbol")
            return
        filled_qty = prev.filled_qty if prev.filled_qty and prev.filled_qty > 0 else event.qty
        if not filled_qty or filled_qty <= 0:
            self._logger.warning(
                "protective_exit_placement_skipped",
                order_link_id=link_id,
                symbol=symbol,
                reason="zero_filled_qty",
            )
            return
        entry_avg_price = prev.avg_price or event.avg_price
        if not entry_avg_price or entry_avg_price <= 0:
            self._logger.warning(
                "protective_exit_placement_skipped",
                order_link_id=link_id,
                symbol=symbol,
                reason="missing_avg_price",
            )
            return
        symbol_spec = self._symbol_specs.get(symbol)
        if symbol_spec is None:
            self._logger.warning(
                "protective_exit_placement_skipped",
                order_link_id=link_id,
                symbol=symbol,
                reason="missing_symbol_spec",
            )
            return
        exit_intent = self._execution_engine.build_protective_limit_exit(
            symbol=symbol,
            side_to_close=side_to_close,
            qty=filled_qty,
            entry_avg_price=entry_avg_price,
            price_tick=symbol_spec.price_tick,
            qty_step=symbol_spec.qty_step,
            now=utc_now(),
        )
        if exit_intent is None:
            self._logger.warning(
                "protective_exit_placement_failed",
                order_link_id=link_id,
                symbol=symbol,
                reason="build_intent_returned_none",
            )
            self._metrics.inc("protective_exit_placement_failed_count")
            await self._ledger.record(
                "protective_exit_placement_failed",
                {
                    "entry_order_link_id": link_id,
                    "symbol": symbol,
                    "reason": "build_intent_returned_none",
                },
            )
            return
        self._metrics.inc("protective_exit_plan_created_count")
        await self._ledger.record(
            "protective_exit_plan_created",
            {
                "entry_order_link_id": link_id,
                "symbol": symbol,
                "qty": str(exit_intent.qty),
                "price": str(exit_intent.price) if exit_intent.price else "",
                "side": exit_intent.side.value,
            },
        )
        self._logger.info(
            "protective_exit_plan_created",
            entry_order_link_id=link_id,
            symbol=symbol,
            qty=str(exit_intent.qty),
            price=str(exit_intent.price) if exit_intent.price else "",
        )
        await self._order_manager.register_intent(exit_intent)
        self._metrics.inc("protective_exit_tracking_registered_count")
        await self._ledger.record(
            "protective_exit_tracking_registered",
            {"order_link_id": exit_intent.order_link_id, "symbol": symbol},
        )
        self._logger.info(
            "protective_exit_tracking_registered",
            order_link_id=exit_intent.order_link_id,
            symbol=symbol,
        )
        if not self._can_place_exchange_orders():
            self._logger.warning(
                "protective_exit_order_not_submitted",
                order_link_id=exit_intent.order_link_id,
                symbol=symbol,
                reason="placement_disabled",
            )
            self._metrics.inc("protective_exit_placement_failed_count")
            await self._ledger.record(
                "protective_exit_placement_failed",
                {
                    "order_link_id": exit_intent.order_link_id,
                    "symbol": symbol,
                    "reason": "placement_disabled",
                },
            )
            return
        self._metrics.inc("protective_exit_order_submitted_count")
        await self._ledger.record(
            "protective_exit_order_submitted",
            {
                "order_link_id": exit_intent.order_link_id,
                "symbol": symbol,
                "qty": str(exit_intent.qty),
            },
        )
        self._logger.info(
            "protective_exit_order_submitted",
            order_link_id=exit_intent.order_link_id,
            symbol=symbol,
            qty=str(exit_intent.qty),
        )
        try:
            await self._submit_intent(exit_intent, is_drill=False)
            order_after = await self._order_manager.get_by_link_id(exit_intent.order_link_id)
            if order_after and order_after.order_id:
                self._metrics.inc("protective_exit_order_ack_received_count")
                await self._ledger.record(
                    "protective_exit_order_ack_received",
                    {"order_link_id": exit_intent.order_link_id, "order_id": order_after.order_id, "symbol": symbol},
                )
                self._logger.info(
                    "protective_exit_order_ack_received",
                    order_link_id=exit_intent.order_link_id,
                    order_id=order_after.order_id,
                    symbol=symbol,
                )
        except BybitRestError as exc:
            reason = str(exc)
            if isinstance(exc, BybitAPIError):
                reason = f"ret_code={exc.ret_code} ret_msg={exc.ret_msg}"
            self._logger.warning(
                "protective_exit_placement_failed",
                order_link_id=exit_intent.order_link_id,
                symbol=symbol,
                reason=reason,
            )
            self._metrics.inc("protective_exit_placement_failed_count")
            await self._ledger.record(
                "protective_exit_placement_failed",
                {
                    "entry_order_link_id": link_id,
                    "order_link_id": exit_intent.order_link_id,
                    "symbol": symbol,
                    "reason": reason,
                },
            )

    async def _submit_intent(self, intent: object, *, is_drill: bool = False) -> None:
        from trading.execution.order_intent import OrderIntent

        if not isinstance(intent, OrderIntent):
            return
        self._metrics.inc("order_submissions_total")
        if not is_drill:
            self._strategy_order_outcomes.submissions += 1
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
        if not is_drill:
            self._logger.info(
                "strategy_order_submitted",
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
            else:
                self._strategy_order_outcomes.acks += 1
                self._logger.info(
                    "strategy_order_ack_received",
                    order_link_id=ack.order_link_id,
                    order_id=ack.order_id,
                )
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

    async def _inspect_startup_exchange_state(self) -> None:
        """Inspect exchange positions and orders at startup; block if dirty state detected."""
        if not self._has_auth_credentials():
            return
        try:
            positions: list = []
            exchange_orders: list = []
            for sym in self._settings.trading.symbols:
                pos_list = await self._rest.get_positions(
                    category=self._settings.trading.category, symbol=sym
                )
                positions.extend(pos_list)
                orders = await self._rest.get_open_orders(
                    category=self._settings.trading.category, symbol=sym
                )
                exchange_orders.extend(orders)
        except BybitRestError as exc:
            self._logger.warning("startup_exchange_inspect_failed", error=str(exc))
            return
        local_open = await self._order_manager.get_open_orders(None)
        local_order_state_empty = len(local_open) == 0
        details: list[dict[str, object]] = []
        dirty = False
        for sym in self._settings.trading.symbols:
            sym_positions = [p for p in positions if p.symbol == sym]
            sym_orders = [o for o in exchange_orders if o.symbol == sym]
            non_flat = any(p.size > 0 for p in sym_positions)
            pos_size = Decimal("0")
            pos_side = ""
            for p in sym_positions:
                if p.size > 0:
                    pos_size = p.size
                    pos_side = p.side or ""
                    break
            reduce_only_count = sum(1 for o in sym_orders if o.reduce_only)
            non_reduce_only_count = len(sym_orders) - reduce_only_count
            order_count = len(sym_orders)
            sym_dirty = non_flat or order_count > 0
            if sym_dirty:
                dirty = True
                d: dict[str, object] = {
                    "symbol": sym,
                    "position_size": float(pos_size),
                    "position_side": pos_side,
                    "open_order_count": order_count,
                    "reduce_only_order_count": reduce_only_count,
                    "non_reduce_only_order_count": non_reduce_only_count,
                    "local_order_state_empty_or_not": local_order_state_empty,
                }
                details.append(d)
        if dirty:
            self._logger.warning(
                "startup_dirty_exchange_state",
                details=details,
                local_order_state_empty=local_order_state_empty,
            )
            if not self._startup_state_blocked:
                self._metrics.inc("startup_state_blocked_count")
            self._startup_state_blocked = True
            self._startup_state_details = details
            self._logger.warning(
                "startup_state_blocked",
                details_count=len(details),
                note="Exchange has non-flat position or open orders. No new entries until resolved.",
            )
            await self._ledger.record(
                "startup_dirty_exchange_state",
                {"details": details, "local_order_state_empty": local_order_state_empty},
            )

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
            if not self._startup_state_blocked:
                self._metrics.inc("startup_state_blocked_count")
            self._startup_state_blocked = True
            orphan_issues = [i for i in report_positions.issues if i.issue_type == "missing_reduce_only_exit"]
            if orphan_issues:
                self._startup_state_details = [
                    {
                        "symbol": i.symbol or "unknown",
                        "position_size": float(i.position_size) if i.position_size is not None else None,
                        "position_side": i.position_side or "unknown",
                        "reason": "reconcile_missing_reduce_only_exit",
                    }
                    for i in orphan_issues
                ]
            if orphan_issues:
                if not self._orphan_position_blocked:
                    self._metrics.inc("orphan_position_blocked_count")
                self._orphan_position_blocked = True
                self._orphan_position_details = [
                    {
                        "symbol": i.symbol or "unknown",
                        "position_size": float(i.position_size) if i.position_size is not None else None,
                        "side": i.position_side or "unknown",
                        "reason": "non_flat_position_no_tracked_reduce_only_exit",
                    }
                    for i in orphan_issues
                ]
                for d in self._orphan_position_details:
                    self._logger.warning(
                        "orphan_position_blocked",
                        symbol=d["symbol"],
                        position_size=d["position_size"],
                        side=d["side"],
                        reason=d["reason"],
                    )
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
            if self._orphan_position_blocked:
                self._orphan_position_blocked = False
                self._metrics.inc("orphan_position_block_cleared_count")
                self._orphan_position_details = []
                self._logger.info("orphan_position_block_cleared", note="position_flat_or_protected")
            if self._startup_state_blocked:
                self._startup_state_blocked = False
                self._metrics.inc("startup_state_block_cleared_count")
                self._startup_state_details = []
                self._logger.info("startup_state_block_cleared", note="exchange_flat_no_unsafe_orders")
            await self._ledger.record(
                "reconcile_ok",
                {"order_issues": 0, "position_issues": 0},
            )
            if self._drill_outcome.order_link_id and self._drill_outcome.attempted:
                post_ack = _drill_post_ack_status(self._drill_outcome)
                payload: dict[str, object] = {
                    "order_link_id": self._drill_outcome.order_link_id,
                    "mismatch": False,
                }
                if post_ack == "resting_open":
                    payload["note"] = "drill_order_resting_open"
                elif post_ack == "ack_only_no_transition":
                    payload["note"] = "drill_ack_received_no_further_transition"
                await self._ledger.record("drill_reconcile_result", payload)
            open_orders = await self._order_manager.get_open_orders(None)
            drill_link = self._drill_outcome.order_link_id or ""
            strategy_resting = [
                o for o in open_orders
                if not o.metadata.get("drill") and o.order_link_id != drill_link
            ]
            if strategy_resting:
                await self._ledger.record(
                    "reconcile_strategy_orders_resting",
                    {
                        "count": len(strategy_resting),
                        "order_link_ids": [o.order_link_id for o in strategy_resting],
                        "note": "strategy_orders_resting_at_reconcile",
                    },
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

    def _record_model_decision(
        self,
        *,
        symbol: str,
        candidate_type: str,
        side: str | None,
        bar_close_time: datetime | None,
        model_probability: float,
        threshold: float,
        shadow_would_block: bool,
        allow: bool | None = None,
        reference_price: Decimal | None = None,
        confidence: Decimal | None = None,
        qty: Decimal | None = None,
    ) -> None:
        """Record per-candidate model decision for shadow evaluation. Bounded to last N."""
        ts = utc_now()
        record: dict[str, object] = {
            "symbol": symbol,
            "candidate_type": candidate_type,
            "side": side,
            "timestamp": ts.isoformat(),
            "bar_close_time": bar_close_time.isoformat() if bar_close_time else None,
            "model_probability": model_probability,
            "threshold": threshold,
            "shadow_would_block": shadow_would_block,
            "strategy_submitted": False,
            "blocking_stage": "model_evaluated",
        }
        if allow is not None:
            record["allow"] = allow
        if reference_price is not None:
            record["reference_price"] = float(reference_price)
        if confidence is not None:
            record["confidence"] = float(confidence)
        if qty is not None:
            record["qty"] = float(qty)
        while len(self._model_shadow_decisions) >= self._model_shadow_decisions_max:
            self._model_shadow_decisions.pop(0)
        self._model_shadow_decisions.append(record)
        if self._model_filter_mode == ModelFilterMode.SHADOW:
            self._logger.info(
                "model_filter_shadow_decision",
                symbol=symbol,
                candidate_type=candidate_type,
                side=side,
                timestamp=record["timestamp"],
                bar_close_time=record["bar_close_time"],
                model_probability=model_probability,
                threshold=threshold,
                shadow_would_block=shadow_would_block,
            )
        else:
            self._logger.info(
                "model_filter_active_decision",
                symbol=symbol,
                candidate_type=candidate_type,
                side=side,
                model_probability=model_probability,
                threshold=threshold,
                allow=allow,
            )

    def _init_model_filter(self) -> None:
        """Load model artifact and enable DEMO-only filter. Never active in LIVE."""
        from trading.models.filter_artifact import load_model_artifact
        from trading.models.filter_predictor import RUNTIME_FEATURE_NAMES
        from trading.research.datasets.prepare import FEATURE_NAMES

        enabled = self._settings.runtime.model_filter_enabled
        path = self._settings.runtime.model_artifact_path

        if self._settings.runtime.mode != RuntimeMode.DEMO:
            if enabled:
                self._logger.warning(
                    "model_filter_disabled_not_demo",
                    mode=self._settings.runtime.mode.value,
                    message="Model filter is DEMO-only; disabled in non-DEMO mode.",
                )
            self._logger.info(
                "model_filter_status",
                model_filter_enabled=False,
                model_loaded=False,
                model_filter_active=False,
            )
            return

        if not enabled:
            self._logger.info(
                "model_filter_status",
                model_filter_enabled=False,
                model_loaded=False,
                model_filter_active=False,
            )
            return

        if path is None or not path:
            self._logger.info(
                "model_filter_status",
                model_filter_enabled=True,
                model_loaded=False,
                model_filter_active=False,
                note="model_artifact_path not set",
            )
            return

        result = load_model_artifact(path)
        if not result.loaded:
            self._logger.info(
                "model_filter_status",
                model_filter_enabled=True,
                model_loaded=False,
                model_filter_active=False,
                path=str(result.path),
                error=result.error,
            )
            return

        if tuple(FEATURE_NAMES) != tuple(RUNTIME_FEATURE_NAMES):
            self._logger.warning(
                "model_filter_feature_mismatch",
                offline_features=list(FEATURE_NAMES),
                runtime_features=list(RUNTIME_FEATURE_NAMES),
                message="Runtime features differ from offline expectations; predictions may be unreliable.",
            )

        self._model_filter_model = result.model
        self._model_filter_active = True
        mf_mode = self._settings.runtime.model_filter_mode
        mf_threshold = self._settings.runtime.model_filter_threshold
        self._model_filter_mode = mf_mode
        self._model_filter_threshold = mf_threshold
        effective_threshold = mf_threshold if mf_threshold is not None else 0.5
        self._strategy_order_outcomes.model_filter.threshold = effective_threshold
        self._strategy_order_outcomes.model_filter.mode = mf_mode.value
        self._logger.info(
            "model_filter_status",
            model_filter_enabled=True,
            model_loaded=True,
            model_filter_active=True,
            path=str(result.path),
            mode=mf_mode.value,
            threshold=effective_threshold,
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

    def _infer_strategy_blocking_stage(self, m: dict[str, float]) -> str:
        """Infer which stage is blocking decisions for operator visibility."""
        if self._startup_state_blocked:
            return "startup_state_blocked"
        intents = int(m.get("order_intents_total", 0))
        if intents > 0:
            return "submitted"
        bars = int(m.get("strategy_bars_confirmed", 0))
        if bars == 0:
            return "no_bars"
        candidates = int(m.get("strategy_candidates_total", 0))
        if candidates == 0:
            return "no_candidates"
        regime_rej = int(m.get("strategy_regime_rejected", 0))
        signal_rej = int(m.get("strategy_signal_rejected", 0))
        sizing_rej = int(m.get("strategy_sizing_rejected", 0))
        risk_rej = int(m.get("strategy_risk_rejected", 0))
        model_reached = int(m.get("strategy_model_filter_reached", 0))
        model_blocked = int(m.get("strategy_model_blocked", 0))
        if regime_rej > 0 and regime_rej >= candidates:
            return "regime_rejected"
        if signal_rej > 0 or sizing_rej > 0:
            passed_regime = candidates - regime_rej
            if signal_rej + sizing_rej >= passed_regime:
                return "signal_rejected"
        if risk_rej > 0:
            return "risk_rejected"
        if self._model_filter_active and model_reached > 0 and model_blocked >= model_reached:
            return "model_blocked"
        return "submitted"

    async def _runtime_summary_cycle(self) -> None:
        """Concise periodic runtime summary for paper/demo modes."""
        health_snap = self._health.snapshot()
        metrics_snap = self._metrics.snapshot()
        equity = float(self._portfolio.equity_usdt)
        decisions = metrics_snap.counters.get("decisions_total", 0)
        c = metrics_snap.counters
        strategy_flow = {
            "bars_confirmed": int(c.get("strategy_bars_confirmed", 0)),
            "candidates": int(c.get("strategy_candidates_total", 0)),
            "regime_rejected": int(c.get("strategy_regime_rejected", 0)),
            "signal_rejected": int(c.get("strategy_signal_rejected", 0)),
            "sizing_rejected": int(c.get("strategy_sizing_rejected", 0)),
            "risk_rejected": int(c.get("strategy_risk_rejected", 0)),
            "model_filter_reached": int(c.get("strategy_model_filter_reached", 0)),
            "model_blocked": int(c.get("strategy_model_blocked", 0)),
            "submitted": int(c.get("order_intents_total", 0)),
        }
        model_filter_calibration: dict[str, object] | None = None
        if self._model_filter_active:
            mf = self._strategy_order_outcomes.model_filter
            model_filter_calibration = {
                "mode": mf.mode,
                "threshold": mf.threshold,
                "blocked": mf.blocked,
                "allowed": mf.allowed,
                "shadow_would_have_blocked": mf.shadow_would_have_blocked,
                "prob_count": mf.prob_count,
                "prob_min": mf.prob_min,
                "prob_max": mf.prob_max,
                "prob_latest": mf.prob_latest,
            }
            decisions = list(self._model_shadow_decisions)
            probs = [float(d.get("model_probability", 0)) for d in decisions if "model_probability" in d]
            if probs:
                from trading.runtime.model_calibration import build_runtime_calibration_stats
                rc = build_runtime_calibration_stats(probs, current_threshold=mf.threshold)
                model_filter_calibration["shadow_calibration"] = {
                    "total_evaluations": rc["total_shadow_evaluations"],
                    "prob_max": rc["probability_distribution"]["max"],
                    "prob_p95": rc["probability_distribution"]["p95"],
                    "prob_p99": rc["probability_distribution"]["p99"],
                    "threshold_above_observed_max": rc["current_threshold_above_observed_max"],
                    "retention_thresholds": rc["retention_thresholds"],
                    "suggested_thresholds_when_above_max": rc.get("suggested_thresholds_when_above_max"),
                }
                sug = rc.get("suggested_thresholds_when_above_max")
                if rc.get("current_threshold_above_observed_max") and sug:
                    self._logger.info(
                        "model_filter_threshold_recommendation",
                        current_threshold=mf.threshold,
                        prob_max=rc["probability_distribution"]["max"],
                        prob_p95=rc["probability_distribution"]["p95"],
                        prob_p99=rc["probability_distribution"]["p99"],
                        threshold_above_observed_max=True,
                        suggested_threshold_near_max=sug.get("threshold_near_max"),
                        suggested_threshold_near_p99=sug.get("threshold_near_p99"),
                        suggested_threshold_near_p95=sug.get("threshold_near_p95"),
                        suggested_threshold_keep_50pct=sug.get("threshold_keep_50pct"),
                        suggested_threshold_keep_25pct=sug.get("threshold_keep_25pct"),
                    )
        blocking_stage = self._infer_strategy_blocking_stage(c)
        candidate_readiness = dict(self._last_candidate_readiness)
        log_payload: dict[str, object] = {
            "mode": self._settings.runtime.mode.value,
            "equity_usdt": equity,
            "ws_public": health_snap.ws_public_connected,
            "ws_private": health_snap.ws_private_connected,
            "private_stream_error": health_snap.private_stream_error,
            "circuit_breaker": health_snap.circuit_breaker_tripped,
            "stale_count": len(health_snap.stale_channels),
            "decisions_total": decisions,
            "strategy_flow": strategy_flow,
            "blocking_stage": blocking_stage,
            "candidate_readiness": candidate_readiness,
        }
        if self._settings.runtime.mode == RuntimeMode.DEMO:
            log_payload["demo_relaxed_candidate_validation"] = self._settings.runtime.demo_relaxed_candidate_validation
            log_payload["demo_validation_candidates_created"] = int(c.get("strategy_relaxed_demo_candidates_created", 0))
            raw = int(c.get("strategy_raw_candidates_total", 0))
            relaxed = int(c.get("strategy_relaxed_demo_candidates_created", 0))
            regime_rej = int(c.get("strategy_regime_rejected", 0))
            signal_rej = int(c.get("strategy_signal_rejected", 0))
            sizing_rej = int(c.get("strategy_sizing_rejected", 0))
            risk_rej = int(c.get("strategy_risk_rejected", 0))
            total_cand = int(c.get("strategy_candidates_total", 0))
            model_reached = int(c.get("strategy_model_filter_reached", 0))
            model_blocked = int(c.get("strategy_model_blocked", 0))
            submitted = int(c.get("order_intents_total", 0))
            log_payload["candidate_pipeline_detail"] = {
                "raw_candidates": raw,
                "relaxed_demo_candidates": relaxed,
                "regime_passed": total_cand - regime_rej,
                "signal_passed": total_cand - regime_rej - signal_rej,
                "sizing_passed": total_cand - regime_rej - signal_rej - sizing_rej,
                "risk_passed": total_cand - regime_rej - signal_rej - sizing_rej - risk_rej,
                "model_reached": model_reached,
                "model_blocked": model_blocked,
                "submitted": submitted,
            }
        if self._orphan_position_blocked:
            log_payload["orphan_position_blocked"] = True
            log_payload["orphan_position_details"] = list(self._orphan_position_details)
        if self._last_risk_rejection is not None:
            log_payload["last_risk_rejection"] = self._last_risk_rejection
        if self._startup_state_blocked:
            log_payload["startup_state_blocked"] = True
            log_payload["startup_state_details"] = self._startup_state_details
        if self._settings.runtime.mode == RuntimeMode.DEMO:
            log_payload["demo_more_opportunities_enabled"] = self._settings.runtime.demo_more_opportunities_enabled
            log_payload["demo_force_marketable_entries"] = self._settings.runtime.demo_force_marketable_entries
        if model_filter_calibration is not None:
            log_payload["model_filter_calibration"] = model_filter_calibration
        self._logger.info("runtime_summary", **log_payload)

    async def _build_session_summary(self) -> dict[str, object]:
        """Build concise session summary from metrics and session state."""
        metrics = self._metrics.snapshot()
        start = self._session_start_time or utc_now()
        end = utc_now()
        so = self._strategy_order_outcomes
        open_orders = await self._order_manager.get_open_orders(None)
        drill_link = self._drill_outcome.order_link_id or ""
        strategy_resting_opens = sum(
            1 for o in open_orders if not o.metadata.get("drill") and o.order_link_id != drill_link
        )
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
            "strategy_flow": {
                "bars_confirmed": int(metrics.counters.get("strategy_bars_confirmed", 0)),
                "candidates": int(metrics.counters.get("strategy_candidates_total", 0)),
                "regime_rejected": int(metrics.counters.get("strategy_regime_rejected", 0)),
                "signal_rejected": int(metrics.counters.get("strategy_signal_rejected", 0)),
                "sizing_rejected": int(metrics.counters.get("strategy_sizing_rejected", 0)),
                "risk_rejected": int(metrics.counters.get("strategy_risk_rejected", 0)),
                "model_filter_reached": int(metrics.counters.get("strategy_model_filter_reached", 0)),
                "model_blocked": int(metrics.counters.get("strategy_model_blocked", 0)),
                "submitted": int(metrics.counters.get("order_intents_total", 0)),
            },
            "blocking_stage": self._infer_strategy_blocking_stage(metrics.counters),
            "candidate_readiness": dict(self._last_candidate_readiness),
            "warmup_results": [
                {"symbol": r.symbol, "timeframe": r.timeframe, "bars_loaded": r.bars_loaded, "min_required": r.min_required, "satisfied": r.satisfied}
                for r in self._warmup_results
            ],
        }
        if self._settings.runtime.mode == RuntimeMode.DEMO:
            summary["demo_more_opportunities_enabled"] = self._settings.runtime.demo_more_opportunities_enabled
            summary["demo_force_marketable_entries"] = self._settings.runtime.demo_force_marketable_entries
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
            if self._drill_outcome.final_status:
                summary["drill_final_status"] = self._drill_outcome.final_status
            summary["drill_post_ack_status"] = _drill_post_ack_status(self._drill_outcome)
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
        if self._last_sizing_rejection is not None:
            summary["last_sizing_rejection"] = dict(self._last_sizing_rejection)
        if self._last_sizing_floor_applied is not None:
            summary["last_sizing_floor_applied"] = dict(self._last_sizing_floor_applied)
        if self._last_regime_rejection is not None:
            summary["last_regime_rejection"] = dict(self._last_regime_rejection)
        if self._last_risk_rejection is not None:
            summary["last_risk_rejection"] = dict(self._last_risk_rejection)
        if self._orphan_position_blocked:
            summary["orphan_position_blocked"] = True
            summary["orphan_position_details"] = list(self._orphan_position_details)
        if self._startup_state_blocked:
            summary["startup_state_blocked"] = True
            summary["startup_state_details"] = list(self._startup_state_details)
        if self._startup_auth_disabled:
            summary["startup_auth_disabled"] = True
        summary["strategy_order_outcomes"] = {
            "intents": so.intents,
            "submissions": so.submissions,
            "acks": so.acks,
            "resting_opens": strategy_resting_opens,
            "partially_filled": so.partially_filled,
            "filled": so.filled,
            "cancelled": so.cancelled,
            "rejected": so.rejected,
        }
        summary["model_filter"] = {
            "enabled": self._settings.runtime.model_filter_enabled,
            "active": self._model_filter_active,
            "model_loaded": self._model_filter_model is not None,
            "mode": so.model_filter.mode,
            "threshold": so.model_filter.threshold,
            "blocked": so.model_filter.blocked,
            "allowed": so.model_filter.allowed,
            "shadow_would_have_blocked": so.model_filter.shadow_would_have_blocked,
            "prediction_unavailable": so.model_filter.prediction_unavailable,
            "prob_min": so.model_filter.prob_min,
            "prob_max": so.model_filter.prob_max,
            "prob_latest": so.model_filter.prob_latest,
            "prob_count": so.model_filter.prob_count,
            "latest_features": dict(so.model_filter.latest_features) if so.model_filter.latest_features else None,
        }
        decisions = list(self._model_shadow_decisions)
        if decisions:
            probs = [float(d.get("model_probability", 0)) for d in decisions if "model_probability" in d]
            active_decisions = [d for d in decisions if "allow" in d]
            active_blocked = sum(1 for d in active_decisions if d.get("allow") is False)
            active_allowed = sum(1 for d in active_decisions if d.get("allow") is True)
            latest_active = active_decisions[-1] if active_decisions else None
            summary["model_shadow_decisions"] = {
                "decisions": decisions,
                "total_model_evaluations": len(decisions),
                "shadow_would_block_count": sum(1 for d in decisions if d.get("shadow_would_block")),
                "shadow_would_allow_count": sum(1 for d in decisions if not d.get("shadow_would_block")),
                "active_blocked_count": active_blocked,
                "active_allowed_count": active_allowed,
                "latest_active_decision": latest_active,
                "avg_probability": round(sum(probs) / len(probs), 6) if probs else None,
                "min_probability": min(probs) if probs else None,
                "max_probability": max(probs) if probs else None,
            }
            so = self._strategy_order_outcomes
            threshold_cfg = self._strategy_order_outcomes.model_filter.threshold
            cal = build_model_calibration_summary(
                decisions,
                threshold_configured=threshold_cfg,
                session_submitted=so.intents,
                session_filled=so.filled,
            )
            summary["model_calibration"] = cal
            if rc := cal.get("runtime_calibration"):
                dist = rc.get("probability_distribution", {})
                summary["promotion_recommendation"] = build_promotion_recommendation(
                    current_threshold=threshold_cfg,
                    observed_max=dist.get("max"),
                    observed_p95=dist.get("p95"),
                    observed_p99=dist.get("p99"),
                    observed_min=dist.get("min"),
                    observed_mean=dist.get("mean"),
                    retention_thresholds=rc.get("retention_thresholds"),
                )
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
            f"- Blocking stage: {summary.get('blocking_stage', 'unknown')}",
            "",
        ]
        if summary.get("mode") == "demo" and (
            summary.get("demo_more_opportunities_enabled") or summary.get("demo_force_marketable_entries")
        ):
            lines.append("## Demo Profile")
            if summary.get("demo_more_opportunities_enabled"):
                lines.append("- More opportunities: enabled")
            if summary.get("demo_force_marketable_entries"):
                lines.append("- Force marketable entries: enabled (MARKET/IOC for validation)")
            lines.append("")
        if warmup := summary.get("warmup_results"):
            lines.append("## Warmup Preload")
            for r in warmup:
                lines.append(f"- {r.get('symbol', '')} {r.get('timeframe', '')}: loaded={r.get('bars_loaded', 0)} required={r.get('min_required', 0)} satisfied={r.get('satisfied', False)}")
            lines.append("")
        if readiness := summary.get("candidate_readiness"):
            lines.append("## Candidate Readiness")
            for sym, r in sorted(readiness.items()):
                lines.append(f"- {sym}: bars_5m={r.get('bars_5m', 0)} bars_1h={r.get('bars_1h', 0)} "
                    f"enough_5m={r.get('has_enough_5m', False)} enough_1h={r.get('has_enough_1h', False)} "
                    f"reason={r.get('reason', '')} candidates={r.get('candidate_count', 0)}")
                if bp := r.get("breakout_precondition"):
                    failed = bp.get("failed_conditions", [])
                    lines.append(f"  - breakout_precondition: failed={failed} "
                        f"up_bps={bp.get('breakout_up_bps')} dn_bps={bp.get('breakout_dn_bps')} "
                        f"move_bps={bp.get('candle_move_bps')} vol_mult={bp.get('vol_multiplier')}")
                if rr := r.get("regime_rejection"):
                    failed = rr.get("failed_conditions", [])
                    lines.append(f"  - regime_rejection: reason={rr.get('reason', '')} failed={failed} "
                        f"state={rr.get('state', '')} trend_bps={rr.get('trend_bps')} vol_bps={rr.get('volatility_bps')}")
                if rkr := r.get("risk_rejection"):
                    failed = rkr.get("failed_conditions", [])
                    lines.append(f"  - risk_rejection: reason={rkr.get('reason', '')} failed={failed} "
                        f"projected_notional={rkr.get('projected_notional')} max_notional={rkr.get('max_total_notional_usdt')}")
            lines.append("")
        if flow := summary.get("strategy_flow"):
            lines.append("## Strategy Flow")
            f = flow
            lines.append(f"- Bars confirmed: {f.get('bars_confirmed', 0)}")
            lines.append(f"- Candidates: {f.get('candidates', 0)}")
            lines.append(f"- Regime rejected: {f.get('regime_rejected', 0)}")
            lines.append(f"- Signal rejected: {f.get('signal_rejected', 0)}")
            lines.append(f"- Sizing rejected: {f.get('sizing_rejected', 0)}")
            lines.append(f"- Risk rejected: {f.get('risk_rejected', 0)}")
            lines.append(f"- Model filter reached: {f.get('model_filter_reached', 0)}")
            lines.append(f"- Model blocked: {f.get('model_blocked', 0)}")
            lines.append(f"- Submitted: {f.get('submitted', 0)}")
            lines.append("")
        if lsr := summary.get("last_sizing_rejection"):
            lines.append("## Last Sizing Rejection")
            lines.append(f"- Symbol: {lsr.get('symbol', '')}")
            lines.append(f"- Reason: {lsr.get('reason', '')}")
            lines.append(f"- equity_usdt={lsr.get('equity_usdt')} confidence={lsr.get('confidence')} volatility_bps={lsr.get('volatility_bps')}")
            lines.append(f"- reference_price={lsr.get('reference_price')} min_qty={lsr.get('min_qty')}")
            lines.append("")
        if lrr := summary.get("last_regime_rejection"):
            lines.append("## Last Regime Rejection")
            lines.append(f"- Symbol: {lrr.get('symbol', '')}")
            lines.append(f"- Reason: {lrr.get('reason', '')}")
            lines.append(f"- Failed conditions: {lrr.get('failed_conditions', [])}")
            lines.append(f"- State: {lrr.get('state', '')} candidate_type: {lrr.get('candidate_type', '')}")
            lines.append(f"- volatility_bps={lrr.get('volatility_bps')} trend_bps={lrr.get('trend_bps')} adaptive_threshold={lrr.get('adaptive_trend_threshold_bps')}")
            lines.append("")
        if lkr := summary.get("last_risk_rejection"):
            lines.append("## Last Risk Rejection")
            lines.append(f"- Symbol: {lkr.get('symbol', '')}")
            lines.append(f"- Reason: {lkr.get('reason', '')}")
            lines.append(f"- Failed conditions: {lkr.get('failed_conditions', [])}")
            lines.append(f"- Side: {lkr.get('side', '')} candidate_type: {lkr.get('candidate_type', '')}")
            lines.append(f"- notional={lkr.get('notional')} projected_notional={lkr.get('projected_notional')} max_total_notional_usdt={lkr.get('max_total_notional_usdt')}")
            lines.append(f"- max_leverage={lkr.get('max_leverage')} effective_leverage={lkr.get('effective_leverage')} position_leverage={lkr.get('position_leverage')}")
            lines.append(f"- confidence={lkr.get('confidence')} min_confidence_threshold={lkr.get('min_confidence_threshold')}")
            lines.append(f"- realized_pnl_today_usdt={lkr.get('realized_pnl_today_usdt')} daily_loss_limit_usdt={lkr.get('daily_loss_limit_usdt')}")
            lines.append("")
        if summary.get("startup_state_blocked"):
            lines.append("## Last Startup Dirty State")
            lines.append("- Non-flat position or unexpected open orders detected at startup/reconcile. No new entries until resolved.")
            for d in summary.get("startup_state_details", []):
                parts = [f"{d.get('symbol', '')}: position_size={d.get('position_size')} side={d.get('position_side', '')}"]
                if "open_order_count" in d:
                    parts.append(f"open_orders={d.get('open_order_count')} reduce_only={d.get('reduce_only_order_count')} non_reduce_only={d.get('non_reduce_only_order_count')}")
                if "local_order_state_empty_or_not" in d:
                    parts.append(f"local_empty={d.get('local_order_state_empty_or_not')}")
                if "reason" in d:
                    parts.append(f"reason={d.get('reason')}")
                lines.append("- " + " ".join(parts))
            lines.append("")
        if summary.get("orphan_position_blocked"):
            lines.append("## Orphan Position Blocked (SAFETY)")
            lines.append("- Non-flat exchange position has no local tracked reduce-only exit order.")
            lines.append("- Trading blocked until operator resolves manually.")
            for d in summary.get("orphan_position_details", []):
                lines.append(f"- {d.get('symbol', '')}: size={d.get('position_size')} side={d.get('side', '')} reason={d.get('reason', '')}")
            lines.append("")
        if lsf := summary.get("last_sizing_floor_applied"):
            lines.append("## Last Sizing Floor Applied (DEMO min-notional)")
            lines.append(f"- Symbol: {lsf.get('symbol', '')}")
            lines.append(f"- original_notional={lsf.get('original_notional')} effective_notional={lsf.get('effective_notional')} qty={lsf.get('qty')}")
            lines.append("")
        if summary.get("drill_enabled"):
            lines.append("## Demo Drill")
            lines.append(f"- Enabled: {summary.get('drill_enabled', False)}")
            lines.append(f"- Attempted: {summary.get('drill_attempted', False)}")
            if summary.get("drill_symbol"):
                lines.append(f"- Symbol/Side/Qty: {summary.get('drill_symbol')} {summary.get('drill_side', '')} {summary.get('drill_qty', '')}")
            lines.append(f"- Ack received: {summary.get('drill_ack_received', False)}")
            if summary.get("drill_final_status"):
                lines.append(f"- Final status: {summary.get('drill_final_status')}")
            lines.append(f"- Post-ack status: {summary.get('drill_post_ack_status', '')}")
            lines.append(f"- Reconcile mismatch: {summary.get('drill_reconcile_mismatch', False)}")
            lines.append(f"- Outcome: {summary.get('drill_outcome', 'pending')}")
            if summary.get("drill_refused_reason"):
                lines.append(f"- Refused reason: {summary.get('drill_refused_reason')}")
            if details := summary.get("drill_abort_details"):
                lines.append("- Abort details:")
                for k, v in details.items():
                    lines.append(f"  - {k}: {v}")
            lines.append("")
        if outcomes := summary.get("strategy_order_outcomes"):
            lines.append("## Strategy Order Outcomes")
            o = outcomes
            lines.append(f"- Intents: {o.get('intents', 0)}")
            lines.append(f"- Submissions: {o.get('submissions', 0)}")
            lines.append(f"- Acks: {o.get('acks', 0)}")
            lines.append(f"- Resting opens: {o.get('resting_opens', 0)}")
            lines.append(f"- Partially filled: {o.get('partially_filled', 0)}")
            lines.append(f"- Filled: {o.get('filled', 0)}")
            lines.append(f"- Cancelled: {o.get('cancelled', 0)}")
            lines.append(f"- Rejected: {o.get('rejected', 0)}")
            lines.append("")
        if mf := summary.get("model_filter"):
            lines.append("## Model Filter (DEMO-only)")
            lines.append(f"- Enabled: {mf.get('enabled', False)}")
            lines.append(f"- Active: {mf.get('active', False)}")
            lines.append(f"- Mode: {mf.get('mode', 'hard_block')}")
            lines.append(f"- Model loaded: {mf.get('model_loaded', False)}")
            lines.append(f"- Threshold: {mf.get('threshold', 0.5)}")
            lines.append(f"- Trades allowed by model: {mf.get('allowed', 0)}")
            lines.append(f"- Trades blocked by model: {mf.get('blocked', 0)}")
            if mf.get("shadow_would_have_blocked", 0) > 0:
                lines.append(f"- Shadow would have blocked: {mf.get('shadow_would_have_blocked', 0)}")
            lines.append(f"- Prediction unavailable: {mf.get('prediction_unavailable', 0)}")
            if mf.get("prob_count", 0) > 0:
                lines.append(f"- Prob stats: min={mf.get('prob_min')} max={mf.get('prob_max')} latest={mf.get('prob_latest')} count={mf.get('prob_count')}")
            if lf := mf.get("latest_features"):
                lines.append(f"- Latest features: reference_price={lf.get('reference_price')} confidence={lf.get('confidence')} qty={lf.get('qty')} ts_ordinal={lf.get('ts_ordinal')}")
            lines.append("")
        if msd := summary.get("model_shadow_decisions"):
            lines.append("## Model Shadow Evaluation")
            lines.append(f"- Total model evaluations: {msd.get('total_model_evaluations', 0)}")
            lines.append(f"- Shadow would block: {msd.get('shadow_would_block_count', 0)}")
            lines.append(f"- Shadow would allow: {msd.get('shadow_would_allow_count', 0)}")
            if (msd.get("active_blocked_count") or 0) > 0 or (msd.get("active_allowed_count") or 0) > 0:
                lines.append(f"- Active blocked: {msd.get('active_blocked_count', 0)}")
                lines.append(f"- Active allowed: {msd.get('active_allowed_count', 0)}")
            if latest := msd.get("latest_active_decision"):
                lines.append(f"- Latest active decision: {latest.get('symbol', '')} {latest.get('candidate_type', '')} "
                    f"side={latest.get('side', '')} prob={latest.get('model_probability')} "
                    f"threshold={latest.get('threshold')} allow={latest.get('allow')}")
            if msd.get("avg_probability") is not None:
                lines.append(f"- Prob: avg={msd.get('avg_probability')} min={msd.get('min_probability')} max={msd.get('max_probability')}")
            lines.append("")
            lines.append("## Recent Model Shadow Decisions")
            for d in msd.get("decisions", [])[-20:]:
                sym = d.get("symbol", "")
                ct = d.get("candidate_type", "")
                side = d.get("side", "")
                prob = d.get("model_probability", "")
                thresh = d.get("threshold", "")
                block = d.get("shadow_would_block", False)
                ts_ = d.get("timestamp", "")[:19] if d.get("timestamp") else ""
                lines.append(f"- {ts_} | {sym} {ct} {side} | prob={prob} thresh={thresh} | would_block={block}")
            lines.append("")
        if cal := summary.get("model_calibration"):
            lines.append("## Model Calibration Review")
            lines.append(f"- Evaluations: {cal.get('total_model_evaluations', 0)} | blocks: {cal.get('total_shadow_blocks', 0)} | allows: {cal.get('total_shadow_allows', 0)}")
            lines.append(f"- Block rate: {cal.get('block_rate', 0)} | mean prob: {cal.get('mean_probability')} | median prob: {cal.get('median_probability')}")
            lines.append(f"- Threshold configured: {cal.get('threshold_configured')}")
            lines.append(f"- Session submitted: {cal.get('session_submitted_count', 0)} | filled: {cal.get('session_filled_count', 0)}")
            lines.append(f"- Outcome linkage: {cal.get('outcome_linkage_note', '')}")
            if buckets := cal.get("probability_buckets"):
                lines.append("- Probability buckets (sample_count | shadow_block | shadow_allow):")
                for b in buckets:
                    lines.append(f"  - {b.get('probability_bucket', '')}: n={b.get('sample_count', 0)} block={b.get('shadow_block_count', 0)} allow={b.get('shadow_allow_count', 0)}")
            if rc := cal.get("runtime_calibration"):
                lines.append("")
                lines.append("## Runtime Probability Distribution")
                dist = rc.get("probability_distribution", {})
                lines.append(f"- Min: {dist.get('min')} | Max: {dist.get('max')} | Mean: {dist.get('mean')} | Median: {dist.get('median')}")
                lines.append(f"- Percentiles: p50={dist.get('p50')} p75={dist.get('p75')} p90={dist.get('p90')} p95={dist.get('p95')} p99={dist.get('p99')}")
                lines.append(f"- Current threshold above observed max: {rc.get('current_threshold_above_observed_max')}")
                if ret := rc.get("retention_thresholds"):
                    lines.append("- Candidate retention thresholds (empirical):")
                    for k, v in ret.items():
                        if v is not None:
                            lines.append(f"  - {k}: {v}")
                if log_buckets := rc.get("probability_buckets_log"):
                    lines.append("- Log-scale probability buckets:")
                    for b in log_buckets:
                        if b.get("count", 0) > 0:
                            lines.append(f"  - {b.get('bucket', '')}: count={b.get('count', 0)}")
            lines.append("")
            lines.append("## Threshold Readiness")
            for row in cal.get("threshold_sweep", []):
                t = row.get("threshold", 0)
                wb = row.get("would_block_count", 0)
                wa = row.get("would_allow_count", 0)
                br = row.get("block_rate", 0)
                lines.append(f"- thresh={t}: would_block={wb} would_allow={wa} block_rate={br}")
            lines.append("")
        if pr := summary.get("promotion_recommendation"):
            lines.append("## Promotion Recommendation")
            lines.append(f"- Current runtime threshold: {pr.get('current_runtime_threshold')}")
            lines.append(f"- Observed max probability: {pr.get('observed_max_probability')}")
            lines.append(f"- Observed p95: {pr.get('observed_p95')} | p99: {pr.get('observed_p99')}")
            lines.append(f"- Current threshold realistic (<= observed max): {pr.get('current_threshold_realistic')}")
            if sug := pr.get("suggested_threshold_shadow"):
                lines.append(f"- Suggested next threshold (shadow, ~75% retain): {sug}")
            if sug := pr.get("suggested_threshold_active_demo"):
                lines.append(f"- Suggested next threshold (active-demo, ~50% retain): {sug}")
            lines.append(f"- **Verdict: {pr.get('verdict', 'remain_shadow')}**")
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
        summary = await self._build_session_summary()
        root = Path(self._parquet_store._root_dir)
        root.mkdir(parents=True, exist_ok=True)
        start = self._session_start_time or utc_now()
        ts = start.strftime("%Y%m%d_%H%M%S")
        report_dir = root / "session_summaries"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"session_{ts}.json"
        md_path = report_dir / f"session_{ts}.md"
        json_ok = False
        md_ok = False
        try:
            json_path.write_text(dumps_json_safe(summary, indent=2), encoding="utf-8")
            json_ok = True
        except OSError as exc:
            self._logger.warning(
                "session_summary_write_failed",
                path=str(json_path),
                file_type="json",
                error=str(exc),
            )
        try:
            md_path.write_text(self._build_markdown_summary(summary), encoding="utf-8")
            md_ok = True
        except OSError as exc:
            self._logger.warning(
                "session_summary_write_failed",
                path=str(md_path),
                file_type="markdown",
                error=str(exc),
            )
        csv_ok = False
        csv_path: Path | None = None
        cal_json_ok = False
        cal_json_path: Path | None = None
        if cal_data := summary.get("model_calibration"):
            _cal_path = report_dir / f"model_calibration_{ts}.json"
            try:
                _cal_path.write_text(dumps_json_safe(cal_data, indent=2), encoding="utf-8")
                cal_json_ok = True
                cal_json_path = _cal_path
            except OSError as exc:
                self._logger.warning(
                    "session_summary_write_failed",
                    path=str(_cal_path),
                    file_type="calibration_json",
                    error=str(exc),
                )
        if decisions := summary.get("model_shadow_decisions", {}).get("decisions"):
            _csv_path = report_dir / f"model_shadow_decisions_{ts}.csv"
            session_id = f"session_{ts}"
            fieldnames = [
                "session_id",
                "timestamp",
                "symbol",
                "candidate_type",
                "side",
                "model_probability",
                "threshold",
                "shadow_would_block",
                "allow",
                "strategy_submitted",
                "blocking_stage",
            ]
            try:
                with _csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    for d in decisions:
                        row = {
                            "session_id": session_id,
                            "timestamp": d.get("timestamp", ""),
                            "symbol": d.get("symbol", ""),
                            "candidate_type": d.get("candidate_type", ""),
                            "side": d.get("side", ""),
                            "model_probability": d.get("model_probability", ""),
                            "threshold": d.get("threshold", ""),
                            "shadow_would_block": d.get("shadow_would_block", False),
                            "allow": d.get("allow") if "allow" in d else "",
                            "strategy_submitted": d.get("strategy_submitted", False),
                            "blocking_stage": d.get("blocking_stage", ""),
                        }
                        writer.writerow(row)
                csv_ok = True
                csv_path = _csv_path
            except OSError as exc:
                self._logger.warning(
                    "session_summary_write_failed",
                    path=str(_csv_path),
                    file_type="csv",
                    error=str(exc),
                )
        if json_ok and md_ok:
            ledger_payload: dict[str, object] = {
                "json_path": str(json_path),
                "md_path": str(md_path),
            }
            if csv_ok and csv_path is not None:
                ledger_payload["csv_path"] = str(csv_path)
            if cal_json_ok and cal_json_path is not None:
                ledger_payload["calibration_json_path"] = str(cal_json_path)
            self._logger.info(
                "session_summary_written",
                path=str(json_path),
                session_ended_cleanly=summary.get("session_ended_cleanly"),
                abort_reasons=summary.get("abort_reasons"),
            )
            await self._ledger.record(
                "session_summary_written",
                ledger_payload,
            )

        soak_result = await self._write_soak_report(summary, report_dir, ts)
        if soak_result is not None:
            soak_json_path, soak_md_path, soak_report = soak_result
            verdict_block = soak_report.get("health_verdict") or {}
            self._logger.info(
                "soak_report_written",
                json_path=str(soak_json_path),
                markdown_path=str(soak_md_path),
                verdict=verdict_block.get("verdict", ""),
                warnings_count=len(verdict_block.get("warnings") or []),
                failures_count=len(verdict_block.get("failures") or []),
            )
            self._logger.info(
                "soak_report_verdict",
                verdict=verdict_block.get("verdict", ""),
                failures=verdict_block.get("failures"),
                warnings=verdict_block.get("warnings"),
            )
            await self._ledger.record(
                "soak_report_written",
                {
                    "json_path": str(soak_json_path),
                    "markdown_path": str(soak_md_path),
                    "verdict": verdict_block.get("verdict", ""),
                    "warnings_count": len(verdict_block.get("warnings") or []),
                    "failures_count": len(verdict_block.get("failures") or []),
                },
            )

    async def _write_soak_report(
        self, summary: dict[str, object], report_dir: Path, ts: str
    ) -> tuple[Path, Path, dict[str, object]] | None:
        """Write soak report JSON and markdown. Returns (json_path, md_path, report) or None."""
        session_id = f"session_{ts}"
        soak_json_path = report_dir / f"soak_report_{session_id}.json"
        soak_md_path = report_dir / f"soak_report_{session_id}.md"
        metrics_snap = self._metrics.snapshot()
        report = build_soak_report(summary, metrics_snap)
        json_ok = False
        md_ok = False
        try:
            soak_json_path.write_text(dumps_json_safe(report, indent=2), encoding="utf-8")
            json_ok = True
        except OSError as exc:
            self._logger.warning(
                "soak_report_write_failed",
                path=str(soak_json_path),
                file_type="json",
                error=str(exc),
            )
        try:
            soak_md_path.write_text(build_soak_markdown(report), encoding="utf-8")
            md_ok = True
        except OSError as exc:
            self._logger.warning(
                "soak_report_write_failed",
                path=str(soak_md_path),
                file_type="markdown",
                error=str(exc),
            )
        if json_ok and md_ok:
            return (soak_json_path, soak_md_path, report)
        return None


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
