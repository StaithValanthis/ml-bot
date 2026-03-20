"""Event-driven backtest engine reusing strategy, risk, and execution contracts."""

from __future__ import annotations

import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import AsyncIterator

from trading.backtest.cost_model import CostModel, CostModelConfig
from trading.backtest.funding_model import FundingModel, FundingModelConfig
from trading.execution.execution_engine import ExecutionEngine
from trading.execution.order_intent import OrderIntent
from trading.journal.ledger import LedgerEvent, LedgerSink
from trading.journal.pnl import PnLRecord
from trading.risk.circuit_breaker import CircuitBreaker
from trading.risk.portfolio_state import PortfolioState, PositionRiskView
from trading.risk.risk_engine import PerSymbolLimit, RiskEngine
from trading.risk.sizing import SizingInputs, VolatilityAwareSizer
from trading.strategy.candidates import BreakoutTrendCandidateGenerator
from trading.strategy.regime_filter import RegimeFilter
from trading.strategy.signal_engine import SignalAction, SignalEngine
from trading.util.time import utc_now
from trading.util.types import MarketSymbol, OHLCVBar, OrderSide, PositionSide


@dataclass(frozen=True, slots=True)
class CandleEvent:
    symbol: str
    bar: OHLCVBar
    funding_rate_bps: Decimal | None = None


@dataclass(slots=True)
class BacktestResult:
    start_time: datetime | None
    end_time: datetime | None
    initial_equity_usdt: Decimal
    final_equity_usdt: Decimal
    total_pnl_usdt: Decimal
    total_costs_usdt: Decimal
    total_funding_usdt: Decimal
    decisions: int
    fills: int
    events: list[LedgerEvent]
    pnl_records: list[PnLRecord]


class BacktestLedgerSink(LedgerSink):
    """In-memory sink for backtest events with explicit timestamps."""

    def __init__(self) -> None:
        self._events: list[LedgerEvent] = []

    async def write_event(self, event: LedgerEvent) -> None:
        self._events.append(event)

    def events(self) -> list[LedgerEvent]:
        return list(self._events)


@dataclass(slots=True)
class BacktestConfig:
    initial_equity_usdt: Decimal = Decimal("10000")
    candle_timeframe: str = "5"
    regime_timeframe: str = "60"
    max_total_notional_usdt: Decimal = Decimal("50000")
    max_leverage: Decimal = Decimal("3")
    daily_loss_limit_usdt: Decimal = Decimal("1000")
    liquidation_buffer_bps: int = 200
    symbol_specs: dict[str, MarketSymbol] = field(default_factory=dict)
    per_symbol_limits: dict[str, PerSymbolLimit] = field(default_factory=dict)
    fill_probability: float = 0.85
    fill_seed: int | None = 42


def _default_symbol_specs() -> dict[str, MarketSymbol]:
    return {
        "BTCUSDT": MarketSymbol(
            symbol="BTCUSDT",
            qty_step=Decimal("0.001"),
            min_qty=Decimal("0.001"),
            price_tick=Decimal("0.10"),
            max_leverage=Decimal("100"),
        ),
        "ETHUSDT": MarketSymbol(
            symbol="ETHUSDT",
            qty_step=Decimal("0.01"),
            min_qty=Decimal("0.01"),
            price_tick=Decimal("0.01"),
            max_leverage=Decimal("100"),
        ),
    }


class BacktestEngine:
    """
    Event-driven backtest coordinator.

    Processes candle events, runs strategy -> risk -> execution flow,
    simulates fills with costs, maintains portfolio state.
    """

    def __init__(
        self,
        config: BacktestConfig | None = None,
        cost_config: CostModelConfig | None = None,
        funding_config: FundingModelConfig | None = None,
    ) -> None:
        self._cfg = config or BacktestConfig()
        specs = self._cfg.symbol_specs or _default_symbol_specs()
        self._cost_model = CostModel(cost_config)
        self._funding_model = FundingModel(funding_config)
        self._circuit_breaker = CircuitBreaker()
        self._risk_engine = RiskEngine(
            max_total_notional_usdt=self._cfg.max_total_notional_usdt,
            max_leverage=self._cfg.max_leverage,
            daily_loss_limit_usdt=self._cfg.daily_loss_limit_usdt,
            liquidation_buffer_bps=self._cfg.liquidation_buffer_bps,
            circuit_breaker=self._circuit_breaker,
            per_symbol_limits=self._cfg.per_symbol_limits,
        )
        self._sizer = VolatilityAwareSizer()
        self._candidate_generator = BreakoutTrendCandidateGenerator()
        self._regime_filter = RegimeFilter()
        self._signal_engine = SignalEngine()
        self._execution_engine = ExecutionEngine(strategy_id="backtest-v1")

        self._bar_history: dict[str, dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=800))
        )
        self._portfolio = PortfolioState(
            equity_usdt=self._cfg.initial_equity_usdt,
            available_balance_usdt=self._cfg.initial_equity_usdt,
            pnl_date=date.today(),
        )
        self._ledger_sink = BacktestLedgerSink()
        self._pnl_records: list[PnLRecord] = []
        self._decisions = 0
        self._fills = 0
        self._total_costs_usdt = Decimal("0")
        self._total_funding_usdt = Decimal("0")
        self._symbol_specs = specs

    async def run(
        self,
        event_source: AsyncIterator[CandleEvent],
        *,
        ledger_sinks: list[LedgerSink] | None = None,
    ) -> BacktestResult:
        if self._cfg.fill_seed is not None:
            random.seed(self._cfg.fill_seed)
        sinks = list(ledger_sinks or []) + [self._ledger_sink]
        start_time: datetime | None = None
        end_time: datetime | None = None

        async for event in event_source:
            if start_time is None:
                start_time = event.bar.close_time
            end_time = event.bar.close_time

            self._bar_history[event.symbol][event.bar.timeframe].append(event.bar)
            if event.bar.timeframe != self._cfg.candle_timeframe:
                continue
            if not event.bar.confirmed:
                continue

            bars_5m = list(self._bar_history[event.symbol][self._cfg.candle_timeframe])
            bars_1h = self._aggregate_to_1h(event.symbol)
            if len(bars_1h) < 24:
                continue

            candidates = self._candidate_generator.on_closed_candle(event.symbol, bars_5m)
            for candidate in candidates:
                regime = self._regime_filter.evaluate(
                    candidate=candidate,
                    bars_1h=bars_1h[-24:],
                    funding_rate_bps=event.funding_rate_bps,
                )
                signal = self._signal_engine.evaluate(candidate, regime)
                if signal.side is None or signal.reference_price is None:
                    continue

                self._decisions += 1
                await self._record(
                    "decision",
                    {
                        "symbol": signal.symbol,
                        "action": signal.action.value,
                        "confidence": str(signal.confidence),
                    },
                    event.bar.close_time,
                )

                symbol_spec = self._symbol_specs.get(signal.symbol)
                if symbol_spec is None:
                    continue

                qty = self._sizer.size_qty(
                    SizingInputs(
                        equity_usdt=max(self._portfolio.equity_usdt, Decimal("1")),
                        confidence=signal.confidence,
                        volatility_bps=regime.volatility_bps,
                        reference_price=signal.reference_price,
                        max_leverage=self._cfg.max_leverage,
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
                    continue

                intent = self._execution_engine.build_entry_intent(
                    signal=signal,
                    qty=qty,
                    reference_price=signal.reference_price,
                    now=event.bar.close_time,
                )
                if intent is None:
                    continue

                await self._record(
                    "order_intent",
                    {
                        "symbol": intent.symbol,
                        "side": intent.side.value,
                        "qty": str(intent.qty),
                        "reference_price": str(signal.reference_price),
                    },
                    event.bar.close_time,
                )

                should_fill = random.random() < float(self._cfg.fill_probability)
                if should_fill:
                    fill_pnl, costs = await self._simulate_fill(intent, event.bar.close_time)
                    self._portfolio = self._apply_fill(intent, fill_pnl, costs, event.bar.close_time)
                    self._fills += 1
                    self._total_costs_usdt += costs
                    fill_price = intent.price or signal.reference_price
                    await self._record(
                        "fill",
                        {
                            "symbol": intent.symbol,
                            "qty": str(intent.qty),
                            "exec_price": str(fill_price),
                        },
                        event.bar.close_time,
                    )
                else:
                    pass

                if should_fill and event.funding_rate_bps is not None:
                    funding = self._funding_model.compute(
                        symbol=intent.symbol,
                        side=PositionSide.LONG if intent.side == OrderSide.BUY else PositionSide.SHORT,
                        notional_usdt=qty * signal.reference_price,
                        rate_bps=event.funding_rate_bps,
                        accrued_at=event.bar.close_time,
                    )
                    self._total_funding_usdt += funding.accrual_usdt
                    self._portfolio.update_realized_pnl(funding.accrual_usdt, event.bar.close_time.date())

            self._snapshot_pnl(event.bar.close_time)

        for evt in self._ledger_sink.events():
            for s in sinks:
                if s is not self._ledger_sink:
                    await s.write_event(evt)

        return BacktestResult(
            start_time=start_time,
            end_time=end_time,
            initial_equity_usdt=self._cfg.initial_equity_usdt,
            final_equity_usdt=self._portfolio.equity_usdt,
            total_pnl_usdt=self._portfolio.equity_usdt - self._cfg.initial_equity_usdt,
            total_costs_usdt=self._total_costs_usdt,
            total_funding_usdt=self._total_funding_usdt,
            decisions=self._decisions,
            fills=self._fills,
            events=self._ledger_sink.events(),
            pnl_records=list(self._pnl_records),
        )

    def _aggregate_to_1h(self, symbol: str) -> list[OHLCVBar]:
        bars_5m = list(self._bar_history[symbol][self._cfg.candle_timeframe])
        if len(bars_5m) < 12:
            return []
        result: list[OHLCVBar] = []
        for i in range(0, len(bars_5m) - 11, 12):
            chunk = bars_5m[i : i + 12]
            if len(chunk) < 12:
                break
            first, last = chunk[0], chunk[-1]
            o, h, l, c = first.open, max(b.high for b in chunk), min(b.low for b in chunk), last.close
            vol = sum(b.volume for b in chunk)
            turn = sum(b.turnover for b in chunk)
            result.append(
                OHLCVBar(
                    symbol=symbol,
                    timeframe="60",
                    open_time=first.open_time,
                    close_time=last.close_time,
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    volume=vol,
                    turnover=turn,
                    confirmed=all(b.confirmed for b in chunk),
                )
            )
        return result

    async def _simulate_fill(
        self, intent: OrderIntent, fill_time: datetime
    ) -> tuple[Decimal, Decimal]:
        price = intent.price or Decimal("0")
        if price <= 0:
            price = intent.metadata.get("reference_price", Decimal("0"))
        notional = intent.qty * price
        cost_breakdown = self._cost_model.compute(
            notional_usdt=notional,
            is_maker=True,
            qty=intent.qty,
            price=price,
        )
        return Decimal("0"), cost_breakdown.total_usdt

    def _apply_fill(
        self, intent: OrderIntent, fill_pnl: Decimal, costs: Decimal, at: datetime
    ) -> PortfolioState:
        price = intent.price or Decimal("0")
        pos_side = PositionSide.LONG if intent.side == OrderSide.BUY else PositionSide.SHORT
        existing = self._portfolio.position_for(intent.symbol)

        if existing is None:
            new_pos = PositionRiskView(
                symbol=intent.symbol,
                side=pos_side,
                qty=intent.qty,
                entry_price=price,
                mark_price=price,
                leverage=Decimal("1"),
                liquidation_price=None,
            )
            positions = {**self._portfolio.positions, intent.symbol: new_pos}
        elif existing.side == pos_side:
            total_qty = existing.qty + intent.qty
            avg = (
                (existing.entry_price * existing.qty + price * intent.qty) / total_qty
                if total_qty
                else price
            )
            new_pos = PositionRiskView(
                symbol=intent.symbol,
                side=pos_side,
                qty=total_qty,
                entry_price=avg,
                mark_price=price,
                leverage=existing.leverage,
                liquidation_price=existing.liquidation_price,
            )
            positions = {**self._portfolio.positions, intent.symbol: new_pos}
        else:
            close_qty = min(abs(existing.qty), abs(intent.qty))
            if pos_side == PositionSide.SHORT:
                pnl = close_qty * (price - existing.entry_price)
            else:
                pnl = close_qty * (existing.entry_price - price)
            self._portfolio.update_realized_pnl(pnl - costs, at.date())
            self._circuit_breaker.record_fill_pnl(pnl - costs)
            remaining = existing.qty + intent.qty
            if remaining == 0:
                positions = {k: v for k, v in self._portfolio.positions.items() if k != intent.symbol}
            else:
                new_pos = PositionRiskView(
                    symbol=intent.symbol,
                    side=PositionSide.LONG if remaining > 0 else PositionSide.SHORT,
                    qty=abs(remaining),
                    entry_price=price,
                    mark_price=price,
                    leverage=Decimal("1"),
                    liquidation_price=None,
                )
                positions = {**self._portfolio.positions, intent.symbol: new_pos}
            fill_pnl = pnl

        equity = self._portfolio.equity_usdt - costs + fill_pnl
        available = self._portfolio.available_balance_usdt - costs
        return PortfolioState(
            equity_usdt=equity,
            available_balance_usdt=available,
            positions=positions,
            realized_pnl_today_usdt=self._portfolio.realized_pnl_today_usdt,
            pnl_date=self._portfolio.pnl_date,
        )

    async def _record(self, event_type: str, payload: dict, timestamp: datetime | None = None) -> None:
        ts = timestamp if timestamp is not None else utc_now()
        evt = LedgerEvent(event_type=event_type, timestamp=ts, payload=payload)
        await self._ledger_sink.write_event(evt)

    def _snapshot_pnl(self, at: datetime) -> None:
        unrealized = sum(
            (p.mark_price - p.entry_price) * p.qty
            if p.side == PositionSide.LONG
            else (p.entry_price - p.mark_price) * p.qty
            for p in self._portfolio.positions.values()
        )
        rec = PnLRecord(
            timestamp=at,
            equity_usdt=self._portfolio.equity_usdt,
            available_usdt=self._portfolio.available_balance_usdt,
            realized_pnl_usdt=self._portfolio.realized_pnl_today_usdt,
            unrealized_pnl_usdt=unrealized,
        )
        self._pnl_records.append(rec)
