"""Unit tests for risk engine."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from trading.risk.circuit_breaker import CircuitBreaker
from trading.risk.portfolio_state import PortfolioState, PositionRiskView
from trading.risk.risk_engine import RiskDecisionReport, RiskEngine, RiskEvaluationContext
from trading.strategy.signal_engine import SignalAction, SignalDecision
from trading.util.types import OrderSide, PositionSide


def _signal(symbol: str = "BTCUSDT", side: OrderSide = OrderSide.BUY) -> SignalDecision:
    return SignalDecision(
        symbol=symbol,
        action=SignalAction.ENTER_LONG if side == OrderSide.BUY else SignalAction.ENTER_SHORT,
        side=side,
        confidence=Decimal("0.8"),
        reference_price=Decimal("60000"),
        stop_price=None,
        reason="test",
        generated_at=datetime.now(UTC),
        metadata={},
    )


def _portfolio(
    equity: Decimal = Decimal("10000"),
    safe_mode: bool = False,
    realized_pnl: Decimal = Decimal("0"),
    positions: dict[str, PositionRiskView] | None = None,
) -> PortfolioState:
    return PortfolioState(
        equity_usdt=equity,
        available_balance_usdt=equity,
        positions=positions or {},
        realized_pnl_today_usdt=realized_pnl,
        safe_mode=safe_mode,
    )


def test_risk_engine_blocks_when_circuit_breaker_tripped() -> None:
    cb = CircuitBreaker()
    cb.trip(reason="test")
    engine = RiskEngine(
        max_total_notional_usdt=Decimal("50000"),
        max_leverage=Decimal("10"),
        daily_loss_limit_usdt=Decimal("500"),
        liquidation_buffer_bps=500,
        circuit_breaker=cb,
    )
    portfolio = _portfolio()
    signal = _signal()
    decision = engine.evaluate(
        signal=signal,
        portfolio=portfolio,
        expected_order_notional=Decimal("6000"),
    )
    assert decision.approved is False
    assert "circuit_breaker" in decision.reason.lower()


def test_risk_engine_blocks_when_safe_mode() -> None:
    engine = RiskEngine(
        max_total_notional_usdt=Decimal("50000"),
        max_leverage=Decimal("10"),
        daily_loss_limit_usdt=Decimal("500"),
        liquidation_buffer_bps=500,
        circuit_breaker=CircuitBreaker(),
    )
    portfolio = _portfolio(safe_mode=True)
    signal = _signal()
    decision = engine.evaluate(
        signal=signal,
        portfolio=portfolio,
        expected_order_notional=Decimal("6000"),
    )
    assert decision.approved is False
    assert "safe_mode" in decision.reason.lower()


def test_risk_engine_blocks_when_daily_loss_limit_reached() -> None:
    engine = RiskEngine(
        max_total_notional_usdt=Decimal("50000"),
        max_leverage=Decimal("10"),
        daily_loss_limit_usdt=Decimal("500"),
        liquidation_buffer_bps=500,
        circuit_breaker=CircuitBreaker(),
    )
    portfolio = _portfolio(realized_pnl=Decimal("-600"))
    signal = _signal()
    decision = engine.evaluate(
        signal=signal,
        portfolio=portfolio,
        expected_order_notional=Decimal("6000"),
    )
    assert decision.approved is False
    assert "daily_loss" in decision.reason.lower()


def test_risk_engine_blocks_when_max_notional_exceeded() -> None:
    engine = RiskEngine(
        max_total_notional_usdt=Decimal("10000"),
        max_leverage=Decimal("10"),
        daily_loss_limit_usdt=Decimal("500"),
        liquidation_buffer_bps=500,
        circuit_breaker=CircuitBreaker(),
    )
    portfolio = _portfolio(equity=Decimal("50000"))
    signal = _signal()
    decision = engine.evaluate(
        signal=signal,
        portfolio=portfolio,
        expected_order_notional=Decimal("15000"),
    )
    assert decision.approved is False
    assert "notional" in decision.reason.lower()


def test_risk_engine_blocks_when_leverage_exceeded() -> None:
    engine = RiskEngine(
        max_total_notional_usdt=Decimal("100000"),
        max_leverage=Decimal("5"),
        daily_loss_limit_usdt=Decimal("500"),
        liquidation_buffer_bps=500,
        circuit_breaker=CircuitBreaker(),
    )
    pos = PositionRiskView(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        qty=Decimal("1"),
        entry_price=Decimal("60000"),
        mark_price=Decimal("60000"),
        leverage=Decimal("10"),
        liquidation_price=Decimal("54000"),
    )
    portfolio = _portfolio(equity=Decimal("10000"), positions={"BTCUSDT": pos})
    signal = _signal()
    decision = engine.evaluate(
        signal=signal,
        portfolio=portfolio,
        expected_order_notional=Decimal("6000"),
    )
    assert decision.approved is False
    assert "leverage" in decision.reason.lower()


def test_risk_engine_approves_valid_signal() -> None:
    engine = RiskEngine(
        max_total_notional_usdt=Decimal("50000"),
        max_leverage=Decimal("10"),
        daily_loss_limit_usdt=Decimal("500"),
        liquidation_buffer_bps=500,
        circuit_breaker=CircuitBreaker(),
    )
    portfolio = _portfolio(safe_mode=False)
    signal = _signal()
    decision = engine.evaluate(
        signal=signal,
        portfolio=portfolio,
        expected_order_notional=Decimal("6000"),
    )
    assert decision.approved is True
    assert decision.reason == "approved"


def test_risk_engine_blocks_very_low_confidence() -> None:
    engine = RiskEngine(
        max_total_notional_usdt=Decimal("50000"),
        max_leverage=Decimal("10"),
        daily_loss_limit_usdt=Decimal("500"),
        liquidation_buffer_bps=500,
        circuit_breaker=CircuitBreaker(),
    )
    portfolio = _portfolio()
    signal = _signal()
    signal = SignalDecision(
        symbol=signal.symbol,
        action=signal.action,
        side=signal.side,
        confidence=Decimal("0.05"),
        reference_price=signal.reference_price,
        stop_price=signal.stop_price,
        reason=signal.reason,
        generated_at=signal.generated_at,
        metadata=signal.metadata,
    )
    decision = engine.evaluate(
        signal=signal,
        portfolio=portfolio,
        expected_order_notional=Decimal("6000"),
    )
    assert decision.approved is False
    assert "confidence" in decision.reason.lower()


def test_evaluate_with_report_returns_decision_and_report() -> None:
    """evaluate_with_report returns (RiskDecision, RiskDecisionReport)."""
    engine = RiskEngine(
        max_total_notional_usdt=Decimal("50000"),
        max_leverage=Decimal("10"),
        daily_loss_limit_usdt=Decimal("500"),
        liquidation_buffer_bps=500,
        circuit_breaker=CircuitBreaker(),
    )
    portfolio = _portfolio()
    signal = _signal()
    decision, report = engine.evaluate_with_report(
        signal=signal,
        portfolio=portfolio,
        expected_order_notional=Decimal("6000"),
    )
    assert decision.approved is True
    assert isinstance(report, RiskDecisionReport)
    assert report.symbol == "BTCUSDT"
    assert report.allow is True
    assert report.reason == "approved"
    assert report.failed_conditions == ()
    d = report.to_log_dict()
    assert "symbol" in d
    assert "allow" in d
    assert "reason" in d
    assert "notional" in d
    assert "projected_notional" in d


def test_evaluate_with_report_rejection_includes_failed_conditions() -> None:
    """Rejection report includes failed_conditions and reason."""
    engine = RiskEngine(
        max_total_notional_usdt=Decimal("10000"),
        max_leverage=Decimal("10"),
        daily_loss_limit_usdt=Decimal("500"),
        liquidation_buffer_bps=500,
        circuit_breaker=CircuitBreaker(),
    )
    portfolio = _portfolio(equity=Decimal("50000"))
    signal = _signal()
    decision, report = engine.evaluate_with_report(
        signal=signal,
        portfolio=portfolio,
        expected_order_notional=Decimal("15000"),
    )
    assert decision.approved is False
    assert report.allow is False
    assert "max_total_notional_exceeded" in report.reason
    assert "max_total_notional_exceeded" in report.failed_conditions
    d = report.to_log_dict()
    assert d["reason"] == "max_total_notional_exceeded"
    assert "max_total_notional_exceeded" in d["failed_conditions"]


def test_evaluate_with_report_context_enriches_report() -> None:
    """RiskEvaluationContext enriches report with candidate_type, orphan_position_blocked."""
    engine = RiskEngine(
        max_total_notional_usdt=Decimal("50000"),
        max_leverage=Decimal("10"),
        daily_loss_limit_usdt=Decimal("500"),
        liquidation_buffer_bps=500,
        circuit_breaker=CircuitBreaker(),
    )
    portfolio = _portfolio()
    signal = _signal()
    ctx = RiskEvaluationContext(
        candidate_type="breakout_long",
        orphan_position_blocked=True,
        current_open_orders_count=2,
    )
    decision, report = engine.evaluate_with_report(
        signal=signal,
        portfolio=portfolio,
        expected_order_notional=Decimal("6000"),
        context=ctx,
    )
    assert report.candidate_type == "breakout_long"
    assert report.orphan_position_blocked is True
    assert report.current_open_orders_count == 2
    d = report.to_log_dict()
    assert d.get("candidate_type") == "breakout_long"
    assert d.get("orphan_position_blocked") is True
    assert d.get("current_open_orders_count") == 2


def test_evaluate_delegates_to_evaluate_with_report() -> None:
    """evaluate() returns same decision as evaluate_with_report()[0]."""
    engine = RiskEngine(
        max_total_notional_usdt=Decimal("50000"),
        max_leverage=Decimal("10"),
        daily_loss_limit_usdt=Decimal("500"),
        liquidation_buffer_bps=500,
        circuit_breaker=CircuitBreaker(),
    )
    portfolio = _portfolio()
    signal = _signal()
    decision_direct = engine.evaluate(
        signal=signal,
        portfolio=portfolio,
        expected_order_notional=Decimal("6000"),
    )
    decision_from_report, _ = engine.evaluate_with_report(
        signal=signal,
        portfolio=portfolio,
        expected_order_notional=Decimal("6000"),
    )
    assert decision_direct.approved == decision_from_report.approved
    assert decision_direct.reason == decision_from_report.reason
