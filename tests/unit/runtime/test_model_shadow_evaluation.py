"""Unit tests for ML shadow evaluation and reporting."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trading.runtime.orchestrator import RuntimeOrchestrator
from trading.settings import load_settings
from trading.strategy.signal_engine import SignalAction, SignalDecision
from trading.util.types import ModelFilterMode, OrderSide


def _make_orchestrator() -> RuntimeOrchestrator:
    mock_rest = MagicMock()
    mock_ws_public = MagicMock()
    mock_ws_public.subscribe = MagicMock()
    mock_ws_public.run_forever = MagicMock()
    mock_ws_public.close = MagicMock()
    mock_ws_private = MagicMock()
    mock_ws_private.subscribe = MagicMock()
    mock_ws_private.run_forever = MagicMock()
    mock_ws_private.close = MagicMock()
    with (
        patch("trading.runtime.orchestrator.BybitRestClient", return_value=mock_rest),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", return_value=mock_ws_public),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", return_value=mock_ws_private),
    ):
        return RuntimeOrchestrator(load_settings())


def test_run_model_filter_stage_evaluates_probe_min_qty() -> None:
    """Paper sizing-rejection probe uses exchange min_qty; predictor must see that quantity."""
    orch = _make_orchestrator()
    orch._model_filter_active = True
    orch._model_filter_mode = ModelFilterMode.SHADOW
    orch._model_filter_threshold = 0.5
    orch._model_filter_model = MagicMock()

    pred = MagicMock()
    pred.available = True
    pred.prob_fill = 0.61
    pred.features_used = None

    signal = SignalDecision(
        symbol="BTCUSDT",
        action=SignalAction.ENTER_LONG,
        side=OrderSide.BUY,
        confidence=Decimal("0.7"),
        reference_price=Decimal("60000"),
        stop_price=None,
        reason="t",
        generated_at=datetime(2025, 1, 1, tzinfo=UTC),
        metadata={},
    )
    cand = MagicMock()
    cand.candidate_type = MagicMock(value="breakout_long")

    with patch("trading.models.filter_predictor.score_for_filter", return_value=(pred, True)) as mock_sf:
        ok = orch._run_model_filter_stage(
            signal=signal,
            qty=Decimal("0.001"),
            candidate=cand,
            bars_5m=[],
        )
    assert ok is True
    assert orch._strategy_order_outcomes.model_filter.prob_count == 1
    mock_sf.assert_called_once()
    assert mock_sf.call_args.kwargs["qty"] == pytest.approx(0.001)


def test_record_model_decision_payload_shape() -> None:
    """Per-candidate logging produces expected dict shape."""
    orch = _make_orchestrator()
    orch._model_filter_mode = ModelFilterMode.SHADOW
    orch._record_model_decision(
        symbol="BTCUSDT",
        candidate_type="breakout_long",
        side="Buy",
        bar_close_time=datetime(2025, 3, 19, 10, 30, 0, tzinfo=UTC),
        model_probability=0.42,
        threshold=0.5,
        shadow_would_block=True,
    )
    assert len(orch._model_shadow_decisions) == 1
    d = orch._model_shadow_decisions[0]
    assert d["symbol"] == "BTCUSDT"
    assert d["candidate_type"] == "breakout_long"
    assert d["side"] == "Buy"
    assert "timestamp" in d
    assert d["bar_close_time"] == "2025-03-19T10:30:00+00:00"
    assert d["model_probability"] == 0.42
    assert d["threshold"] == 0.5
    assert d["shadow_would_block"] is True
    assert d["strategy_submitted"] is False
    assert d["blocking_stage"] == "model_evaluated"


def test_record_model_decision_bounded_to_max() -> None:
    """Decisions list is bounded to _model_shadow_decisions_max."""
    orch = _make_orchestrator()
    orch._model_shadow_decisions_max = 3
    orch._model_filter_mode = ModelFilterMode.SHADOW
    for i in range(5):
        orch._record_model_decision(
            symbol="BTCUSDT",
            candidate_type="breakout_long",
            side="Buy",
            bar_close_time=None,
            model_probability=0.5 + i * 0.01,
            threshold=0.5,
            shadow_would_block=False,
        )
    assert len(orch._model_shadow_decisions) == 3
    probs = [d["model_probability"] for d in orch._model_shadow_decisions]
    assert probs == [0.52, 0.53, 0.54]


@pytest.mark.asyncio
async def test_session_summary_model_shadow_decisions_section() -> None:
    """Session summary includes model_shadow_decisions with aggregates."""
    orch = _make_orchestrator()
    orch._model_shadow_decisions = [
        {"symbol": "BTCUSDT", "model_probability": 0.3, "shadow_would_block": True},
        {"symbol": "ETHUSDT", "model_probability": 0.8, "shadow_would_block": False},
        {"symbol": "BTCUSDT", "model_probability": 0.55, "shadow_would_block": False},
    ]
    summary = await orch._build_session_summary()
    msd = summary.get("model_shadow_decisions")
    assert msd is not None
    assert msd["total_model_evaluations"] == 3
    assert msd["shadow_would_block_count"] == 1
    assert msd["shadow_would_allow_count"] == 2
    assert msd["avg_probability"] == pytest.approx(0.55)
    assert msd["min_probability"] == 0.3
    assert msd["max_probability"] == 0.8
    assert len(msd["decisions"]) == 3


@pytest.mark.asyncio
async def test_session_summary_no_model_shadow_decisions_when_empty() -> None:
    """model_shadow_decisions omitted when no decisions."""
    orch = _make_orchestrator()
    orch._model_shadow_decisions = []
    summary = await orch._build_session_summary()
    assert "model_shadow_decisions" not in summary


def test_markdown_summary_model_shadow_evaluation_sections() -> None:
    """Markdown includes Model Shadow Evaluation and Recent Model Shadow Decisions."""
    orch = _make_orchestrator()
    summary = {
        "model_shadow_decisions": {
            "total_model_evaluations": 5,
            "shadow_would_block_count": 2,
            "shadow_would_allow_count": 3,
            "active_blocked_count": 2,
            "active_allowed_count": 3,
            "latest_active_decision": {
                "symbol": "BTCUSDT",
                "candidate_type": "breakout_long",
                "side": "Buy",
                "model_probability": 0.52,
                "threshold": 0.45,
                "allow": True,
            },
            "avg_probability": 0.48,
            "min_probability": 0.2,
            "max_probability": 0.75,
            "decisions": [
                {
                    "symbol": "BTCUSDT",
                    "candidate_type": "breakout_long",
                    "side": "Buy",
                    "timestamp": "2025-03-19T10:30:00.123456+00:00",
                    "model_probability": 0.42,
                    "threshold": 0.5,
                    "shadow_would_block": True,
                },
            ],
        },
    }
    md = orch._build_markdown_summary(summary)
    assert "## Model Shadow Evaluation" in md
    assert "## Recent Model Shadow Decisions" in md
    assert "Total model evaluations: 5" in md
    assert "Shadow would block: 2" in md
    assert "Shadow would allow: 3" in md
    assert "Active blocked: 2" in md
    assert "Active allowed: 3" in md
    assert "Latest active decision:" in md
    assert "prob=0.52" in md
    assert "threshold=0.45" in md
    assert "allow=True" in md
    assert "Prob: avg=0.48 min=0.2 max=0.75" in md
    assert "BTCUSDT" in md
    assert "breakout_long" in md
    assert "would_block=True" in md


@pytest.mark.asyncio
async def test_csv_artifact_written_with_correct_columns(tmp_path: Path) -> None:
    """CSV artifact has expected columns and row structure."""
    from trading.util.types import RuntimeMode

    orch = _make_orchestrator()
    orch._parquet_store._root_dir = tmp_path
    orch._session_start_time = datetime(2025, 3, 19, 9, 0, 0, tzinfo=UTC)
    orch._settings.runtime.mode = RuntimeMode.DEMO
    orch._model_shadow_decisions = [
        {
            "timestamp": "2025-03-19T10:30:00+00:00",
            "symbol": "BTCUSDT",
            "candidate_type": "breakout_long",
            "side": "Buy",
            "model_probability": 0.42,
            "threshold": 0.5,
            "shadow_would_block": True,
            "strategy_submitted": False,
            "blocking_stage": "model_evaluated",
        },
    ]
    await orch._write_session_summary()
    report_dir = tmp_path / "session_summaries"
    csv_files = list(report_dir.glob("model_shadow_decisions_*.csv"))
    assert len(csv_files) == 1
    content = csv_files[0].read_text(encoding="utf-8")
    lines = content.strip().split("\n")
    assert "session_id" in lines[0]
    assert "allow" in lines[0]
    assert "session_20250319" in lines[1]
    assert "BTCUSDT" in lines[1]
    assert "breakout_long" in lines[1]
    assert "Buy" in lines[1]
    assert "0.42" in lines[1]
    assert "0.5" in lines[1]
    assert "True" in lines[1]
    assert "False" in lines[1]
    assert "model_evaluated" in lines[1]


def test_shadow_mode_does_not_gate_behavior() -> None:
    """In SHADOW mode, _record_model_decision logs but does not change flow (reporting only)."""
    orch = _make_orchestrator()
    orch._model_filter_mode = ModelFilterMode.SHADOW
    orch._record_model_decision(
        symbol="BTCUSDT",
        candidate_type="breakout_long",
        side="Buy",
        bar_close_time=None,
        model_probability=0.1,
        threshold=0.5,
        shadow_would_block=True,
    )
    assert len(orch._model_shadow_decisions) == 1
    assert orch._model_shadow_decisions[0]["shadow_would_block"] is True
    assert orch._model_shadow_decisions[0]["strategy_submitted"] is False
    assert "allow" not in orch._model_shadow_decisions[0]


def test_record_model_decision_active_mode_includes_allow() -> None:
    """In active (HARD_BLOCK) mode, _record_model_decision stores allow for CSV export."""
    orch = _make_orchestrator()
    orch._model_filter_mode = ModelFilterMode.HARD_BLOCK
    orch._record_model_decision(
        symbol="ETHUSDT",
        candidate_type="breakout_short",
        side="Sell",
        bar_close_time=None,
        model_probability=0.52,
        threshold=0.45,
        shadow_would_block=False,
        allow=True,
    )
    assert len(orch._model_shadow_decisions) == 1
    assert orch._model_shadow_decisions[0]["allow"] is True
    orch._record_model_decision(
        symbol="BTCUSDT",
        candidate_type="breakout_long",
        side="Buy",
        bar_close_time=None,
        model_probability=0.3,
        threshold=0.45,
        shadow_would_block=True,
        allow=False,
    )
    assert orch._model_shadow_decisions[1]["allow"] is False


@pytest.mark.asyncio
async def test_session_summary_includes_active_blocked_allowed() -> None:
    """Session summary model_shadow_decisions includes active_blocked_count and active_allowed_count."""
    orch = _make_orchestrator()
    orch._model_shadow_decisions = [
        {"symbol": "A", "model_probability": 0.3, "shadow_would_block": True, "allow": False},
        {"symbol": "B", "model_probability": 0.8, "shadow_would_block": False, "allow": True},
        {"symbol": "C", "model_probability": 0.5, "shadow_would_block": False, "allow": True},
    ]
    summary = await orch._build_session_summary()
    msd = summary.get("model_shadow_decisions")
    assert msd is not None
    assert msd["active_blocked_count"] == 1
    assert msd["active_allowed_count"] == 2
    assert msd["latest_active_decision"]["symbol"] == "C"
    assert msd["latest_active_decision"]["allow"] is True


@pytest.mark.asyncio
async def test_runtime_summary_includes_active_blocked_allowed() -> None:
    """Runtime summary model_filter_calibration includes blocked/allowed when model filter active."""
    captured: list[dict] = []

    def capture_log(event: str, **kwargs: object) -> None:
        if event == "runtime_summary":
            captured.append(dict(kwargs))

    orch = _make_orchestrator()
    orch._model_filter_active = True
    orch._strategy_order_outcomes.model_filter.blocked = 3
    orch._strategy_order_outcomes.model_filter.allowed = 7
    orch._strategy_order_outcomes.model_filter.threshold = 0.45
    orch._strategy_order_outcomes.model_filter.mode = "hard_block"
    orch._logger.info = capture_log

    await orch._runtime_summary_cycle()

    assert len(captured) == 1
    mfc = captured[0].get("model_filter_calibration")
    assert mfc is not None
    assert mfc.get("blocked") == 3
    assert mfc.get("allowed") == 7
    assert mfc.get("threshold") == 0.45


@pytest.mark.asyncio
async def test_csv_export_includes_allow_column_for_active_decisions(tmp_path: Path) -> None:
    """CSV artifact includes allow column when decisions have active (allow) field."""
    from trading.util.types import RuntimeMode

    orch = _make_orchestrator()
    orch._parquet_store._root_dir = tmp_path
    orch._session_start_time = datetime(2025, 3, 19, 9, 0, 0, tzinfo=UTC)
    orch._settings.runtime.mode = RuntimeMode.DEMO
    orch._model_shadow_decisions = [
        {
            "timestamp": "2025-03-19T10:30:00+00:00",
            "symbol": "BTCUSDT",
            "candidate_type": "breakout_long",
            "side": "Buy",
            "model_probability": 0.4,
            "threshold": 0.45,
            "shadow_would_block": True,
            "allow": False,
            "strategy_submitted": False,
            "blocking_stage": "model_evaluated",
        },
    ]
    await orch._write_session_summary()
    report_dir = tmp_path / "session_summaries"
    csv_files = list(report_dir.glob("model_shadow_decisions_*.csv"))
    assert len(csv_files) == 1
    content = csv_files[0].read_text(encoding="utf-8")
    assert "allow" in content.split("\n")[0]
    assert "False" in content
