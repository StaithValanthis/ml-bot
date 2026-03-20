"""Unit tests for DEMO-only model-assisted trade filter."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trading.models.filter_artifact import load_model_artifact, ModelArtifactLoadResult
from trading.models.filter_predictor import (
    FilterPredictionResult,
    build_runtime_features,
    predict_proba_fill,
    score_for_filter,
)
from trading.runtime.strategy_orders import ModelFilterOutcomes, StrategyOrderOutcomes
from trading.util.types import ModelFilterMode, RuntimeMode


def test_load_model_artifact_no_path() -> None:
    """When path is None, model is not loaded."""
    result = load_model_artifact(None)
    assert isinstance(result, ModelArtifactLoadResult)
    assert result.loaded is False
    assert result.error == "no_path"


def test_load_model_artifact_path_not_found(tmp_path: Path) -> None:
    """When path does not exist, model is not loaded."""
    missing = tmp_path / "nonexistent.pkl"
    result = load_model_artifact(missing)
    assert result.loaded is False
    assert result.error == "path_not_found"


def test_load_model_artifact_success(tmp_path: Path) -> None:
    """When valid joblib artifact exists, model loads."""
    try:
        import joblib
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        pytest.skip("sklearn/joblib not available")

    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit([[0, 0], [1, 1]], [0, 1])
    path = tmp_path / "model.pkl"
    joblib.dump(clf, path)

    result = load_model_artifact(path)
    assert result.loaded is True
    assert result.model is not None
    assert result.error is None


def test_build_runtime_features_missing_required_returns_none() -> None:
    """When reference_price or confidence missing, returns None."""
    out = build_runtime_features(
        symbol="BTCUSDT",
        action="enter_long",
        side="Buy",
        qty=0.001,
        risk_approved=True,
        reference_price=None,
        confidence=None,
        ts_utc=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
    )
    assert out is None

    out2 = build_runtime_features(
        symbol="BTCUSDT",
        action="enter_long",
        side="Buy",
        qty=0.001,
        risk_approved=True,
        reference_price=Decimal("50000"),
        confidence=None,
        ts_utc=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
    )
    assert out2 is None


def test_build_runtime_features_complete() -> None:
    """When all required features present, returns feature dict."""
    out = build_runtime_features(
        symbol="BTCUSDT",
        action="enter_long",
        side="Buy",
        qty=0.001,
        risk_approved=True,
        reference_price=Decimal("50000"),
        confidence=Decimal("0.8"),
        ts_utc=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
    )
    assert out is not None
    assert "ts_ordinal" in out
    assert "reference_price" in out
    assert "confidence" in out
    assert out["risk_approved"] == 1.0


def test_score_for_filter_no_model_allows() -> None:
    """When no model, allow_trade is True."""
    result, allow = score_for_filter(
        None,
        symbol="BTCUSDT",
        action="enter_long",
        side="Buy",
        qty=0.001,
        risk_approved=True,
        reference_price=Decimal("50000"),
        confidence=Decimal("0.8"),
        ts_utc=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
    )
    assert result.available is False
    assert allow is True


def test_score_for_filter_prediction_unavailable_allows() -> None:
    """When required features missing, allow_trade is True (no prediction)."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        pytest.skip("sklearn not available")

    clf = LogisticRegression(max_iter=100, random_state=42)
    X = [[0, 0, 1, 0.001, 1, 1, 50000, 0.8] for _ in range(10)]
    y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    clf.fit(X, y)

    result, allow = score_for_filter(
        clf,
        symbol="BTCUSDT",
        action="enter_long",
        side="Buy",
        qty=0.001,
        risk_approved=True,
        reference_price=None,
        confidence=Decimal("0.8"),
        ts_utc=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
    )
    assert result.available is False
    assert allow is True


def test_score_for_filter_trade_blocked() -> None:
    """When prob_fill < threshold, allow_trade is False."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        pytest.skip("sklearn not available")

    clf = LogisticRegression(max_iter=100, random_state=42)
    X = [[0, 0, 1, 0.001, 1, 1, 50000, 0.1] for _ in range(10)] + [
        [0, 0, 1, 0.001, 1, 1, 50000, 0.9] for _ in range(10)
    ]
    y = [0] * 10 + [1] * 10
    clf.fit(X, y)

    result, allow = score_for_filter(
        clf,
        symbol="BTCUSDT",
        action="enter_long",
        side="Buy",
        qty=0.001,
        risk_approved=True,
        reference_price=Decimal("50000"),
        confidence=Decimal("0.1"),
        ts_utc=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
        threshold=0.5,
    )
    assert result.available is True
    assert result.prob_fill < 0.5
    assert allow is False


def test_score_for_filter_trade_allowed() -> None:
    """When prob_fill >= threshold, allow_trade is True."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        pytest.skip("sklearn not available")

    clf = LogisticRegression(max_iter=100, random_state=42)
    X = [[0, 0, 1, 0.001, 1, 1, 50000, 0.1] for _ in range(10)] + [
        [0, 0, 1, 0.001, 1, 1, 50000, 0.9] for _ in range(10)
    ]
    y = [0] * 10 + [1] * 10
    clf.fit(X, y)

    result, allow = score_for_filter(
        clf,
        symbol="BTCUSDT",
        action="enter_long",
        side="Buy",
        qty=0.001,
        risk_approved=True,
        reference_price=Decimal("50000"),
        confidence=Decimal("0.9"),
        ts_utc=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
        threshold=0.5,
    )
    assert result.available is True
    assert result.prob_fill >= 0.5
    assert allow is True


def test_model_filter_outcomes_defaults() -> None:
    """ModelFilterOutcomes has expected defaults for calibration visibility."""
    mf = ModelFilterOutcomes()
    assert mf.blocked == 0
    assert mf.allowed == 0
    assert mf.prediction_unavailable == 0
    assert mf.shadow_would_have_blocked == 0
    assert mf.threshold == 0.5
    assert mf.mode == "hard_block"
    assert mf.prob_min is None
    assert mf.prob_max is None
    assert mf.prob_latest is None
    assert mf.prob_count == 0
    assert mf.latest_features is None


def test_strategy_order_outcomes_includes_model_filter() -> None:
    """StrategyOrderOutcomes has model_filter field."""
    so = StrategyOrderOutcomes()
    assert hasattr(so, "model_filter")
    assert isinstance(so.model_filter, ModelFilterOutcomes)


def test_demo_only_gating_orchestrator_init_model_filter() -> None:
    """Model filter is disabled when mode is not DEMO."""
    from trading.runtime.orchestrator import RuntimeOrchestrator
    from trading.settings import load_settings

    settings = load_settings()
    runtime = MagicMock()
    runtime.mode = RuntimeMode.LIVE
    runtime.model_filter_enabled = True
    runtime.model_artifact_path = Path("/some/model.pkl")
    runtime.demo_drill = MagicMock()
    settings.runtime = runtime

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
        orch = RuntimeOrchestrator(settings)
        orch._init_model_filter()

    assert orch._model_filter_active is False
    assert orch._model_filter_model is None


def test_score_for_filter_returns_features_used_when_available() -> None:
    """When prediction is available, result includes features_used for operator visibility."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        pytest.skip("sklearn not available")

    clf = LogisticRegression(max_iter=100, random_state=42)
    X = [[0, 0, 1, 0.001, 1, 1, 50000, 0.8] for _ in range(10)]
    y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    clf.fit(X, y)

    result, allow = score_for_filter(
        clf,
        symbol="BTCUSDT",
        action="enter_long",
        side="Buy",
        qty=0.001,
        risk_approved=True,
        reference_price=Decimal("50000"),
        confidence=Decimal("0.8"),
        ts_utc=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
        threshold=0.5,
    )
    assert result.available is True
    assert result.features_used is not None
    assert result.features_used["reference_price"] == 50000.0
    assert result.features_used["confidence"] == 0.8
    assert result.features_used["qty"] == 0.001


@pytest.mark.asyncio
async def test_session_summary_model_filter_reporting() -> None:
    """Session summary and markdown include model filter fields and calibration visibility."""
    import asyncio

    from trading.runtime.orchestrator import RuntimeOrchestrator
    from trading.settings import load_settings

    settings = load_settings()
    mock_rest = MagicMock()
    mock_ws_public = MagicMock()
    mock_ws_public.subscribe = MagicMock()
    mock_ws_public.run_forever = MagicMock(side_effect=lambda: asyncio.sleep(0.01))
    mock_ws_public.close = MagicMock()
    mock_ws_private = MagicMock()
    mock_ws_private.subscribe = MagicMock()
    mock_ws_private.run_forever = MagicMock(return_value=None)
    mock_ws_private.close = MagicMock()

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", return_value=mock_rest),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", return_value=mock_ws_public),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", return_value=mock_ws_private),
    ):
        orch = RuntimeOrchestrator(settings)
        orch._strategy_order_outcomes.model_filter.blocked = 2
        orch._strategy_order_outcomes.model_filter.allowed = 5
        orch._strategy_order_outcomes.model_filter.prediction_unavailable = 1
        orch._strategy_order_outcomes.model_filter.shadow_would_have_blocked = 0
        orch._strategy_order_outcomes.model_filter.threshold = 0.5
        orch._strategy_order_outcomes.model_filter.mode = "hard_block"
        orch._strategy_order_outcomes.model_filter.prob_min = 1.9e-07
        orch._strategy_order_outcomes.model_filter.prob_max = 2.1e-07
        orch._strategy_order_outcomes.model_filter.prob_latest = 2.0e-07
        orch._strategy_order_outcomes.model_filter.prob_count = 3
        orch._strategy_order_outcomes.model_filter.latest_features = {
            "reference_price": 69850.0,
            "confidence": 0.55,
            "qty": 0.001,
            "ts_ordinal": 1710000000.0,
        }

        summary = await orch._build_session_summary()

    mf = summary.get("model_filter")
    assert mf is not None
    assert mf.get("blocked") == 2
    assert mf.get("allowed") == 5
    assert mf.get("prediction_unavailable") == 1
    assert mf.get("mode") == "hard_block"
    assert mf.get("threshold") == 0.5
    assert mf.get("prob_min") == 1.9e-07
    assert mf.get("prob_max") == 2.1e-07
    assert mf.get("prob_latest") == 2.0e-07
    assert mf.get("prob_count") == 3
    assert mf.get("latest_features") == {
        "reference_price": 69850.0,
        "confidence": 0.55,
        "qty": 0.001,
        "ts_ordinal": 1710000000.0,
    }

    markdown = orch._build_markdown_summary(summary)
    assert "## Model Filter (DEMO-only)" in markdown
    assert "Mode: hard_block" in markdown
    assert "Trades allowed by model: 5" in markdown
    assert "Trades blocked by model: 2" in markdown
    assert "Threshold: 0.5" in markdown
    assert "Prob stats:" in markdown
    assert "Latest features:" in markdown


def test_model_filter_mode_enum_values() -> None:
    """ModelFilterMode has expected DEMO-only modes."""
    assert ModelFilterMode.HARD_BLOCK.value == "hard_block"
    assert ModelFilterMode.SHADOW.value == "shadow"
    assert ModelFilterMode.THRESHOLD_OVERRIDE.value == "threshold_override"


def test_orchestrator_model_filter_init_sets_mode_and_threshold(tmp_path: Path) -> None:
    """When model filter loads in DEMO, mode and threshold are set from settings."""
    try:
        import joblib
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        pytest.skip("sklearn/joblib not available")

    from trading.runtime.orchestrator import RuntimeOrchestrator
    from trading.settings import load_settings

    settings = load_settings()
    runtime = MagicMock()
    runtime.mode = RuntimeMode.DEMO
    runtime.model_filter_enabled = True
    runtime.model_filter_mode = ModelFilterMode.SHADOW
    runtime.model_filter_threshold = 0.01
    runtime.model_artifact_path = None
    runtime.demo_drill = MagicMock()
    runtime.demo_candidate_overrides = None
    settings.runtime = runtime

    model_path = tmp_path / "model.pkl"
    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit([[0, 0], [1, 1]], [0, 1])
    joblib.dump(clf, model_path)
    runtime.model_artifact_path = model_path

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
        orch = RuntimeOrchestrator(settings)
        orch._init_model_filter()

    assert orch._model_filter_active is True
    assert orch._model_filter_mode == ModelFilterMode.SHADOW
    assert orch._model_filter_threshold == 0.01
    assert orch._strategy_order_outcomes.model_filter.mode == "shadow"


@pytest.mark.asyncio
async def test_session_summary_includes_shadow_would_have_blocked() -> None:
    """Session summary and markdown include shadow_would_have_blocked when nonzero."""
    import asyncio

    from trading.runtime.orchestrator import RuntimeOrchestrator
    from trading.settings import load_settings

    settings = load_settings()
    mock_rest = MagicMock()
    mock_ws_public = MagicMock()
    mock_ws_public.subscribe = MagicMock()
    mock_ws_public.run_forever = MagicMock(side_effect=lambda: asyncio.sleep(0.01))
    mock_ws_public.close = MagicMock()
    mock_ws_private = MagicMock()
    mock_ws_private.subscribe = MagicMock()
    mock_ws_private.run_forever = MagicMock(return_value=None)
    mock_ws_private.close = MagicMock()

    with (
        patch("trading.runtime.orchestrator.BybitRestClient", return_value=mock_rest),
        patch("trading.runtime.orchestrator.BybitWsPublicClient", return_value=mock_ws_public),
        patch("trading.runtime.orchestrator.BybitWsPrivateClient", return_value=mock_ws_private),
    ):
        orch = RuntimeOrchestrator(settings)
        orch._strategy_order_outcomes.model_filter.mode = "shadow"
        orch._strategy_order_outcomes.model_filter.shadow_would_have_blocked = 7

        summary = await orch._build_session_summary()
        md = orch._build_markdown_summary(summary)

    assert summary.get("model_filter", {}).get("shadow_would_have_blocked") == 7
    assert "Shadow would have blocked: 7" in md
