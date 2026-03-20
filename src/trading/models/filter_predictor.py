"""
Prediction adapter for runtime model-assisted filter.

Uses same feature preparation assumptions as offline training.
Returns no prediction when required features are missing.

Offline-to-runtime consistency:
- RUNTIME_FEATURE_NAMES must match trading.research.datasets.prepare.FEATURE_NAMES.
- Required at runtime: reference_price, confidence (no MISSING_SENTINEL).
- If runtime features differ from offline expectations, predictions may be unreliable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

# Feature provenance: must match offline prepare.FEATURE_NAMES order.
# Offline-to-runtime consistency: same names and order as trading.research.datasets.prepare.
RUNTIME_FEATURE_NAMES = (
    "ts_ordinal",
    "symbol_hash",
    "action_encoded",
    "qty",
    "risk_approved",
    "side_encoded",
    "reference_price",
    "confidence",
)
MISSING_SENTINEL = -1.0


@dataclass(frozen=True, slots=True)
class FilterPredictionResult:
    """Result of model filter prediction."""

    prob_fill: float
    available: bool
    feature_missing_note: str | None = None


def _action_to_int(action: str) -> int:
    mapping = {
        "entry_long": 1,
        "entry_short": 2,
        "exit_long": 3,
        "exit_short": 4,
        "enter_long": 1,
        "enter_short": 2,
    }
    return mapping.get(action, 0)


def _side_to_int(side: str | None) -> float:
    if side is None:
        return 0.0
    s = str(side).lower()
    if s in ("buy", "long"):
        return 1.0
    if s in ("sell", "short"):
        return -1.0
    return 0.0


def _symbol_hash(symbol: str) -> float:
    return float(hash(symbol) % 10000)


def _safe_float(val: Decimal | float | str | None) -> float:
    if val is None:
        return MISSING_SENTINEL
    try:
        return float(val)
    except (ValueError, TypeError):
        return MISSING_SENTINEL


def build_runtime_features(
    *,
    symbol: str,
    action: str,
    side: str | None,
    qty: float,
    risk_approved: bool,
    reference_price: Decimal | float | None,
    confidence: Decimal | float | None,
    ts_utc: datetime,
) -> dict[str, float] | None:
    """
    Build features for runtime prediction. Same provenance as offline prepare.

    Returns None if required features are missing (reference_price, confidence).
    Offline-to-runtime consistency: use MISSING_SENTINEL only when data absent.
    """
    ref = _safe_float(reference_price)
    conf = _safe_float(confidence)
    features = {
        "ts_ordinal": ts_utc.timestamp(),
        "symbol_hash": _symbol_hash(symbol),
        "action_encoded": float(_action_to_int(action)),
        "qty": qty,
        "risk_approved": 1.0 if risk_approved else 0.0,
        "side_encoded": _side_to_int(side),
        "reference_price": ref if ref != MISSING_SENTINEL else MISSING_SENTINEL,
        "confidence": conf if conf != MISSING_SENTINEL else MISSING_SENTINEL,
    }
    missing: list[str] = []
    if features["reference_price"] == MISSING_SENTINEL:
        missing.append("reference_price")
    if features["confidence"] == MISSING_SENTINEL:
        missing.append("confidence")
    if missing:
        return None
    return features


def predict_proba_fill(
    model: object,
    features: dict[str, float],
) -> FilterPredictionResult:
    """
    Score using model. Returns prob of fill (class 1).

    If model has predict_proba, use it. Else if predict, use 1.0/0.0.
    Returns available=False when model lacks required interface.
    """
    try:
        if hasattr(model, "predict_proba"):
            X = [[features.get(k, MISSING_SENTINEL) for k in RUNTIME_FEATURE_NAMES]]
            proba = model.predict_proba(X)[0]
            if len(proba) >= 2:
                return FilterPredictionResult(prob_fill=float(proba[1]), available=True)
            return FilterPredictionResult(prob_fill=float(proba[0]), available=True)
        if hasattr(model, "predict"):
            X = [[features.get(k, MISSING_SENTINEL) for k in RUNTIME_FEATURE_NAMES]]
            pred = model.predict(X)[0]
            return FilterPredictionResult(prob_fill=1.0 if int(pred) == 1 else 0.0, available=True)
    except Exception:
        pass
    return FilterPredictionResult(prob_fill=0.0, available=False, feature_missing_note="model_predict_failed")


def score_for_filter(
    model: object | None,
    *,
    symbol: str,
    action: str,
    side: str | None,
    qty: float,
    risk_approved: bool,
    reference_price: Decimal | float | None,
    confidence: Decimal | float | None,
    ts_utc: datetime,
    threshold: float = 0.5,
) -> tuple[FilterPredictionResult, bool]:
    """
    Score a candidate for model filter. Returns (result, allow_trade).

    allow_trade=True when: no model, no prediction, or prob_fill >= threshold.
    allow_trade=False when: prediction available and prob_fill < threshold.
    """
    if model is None:
        return (FilterPredictionResult(prob_fill=0.0, available=False, feature_missing_note="no_model"), True)
    features = build_runtime_features(
        symbol=symbol,
        action=action,
        side=side,
        qty=qty,
        risk_approved=risk_approved,
        reference_price=reference_price,
        confidence=confidence,
        ts_utc=ts_utc,
    )
    if features is None:
        return (
            FilterPredictionResult(prob_fill=0.0, available=False, feature_missing_note="required_features_missing"),
            True,
        )
    result = predict_proba_fill(model, features)
    if not result.available:
        return (result, True)
    allow = result.prob_fill >= threshold
    return (result, allow)
