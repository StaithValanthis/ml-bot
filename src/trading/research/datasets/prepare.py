"""Convert exported decision/outcome records into model-ready tabular rows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from trading.research.datasets.export import DecisionExportRecord

# Explicit feature boundaries. Provenance: see docstrings in prepare_training_rows.
FEATURE_NAMES = (
    "ts_ordinal",
    "symbol_hash",
    "action_encoded",
    "qty",
    "risk_approved",
    "side_encoded",
    "reference_price",
    "confidence",
)
LABEL_NAME = "filled"

# Optional label names when derivable.
LABEL_PROFITABLE_FILL = "profitable_fill"
LABEL_INTENT_WITHOUT_FILL = "intent_without_fill"

# Sentinel for missing numeric features (used when value typically non-negative).
MISSING_SENTINEL = -1.0


@dataclass(frozen=True, slots=True)
class OptionalLabels:
    """Optional labels when data supports derivation. None when not derivable."""

    profitable_fill: int | None
    intent_without_fill: int | None


@dataclass(frozen=True, slots=True)
class ModelReadyRow:
    """
    Single model-ready row with explicit feature and label boundaries.

    Features include provenance-aware fields; optional_labels when derivable.
    """

    ts_utc: datetime
    symbol: str
    features: dict[str, float]
    label: int
    optional_labels: OptionalLabels | None = None


def _action_to_int(action: str) -> int:
    """Encode action as integer. Honest scaffold: minimal mapping."""
    mapping = {"entry_long": 1, "entry_short": 2, "exit_long": 3, "exit_short": 4}
    return mapping.get(action, 0)


def _side_to_int(side: str | None) -> float:
    """Encode side/direction. Buy=1, Sell=-1, None=0."""
    if side is None:
        return 0.0
    if str(side).lower() in ("buy", "long"):
        return 1.0
    if str(side).lower() in ("sell", "short"):
        return -1.0
    return 0.0


def _symbol_hash(symbol: str) -> float:
    """Hash symbol to numeric. Simple and deterministic."""
    return float(hash(symbol) % 10000)


def _safe_float(s: str | None, sentinel: float = MISSING_SENTINEL) -> float:
    """Parse to float; return sentinel when invalid or missing."""
    if not s or not str(s).strip():
        return sentinel
    try:
        return float(s)
    except (ValueError, TypeError):
        return sentinel


def _derive_profitable_fill(r: DecisionExportRecord) -> int | None:
    """
    Derive profitable_fill when reference_price and fill_price available.

    Long (Buy): profitable if fill_price > reference_price.
    Short (Sell): profitable if fill_price < reference_price.
    Returns None when not derivable.
    """
    ref = _safe_float(r.reference_price)
    fill = _safe_float(r.fill_price)
    if ref <= 0 or fill <= 0 or not r.filled:
        return None
    side = _side_to_int(r.side)
    if side > 0:
        profitable = fill > ref
    elif side < 0:
        profitable = fill < ref
    else:
        return None
    return 1 if profitable else 0


def _derive_intent_without_fill(r: DecisionExportRecord) -> int | None:
    """
    Derive intent_without_fill: 1 when risk_approved and not filled, 0 when filled.

    Always derivable from risk_approved and filled.
    """
    if r.risk_approved and not r.filled:
        return 1
    if r.filled:
        return 0
    return 0


def prepare_training_rows(records: list[DecisionExportRecord]) -> list[ModelReadyRow]:
    """
    Convert decision export records into model-ready rows.

    Features (provenance):
    - ts_ordinal: from record.ts_utc
    - symbol_hash: from record.symbol
    - action_encoded: from record.action
    - qty: from record.qty
    - risk_approved: from record.risk_approved
    - side_encoded: from record.side (Buy/Sell)
    - reference_price: from record.reference_price; MISSING_SENTINEL when absent
    - confidence: from record.confidence; MISSING_SENTINEL when absent

    Labels:
    - filled: primary, always available
    - profitable_fill: optional, when ref+fill price available
    - intent_without_fill: optional, when risk_approved/filled available
    """
    rows: list[ModelReadyRow] = []
    for r in records:
        try:
            qty = float(r.qty) if r.qty else 0.0
        except (ValueError, TypeError):
            qty = 0.0
        ref_price = _safe_float(r.reference_price)
        conf = _safe_float(r.confidence)
        features = {
            "ts_ordinal": r.ts_utc.timestamp(),
            "symbol_hash": _symbol_hash(r.symbol),
            "action_encoded": float(_action_to_int(r.action)),
            "qty": qty,
            "risk_approved": 1.0 if r.risk_approved else 0.0,
            "side_encoded": _side_to_int(r.side),
            "reference_price": ref_price,
            "confidence": conf,
        }
        label = 1 if r.filled else 0
        opt_labels = OptionalLabels(
            profitable_fill=_derive_profitable_fill(r),
            intent_without_fill=_derive_intent_without_fill(r),
        )
        rows.append(
            ModelReadyRow(
                ts_utc=r.ts_utc,
                symbol=r.symbol,
                features=features,
                label=label,
                optional_labels=opt_labels,
            )
        )
    return rows


def write_training_rows_csv(rows: list[ModelReadyRow], path: Path) -> None:
    """Write model-ready rows to CSV for baseline tabular export."""
    import csv

    if not rows:
        path.write_text("ts_utc,symbol," + ",".join(FEATURE_NAMES) + "," + LABEL_NAME + "\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ts_utc", "symbol"] + list(FEATURE_NAMES) + [LABEL_NAME])
        for row in rows:
            feat_vals = [row.features.get(k, 0.0) for k in FEATURE_NAMES]
            writer.writerow([row.ts_utc.isoformat(), row.symbol] + feat_vals + [row.label])


def compute_feature_coverage(rows: list[ModelReadyRow]) -> dict[str, float]:
    """Compute feature coverage (non-missing fraction) for reporting."""
    if not rows:
        return {}
    n = len(rows)
    coverage: dict[str, float] = {}
    for name in FEATURE_NAMES:
        non_missing = sum(1 for r in rows if r.features.get(name, MISSING_SENTINEL) != MISSING_SENTINEL)
        coverage[name] = non_missing / n if n else 0.0
    return coverage


def compute_label_trust(rows: list[ModelReadyRow]) -> dict[str, str]:
    """Compute which labels are trustworthy vs scaffolded for reporting."""
    trust: dict[str, str] = {"filled": "trustworthy"}
    if not rows:
        return trust
    n = len(rows)
    prof_ok = sum(1 for r in rows if r.optional_labels and r.optional_labels.profitable_fill is not None)
    trust[LABEL_PROFITABLE_FILL] = "trustworthy" if prof_ok == n else f"scaffold_{n - prof_ok}_missing"
    intent_ok = sum(1 for r in rows if r.optional_labels and r.optional_labels.intent_without_fill is not None)
    trust[LABEL_INTENT_WITHOUT_FILL] = "trustworthy" if intent_ok == n else f"scaffold_{n - intent_ok}_missing"
    return trust
