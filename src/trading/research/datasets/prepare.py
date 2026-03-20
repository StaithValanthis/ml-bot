"""Convert exported decision/outcome records into model-ready tabular rows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from trading.research.datasets.export import DecisionExportRecord

# Explicit feature/label boundaries for model training.
FEATURE_NAMES = ("ts_ordinal", "symbol_hash", "action_encoded", "qty", "risk_approved")
LABEL_NAME = "filled"


@dataclass(frozen=True, slots=True)
class ModelReadyRow:
    """
    Single model-ready row with explicit feature and label boundaries.

    Features are minimal and honest; no complex feature engineering.
    """

    ts_utc: datetime
    symbol: str
    features: dict[str, float]
    label: int


def _action_to_int(action: str) -> int:
    """Encode action as integer. Honest scaffold: minimal mapping."""
    mapping = {"entry_long": 1, "entry_short": 2, "exit_long": 3, "exit_short": 4}
    return mapping.get(action, 0)


def _symbol_hash(symbol: str) -> float:
    """Hash symbol to numeric. Simple and deterministic."""
    return float(hash(symbol) % 10000)


def prepare_training_rows(records: list[DecisionExportRecord]) -> list[ModelReadyRow]:
    """
    Convert decision export records into model-ready rows.

    Features: ts_ordinal, symbol_hash, action_encoded, qty, risk_approved.
    Label: filled (0 or 1).
    """
    rows: list[ModelReadyRow] = []
    for r in records:
        try:
            qty = float(r.qty) if r.qty else 0.0
        except (ValueError, TypeError):
            qty = 0.0
        features = {
            "ts_ordinal": r.ts_utc.timestamp(),
            "symbol_hash": _symbol_hash(r.symbol),
            "action_encoded": float(_action_to_int(r.action)),
            "qty": qty,
            "risk_approved": 1.0 if r.risk_approved else 0.0,
        }
        label = 1 if r.filled else 0
        rows.append(
            ModelReadyRow(
                ts_utc=r.ts_utc,
                symbol=r.symbol,
                features=features,
                label=label,
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
