"""Unit tests for reconcile diagnostics and aggregation."""

from __future__ import annotations

from trading.settings import load_settings
from trading.runtime.reconcile_diagnostics import (
    aggregate_reconcile_issues,
    bucket_details,
    build_reconcile_diagnostics_summary,
)


def _sym() -> str:
    return load_settings().trading.symbols[0]


def test_bucket_details_missing_on_exchange() -> None:
    """Details mentioning missing on exchange map to missing_on_exchange."""
    assert bucket_details("Local open order not found remotely: link_id=x") == "missing_on_exchange"
    assert bucket_details("missing_on_exchange") == "missing_on_exchange"


def test_bucket_details_missing_locally() -> None:
    """Details mentioning missing locally map to missing_locally."""
    assert bucket_details("Exchange open order not tracked locally") == "missing_locally"


def test_bucket_details_missing_reduce_only_exit() -> None:
    """Details mentioning reduce-only exit map to missing_reduce_only_exit."""
    assert bucket_details("Non-flat position has no local tracked reduce-only exit") == "missing_reduce_only_exit"


def test_bucket_details_qty_mismatch() -> None:
    """Details mentioning qty mismatch map to qty_mismatch."""
    assert bucket_details("qty mismatch link_id=x local=0.1 remote=0.2") == "qty_mismatch"


def test_bucket_details_other() -> None:
    """Unknown details map to other."""
    assert bucket_details("something else") == "other"


def test_aggregate_reconcile_issues_empty() -> None:
    """Empty accumulator yields zeroed diagnostics."""
    agg = aggregate_reconcile_issues([])
    assert agg["total_occurrences"] == 0
    assert agg["first_occurrence_at"] is None
    assert agg["by_issue_type"] == {}
    assert agg["top_issue_type"] is None
    assert agg["top_symbol"] is None


def test_aggregate_reconcile_issues_by_type_and_symbol() -> None:
    """Aggregation counts by issue_type and symbol."""
    other = next(s for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT") if s != _sym())
    accumulated = [
        {"issue_type": "missing_on_exchange", "symbol": _sym(), "details": "Local not found", "occurred_at": "2025-01-01T10:00:00Z"},
        {"issue_type": "missing_on_exchange", "symbol": _sym(), "details": "Local not found 2", "occurred_at": "2025-01-01T10:01:00Z"},
        {"issue_type": "missing_locally", "symbol": other, "details": "Not tracked", "occurred_at": "2025-01-01T10:02:00Z"},
    ]
    agg = aggregate_reconcile_issues(accumulated)
    assert agg["total_occurrences"] == 3
    assert agg["by_issue_type"]["missing_on_exchange"] == 2
    assert agg["by_issue_type"]["missing_locally"] == 1
    assert agg["by_symbol"][_sym()] == 2
    assert agg["by_symbol"][other] == 1
    assert agg["top_issue_type"] == "missing_on_exchange"
    assert agg["top_symbol"] == _sym()


def test_build_reconcile_diagnostics_summary_top_3() -> None:
    """Summary includes top_3_reason_buckets."""
    agg = {
        "by_reason_bucket": {"missing_on_exchange": 5, "missing_locally": 2, "qty_mismatch": 1},
    }
    summary = build_reconcile_diagnostics_summary(agg)
    top3 = summary.get("top_3_reason_buckets") or []
    assert len(top3) == 3
    assert top3[0]["reason_bucket"] == "missing_on_exchange"
    assert top3[0]["count"] == 5
    assert top3[1]["reason_bucket"] == "missing_locally"
    assert top3[2]["reason_bucket"] == "qty_mismatch"
