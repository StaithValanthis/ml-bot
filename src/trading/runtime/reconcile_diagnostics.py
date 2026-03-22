"""Reconcile mismatch aggregation and diagnostics."""

from __future__ import annotations

from typing import Any


def bucket_details(details: str) -> str:
    """Normalize details string into a reason bucket."""
    d = (details or "").lower()
    if "missing_on_exchange" in d or "not found remotely" in d:
        return "missing_on_exchange"
    if "missing_locally" in d or "not tracked locally" in d:
        return "missing_locally"
    if "missing_reduce_only_exit" in d or "no tracked reduce-only" in d or "reduce-only exit" in d or "no local tracked reduce-only" in d:
        return "missing_reduce_only_exit"
    if "qty_mismatch" in d or "qty mismatch" in d:
        return "qty_mismatch"
    return "other"


def aggregate_reconcile_issues(
    accumulated: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate reconcile mismatch events into structured diagnostics.
    Each accumulated entry: {issue_type, symbol, details, occurred_at}
    """
    if not accumulated:
        return {
            "total_occurrences": 0,
            "first_occurrence_at": None,
            "last_occurrence_at": None,
            "by_issue_type": {},
            "by_symbol": {},
            "by_reason_bucket": {},
            "top_issue_type": None,
            "top_symbol": None,
            "top_reason_bucket": None,
        }

    first_ts = None
    last_ts = None
    by_type: dict[str, int] = {}
    by_symbol: dict[str, int] = {}
    by_bucket: dict[str, int] = {}

    for e in accumulated:
        ts = e.get("occurred_at")
        if ts:
            first_ts = first_ts or ts
            last_ts = ts
        it = e.get("issue_type") or "unknown"
        sym = e.get("symbol") or "_unknown_"
        details = e.get("details") or ""
        bucket = bucket_details(details)

        by_type[it] = by_type.get(it, 0) + 1
        by_symbol[sym] = by_symbol.get(sym, 0) + 1
        by_bucket[bucket] = by_bucket.get(bucket, 0) + 1

    def _top(d: dict[str, int]) -> str | None:
        if not d:
            return None
        return max(d, key=d.get)

    return {
        "total_occurrences": len(accumulated),
        "first_occurrence_at": first_ts,
        "last_occurrence_at": last_ts,
        "by_issue_type": by_type,
        "by_symbol": by_symbol,
        "by_reason_bucket": by_bucket,
        "top_issue_type": _top(by_type),
        "top_symbol": _top(by_symbol),
        "top_reason_bucket": _top(by_bucket),
    }


def build_reconcile_diagnostics_summary(aggregated: dict[str, Any]) -> dict[str, Any]:
    """Build promotion-ready reconcile diagnostics with top_N."""
    top_3_buckets = []
    by_bucket = aggregated.get("by_reason_bucket") or {}
    for bucket, count in sorted(by_bucket.items(), key=lambda x: -x[1])[:3]:
        top_3_buckets.append({"reason_bucket": bucket, "count": count})
    return {
        **aggregated,
        "top_3_reason_buckets": top_3_buckets,
    }
