from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(UTC)


def ensure_utc(dt: datetime) -> datetime:
    """Normalize a datetime into UTC."""
    if dt.tzinfo is None:
        raise ValueError("Naive datetime is not allowed; expected UTC-aware datetime.")
    return dt.astimezone(UTC)
