"""JSON-safe serialization for structured payloads (ledger, logs, summaries)."""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from typing import Any


def _json_default(obj: object) -> str:
    """Fallback for json.dumps when encountering non-serializable types."""
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def json_safe(obj: Any) -> Any:
    """
    Recursively normalize a payload for JSON serialization.
    Converts Decimal -> str, datetime -> isoformat string.
    Returns a new structure safe for json.dumps without a default.
    """
    if isinstance(obj, Decimal):
        s = format(obj, "f")
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s if s else "0"
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def dumps_json_safe(obj: Any, **kwargs: Any) -> str:
    """Serialize obj to JSON string, handling Decimal and datetime."""
    return json.dumps(obj, default=_json_default, **kwargs)
