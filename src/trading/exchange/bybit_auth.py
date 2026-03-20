from __future__ import annotations

import hashlib
import hmac
import json
from time import time
from urllib.parse import urlencode

from trading.settings import ExchangeSettings
from trading.util.json_util import _json_default


def bybit_timestamp_ms() -> str:
    return str(int(time() * 1000))


def canonical_query_string(params: dict[str, object] | None) -> str:
    if not params:
        return ""
    normalized = {k: _stringify_value(v) for k, v in sorted(params.items(), key=lambda item: item[0])}
    return urlencode(normalized)


def canonical_json_body(body: dict[str, object] | None) -> str:
    """Serialize request body to JSON for signing and outbound POST. Handles Decimal/datetime."""
    if not body:
        return ""
    return json.dumps(body, separators=(",", ":"), sort_keys=True, default=_json_default)


def sign_v5(
    *,
    secret: str,
    timestamp_ms: str,
    api_key: str,
    recv_window_ms: int,
    payload: str,
) -> str:
    plain_text = f"{timestamp_ms}{api_key}{recv_window_ms}{payload}"
    return hmac.new(secret.encode("utf-8"), plain_text.encode("utf-8"), hashlib.sha256).hexdigest()


def build_v5_auth_headers(
    *,
    settings: ExchangeSettings,
    payload: str,
    timestamp_ms: str | None = None,
) -> dict[str, str]:
    if settings.bybit_api_key is None or settings.bybit_api_secret is None:
        raise ValueError("Bybit API credentials are required for authenticated requests.")

    ts = timestamp_ms or bybit_timestamp_ms()
    api_key = settings.bybit_api_key.get_secret_value()
    api_secret = settings.bybit_api_secret.get_secret_value()
    signature = sign_v5(
        secret=api_secret,
        timestamp_ms=ts,
        api_key=api_key,
        recv_window_ms=settings.recv_window_ms,
        payload=payload,
    )
    return {
        "X-BAPI-API-KEY": api_key,
        "X-BAPI-TIMESTAMP": ts,
        "X-BAPI-SIGN": signature,
        "X-BAPI-RECV-WINDOW": str(settings.recv_window_ms),
    }


def _stringify_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
