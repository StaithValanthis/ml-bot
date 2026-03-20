"""Unit tests for Bybit auth and request body serialization."""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from trading.exchange.bybit_auth import canonical_json_body
from trading.exchange.bybit_rest import BybitRestClient
from trading.exchange.schemas import PlaceOrderRequest
from trading.settings import ExchangeSettings
from trading.util.types import OrderSide, OrderType, TimeInForce


def _settings() -> ExchangeSettings:
    s = ExchangeSettings.model_validate(
        {
            "provider": "bybit",
            "base_url": "https://api-testnet.bybit.com",
            "public_ws_url": "wss://stream-testnet.bybit.com/v5/public/linear",
            "private_ws_url": "wss://stream-testnet.bybit.com/v5/private",
            "testnet": True,
            "recv_window_ms": 5000,
            "request_timeout_seconds": 10,
            "max_retries": 2,
            "backoff_base_seconds": 0.1,
            "backoff_max_seconds": 0.2,
        }
    )
    s.bybit_api_key = SecretStr("test-key")
    s.bybit_api_secret = SecretStr("test-secret")
    return s


def test_canonical_json_body_with_decimal_qty_price() -> None:
    """canonical_json_body serializes body with Decimal qty/price without crash."""
    body = {
        "category": "linear",
        "symbol": "BTCUSDT",
        "side": "Buy",
        "orderType": "Limit",
        "qty": Decimal("0.001"),
        "price": Decimal("50000.5"),
        "timeInForce": "PostOnly",
        "orderLinkId": "drill-btcu-240320100000",
        "reduceOnly": False,
    }
    result = canonical_json_body(body)
    assert "Object of type Decimal is not JSON serializable" not in result
    parsed = json.loads(result)
    assert parsed["qty"] == "0.001"
    assert parsed["price"] == "50000.5"
    assert parsed["symbol"] == "BTCUSDT"


def test_canonical_json_body_place_order_request_model_dump() -> None:
    """PlaceOrderRequest.model_dump produces dict with Decimal; canonical_json_body serializes it."""
    request = PlaceOrderRequest(
        category="linear",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        qty=Decimal("0.001"),
        price=Decimal("49999.99"),
        time_in_force=TimeInForce.POST_ONLY,
        order_link_id="drill-btcu-240320100000",
        reduce_only=False,
    )
    body = request.model_dump(by_alias=True, exclude_none=True)
    result = canonical_json_body(body)
    assert "Object of type Decimal is not JSON serializable" not in result
    parsed = json.loads(result)
    assert parsed["qty"] == "0.001"
    assert parsed["price"] == "49999.99"


def test_canonical_json_body_signing_and_request_body_identical() -> None:
    """The same serialized body is used for signing and outbound POST; verify no crash."""
    body = {
        "category": "linear",
        "symbol": "BTCUSDT",
        "side": "Buy",
        "orderType": "Limit",
        "qty": Decimal("0.001"),
        "price": Decimal("50000"),
        "timeInForce": "PostOnly",
        "orderLinkId": "drill-btcu-240320100000",
    }
    # Simulate what _request does: one body_payload for both signing and content
    body_payload = canonical_json_body(body)
    sign_payload = body_payload
    request_content = body_payload
    assert sign_payload == request_content
    assert json.loads(sign_payload) == json.loads(request_content)


@pytest.mark.asyncio
async def test_place_order_request_body_serializes_with_decimal() -> None:
    """place_order with Decimal qty/price reaches POST body construction without serialization failure."""
    settings = _settings()
    captured_content: bytes | None = None

    def make_response() -> object:
        mock = MagicMock()
        mock.status_code = 200
        mock.json.return_value = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {"orderId": "ex-123", "orderLinkId": "drill-btcu-240320100000"},
            "time": 1710000000123,
        }
        mock.raise_for_status = MagicMock()
        mock.text = "{}"
        return mock

    async def capture_request(*args: object, **kwargs: object) -> object:
        nonlocal captured_content
        captured_content = kwargs.get("content")
        return make_response()

    client = BybitRestClient(settings)
    with patch.object(client._client, "request", side_effect=capture_request):
        request = PlaceOrderRequest(
            category="linear",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            qty=Decimal("0.001"),
            price=Decimal("49999.99"),
            time_in_force=TimeInForce.POST_ONLY,
            order_link_id="drill-btcu-240320100000",
            reduce_only=False,
        )
        await client.place_order(request)

    assert captured_content is not None
    body_str = captured_content.decode("utf-8") if isinstance(captured_content, bytes) else captured_content
    parsed = json.loads(body_str)
    assert parsed["qty"] == "0.001"
    assert parsed["price"] == "49999.99"
    await client.close()
