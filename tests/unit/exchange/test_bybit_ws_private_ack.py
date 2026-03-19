"""Unit tests for private WebSocket auth and subscribe ack handling."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from trading.exchange.bybit_ws_private import BybitWsPrivateClient
from trading.settings import ExchangeSettings


def _settings() -> ExchangeSettings:
    return ExchangeSettings.model_validate(
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


@pytest.mark.asyncio
async def test_handle_control_message_resolves_auth_ack() -> None:
    handler = AsyncMock()
    client = BybitWsPrivateClient(_settings(), message_handler=handler, ack_timeout_seconds=2.0)
    future: asyncio.Future[dict] = asyncio.get_running_loop().create_future()
    client._pending_auth_ack = future

    payload = {"op": "auth", "success": True, "ret_msg": "OK"}
    handled = client._handle_control_message(payload)

    assert handled is True
    assert future.done()
    assert future.result() == payload


@pytest.mark.asyncio
async def test_handle_control_message_resolves_subscribe_ack() -> None:
    handler = AsyncMock()
    client = BybitWsPrivateClient(_settings(), message_handler=handler, ack_timeout_seconds=2.0)
    req_id = "sub-private-123.456"
    future: asyncio.Future[dict] = asyncio.get_running_loop().create_future()
    client._pending_subscribe_acks[req_id] = future

    payload = {"op": "subscribe", "req_id": req_id, "success": True, "ret_msg": "OK"}
    handled = client._handle_control_message(payload)

    assert handled is True
    assert future.done()
    assert future.result() == payload
    assert req_id not in client._pending_subscribe_acks


@pytest.mark.asyncio
async def test_handle_control_message_auth_takes_precedence_over_subscribe() -> None:
    handler = AsyncMock()
    client = BybitWsPrivateClient(_settings(), message_handler=handler)
    auth_future: asyncio.Future[dict] = asyncio.get_running_loop().create_future()
    client._pending_auth_ack = auth_future

    payload = {"op": "auth", "success": True}
    handled = client._handle_control_message(payload)
    assert handled is True
    assert auth_future.done()


@pytest.mark.asyncio
async def test_reset_ack_state_clears_all_pending() -> None:
    handler = AsyncMock()
    client = BybitWsPrivateClient(_settings(), message_handler=handler)
    auth_future: asyncio.Future[dict] = asyncio.get_running_loop().create_future()
    sub_future: asyncio.Future[dict] = asyncio.get_running_loop().create_future()
    client._pending_auth_ack = auth_future
    client._pending_subscribe_acks["req-1"] = sub_future

    client._reset_ack_state()

    assert auth_future.cancelled()
    assert sub_future.cancelled()
    assert client._pending_auth_ack is None
    assert len(client._pending_subscribe_acks) == 0
