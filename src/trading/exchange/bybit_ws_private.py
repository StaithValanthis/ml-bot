from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from collections.abc import Awaitable, Callable
from time import time
from typing import Any

import websockets
from websockets.client import WebSocketClientProtocol

from trading.marketdata.normalizers import NormalizedEvent, normalize_private_message
from trading.settings import ExchangeSettings
from trading.util.logging import get_logger

PrivateMessageHandler = Callable[[list[NormalizedEvent]], Awaitable[None]]


class BybitWsPrivateClient:
    """
    Private V5 websocket client with auth, reconnect, resubscribe, and ping.
    """

    def __init__(
        self,
        settings: ExchangeSettings,
        *,
        message_handler: PrivateMessageHandler,
        connection_state_handler: Callable[[bool], None] | None = None,
        ping_interval_seconds: float = 20.0,
        ack_timeout_seconds: float = 8.0,
        reconnect_base_seconds: float = 1.0,
        reconnect_max_seconds: float = 30.0,
    ) -> None:
        self._settings = settings
        self._message_handler = message_handler
        self._ping_interval_seconds = ping_interval_seconds
        self._ack_timeout_seconds = ack_timeout_seconds
        self._reconnect_base_seconds = reconnect_base_seconds
        self._reconnect_max_seconds = reconnect_max_seconds
        self._connection_state_handler = connection_state_handler
        self._logger = get_logger("trading.exchange.bybit_ws_private")
        self._stop_event = asyncio.Event()
        self._subscriptions: set[str] = set()
        self._connection: WebSocketClientProtocol | None = None
        self._pending_auth_ack: asyncio.Future[dict[str, Any]] | None = None
        self._pending_subscribe_acks: dict[str, asyncio.Future[dict[str, Any]]] = {}

    async def subscribe(self, topics: list[str]) -> None:
        self._subscriptions.update(topics)
        if self._connection is not None:
            await self._send_subscribe(self._connection, topics)

    async def run_forever(self) -> None:
        delay = self._reconnect_base_seconds
        while not self._stop_event.is_set():
            try:
                await self._run_once()
                delay = self._reconnect_base_seconds
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._logger.exception("private_ws_loop_error", error=str(exc))
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._reconnect_max_seconds)

    async def close(self) -> None:
        self._stop_event.set()
        if self._connection is not None:
            await self._connection.close()

    async def _run_once(self) -> None:
        async with websockets.connect(
            self._settings.private_ws_url,
            ping_interval=None,
            close_timeout=3.0,
            max_size=2**22,
        ) as ws:
            self._connection = ws
            self._logger.info("private_ws_connected", url=self._settings.private_ws_url)
            reader_task = asyncio.create_task(self._read_loop(ws), name="bybit-private-reader")
            await self._authenticate(ws)
            if self._subscriptions:
                await self._send_subscribe(ws, list(self._subscriptions))
            ping_task = asyncio.create_task(self._ping_loop(ws), name="bybit-private-ping")
            self._emit_connection_state(True)
            done, pending = await asyncio.wait(
                [reader_task, ping_task],
                return_when=asyncio.FIRST_EXCEPTION,
            )
            for task in pending:
                task.cancel()
            for task in done:
                exception = task.exception()
                if exception is not None:
                    raise exception
        self._connection = None
        self._reset_ack_state()
        self._emit_connection_state(False)
        self._logger.warning("private_ws_disconnected")

    async def _authenticate(self, ws: WebSocketClientProtocol) -> None:
        if self._settings.bybit_api_key is None or self._settings.bybit_api_secret is None:
            raise RuntimeError("Bybit API credentials are required for private websocket.")
        self._pending_auth_ack = asyncio.get_running_loop().create_future()
        api_key = self._settings.bybit_api_key.get_secret_value()
        secret = self._settings.bybit_api_secret.get_secret_value()
        expires = int((time() + 10) * 1000)
        signature = hmac.new(
            secret.encode("utf-8"),
            f"GET/realtime{expires}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        payload = {"op": "auth", "args": [api_key, expires, signature]}
        await ws.send(json.dumps(payload))
        self._logger.info("private_ws_auth_sent")
        try:
            ack = await asyncio.wait_for(self._pending_auth_ack, timeout=self._ack_timeout_seconds)
        except TimeoutError as exc:
            self._pending_auth_ack = None
            raise RuntimeError("Private auth ack timeout.") from exc
        if ack.get("success") is not True:
            self._pending_auth_ack = None
            raise RuntimeError(f"Private websocket auth failed: {ack}")
        self._pending_auth_ack = None

    async def _send_subscribe(self, ws: WebSocketClientProtocol, topics: list[str]) -> None:
        if not topics:
            return
        req_id = f"sub-private-{asyncio.get_running_loop().time():.6f}"
        ack_future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending_subscribe_acks[req_id] = ack_future
        payload = {"op": "subscribe", "req_id": req_id, "args": topics}
        await ws.send(json.dumps(payload))
        self._logger.info("private_ws_subscribe_sent", topics=topics, req_id=req_id)
        try:
            ack = await asyncio.wait_for(ack_future, timeout=self._ack_timeout_seconds)
        except TimeoutError as exc:
            self._pending_subscribe_acks.pop(req_id, None)
            raise RuntimeError(f"Private subscribe ack timeout req_id={req_id}") from exc
        if ack.get("success") is not True:
            raise RuntimeError(f"Private subscribe rejected req_id={req_id}: {ack}")

    async def _ping_loop(self, ws: WebSocketClientProtocol) -> None:
        while not self._stop_event.is_set():
            await asyncio.sleep(self._ping_interval_seconds)
            await ws.send(json.dumps({"op": "ping"}))

    async def _read_loop(self, ws: WebSocketClientProtocol) -> None:
        async for raw_msg in ws:
            payload = self._decode_json(raw_msg)
            if payload is None:
                continue
            if self._handle_control_message(payload):
                continue
            if _is_auth_failure(payload):
                raise RuntimeError(f"Private websocket auth failed: {payload}")
            if _is_control_message(payload):
                continue
            events = normalize_private_message(payload)
            if events:
                # private payloads can be partial; normalizer returns optional fields.
                await self._message_handler(events)

    def _decode_json(self, raw_msg: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(raw_msg)
        except json.JSONDecodeError:
            self._logger.warning("private_ws_json_decode_failed")
            return None
        if not isinstance(payload, dict):
            self._logger.warning("private_ws_non_object_payload")
            return None
        return payload

    def _emit_connection_state(self, connected: bool) -> None:
        if self._connection_state_handler is not None:
            self._connection_state_handler(connected)

    def _handle_control_message(self, payload: dict[str, Any]) -> bool:
        op = payload.get("op")
        if op == "auth" and self._pending_auth_ack is not None:
            if not self._pending_auth_ack.done():
                self._pending_auth_ack.set_result(payload)
            return True
        if op == "subscribe":
            req_id = payload.get("req_id") or payload.get("reqId")
            if isinstance(req_id, str):
                pending = self._pending_subscribe_acks.pop(req_id, None)
                if pending is not None:
                    if not pending.done():
                        pending.set_result(payload)
                    return True
        return False

    def _reset_ack_state(self) -> None:
        if self._pending_auth_ack is not None and not self._pending_auth_ack.done():
            self._pending_auth_ack.cancel()
        self._pending_auth_ack = None
        for future in self._pending_subscribe_acks.values():
            if not future.done():
                future.cancel()
        self._pending_subscribe_acks.clear()


def _is_auth_failure(payload: dict[str, Any]) -> bool:
    if payload.get("op") != "auth":
        return False
    success = payload.get("success")
    return success is False


def _is_control_message(payload: dict[str, Any]) -> bool:
    op = payload.get("op")
    if op in {"pong", "ping", "subscribe", "unsubscribe", "auth"}:
        return True
    if "success" in payload and "ret_msg" in payload:
        return True
    return False
