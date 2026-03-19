from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import websockets
from websockets.client import WebSocketClientProtocol

from trading.marketdata.normalizers import NormalizedEvent, normalize_public_message
from trading.settings import ExchangeSettings
from trading.util.logging import get_logger

PublicMessageHandler = Callable[[list[NormalizedEvent]], Awaitable[None]]


class BybitWsPublicClient:
    """
    Public V5 websocket client with reconnect, resubscribe, ping, and dispatch.
    """

    def __init__(
        self,
        settings: ExchangeSettings,
        *,
        message_handler: PublicMessageHandler,
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
        self._logger = get_logger("trading.exchange.bybit_ws_public")
        self._stop_event = asyncio.Event()
        self._subscriptions: set[str] = set()
        self._connection: WebSocketClientProtocol | None = None
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
                self._logger.exception("public_ws_loop_error", error=str(exc))
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._reconnect_max_seconds)

    async def close(self) -> None:
        self._stop_event.set()
        if self._connection is not None:
            await self._connection.close()

    async def _run_once(self) -> None:
        async with websockets.connect(
            self._settings.public_ws_url,
            ping_interval=None,
            close_timeout=3.0,
            max_size=2**22,
        ) as ws:
            self._connection = ws
            self._logger.info("public_ws_connected", url=self._settings.public_ws_url)
            reader_task = asyncio.create_task(self._read_loop(ws), name="bybit-public-reader")
            if self._subscriptions:
                await self._send_subscribe(ws, list(self._subscriptions))
            ping_task = asyncio.create_task(self._ping_loop(ws), name="bybit-public-ping")
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
        self._logger.warning("public_ws_disconnected")

    async def _send_subscribe(self, ws: WebSocketClientProtocol, topics: list[str]) -> None:
        if not topics:
            return
        req_id = f"sub-public-{asyncio.get_running_loop().time():.6f}"
        ack_future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending_subscribe_acks[req_id] = ack_future
        payload = {"op": "subscribe", "req_id": req_id, "args": topics}
        await ws.send(json.dumps(payload))
        self._logger.info("public_ws_subscribe_sent", topics=topics, req_id=req_id)
        try:
            ack = await asyncio.wait_for(ack_future, timeout=self._ack_timeout_seconds)
        except TimeoutError as exc:
            self._pending_subscribe_acks.pop(req_id, None)
            raise RuntimeError(f"Public subscribe ack timeout req_id={req_id}") from exc
        if ack.get("success") is not True:
            raise RuntimeError(f"Public subscribe rejected req_id={req_id}: {ack}")

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
            if _is_control_message(payload):
                continue
            events = normalize_public_message(payload)
            if events:
                await self._message_handler(events)

    def _decode_json(self, raw_msg: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(raw_msg)
        except json.JSONDecodeError:
            self._logger.warning("public_ws_json_decode_failed")
            return None
        if not isinstance(payload, dict):
            self._logger.warning("public_ws_non_object_payload")
            return None
        return payload

    def _emit_connection_state(self, connected: bool) -> None:
        if self._connection_state_handler is not None:
            self._connection_state_handler(connected)

    def _handle_control_message(self, payload: dict[str, Any]) -> bool:
        op = payload.get("op")
        if op != "subscribe":
            return False
        req_id = payload.get("req_id") or payload.get("reqId")
        if not isinstance(req_id, str):
            return False
        pending = self._pending_subscribe_acks.pop(req_id, None)
        if pending is None:
            return False
        if not pending.done():
            pending.set_result(payload)
        return True

    def _reset_ack_state(self) -> None:
        for future in self._pending_subscribe_acks.values():
            if not future.done():
                future.cancel()
        self._pending_subscribe_acks.clear()


def _is_control_message(payload: dict[str, Any]) -> bool:
    op = payload.get("op")
    if op in {"pong", "ping", "subscribe", "unsubscribe", "auth"}:
        return True
    if "success" in payload and "ret_msg" in payload:
        return True
    return False
