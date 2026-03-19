from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from trading.exchange.bybit_auth import (
    build_v5_auth_headers,
    canonical_json_body,
    canonical_query_string,
)
from trading.exchange.rate_limiter import EndpointRateLimiter, build_default_bybit_limiter
from trading.exchange.schemas import (
    AmendOrderRequest,
    BybitEnvelope,
    BybitListResult,
    CancelOrderRequest,
    FeeRateItem,
    FundingHistoryItem,
    KlineItem,
    OpenInterestItem,
    OpenOrderItem,
    OrderAck,
    PlaceOrderRequest,
    PositionItem,
    ServerTimeResult,
    WalletBalanceItem,
)
from trading.settings import ExchangeSettings
from trading.util.logging import get_logger
from trading.util.retry import build_async_retrying


class BybitRestError(Exception):
    """Base exception for Bybit REST interactions."""


class BybitAuthenticationError(BybitRestError):
    """Raised when authenticated request cannot be signed or authorized."""


class BybitHTTPError(BybitRestError):
    def __init__(self, message: str, *, status_code: int, response_text: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


class BybitAPIError(BybitRestError):
    def __init__(
        self,
        message: str,
        *,
        ret_code: int,
        ret_msg: str,
        payload: dict[str, Any],
        operation: str = "",
        scope: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.ret_code = ret_code
        self.ret_msg = ret_msg
        self.payload = payload
        self.operation = operation
        self.scope = scope or {}


class BybitResponseDecodeError(BybitRestError):
    """Raised when response body is invalid or incompatible with schema."""


class BybitRestClient:
    def __init__(
        self,
        settings: ExchangeSettings,
        *,
        rate_limiter: EndpointRateLimiter | None = None,
    ) -> None:
        self._settings = settings
        self._rate_limiter = rate_limiter or build_default_bybit_limiter()
        self._logger = get_logger("trading.exchange.bybit_rest")
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=httpx.Timeout(settings.request_timeout_seconds),
            headers={"Content-Type": "application/json"},
        )
        self._retrying = build_async_retrying(
            max_attempts=settings.max_retries,
            base_seconds=settings.backoff_base_seconds,
            max_seconds=settings.backoff_max_seconds,
            logger=lambda msg: self._logger.warning("http_retry", message=msg),
        )

    async def __aenter__(self) -> BybitRestClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def get_server_time(self) -> ServerTimeResult:
        data = await self._request("GET", "/v5/market/time", endpoint_group="market", auth=False)
        result = self._validate_envelope(data, "server time")
        return ServerTimeResult.model_validate(result)

    async def get_fee_rate(self, *, category: str, symbol: str) -> list[FeeRateItem]:
        data = await self._request(
            "GET",
            "/v5/account/fee-rate",
            params={"category": category, "symbol": symbol},
            endpoint_group="account",
            auth=True,
        )
        result = self._validate_envelope(data, "fee rate")
        items = self._extract_list(result, "fee rate")
        return [FeeRateItem.model_validate(item) for item in items]

    async def get_open_orders(self, *, category: str, symbol: str | None = None) -> list[OpenOrderItem]:
        params: dict[str, Any] = {"category": category}
        if symbol is not None:
            params["symbol"] = symbol
        data = await self._request(
            "GET",
            "/v5/order/realtime",
            params=params,
            endpoint_group="trade",
            auth=True,
        )
        result = self._validate_envelope(data, "open_orders", scope={"category": category, "symbol": symbol})
        result = BybitListResult[OpenOrderItem].model_validate(result)
        return result.items

    async def get_positions(self, *, category: str, symbol: str | None = None) -> list[PositionItem]:
        params: dict[str, Any] = {"category": category}
        if symbol is not None:
            params["symbol"] = symbol
        data = await self._request(
            "GET",
            "/v5/position/list",
            params=params,
            endpoint_group="position",
            auth=True,
        )
        result = self._validate_envelope(data, "position_list", scope={"category": category, "symbol": symbol})
        result = BybitListResult[PositionItem].model_validate(result)
        return result.items

    async def get_wallet(self, *, account_type: str = "UNIFIED", coin: str | None = None) -> list[WalletBalanceItem]:
        params: dict[str, Any] = {"accountType": account_type}
        if coin is not None:
            params["coin"] = coin
        data = await self._request(
            "GET",
            "/v5/account/wallet-balance",
            params=params,
            endpoint_group="account",
            auth=True,
        )
        result = self._validate_envelope(data, "wallet balance")
        result = BybitListResult[WalletBalanceItem].model_validate(result)
        return result.items

    async def place_order(self, request: PlaceOrderRequest) -> OrderAck:
        data = await self._request(
            "POST",
            "/v5/order/create",
            body=request.model_dump(by_alias=True, exclude_none=True),
            endpoint_group="trade",
            auth=True,
        )
        result = self._validate_envelope(
            data, "place_order", scope={"category": request.category, "symbol": request.symbol}
        )
        return OrderAck.model_validate(result)

    async def amend_order(self, request: AmendOrderRequest) -> OrderAck:
        data = await self._request(
            "POST",
            "/v5/order/amend",
            body=request.model_dump(by_alias=True, exclude_none=True),
            endpoint_group="trade",
            auth=True,
        )
        result = self._validate_envelope(data, "amend order")
        return OrderAck.model_validate(result)

    async def cancel_order(self, request: CancelOrderRequest) -> OrderAck:
        data = await self._request(
            "POST",
            "/v5/order/cancel",
            body=request.model_dump(by_alias=True, exclude_none=True),
            endpoint_group="trade",
            auth=True,
        )
        result = self._validate_envelope(data, "cancel order")
        return OrderAck.model_validate(result)

    async def get_kline(
        self,
        *,
        category: str,
        symbol: str,
        interval: str,
        limit: int = 200,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[KlineItem]:
        params: dict[str, Any] = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_ms is not None:
            params["start"] = start_ms
        if end_ms is not None:
            params["end"] = end_ms
        data = await self._request(
            "GET",
            "/v5/market/kline",
            params=params,
            endpoint_group="market",
            auth=False,
        )
        result = self._validate_envelope(data, "kline")
        raw_rows = self._extract_list(result, "kline")
        return [KlineItem.from_raw(row) for row in raw_rows]

    async def get_funding_history(
        self,
        *,
        category: str,
        symbol: str,
        limit: int = 200,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[FundingHistoryItem]:
        params: dict[str, Any] = {"category": category, "symbol": symbol, "limit": limit}
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms
        data = await self._request(
            "GET",
            "/v5/market/funding/history",
            params=params,
            endpoint_group="market",
            auth=False,
        )
        result = self._validate_envelope(data, "funding history")
        list_result = result.get("list", [])
        if not isinstance(list_result, list):
            raise BybitResponseDecodeError("Unexpected funding history payload shape.")
        return [FundingHistoryItem.model_validate(item) for item in list_result]

    async def get_open_interest(
        self,
        *,
        category: str,
        symbol: str,
        interval_time: str,
        limit: int = 200,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[OpenInterestItem]:
        params: dict[str, Any] = {
            "category": category,
            "symbol": symbol,
            "intervalTime": interval_time,
            "limit": limit,
        }
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms
        data = await self._request(
            "GET",
            "/v5/market/open-interest",
            params=params,
            endpoint_group="market",
            auth=False,
        )
        result = self._validate_envelope(data, "open interest")
        list_result = result.get("list", [])
        if not isinstance(list_result, list):
            raise BybitResponseDecodeError("Unexpected open interest payload shape.")
        return [OpenInterestItem.model_validate(item) for item in list_result]

    async def _request(
        self,
        method: str,
        path: str,
        *,
        endpoint_group: str,
        auth: bool,
        params: Mapping[str, Any] | None = None,
        body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        await self._rate_limiter.acquire(endpoint_group)
        query = canonical_query_string(dict(params) if params else None)
        body_payload = canonical_json_body(dict(body) if body else None)
        sign_payload = query if method.upper() == "GET" else body_payload

        headers: dict[str, str] = {}
        if auth:
            try:
                headers = build_v5_auth_headers(settings=self._settings, payload=sign_payload)
            except ValueError as exc:
                raise BybitAuthenticationError(str(exc)) from exc

        async for attempt in self._retrying:
            with attempt:
                response = await self._client.request(
                    method=method,
                    url=path,
                    params=dict(params) if params else None,
                    content=body_payload if method.upper() != "GET" and body_payload else None,
                    headers=headers,
                )
                if response.status_code >= 500:
                    self._logger.warning(
                        "bybit_http_server_error",
                        status_code=response.status_code,
                        path=path,
                        body=response.text,
                    )
                    response.raise_for_status()
                if response.status_code >= 400:
                    raise BybitHTTPError(
                        f"Bybit HTTP error for {path}",
                        status_code=response.status_code,
                        response_text=response.text,
                    )
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise BybitResponseDecodeError("Failed to decode JSON response from Bybit.") from exc
                if not isinstance(payload, dict):
                    raise BybitResponseDecodeError("Bybit response payload is not a JSON object.")
                return payload

        raise BybitRestError("Unreachable retry termination in _request.")

    def _validate_envelope(
        self,
        payload: dict[str, Any],
        operation: str,
        scope: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            envelope = BybitEnvelope[dict[str, Any]].model_validate(payload)
        except Exception as exc:  # pragma: no cover - pydantic details are tested via integration tests
            raise BybitResponseDecodeError(f"Unexpected Bybit payload for {operation}.") from exc
        if envelope.ret_code != 0:
            raise BybitAPIError(
                f"Bybit API error during {operation}",
                ret_code=envelope.ret_code,
                ret_msg=envelope.ret_msg,
                payload=payload,
                operation=operation,
                scope=scope or {},
            )
        return envelope.result

    @staticmethod
    def _extract_list(result: dict[str, Any], operation: str) -> list[Any]:
        raw_list = result.get("list", [])
        if not isinstance(raw_list, list):
            raise BybitResponseDecodeError(f"Unexpected list payload shape for {operation}.")
        return raw_list
