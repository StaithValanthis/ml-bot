from __future__ import annotations

from decimal import Decimal

import pytest

from trading.exchange.bybit_rest import BybitAPIError, BybitRestClient
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
async def test_get_server_time_uses_validated_result_body() -> None:
    client = BybitRestClient(_settings())

    async def fake_request(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "retCode": 0,
            "retMsg": "OK",
            "result": {"timeSecond": "1710000000", "timeNano": "1710000000000000000"},
            "time": 1710000000123,
        }

    client._request = fake_request  # type: ignore[method-assign]
    result = await client.get_server_time()
    await client.close()

    assert result.time_second == "1710000000"
    assert result.time_nano == "1710000000000000000"


@pytest.mark.asyncio
async def test_get_kline_uses_validated_result_body() -> None:
    client = BybitRestClient(_settings())

    async def fake_request(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "category": "linear",
                "symbol": "BTCUSDT",
                "list": [
                    ["1710000000000", "60000", "60200", "59900", "60100", "123.4", "7412340.0"],
                ],
            },
            "time": 1710000000123,
        }

    client._request = fake_request  # type: ignore[method-assign]
    result = await client.get_kline(category="linear", symbol="BTCUSDT", interval="5")
    await client.close()

    assert len(result) == 1
    assert result[0].start_time_ms == 1710000000000
    assert result[0].close_price == Decimal("60100")


@pytest.mark.asyncio
async def test_validate_envelope_raises_on_nonzero_ret_code() -> None:
    client = BybitRestClient(_settings())

    with pytest.raises(BybitAPIError):
        client._validate_envelope(
            {
                "retCode": 10001,
                "retMsg": "param error",
                "result": {},
                "time": 1710000000123,
            },
            "test",
        )

    await client.close()
