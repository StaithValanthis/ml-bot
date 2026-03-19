"""Unit tests for DEMO endpoint selection."""

from __future__ import annotations

import os

import pytest
import yaml

from trading.settings import load_settings


def test_demo_uses_mainnet_public_ws() -> None:
    """DEMO mode must use stream.bybit.com for public WS (stream-demo returns 404)."""
    with pytest.MonkeyPatch.context() as m:
        m.setenv("TRADING_MODE", "demo")
        m.setenv("TRADING_ENV", "bybit_demo")
        settings = load_settings()
    assert "stream.bybit.com" in settings.exchange.public_ws_url
    assert "stream-demo" not in settings.exchange.public_ws_url


def test_demo_uses_demo_private_ws() -> None:
    """DEMO mode uses stream-demo for private WS."""
    with pytest.MonkeyPatch.context() as m:
        m.setenv("TRADING_MODE", "demo")
        m.setenv("TRADING_ENV", "bybit_demo")
        settings = load_settings()
    assert "stream-demo.bybit.com" in settings.exchange.private_ws_url


def test_demo_uses_demo_rest() -> None:
    """DEMO mode uses api-demo.bybit.com for REST."""
    with pytest.MonkeyPatch.context() as m:
        m.setenv("TRADING_MODE", "demo")
        m.setenv("TRADING_ENV", "bybit_demo")
        settings = load_settings()
    assert "api-demo.bybit.com" in settings.exchange.base_url


def test_bybit_demo_yaml_public_ws_is_mainnet() -> None:
    """configs/bybit_demo.yaml explicitly uses mainnet public stream."""
    from pathlib import Path

    config_dir = Path("configs")
    demo_path = config_dir / "bybit_demo.yaml"
    if not demo_path.exists():
        pytest.skip("configs/bybit_demo.yaml not found")
    data = yaml.safe_load(demo_path.read_text(encoding="utf-8"))
    public_ws = data.get("exchange", {}).get("public_ws_url", "")
    assert "stream.bybit.com" in public_ws
    assert "stream-demo" not in public_ws
