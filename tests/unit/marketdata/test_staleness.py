"""Unit tests for feed staleness watchdog."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from trading.marketdata.staleness import FeedStalenessLimit, FeedStalenessWatchdog, build_default_watchdog


@pytest.mark.asyncio
async def test_stale_channels_trigger_streams_excludes_private() -> None:
    """Idle private stream does not appear in stale_channels when trigger_streams=public."""
    limit_1s = FeedStalenessLimit(max_age=timedelta(seconds=1))
    watchdog = FeedStalenessWatchdog(
        limits={"public": limit_1s, "private": limit_1s},
        default_limit=limit_1s,
    )
    watchdog.set_expected_channels({"public:BTCUSDT", "private:BTCUSDT"})

    now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    past = now - timedelta(seconds=10)

    with patch("trading.marketdata.staleness.utc_now", return_value=now):
        # Simulate both channels stale by setting last_seen to past
        async with watchdog._lock:
            watchdog._last_seen["public:BTCUSDT"] = past
            watchdog._last_seen["private:BTCUSDT"] = past

        stale_public_only = await watchdog.stale_channels(trigger_streams={"public"})
        assert "public:BTCUSDT" in stale_public_only
        assert "private:BTCUSDT" not in stale_public_only


@pytest.mark.asyncio
async def test_stale_channels_trigger_streams_includes_public() -> None:
    """Stale public channels are returned when trigger_streams=public."""
    limit_1s = FeedStalenessLimit(max_age=timedelta(seconds=1))
    watchdog = FeedStalenessWatchdog(
        limits={"public": limit_1s, "private": limit_1s},
        default_limit=limit_1s,
    )
    watchdog.set_expected_channels({"public:BTCUSDT"})

    now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    past = now - timedelta(seconds=10)

    with patch("trading.marketdata.staleness.utc_now", return_value=now):
        async with watchdog._lock:
            watchdog._last_seen["public:BTCUSDT"] = past

        stale = await watchdog.stale_channels(trigger_streams={"public"})
        assert stale == ["public:BTCUSDT"]


@pytest.mark.asyncio
async def test_stale_channels_all_streams_when_no_filter() -> None:
    """stale_channels(trigger_streams=None) returns all stale channels."""
    limit_1s = FeedStalenessLimit(max_age=timedelta(seconds=1))
    watchdog = FeedStalenessWatchdog(
        limits={"public": limit_1s, "private": limit_1s},
        default_limit=limit_1s,
    )
    watchdog.set_expected_channels({"public:BTCUSDT", "private:BTCUSDT"})

    now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    past = now - timedelta(seconds=10)

    with patch("trading.marketdata.staleness.utc_now", return_value=now):
        async with watchdog._lock:
            watchdog._last_seen["public:BTCUSDT"] = past
            watchdog._last_seen["private:BTCUSDT"] = past

        stale = await watchdog.stale_channels(trigger_streams=None)
        assert "public:BTCUSDT" in stale
        assert "private:BTCUSDT" in stale


@pytest.mark.asyncio
async def test_assert_healthy_only_checks_public() -> None:
    """assert_healthy raises only for stale public channels, not idle private."""
    limit_1s = FeedStalenessLimit(max_age=timedelta(seconds=1))
    watchdog = FeedStalenessWatchdog(
        limits={"public": limit_1s, "private": limit_1s},
        default_limit=limit_1s,
    )
    watchdog.set_expected_channels({"public:BTCUSDT", "private:BTCUSDT"})

    now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    past = now - timedelta(seconds=10)

    with patch("trading.marketdata.staleness.utc_now", return_value=now):
        async with watchdog._lock:
            watchdog._last_seen["public:BTCUSDT"] = now  # fresh
            watchdog._last_seen["private:BTCUSDT"] = past  # stale but ignored

        await watchdog.assert_healthy()

    with patch("trading.marketdata.staleness.utc_now", return_value=now):
        async with watchdog._lock:
            watchdog._last_seen["public:BTCUSDT"] = past  # stale

        with pytest.raises(RuntimeError, match="Feed staleness detected"):
            await watchdog.assert_healthy()


def test_build_default_watchdog() -> None:
    """build_default_watchdog returns configured watchdog."""
    w = build_default_watchdog()
    assert isinstance(w, FeedStalenessWatchdog)
    assert w._limits["public"].max_age == timedelta(seconds=15)
    assert w._limits["private"].max_age == timedelta(seconds=30)
