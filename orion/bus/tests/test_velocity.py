from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from orion.bus.velocity import read_channel_velocity, scan_active_channels
from orion.core.bus.velocity_keys import velocity_window_keys


def _async_iter(items):
    async def gen():
        for item in items:
            yield item

    return gen()


@pytest.mark.asyncio
async def test_read_channel_velocity_sums_buckets_into_rate() -> None:
    now = datetime(2026, 7, 23, 14, 5, 30, tzinfo=timezone.utc)
    redis = AsyncMock()
    # 3 one-minute buckets: 60 + 30 + 30 = 120 messages over 180 seconds.
    redis.mget = AsyncMock(return_value=[b"60", b"30", b"30"])

    rate = await read_channel_velocity(redis, "ch", window_minutes=3, now=now)

    assert rate == pytest.approx(120 / 180)
    redis.mget.assert_awaited_once_with(
        velocity_window_keys("ch", now=now, window_minutes=3)
    )


@pytest.mark.asyncio
async def test_read_channel_velocity_treats_missing_buckets_as_zero() -> None:
    redis = AsyncMock()
    redis.mget = AsyncMock(return_value=[None, b"10", None])

    rate = await read_channel_velocity(redis, "ch", window_minutes=3)

    assert rate == pytest.approx(10 / 180)


@pytest.mark.asyncio
async def test_read_channel_velocity_fails_open_on_redis_error() -> None:
    redis = AsyncMock()
    redis.mget = AsyncMock(side_effect=ConnectionError("down"))

    rate = await read_channel_velocity(redis, "ch", window_minutes=5)

    assert rate == 0.0


@pytest.mark.asyncio
async def test_read_channel_velocity_zero_window_short_circuits() -> None:
    redis = AsyncMock()
    rate = await read_channel_velocity(redis, "ch", window_minutes=0)
    assert rate == 0.0
    redis.mget.assert_not_called()


@pytest.mark.asyncio
async def test_scan_active_channels_discovers_and_aggregates_per_channel() -> None:
    now = datetime(2026, 7, 23, 17, 28, 30, tzinfo=timezone.utc)
    redis = AsyncMock()
    keys = [
        "orion:bus:velocity:orion:system:health:20260723T1727Z",
        "orion:bus:velocity:orion:system:health:20260723T1728Z",
        "orion:bus:velocity:orion:exec:result:LLMGatewayService:abc-123:20260723T1728Z",
    ]
    redis.scan_iter = lambda **kwargs: _async_iter(keys)
    redis.mget = AsyncMock(return_value=[b"1035", b"732", b"2"])

    result = await scan_active_channels(redis, window_minutes=5, now=now)

    assert result["orion:system:health"] == pytest.approx((1035 + 732) / 300.0)
    assert result["orion:exec:result:LLMGatewayService:abc-123"] == pytest.approx(2 / 300.0)


@pytest.mark.asyncio
async def test_scan_active_channels_skips_buckets_outside_window() -> None:
    now = datetime(2026, 7, 23, 17, 28, 30, tzinfo=timezone.utc)
    redis = AsyncMock()
    keys = [
        "orion:bus:velocity:orion:old:channel:20260723T1700Z",  # 28 minutes stale
    ]
    redis.scan_iter = lambda **kwargs: _async_iter(keys)
    redis.mget = AsyncMock(return_value=[])

    result = await scan_active_channels(redis, window_minutes=5, now=now)

    assert result == {}
    redis.mget.assert_not_called()


@pytest.mark.asyncio
async def test_scan_active_channels_ignores_malformed_keys() -> None:
    redis = AsyncMock()
    keys = ["orion:bus:velocity:not-a-bucket-suffix", "some:other:key:entirely"]
    redis.scan_iter = lambda **kwargs: _async_iter(keys)
    redis.mget = AsyncMock(return_value=[])

    result = await scan_active_channels(redis, window_minutes=5)

    assert result == {}
    redis.mget.assert_not_called()


@pytest.mark.asyncio
async def test_scan_active_channels_ignores_empty_channel_name() -> None:
    redis = AsyncMock()
    # A key with nothing between the prefix and the bucket suffix -- would
    # parse to an empty-string channel if _parse_velocity_key's guard didn't
    # reject it explicitly.
    keys = ["orion:bus:velocity::20260723T1728Z"]
    redis.scan_iter = lambda **kwargs: _async_iter(keys)
    redis.mget = AsyncMock(return_value=[])

    result = await scan_active_channels(redis, window_minutes=5)

    assert result == {}
    redis.mget.assert_not_called()


@pytest.mark.asyncio
async def test_scan_active_channels_fails_open_on_redis_error() -> None:
    redis = AsyncMock()

    def _raise(**kwargs):
        raise ConnectionError("down")

    redis.scan_iter = _raise

    result = await scan_active_channels(redis, window_minutes=5)

    assert result == {}


@pytest.mark.asyncio
async def test_scan_active_channels_zero_window_short_circuits() -> None:
    redis = AsyncMock()
    redis.scan_iter = lambda **kwargs: _async_iter([])
    redis.mget = AsyncMock(return_value=[])

    result = await scan_active_channels(redis, window_minutes=0)

    assert result == {}
    redis.mget.assert_not_called()
