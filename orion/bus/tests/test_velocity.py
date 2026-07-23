from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from orion.bus.velocity import read_channel_velocity
from orion.core.bus.velocity_keys import velocity_window_keys


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
