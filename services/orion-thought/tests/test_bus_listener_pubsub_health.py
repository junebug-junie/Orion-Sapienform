from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app import bus_listener


@pytest.mark.asyncio
async def test_thought_channel_subscribers_returns_count() -> None:
    bus = AsyncMock()
    bus.redis.pubsub_numsub = AsyncMock(return_value=[("orion:thought:request", 1)])
    assert await bus_listener._thought_channel_subscribers(bus, "orion:thought:request") == 1


@pytest.mark.asyncio
async def test_thought_channel_subscribers_returns_zero_when_missing() -> None:
    bus = AsyncMock()
    bus.redis.pubsub_numsub = AsyncMock(return_value=[("orion:thought:request", 0)])
    assert await bus_listener._thought_channel_subscribers(bus, "orion:thought:request") == 0


@pytest.mark.asyncio
async def test_thought_channel_subscribers_probe_failure_is_fail_open() -> None:
    bus = AsyncMock()
    bus.redis.pubsub_numsub = AsyncMock(side_effect=ConnectionError("redis down"))
    assert await bus_listener._thought_channel_subscribers(bus, "orion:thought:request") == -1
