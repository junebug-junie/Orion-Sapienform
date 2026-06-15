from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from orion.bus.consumer_readiness import (
    BusConsumerReadinessResult,
    bus_consumer_readiness_v1,
    check_bus_consumer_readiness,
    redis_pubsub_numsub,
)


@pytest.mark.asyncio
async def test_redis_pubsub_numsub_parses_pairs() -> None:
    redis = AsyncMock()
    channel = "orion:exec:request:RecallService"
    redis.execute_command = AsyncMock(
        return_value=[channel.encode("utf-8"), 2, b"other:channel", 0],
    )

    out = await redis_pubsub_numsub(redis, [channel, "other:channel"])

    assert out == {channel: 2, "other:channel": 0}
    redis.execute_command.assert_awaited_once_with("PUBSUB", "NUMSUB", channel, "other:channel")


@pytest.mark.asyncio
async def test_redis_pubsub_numsub_empty_channels() -> None:
    redis = AsyncMock()
    assert await redis_pubsub_numsub(redis, []) == {}
    redis.execute_command.assert_not_called()


@pytest.mark.asyncio
async def test_check_bus_consumer_readiness_no_subscribers() -> None:
    intake = "orion:exec:request:RecallService"
    redis = AsyncMock()
    redis.ping = AsyncMock(return_value=True)
    redis.execute_command = AsyncMock(return_value=[intake, 0])

    result = await check_bus_consumer_readiness(
        redis,
        intake_channel=intake,
        service_name="recall",
        check_heartbeat=False,
    )

    assert result.ok is False
    assert result.bus_consumer_ready is False
    assert result.subscriber_count == 0
    assert result.intake_channel == intake
    assert result.dependency_status == "unavailable"
    assert result.error == f"no subscribers on intake channel: {intake}"


def test_bus_consumer_readiness_v1_sets_http_alive_without_duplicate_kwarg() -> None:
    result = BusConsumerReadinessResult(
        ok=True,
        bus_consumer_ready=True,
        intake_channel="orion:exec:request:LLMGatewayService",
        subscriber_count=1,
        dependency_status="available",
        http_alive=None,
    )
    body = bus_consumer_readiness_v1(result, http_alive=True)
    assert body.http_alive is True
    assert body.ok is True
    assert body.subscriber_count == 1
