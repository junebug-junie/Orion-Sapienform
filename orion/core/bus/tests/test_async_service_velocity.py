from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.velocity_keys import DEFAULT_BUCKET_TTL_SEC


def _bus_with_mock_redis(*, track_velocity: bool) -> tuple[OrionBusAsync, AsyncMock, MagicMock]:
    bus = OrionBusAsync("redis://unused:6379/0", track_velocity=track_velocity)
    redis = AsyncMock()
    pipe = AsyncMock()
    pipe.incr = MagicMock(return_value=None)
    pipe.expire = MagicMock(return_value=None)
    redis.pipeline = MagicMock(return_value=pipe)
    bus._redis = redis
    return bus, redis, pipe


@pytest.mark.asyncio
async def test_publish_does_not_track_velocity_when_disabled() -> None:
    bus, redis, pipe = _bus_with_mock_redis(track_velocity=False)

    await bus.publish("orion:core:events", {"foo": "bar"})

    redis.publish.assert_awaited_once()
    redis.pipeline.assert_not_called()
    pipe.execute.assert_not_called()


@pytest.mark.asyncio
async def test_publish_tracks_velocity_when_enabled() -> None:
    bus, redis, pipe = _bus_with_mock_redis(track_velocity=True)

    await bus.publish("orion:core:events", {"foo": "bar"})

    redis.publish.assert_awaited_once()
    redis.pipeline.assert_called_once_with(transaction=False)
    pipe.incr.assert_called_once()
    (incr_key,), _ = pipe.incr.call_args
    assert incr_key.startswith("orion:bus:velocity:orion:core:events:")
    pipe.expire.assert_called_once()
    (expire_key, ttl), _ = pipe.expire.call_args
    assert expire_key == incr_key
    assert ttl == DEFAULT_BUCKET_TTL_SEC
    pipe.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_publish_swallows_velocity_tracking_errors() -> None:
    bus, redis, pipe = _bus_with_mock_redis(track_velocity=True)
    pipe.execute = AsyncMock(side_effect=ConnectionError("redis down"))

    # A broken counter must not break a real publish that already succeeded.
    await bus.publish("orion:core:events", {"foo": "bar"})

    redis.publish.assert_awaited_once()


@pytest.mark.asyncio
async def test_publish_skips_velocity_entirely_when_bus_disabled() -> None:
    bus, redis, pipe = _bus_with_mock_redis(track_velocity=True)
    bus.enabled = False

    await bus.publish("orion:core:events", {"foo": "bar"})

    redis.publish.assert_not_awaited()
    redis.pipeline.assert_not_called()


def test_track_velocity_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORION_BUS_VELOCITY_TRACKING_ENABLED", raising=False)
    bus = OrionBusAsync("redis://unused:6379/0")
    assert bus.track_velocity is False


def test_track_velocity_reads_env_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORION_BUS_VELOCITY_TRACKING_ENABLED", "true")
    bus = OrionBusAsync("redis://unused:6379/0")
    assert bus.track_velocity is True


def test_track_velocity_constructor_override_wins_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORION_BUS_VELOCITY_TRACKING_ENABLED", "true")
    bus = OrionBusAsync("redis://unused:6379/0", track_velocity=False)
    assert bus.track_velocity is False
