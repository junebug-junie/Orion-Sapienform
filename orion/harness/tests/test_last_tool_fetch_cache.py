from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from orion.harness.last_tool_fetch_cache import publish_last_tool_fetch, read_last_tool_fetch


def _bus_with_redis() -> MagicMock:
    bus = MagicMock()
    bus.redis = AsyncMock()
    return bus


@pytest.mark.asyncio
async def test_noop_when_session_id_missing() -> None:
    bus = _bus_with_redis()

    await publish_last_tool_fetch(
        bus,
        session_id=None,
        correlation_id="c-1",
        tool_names=["mcp__github__get_file_contents"],
    )

    bus.redis.setex.assert_not_called()


@pytest.mark.asyncio
async def test_noop_when_tool_names_empty() -> None:
    bus = _bus_with_redis()

    await publish_last_tool_fetch(
        bus,
        session_id="sess-1",
        correlation_id="c-1",
        tool_names=[],
    )

    bus.redis.setex.assert_not_called()


@pytest.mark.asyncio
async def test_writes_setex_with_exact_key_and_payload_shape() -> None:
    bus = _bus_with_redis()

    await publish_last_tool_fetch(
        bus,
        session_id="sess-42",
        correlation_id="c-99",
        tool_names=["mcp__github__get_file_contents", "WebFetch"],
    )

    bus.redis.setex.assert_awaited_once()
    args, kwargs = bus.redis.setex.await_args
    key, ttl, payload_raw = args
    assert key == "orion:harness:last_tool_fetch:sess-42"
    assert ttl == 600
    payload = json.loads(payload_raw)
    assert set(payload.keys()) == {"tool_names", "correlation_id", "at"}
    assert payload["tool_names"] == ["mcp__github__get_file_contents", "WebFetch"]
    assert payload["correlation_id"] == "c-99"
    assert isinstance(payload["at"], str) and payload["at"]


@pytest.mark.asyncio
async def test_custom_ttl_is_passed_through() -> None:
    bus = _bus_with_redis()

    await publish_last_tool_fetch(
        bus,
        session_id="sess-7",
        correlation_id="c-7",
        tool_names=["WebFetch"],
        ttl_seconds=30,
    )

    args, _ = bus.redis.setex.await_args
    assert args[1] == 30


@pytest.mark.asyncio
async def test_never_raises_when_redis_setex_fails(caplog: pytest.LogCaptureFixture) -> None:
    bus = _bus_with_redis()
    bus.redis.setex.side_effect = RuntimeError("redis is down")

    with caplog.at_level(logging.WARNING, logger="orion.harness.last_tool_fetch_cache"):
        await publish_last_tool_fetch(
            bus,
            session_id="sess-1",
            correlation_id="c-1",
            tool_names=["WebFetch"],
        )

    assert any("last_tool_fetch_write_failed" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_read_returns_none_when_session_id_missing() -> None:
    bus = _bus_with_redis()
    assert await read_last_tool_fetch(bus, session_id=None) is None
    bus.redis.get.assert_not_called()


@pytest.mark.asyncio
async def test_read_returns_none_when_key_absent() -> None:
    bus = _bus_with_redis()
    bus.redis.get.return_value = None
    assert await read_last_tool_fetch(bus, session_id="sess-1") is None


@pytest.mark.asyncio
async def test_read_round_trips_what_write_wrote() -> None:
    """Same key format both directions: write with publish_last_tool_fetch,
    read back with read_last_tool_fetch, using one shared fake Redis."""
    store: dict[str, str] = {}

    async def _setex(key, ttl, value):
        store[key] = value

    async def _get(key):
        return store.get(key)

    bus = MagicMock()
    bus.redis = MagicMock()
    bus.redis.setex = AsyncMock(side_effect=_setex)
    bus.redis.get = AsyncMock(side_effect=_get)

    await publish_last_tool_fetch(
        bus,
        session_id="sess-round-trip",
        correlation_id="c-1",
        tool_names=["mcp__github__get_file_contents"],
    )
    result = await read_last_tool_fetch(bus, session_id="sess-round-trip")

    assert result is not None
    assert result["tool_names"] == ["mcp__github__get_file_contents"]
    assert result["correlation_id"] == "c-1"


@pytest.mark.asyncio
async def test_read_returns_none_on_malformed_json() -> None:
    bus = _bus_with_redis()
    bus.redis.get.return_value = "not json"
    assert await read_last_tool_fetch(bus, session_id="sess-1") is None


@pytest.mark.asyncio
async def test_read_returns_none_when_missing_expected_keys() -> None:
    bus = _bus_with_redis()
    bus.redis.get.return_value = json.dumps({"tool_names": ["WebFetch"]})  # missing correlation_id/at
    assert await read_last_tool_fetch(bus, session_id="sess-1") is None


@pytest.mark.asyncio
async def test_read_returns_none_when_tool_names_not_a_list() -> None:
    bus = _bus_with_redis()
    bus.redis.get.return_value = json.dumps({"tool_names": "WebFetch", "correlation_id": "c-1", "at": "now"})
    assert await read_last_tool_fetch(bus, session_id="sess-1") is None


@pytest.mark.asyncio
async def test_read_returns_none_when_tool_names_has_non_string_element() -> None:
    """Regression: a non-string element would reach prefix.py's
    ", ".join(prior_tool_fetch_names) uncaught -- validation belongs here."""
    bus = _bus_with_redis()
    bus.redis.get.return_value = json.dumps(
        {"tool_names": ["WebFetch", 42], "correlation_id": "c-1", "at": "now"}
    )
    assert await read_last_tool_fetch(bus, session_id="sess-1") is None


@pytest.mark.asyncio
async def test_read_never_raises_on_redis_error(caplog: pytest.LogCaptureFixture) -> None:
    bus = _bus_with_redis()
    bus.redis.get.side_effect = RuntimeError("redis is down")

    with caplog.at_level(logging.WARNING, logger="orion.harness.last_tool_fetch_cache"):
        result = await read_last_tool_fetch(bus, session_id="sess-1")

    assert result is None
    assert any("redis_error" in record.message for record in caplog.records)
