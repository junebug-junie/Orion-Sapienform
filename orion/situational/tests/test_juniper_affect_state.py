from __future__ import annotations

from datetime import datetime, timezone

import pytest

import orion.situational.juniper_affect_state as juniper_affect_state
from orion.situational.juniper_affect_state import (
    _KEY,
    bind_juniper_affect_state_bus,
    read_latest_juniper_affect,
    reset_juniper_affect_state_bus_for_tests,
    write_latest_juniper_affect,
)


@pytest.fixture(autouse=True)
def _reset_bus():
    reset_juniper_affect_state_bus_for_tests()
    yield
    reset_juniper_affect_state_bus_for_tests()


class _FakeRedis:
    """Mirrors test_session_turn_phase.py's _FakeRedis -- stores raw bytes,
    like a real redis.asyncio client without decode_responses=True."""

    def __init__(self, store: dict[str, bytes] | None = None) -> None:
        self.store: dict[str, bytes] = store if store is not None else {}
        self.setex_calls: list[tuple[str, int, str]] = []

    async def get(self, key: str):
        return self.store.get(key)

    async def setex(self, key: str, ttl_seconds: int, payload: str):
        self.setex_calls.append((key, ttl_seconds, payload))
        self.store[key] = payload.encode("utf-8")


class _FakeBus:
    def __init__(self, redis: _FakeRedis) -> None:
        self.redis = redis


class _RaisingRedis:
    async def get(self, key: str):
        raise ConnectionError("redis unreachable")

    async def setex(self, key: str, ttl_seconds: int, payload: str):
        raise ConnectionError("redis unreachable")


# --- read side: unbound / absent / malformed -------------------------------


@pytest.mark.asyncio
async def test_read_with_unbound_bus_returns_unknown_not_raise() -> None:
    state = await read_latest_juniper_affect()
    assert state.ok is False
    assert state.summary is None


@pytest.mark.asyncio
async def test_read_with_no_key_returns_confirmed_empty() -> None:
    bind_juniper_affect_state_bus(_FakeBus(_FakeRedis()))
    state = await read_latest_juniper_affect()
    assert state.ok is True
    assert state.summary is None


@pytest.mark.asyncio
async def test_read_malformed_json_returns_confirmed_empty_not_unknown() -> None:
    redis = _FakeRedis({_KEY: b"not-json{{{"})
    bind_juniper_affect_state_bus(_FakeBus(redis))
    state = await read_latest_juniper_affect()
    assert state.ok is True
    assert state.summary is None


@pytest.mark.asyncio
async def test_read_redis_error_returns_unknown() -> None:
    bind_juniper_affect_state_bus(_FakeBus(_RaisingRedis()))
    state = await read_latest_juniper_affect()
    assert state.ok is False


# --- write/read round trip --------------------------------------------------


@pytest.mark.asyncio
async def test_write_then_read_round_trips_all_fields() -> None:
    redis = _FakeRedis()
    bus = _FakeBus(redis)
    observed_at = datetime(2026, 8, 25, 3, 0, 0, tzinfo=timezone.utc)

    await write_latest_juniper_affect(
        bus,
        summary="Juniper appears focused, leaning toward the screen.",
        observed_at=observed_at,
        trigger="manual",
        subtitle_source="transcribed",
    )

    bind_juniper_affect_state_bus(bus)
    state = await read_latest_juniper_affect()

    assert state.ok is True
    assert state.summary == "Juniper appears focused, leaning toward the screen."
    assert state.observed_at == observed_at
    assert state.trigger == "manual"
    assert state.subtitle_source == "transcribed"


@pytest.mark.asyncio
async def test_write_always_sets_a_ttl() -> None:
    """SETEX, not SET -- a crashed/misconfigured producer must not leave a
    stale read alive forever. Same reasoning as write_session_turn_state's
    own docstring for why SETEX over SET+EXPIRE."""
    redis = _FakeRedis()
    await write_latest_juniper_affect(
        _FakeBus(redis),
        summary="calm",
        observed_at=datetime.now(timezone.utc),
        trigger="ambient",
        subtitle_source="none",
    )
    assert len(redis.setex_calls) == 1
    key, ttl_seconds, _payload = redis.setex_calls[0]
    assert key == _KEY
    assert ttl_seconds == juniper_affect_state._WRITE_TTL_SECONDS
    assert ttl_seconds > 0


@pytest.mark.asyncio
async def test_write_failure_is_fail_open_never_raises() -> None:
    # Must not raise -- a failed mirror-write must never break the caller's
    # own real bus-publish path.
    await write_latest_juniper_affect(
        _FakeBus(_RaisingRedis()),
        summary="calm",
        observed_at=datetime.now(timezone.utc),
        trigger="manual",
        subtitle_source="none",
    )


# --- cross-process durability (mirrors session_turn_phase's own test) ------


@pytest.mark.asyncio
async def test_second_process_sees_first_processs_write_via_shared_redis() -> None:
    shared_store: dict[str, bytes] = {}
    now = datetime(2026, 8, 25, 4, 0, 0, tzinfo=timezone.utc)

    # "Process A" (orion-juniper-affective-state) writes.
    await write_latest_juniper_affect(
        _FakeBus(_FakeRedis(shared_store)),
        summary="Juniper seems relaxed.",
        observed_at=now,
        trigger="ambient",
        subtitle_source="none",
    )

    # "Process B" (orion-hub or orion-cortex-exec building a situation
    # brief) -- fresh bind, same underlying store, exactly as every real
    # process shares one real Redis.
    reset_juniper_affect_state_bus_for_tests()
    assert juniper_affect_state._BUS is None
    bind_juniper_affect_state_bus(_FakeBus(_FakeRedis(shared_store)))

    state = await read_latest_juniper_affect()
    assert state.ok is True
    assert state.summary == "Juniper seems relaxed."
    assert state.observed_at == now
