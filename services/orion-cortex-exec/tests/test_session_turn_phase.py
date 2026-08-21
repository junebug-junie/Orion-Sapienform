from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import pytest

import app.session_turn_phase as session_turn_phase
from app.session_turn_phase import (
    SessionTurnState,
    bind_session_turn_phase_bus,
    read_session_turn_state,
    reset_session_turn_phase_bus_for_tests,
    write_session_turn_state,
)


@pytest.fixture(autouse=True)
def _reset_bus():
    reset_session_turn_phase_bus_for_tests()
    yield
    reset_session_turn_phase_bus_for_tests()


class _FakeRedis:
    """Minimal async stand-in for the subset of `bus.redis` this module uses.

    Stores raw bytes, like a real `redis.asyncio` client without
    `decode_responses=True` -- exercises the same bytes-decoding path
    `read_session_turn_state` uses in production.
    """

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


# --- the actual bug this fixes: cross-process durability -------------------


@pytest.mark.asyncio
async def test_second_process_sees_first_processs_write_via_shared_redis() -> None:
    """This is the test that would have failed against the old in-memory-
    dict code (`_SESSION_LAST_USER_TURN`/`_SESSION_LAST_ORION_TURN`): those
    dicts were private per-process state, so a second cortex-exec replica
    (or the same process after a restart) could never see a first
    process's write. Here, "different process" is simulated by resetting
    this module's own bound-bus state (`reset_session_turn_phase_bus_for_tests`)
    between the two calls -- so nothing Python-process-local survives across
    them -- while both "processes" share the same underlying fake Redis
    store, exactly as every real cortex-exec replica shares one real Redis.
    """
    shared_store: dict[str, bytes] = {}

    # "Process A" writes.
    bind_session_turn_phase_bus(_FakeBus(_FakeRedis(shared_store)))
    now = datetime(2026, 8, 21, 3, 0, 0, tzinfo=timezone.utc)
    await write_session_turn_state(
        "sid-cross-process",
        last_user_turn_at=now,
        last_orion_turn_at=None,
    )

    # Simulate a fresh process: no shared Python state at all, only a fresh
    # bus binding into the same underlying Redis store.
    reset_session_turn_phase_bus_for_tests()
    assert session_turn_phase._BUS is None

    bind_session_turn_phase_bus(_FakeBus(_FakeRedis(shared_store)))
    state = await read_session_turn_state("sid-cross-process")

    assert state.last_user_turn_at == now
    assert state.last_orion_turn_at is None


@pytest.mark.asyncio
async def test_restart_survives_because_state_is_not_in_process_memory() -> None:
    """A restart wipes an in-process dict but not a Redis key with a live
    TTL -- this is exactly the 2026-08-21T03:51 incident (all four
    cortex-exec containers restarted simultaneously, wiping every replica's
    private dict at once)."""
    shared_store: dict[str, bytes] = {}
    bind_session_turn_phase_bus(_FakeBus(_FakeRedis(shared_store)))
    now = datetime(2026, 8, 21, 3, 51, 0, tzinfo=timezone.utc)
    await write_session_turn_state("sid-restart", last_user_turn_at=now, last_orion_turn_at=now)

    # "Restart": rebind entirely, but the fake Redis store (standing in for
    # the real, separate Redis process) is untouched.
    reset_session_turn_phase_bus_for_tests()
    bind_session_turn_phase_bus(_FakeBus(_FakeRedis(shared_store)))

    state = await read_session_turn_state("sid-restart")
    assert state.last_user_turn_at == now
    assert state.last_orion_turn_at == now


# --- write behavior ----------------------------------------------------------


@pytest.mark.asyncio
async def test_write_uses_a_single_setex_call_with_the_given_ttl() -> None:
    redis = _FakeRedis()
    bind_session_turn_phase_bus(_FakeBus(redis))
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    await write_session_turn_state("sid-ttl", last_user_turn_at=now, last_orion_turn_at=None, ttl_seconds=1234)

    assert len(redis.setex_calls) == 1
    key, ttl, payload = redis.setex_calls[0]
    assert key == "orion:cortex-exec:session_turn_phase:sid-ttl"
    assert ttl == 1234
    parsed = json.loads(payload)
    assert parsed["last_user_turn_at"] == now.isoformat()
    assert parsed["last_orion_turn_at"] is None


@pytest.mark.asyncio
async def test_write_default_ttl_comfortably_exceeds_the_stale_thread_threshold() -> None:
    redis = _FakeRedis()
    bind_session_turn_phase_bus(_FakeBus(redis))
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    await write_session_turn_state("sid-default-ttl", last_user_turn_at=now, last_orion_turn_at=None)

    _, ttl, _ = redis.setex_calls[0]
    stale_thread_threshold_seconds = 48 * 3600
    assert ttl > stale_thread_threshold_seconds
    assert ttl == 7 * 24 * 3600


# --- fail-open behavior ------------------------------------------------------


@pytest.mark.asyncio
async def test_read_fails_open_when_redis_raises(caplog) -> None:
    bind_session_turn_phase_bus(_FakeBus(_RaisingRedis()))
    with caplog.at_level(logging.WARNING, logger="orion.cortex.session_turn_phase"):
        state = await read_session_turn_state("sid-read-error")
    assert state == SessionTurnState(last_user_turn_at=None, last_orion_turn_at=None)
    assert any("session_turn_phase_read_failed" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_write_fails_open_when_redis_raises_and_does_not_propagate(caplog) -> None:
    bind_session_turn_phase_bus(_FakeBus(_RaisingRedis()))
    with caplog.at_level(logging.WARNING, logger="orion.cortex.session_turn_phase"):
        await write_session_turn_state(
            "sid-write-error",
            last_user_turn_at=datetime.now(timezone.utc),
            last_orion_turn_at=None,
        )  # must not raise
    assert any("session_turn_phase_write_failed" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_read_fails_open_when_bus_unbound_with_distinguishable_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="orion.cortex.session_turn_phase"):
        state = await read_session_turn_state("sid-unbound")
    assert state == SessionTurnState(last_user_turn_at=None, last_orion_turn_at=None)
    assert any("bus_unbound" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_write_fails_open_when_bus_unbound_with_distinguishable_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="orion.cortex.session_turn_phase"):
        await write_session_turn_state(
            "sid-unbound-write", last_user_turn_at=datetime.now(timezone.utc), last_orion_turn_at=None
        )
    assert any("bus_unbound" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_missing_key_is_not_a_failure_and_does_not_warn(caplog) -> None:
    """A key genuinely absent (first-ever turn of a session) is an expected,
    common case -- must be logged distinctly from an actual read failure,
    same convention as last_tool_fetch_cache.py's found=False case."""
    bind_session_turn_phase_bus(_FakeBus(_FakeRedis()))
    with caplog.at_level(logging.INFO, logger="orion.cortex.session_turn_phase"):
        state = await read_session_turn_state("sid-never-seen")
    assert state == SessionTurnState(last_user_turn_at=None, last_orion_turn_at=None)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == []


@pytest.mark.asyncio
async def test_malformed_payload_fails_open(caplog) -> None:
    redis = _FakeRedis({"orion:cortex-exec:session_turn_phase:sid-garbage": b"not json at all"})
    bind_session_turn_phase_bus(_FakeBus(redis))
    with caplog.at_level(logging.WARNING, logger="orion.cortex.session_turn_phase"):
        state = await read_session_turn_state("sid-garbage")
    assert state == SessionTurnState(last_user_turn_at=None, last_orion_turn_at=None)
    assert any("session_turn_phase_read_failed" in r.message for r in caplog.records)


# --- round trip ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_then_read_round_trips_both_timestamps() -> None:
    redis = _FakeRedis()
    bind_session_turn_phase_bus(_FakeBus(redis))
    last_user = datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)
    last_orion = last_user + timedelta(seconds=5)
    await write_session_turn_state("sid-roundtrip", last_user_turn_at=last_user, last_orion_turn_at=last_orion)

    state = await read_session_turn_state("sid-roundtrip")
    assert state.last_user_turn_at == last_user
    assert state.last_orion_turn_at == last_orion
