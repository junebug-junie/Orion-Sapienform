from __future__ import annotations

import pytest

import orion.situational.identity_ask_cooldown as identity_ask_cooldown
from orion.situational.identity_ask_cooldown import (
    bind_identity_ask_cooldown_bus,
    identity_ask_in_cooldown,
    mark_identity_ask_offered,
    reset_identity_ask_cooldown_bus_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_bus():
    reset_identity_ask_cooldown_bus_for_tests()
    yield
    reset_identity_ask_cooldown_bus_for_tests()


class _FakeRedis:
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


# --- the core contract -------------------------------------------------------


@pytest.mark.asyncio
async def test_not_in_cooldown_before_any_ask() -> None:
    bind_identity_ask_cooldown_bus(_FakeBus(_FakeRedis()))
    assert await identity_ask_in_cooldown("cam0") is False


@pytest.mark.asyncio
async def test_in_cooldown_immediately_after_marking() -> None:
    bind_identity_ask_cooldown_bus(_FakeBus(_FakeRedis()))
    await mark_identity_ask_offered("cam0")
    assert await identity_ask_in_cooldown("cam0") is True


@pytest.mark.asyncio
async def test_cooldown_is_scoped_per_stream_not_global() -> None:
    """A "carbon" ask must not silence "cam0" -- these are different
    cameras, each with its own sit-down."""
    bind_identity_ask_cooldown_bus(_FakeBus(_FakeRedis()))
    await mark_identity_ask_offered("carbon")
    assert await identity_ask_in_cooldown("cam0") is False
    assert await identity_ask_in_cooldown("carbon") is True


@pytest.mark.asyncio
async def test_write_uses_the_configured_ttl() -> None:
    redis = _FakeRedis()
    bind_identity_ask_cooldown_bus(_FakeBus(redis))
    await mark_identity_ask_offered("cam0", ttl_seconds=42)
    assert redis.setex_calls == [("orion:cortex-exec:identity_ask_cooldown:cam0", 42, "1")]


# --- cross-process durability, mirroring session_turn_phase's own precedent -


@pytest.mark.asyncio
async def test_second_replica_sees_the_first_replicas_mark_via_shared_redis() -> None:
    """This is the failure class the module docstring names: there are FOUR
    independent cortex-exec replicas, so an in-process flag would let a
    different replica ask again on the very next turn. "Different process"
    is simulated the same way test_session_turn_phase.py does it: reset this
    module's bound-bus state between calls (no Python state survives), while
    both "processes" share one underlying fake Redis store."""
    shared_store: dict[str, bytes] = {}

    bind_identity_ask_cooldown_bus(_FakeBus(_FakeRedis(shared_store)))
    await mark_identity_ask_offered("cam0")

    reset_identity_ask_cooldown_bus_for_tests()
    assert identity_ask_cooldown._BUS is None

    bind_identity_ask_cooldown_bus(_FakeBus(_FakeRedis(shared_store)))
    assert await identity_ask_in_cooldown("cam0") is True


# --- fail-open, toward asking not toward silence -----------------------------


@pytest.mark.asyncio
async def test_read_fails_open_to_not_in_cooldown_on_redis_error() -> None:
    bind_identity_ask_cooldown_bus(_FakeBus(_RaisingRedis()))
    assert await identity_ask_in_cooldown("cam0") is False


@pytest.mark.asyncio
async def test_read_fails_open_to_not_in_cooldown_when_bus_unbound() -> None:
    assert await identity_ask_in_cooldown("cam0") is False


@pytest.mark.asyncio
async def test_write_never_raises_on_redis_error() -> None:
    bind_identity_ask_cooldown_bus(_FakeBus(_RaisingRedis()))
    await mark_identity_ask_offered("cam0")  # must not raise


@pytest.mark.asyncio
async def test_write_never_raises_when_bus_unbound() -> None:
    await mark_identity_ask_offered("cam0")  # must not raise
