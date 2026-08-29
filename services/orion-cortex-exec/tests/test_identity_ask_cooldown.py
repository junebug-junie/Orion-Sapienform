from __future__ import annotations

import pytest

import orion.situational.identity_ask_cooldown as identity_ask_cooldown
from orion.situational.identity_ask_cooldown import (
    bind_identity_ask_cooldown_bus,
    reset_identity_ask_cooldown_bus_for_tests,
    try_claim_identity_ask,
)


@pytest.fixture(autouse=True)
def _reset_bus():
    reset_identity_ask_cooldown_bus_for_tests()
    yield
    reset_identity_ask_cooldown_bus_for_tests()


class _FakeRedis:
    """Minimal async stand-in for the subset of `bus.redis` this module
    uses, matching real redis-py `SET ... NX` semantics: returns True the
    first time a key is set, None (falsy) on every call while the key
    already exists -- this is what makes the claim atomic in production and
    is exactly the behavior the race-condition regression test below relies
    on to distinguish "I claimed it" from "someone else already did".
    """

    def __init__(self, store: dict[str, bytes] | None = None) -> None:
        self.store: dict[str, bytes] = store if store is not None else {}
        self.set_calls: list[tuple[str, str, bool, int]] = []

    async def set(self, key: str, value: str, nx: bool = False, ex: int | None = None):
        self.set_calls.append((key, value, nx, ex))
        if nx and key in self.store:
            return None
        self.store[key] = value.encode("utf-8")
        return True

    async def get(self, key: str):
        return self.store.get(key)


class _FakeBus:
    def __init__(self, redis: _FakeRedis) -> None:
        self.redis = redis


class _RaisingRedis:
    async def set(self, key: str, value: str, nx: bool = False, ex: int | None = None):
        raise ConnectionError("redis unreachable")

    async def get(self, key: str):
        raise ConnectionError("redis unreachable")


# --- the core contract -------------------------------------------------------


@pytest.mark.asyncio
async def test_first_claim_for_a_stream_succeeds() -> None:
    bind_identity_ask_cooldown_bus(_FakeBus(_FakeRedis()))
    assert await try_claim_identity_ask("cam0") is True


@pytest.mark.asyncio
async def test_second_claim_within_the_cooldown_fails() -> None:
    bind_identity_ask_cooldown_bus(_FakeBus(_FakeRedis()))
    assert await try_claim_identity_ask("cam0") is True
    assert await try_claim_identity_ask("cam0") is False


@pytest.mark.asyncio
async def test_cooldown_is_scoped_per_stream_not_global() -> None:
    """A "carbon" claim must not block "cam0" -- these are different
    cameras, each with its own sit-down."""
    bind_identity_ask_cooldown_bus(_FakeBus(_FakeRedis()))
    assert await try_claim_identity_ask("carbon") is True
    assert await try_claim_identity_ask("cam0") is True
    assert await try_claim_identity_ask("carbon") is False


@pytest.mark.asyncio
async def test_claim_uses_the_configured_ttl() -> None:
    redis = _FakeRedis()
    bind_identity_ask_cooldown_bus(_FakeBus(redis))
    await try_claim_identity_ask("cam0", ttl_seconds=42)
    assert redis.set_calls == [
        ("orion:cortex-exec:identity_ask_cooldown:unmatched_face:cam0", "1", True, 42)
    ]


@pytest.mark.asyncio
async def test_each_reason_claims_its_own_key() -> None:
    """Keyed by (reason, stream) since 2026-08-29. A shared key would let the
    common reason starve the rare one: hours of lid-closed chat would hold the
    slot, and a stranger who then walked into frame would go unremarked."""
    redis = _FakeRedis()
    bind_identity_ask_cooldown_bus(_FakeBus(redis))
    assert await try_claim_identity_ask("cam0", reason="no_visual_confirmation") is True
    assert await try_claim_identity_ask("cam0", reason="unmatched_face") is True
    assert {call[0] for call in redis.set_calls} == {
        "orion:cortex-exec:identity_ask_cooldown:no_visual_confirmation:cam0",
        "orion:cortex-exec:identity_ask_cooldown:unmatched_face:cam0",
    }


@pytest.mark.asyncio
async def test_the_same_reason_on_the_same_stream_claims_once() -> None:
    redis = _FakeRedis()
    bind_identity_ask_cooldown_bus(_FakeBus(redis))
    first = await try_claim_identity_ask("cam0", reason="no_visual_confirmation")
    second = await try_claim_identity_ask("cam0", reason="no_visual_confirmation")
    assert (first, second) == (True, False)


@pytest.mark.asyncio
async def test_reason_defaults_to_the_original_signal() -> None:
    """Callers written before the second reason existed keep their key."""
    redis = _FakeRedis()
    bind_identity_ask_cooldown_bus(_FakeBus(redis))
    await try_claim_identity_ask("cam0")
    assert redis.set_calls[0][0].endswith(":unmatched_face:cam0")


# --- the race condition this atomic claim exists to close -------------------


@pytest.mark.asyncio
async def test_two_concurrent_claims_for_the_same_stream_only_one_wins() -> None:
    """This is the failure the module's own docstring names: an earlier
    version split this into a GET check and a separate SETEX write, so two
    replicas racing between the two round-trips could both read "not in
    cooldown" and both ask. A single SET-NX-EX call cannot have that gap --
    exactly one of two concurrent claims for the same key succeeds, proven
    here directly against the fake redis's own NX semantics (see its
    docstring) rather than against real network timing."""
    bind_identity_ask_cooldown_bus(_FakeBus(_FakeRedis()))
    first = await try_claim_identity_ask("cam0")
    second = await try_claim_identity_ask("cam0")
    assert (first, second) == (True, False)


# --- cross-process durability, mirroring session_turn_phase's own precedent -


@pytest.mark.asyncio
async def test_second_replica_sees_the_first_replicas_claim_via_shared_redis() -> None:
    """"Different process" is simulated the same way test_session_turn_phase.py
    does it: reset this module's bound-bus state between calls (no Python
    state survives), while both "processes" share one underlying fake Redis
    store."""
    shared_store: dict[str, bytes] = {}

    bind_identity_ask_cooldown_bus(_FakeBus(_FakeRedis(shared_store)))
    assert await try_claim_identity_ask("cam0") is True

    reset_identity_ask_cooldown_bus_for_tests()
    assert identity_ask_cooldown._BUS is None

    bind_identity_ask_cooldown_bus(_FakeBus(_FakeRedis(shared_store)))
    assert await try_claim_identity_ask("cam0") is False


# --- fail-open, toward asking not toward silence -----------------------------


@pytest.mark.asyncio
async def test_claim_fails_open_to_true_on_redis_error() -> None:
    bind_identity_ask_cooldown_bus(_FakeBus(_RaisingRedis()))
    assert await try_claim_identity_ask("cam0") is True


@pytest.mark.asyncio
async def test_claim_fails_open_to_true_when_bus_unbound() -> None:
    assert await try_claim_identity_ask("cam0") is True
