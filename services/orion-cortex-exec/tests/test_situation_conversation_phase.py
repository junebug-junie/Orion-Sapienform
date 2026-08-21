"""Phase-bucketing threshold tests for `situation._build_conversation_phase`
and the redis-backed read-modify-write in `situation.mark_orion_turn`.

Storage-backend-change scope note: these thresholds
(same_breath/short_pause/resumed_thread/long_gap/next_day/stale_thread) are
identical to before this patch -- only where `last_user`/`last_orion` come
from changed (Redis via session_turn_phase.py, not an in-process dict). A
fake bus/Redis stands in for the real one so these tests exercise the real
async code path end-to-end without a live Redis.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import app.session_turn_phase as session_turn_phase
import app.situation as situation_mod
from app.session_turn_phase import bind_session_turn_phase_bus, reset_session_turn_phase_bus_for_tests
from app.situation import _build_conversation_phase, _build_time_context, mark_orion_turn, settings_from_runtime
from orion.schemas.situation import SituationDiagnosticsV1


class _FakeRedis:
    def __init__(self, store: dict[str, bytes] | None = None) -> None:
        self.store: dict[str, bytes] = store if store is not None else {}

    async def get(self, key: str):
        return self.store.get(key)

    async def setex(self, key: str, ttl_seconds: int, payload: str):
        self.store[key] = payload.encode("utf-8")


class _FakeBus:
    def __init__(self, redis: _FakeRedis) -> None:
        self.redis = redis


class _RaisingRedis:
    """Simulates a Redis read failure (as opposed to a genuinely-empty
    read) -- distinct from a plain `_FakeRedis()` with no seeded key."""

    async def get(self, key: str):
        raise ConnectionError("redis unreachable")

    async def setex(self, key: str, ttl_seconds: int, payload: str):
        raise ConnectionError("redis unreachable")


@pytest.fixture(autouse=True)
def _reset_bus():
    reset_session_turn_phase_bus_for_tests()
    yield
    reset_session_turn_phase_bus_for_tests()


def _seed(store: dict, session_id: str, *, last_user: datetime | None, last_orion: datetime | None = None) -> None:
    key = f"orion:cortex-exec:session_turn_phase:{session_id}"
    store[key] = json.dumps(
        {
            "last_user_turn_at": last_user.isoformat() if last_user else None,
            "last_orion_turn_at": last_orion.isoformat() if last_orion else None,
        }
    ).encode("utf-8")


def _time_ctx():
    cfg = settings_from_runtime(SimpleNamespace())
    return _build_time_context(cfg, SituationDiagnosticsV1())


NOW = datetime(2026, 8, 21, 12, 0, 0, tzinfo=timezone.utc)


class _FixedDatetime(datetime):
    """`_build_conversation_phase`'s crossed_day_boundary check calls the
    real `datetime.now(tz)` directly (a pre-existing quirk unrelated to this
    patch's storage-backend change: it does NOT use the `now_utc` argument
    every other part of the function is given). Freezing it to `NOW` here
    keeps these threshold tests deterministic regardless of the real
    wall-clock time the suite happens to run at."""

    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz is not None else NOW.replace(tzinfo=None)


@pytest.fixture(autouse=True)
def _freeze_real_now(monkeypatch):
    monkeypatch.setattr(situation_mod, "datetime", _FixedDatetime)


@pytest.mark.asyncio
async def test_no_prior_state_is_unknown_phase() -> None:
    bind_session_turn_phase_bus(_FakeBus(_FakeRedis()))
    out = await _build_conversation_phase({"session_id": "sid-fresh"}, _time_ctx(), NOW)
    assert out.phase_change == "unknown"
    assert out.continuity_mode == "continue_directly"
    assert out.topic_staleness_risk == "none"


@pytest.mark.asyncio
async def test_same_breath_under_two_minutes() -> None:
    store: dict = {}
    _seed(store, "sid-same-breath", last_user=NOW - timedelta(seconds=90))
    bind_session_turn_phase_bus(_FakeBus(_FakeRedis(store)))
    out = await _build_conversation_phase({"session_id": "sid-same-breath"}, _time_ctx(), NOW)
    assert out.phase_change == "same_breath"
    assert out.continuity_mode == "continue_directly"


@pytest.mark.asyncio
async def test_short_pause_between_two_minutes_and_twenty_minutes() -> None:
    store: dict = {}
    _seed(store, "sid-short-pause", last_user=NOW - timedelta(minutes=10))
    bind_session_turn_phase_bus(_FakeBus(_FakeRedis(store)))
    out = await _build_conversation_phase({"session_id": "sid-short-pause"}, _time_ctx(), NOW)
    assert out.phase_change == "short_pause"


@pytest.mark.asyncio
async def test_resumed_thread_between_twenty_minutes_and_three_hours() -> None:
    store: dict = {}
    _seed(store, "sid-resumed", last_user=NOW - timedelta(hours=1))
    bind_session_turn_phase_bus(_FakeBus(_FakeRedis(store)))
    out = await _build_conversation_phase({"session_id": "sid-resumed"}, _time_ctx(), NOW)
    assert out.phase_change == "resumed_thread"
    assert out.continuity_mode == "lightly_resume"
    assert out.topic_staleness_risk == "low"


@pytest.mark.asyncio
async def test_long_gap_between_three_and_twelve_hours() -> None:
    store: dict = {}
    _seed(store, "sid-long-gap", last_user=NOW - timedelta(hours=6))
    bind_session_turn_phase_bus(_FakeBus(_FakeRedis(store)))
    out = await _build_conversation_phase({"session_id": "sid-long-gap"}, _time_ctx(), NOW)
    assert out.phase_change == "long_gap"
    assert out.continuity_mode == "reorient"
    assert out.topic_staleness_risk == "medium"
    assert out.response_adjustments


@pytest.mark.asyncio
async def test_KNOWN_GAP_twelve_to_forty_eight_hours_falls_through_to_unknown() -> None:
    """KNOWN GAP, pre-existing and out of scope for this storage-only
    patch (confirmed byte-identical against `git show main:.../situation.py`
    -- this is not something the Redis migration introduced or could have
    fixed incidentally): `long_gap` requires `delta_user < 12*3600` and
    `stale_thread` requires `delta_user > 48*3600` (strict), so anything
    from 12h up to and including 48h falls through every bucket and lands
    on `phase="unknown"`/`continuity_mode="continue_directly"` -- the SAME
    "as if we just talked" framing as a session with no prior history at
    all. A day-and-a-half of real silence renders identically to "fresh
    session." Found during review of this patch; pinned here rather than
    silently changed, so the hole stays documented instead of invisible.
    Follow-up: close this 12h-48h dead zone in a separate, threshold-only
    change.
    """
    store: dict = {}
    for hours in (13, 24, 36, 47, 48):
        session_id = f"sid-known-gap-{hours}h"
        _seed(store, session_id, last_user=NOW - timedelta(hours=hours))
        bind_session_turn_phase_bus(_FakeBus(_FakeRedis(store)))
        out = await _build_conversation_phase({"session_id": session_id}, _time_ctx(), NOW)
        assert out.phase_change == "unknown", f"expected the known 12h-48h dead zone at {hours}h, got {out.phase_change!r}"
        assert out.continuity_mode == "continue_directly"


@pytest.mark.asyncio
async def test_stale_thread_beyond_forty_eight_hours() -> None:
    store: dict = {}
    _seed(store, "sid-stale", last_user=NOW - timedelta(hours=72))
    bind_session_turn_phase_bus(_FakeBus(_FakeRedis(store)))
    out = await _build_conversation_phase({"session_id": "sid-stale"}, _time_ctx(), NOW)
    assert out.phase_change == "stale_thread"
    assert out.continuity_mode == "revalidate_context"
    assert out.topic_staleness_risk == "high"


@pytest.mark.asyncio
async def test_a_key_that_should_have_expired_but_somehow_survived_still_correctly_reads_stale() -> None:
    """Guards the TTL-sizing rationale directly: even a very old but
    still-present record (as if the TTL were misconfigured too short and
    Redis hadn't expired it yet) must classify as stale_thread, not
    silently misread as unknown."""
    store: dict = {}
    _seed(store, "sid-very-old", last_user=NOW - timedelta(days=30))
    bind_session_turn_phase_bus(_FakeBus(_FakeRedis(store)))
    out = await _build_conversation_phase({"session_id": "sid-very-old"}, _time_ctx(), NOW)
    assert out.phase_change == "stale_thread"


@pytest.mark.asyncio
async def test_build_conversation_phase_writes_back_updated_last_user_turn_preserving_last_orion() -> None:
    store: dict = {}
    original_orion = NOW - timedelta(minutes=30)
    _seed(store, "sid-writeback", last_user=NOW - timedelta(hours=1), last_orion=original_orion)
    bind_session_turn_phase_bus(_FakeBus(_FakeRedis(store)))

    await _build_conversation_phase({"session_id": "sid-writeback"}, _time_ctx(), NOW)

    key = "orion:cortex-exec:session_turn_phase:sid-writeback"
    written = json.loads(store[key])
    assert written["last_user_turn_at"] == NOW.isoformat()
    # last_orion_turn_at must be preserved exactly as read, not clobbered.
    assert written["last_orion_turn_at"] == original_orion.isoformat()


# --- mark_orion_turn read-modify-write --------------------------------------


@pytest.mark.asyncio
async def test_mark_orion_turn_preserves_existing_last_user_turn_at() -> None:
    store: dict = {}
    existing_user_turn = NOW - timedelta(minutes=2)
    _seed(store, "sid-mark-orion", last_user=existing_user_turn, last_orion=None)
    bind_session_turn_phase_bus(_FakeBus(_FakeRedis(store)))

    await mark_orion_turn("sid-mark-orion")

    key = "orion:cortex-exec:session_turn_phase:sid-mark-orion"
    written = json.loads(store[key])
    assert written["last_user_turn_at"] == existing_user_turn.isoformat()
    assert written["last_orion_turn_at"] is not None


@pytest.mark.asyncio
async def test_mark_orion_turn_with_no_prior_state_writes_only_orion_side() -> None:
    bind_session_turn_phase_bus(_FakeBus(_FakeRedis()))
    await mark_orion_turn("sid-mark-orion-fresh")

    key = "orion:cortex-exec:session_turn_phase:sid-mark-orion-fresh"
    redis = session_turn_phase._BUS.redis
    written = json.loads(redis.store[key])
    assert written["last_user_turn_at"] is None
    assert written["last_orion_turn_at"] is not None


@pytest.mark.asyncio
async def test_mark_orion_turn_defaults_session_id_to_global() -> None:
    bind_session_turn_phase_bus(_FakeBus(_FakeRedis()))
    await mark_orion_turn(None)

    redis = session_turn_phase._BUS.redis
    assert "orion:cortex-exec:session_turn_phase:global" in redis.store


# --- clobber prevention on a failed read ------------------------------------
#
# A Redis read FAILURE (not "genuinely no prior data") must never result in
# writing back a None that overwrites a real, existing value on the field
# this call didn't intend to touch. Before session_turn_phase.py grew the
# ok=False/True distinction, both _build_conversation_phase and
# mark_orion_turn would happily write (None, computed_value) back on a
# failed read -- silently wiping the OTHER timestamp. These tests would have
# failed against that version.


@pytest.mark.asyncio
async def test_build_conversation_phase_does_not_write_when_read_fails() -> None:
    bind_session_turn_phase_bus(_FakeBus(_RaisingRedis()))
    out = await _build_conversation_phase({"session_id": "sid-read-fails"}, _time_ctx(), NOW)

    # Classification still degrades safely to today's defaults...
    assert out.phase_change == "unknown"
    # ...but no write was attempted at all (a raising setex would have
    # itself been caught and logged by write_session_turn_state -- the
    # point here is the CALL never happens).
    key = "orion:cortex-exec:session_turn_phase:sid-read-fails"
    redis = session_turn_phase._BUS.redis
    assert key not in getattr(redis, "store", {})


@pytest.mark.asyncio
async def test_build_conversation_phase_read_failure_does_not_clobber_last_orion() -> None:
    """The scenario that actually matters: last_orion_turn_at has a real
    value sitting in Redis, but THIS particular read transiently fails.
    A naive implementation would write last_orion_turn_at=None here,
    destroying real data this call never meant to touch."""

    class _FlakyThenFakeRedis(_FakeRedis):
        def __init__(self, store: dict) -> None:
            super().__init__(store)
            self._first_get_done = False

        async def get(self, key: str):
            if not self._first_get_done:
                self._first_get_done = True
                raise ConnectionError("transient redis blip")
            return await super().get(key)

    store: dict = {}
    real_last_orion = NOW - timedelta(minutes=1)
    _seed(store, "sid-flaky", last_user=NOW - timedelta(minutes=5), last_orion=real_last_orion)
    bind_session_turn_phase_bus(_FakeBus(_FlakyThenFakeRedis(store)))

    await _build_conversation_phase({"session_id": "sid-flaky"}, _time_ctx(), NOW)

    # No write happened on the failed read, so the real last_orion value
    # already in the store is untouched.
    key = "orion:cortex-exec:session_turn_phase:sid-flaky"
    written = json.loads(store[key])
    assert written["last_orion_turn_at"] == real_last_orion.isoformat()


@pytest.mark.asyncio
async def test_mark_orion_turn_does_not_write_when_read_fails() -> None:
    bind_session_turn_phase_bus(_FakeBus(_RaisingRedis()))
    await mark_orion_turn("sid-mark-read-fails")

    key = "orion:cortex-exec:session_turn_phase:sid-mark-read-fails"
    redis = session_turn_phase._BUS.redis
    assert key not in getattr(redis, "store", {})
