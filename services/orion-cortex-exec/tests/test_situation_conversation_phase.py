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
