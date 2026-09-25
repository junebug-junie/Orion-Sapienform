"""Only the user's own turns may stamp "the user just spoke".

Once orion-hub bound the conversation-phase store (orion/situational/
state_buses.py), every unified turn's situation build started recording
last_user_turn_at -- including turns Orion authors itself in Juniper's live
session (endogenous outreach). An outreach tick six hours into her absence
then made her reply 30 minutes later read as resumed_thread instead of
long_gap. Those builds now read the phase without recording (the Hub side
sets ``record_user_turn`` from ``utterance_origin``).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import orion.situational.context as situation_mod
import orion.situational.session_turn_phase as session_turn_phase
from orion.schemas.situation import SituationDiagnosticsV1
from orion.situational.context import (
    _build_conversation_phase,
    _build_time_context,
    _situation_cache_key,
    settings_from_runtime,
)

NOW = datetime(2026, 9, 25, 15, 0, 0, tzinfo=timezone.utc)
SID = "orion_sid_1"
KEY = f"orion:cortex-exec:session_turn_phase:{SID}"


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz is not None else NOW.replace(tzinfo=None)


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    async def get(self, key: str):
        return self.store.get(key)

    async def setex(self, key: str, ttl_seconds: int, payload: str):
        self.store[key] = payload.encode("utf-8")


@pytest.fixture
def redis(monkeypatch):
    monkeypatch.setattr(situation_mod, "datetime", _FixedDatetime)
    fake = _FakeRedis()
    fake.store[KEY] = json.dumps(
        {"last_user_turn_at": (NOW - timedelta(hours=6)).isoformat(), "last_orion_turn_at": None}
    ).encode("utf-8")
    monkeypatch.setattr(session_turn_phase, "_BUS", SimpleNamespace(redis=fake))
    return fake


def _time_ctx():
    return _build_time_context(settings_from_runtime(SimpleNamespace()), SituationDiagnosticsV1())


def _last_user(redis: _FakeRedis) -> str:
    return json.loads(redis.store[KEY])["last_user_turn_at"]


@pytest.mark.asyncio
async def test_orion_authored_turn_reads_phase_without_recording(redis):
    before = _last_user(redis)

    phase = await _build_conversation_phase(
        {"session_id": SID, "record_user_turn": False}, _time_ctx(), NOW
    )

    assert phase.phase_change == "long_gap"
    assert _last_user(redis) == before


@pytest.mark.asyncio
async def test_outreach_tick_does_not_mask_the_gap_before_her_reply(redis):
    await _build_conversation_phase({"session_id": SID, "record_user_turn": False}, _time_ctx(), NOW)

    reply_at = NOW + timedelta(minutes=30)
    phase = await _build_conversation_phase(
        {"session_id": SID, "record_user_turn": True}, _time_ctx(), reply_at
    )

    assert phase.phase_change == "long_gap"  # 6.5h since she last spoke, not 30 min
    assert _last_user(redis) == reply_at.isoformat()


@pytest.mark.asyncio
async def test_builds_without_the_flag_still_record(redis):
    """Legacy cortex-exec chat-verb builds never set the flag: every one is a
    user turn, unchanged."""
    await _build_conversation_phase({"session_id": SID}, _time_ctx(), NOW)

    assert _last_user(redis) == NOW.isoformat()


def test_read_only_builds_get_their_own_cache_entry():
    """A cache hit skips _build_conversation_phase entirely, so an outreach
    build sharing her entry would swallow her real reply's record."""
    cfg = settings_from_runtime(SimpleNamespace())
    user_turn = _situation_cache_key({"session_id": SID}, cfg)

    assert _situation_cache_key({"session_id": SID, "record_user_turn": True}, cfg) == user_turn
    assert _situation_cache_key({"session_id": SID, "record_user_turn": False}, cfg) != user_turn


# Every provider that reaches outside the process switched off, so the real
# builder -- cache included -- runs on nothing but the phase store.
_ISOLATED_RUNTIME = SimpleNamespace(
    orion_situation_enabled=True,
    orion_situation_weather_enabled=False,
    orion_situation_agenda_enabled=False,
    orion_situation_lab_context_enabled=False,
    orion_situation_perception_enabled=False,
    orion_situation_affect_enabled=False,
    orion_situation_curiosity_enabled=False,
    orion_situation_reverie_enabled=False,
    orion_situation_cabinet_enabled=False,
    orion_situation_runtime_enabled=False,
)


@pytest.mark.asyncio
async def test_outreach_brief_cached_moments_earlier_does_not_swallow_her_reply(redis, monkeypatch):
    """End to end through build_situation_for_ctx's cache: outreach builds
    (and caches) a brief for her session, then her reply arrives inside the
    same TTL window. Her build must still run the phase and record her."""
    monkeypatch.setattr(situation_mod, "_SITUATION_CACHE", {})
    six_hours_ago = _last_user(redis)

    outreach_brief, _ = await situation_mod.build_situation_for_ctx(
        {"session_id": SID, "record_user_turn": False}, _ISOLATED_RUNTIME
    )
    assert outreach_brief["conversation_phase"]["phase_change"] == "long_gap"
    assert _last_user(redis) == six_hours_ago  # outreach read, did not record

    reply_brief, _ = await situation_mod.build_situation_for_ctx(
        {"session_id": SID, "record_user_turn": True}, _ISOLATED_RUNTIME
    )
    assert reply_brief["conversation_phase"]["phase_change"] == "long_gap"
    assert _last_user(redis) == NOW.isoformat()  # her reply recorded despite the cached brief
