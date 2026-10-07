"""Memory episode boundary Fix 1: the per-turn conversation_phase stamp.

The Hub persists ``spark_meta.conversation_phase`` on every chat turn so
orion-memory-consolidation's boundary rule can read the wall clock (before
this, 0 of 3,586 window turns carried a phase). The stamp must be right on a
situation-cache hit too: the cache (300 s live) would otherwise hand the
morning's ``next_day`` phase to Juniper's second message four minutes later,
and under boundary Rule 3 that reads as a new conversation.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import orion.situational.context as situation_mod
import orion.situational.session_turn_phase as session_turn_phase
from orion.situational.context import (
    classify_conversation_phase,
    conversation_phase_stamp,
    read_conversation_phase_stamp,
)

TZ = "America/Denver"
SID = "orion_sid_stamp"
KEY = f"orion:cortex-exec:session_turn_phase:{SID}"
# 2026-09-28 06:26 MDT == 12:26 UTC, the Austin morning's first turn.
MORNING = datetime(2026, 9, 28, 12, 26, 0, tzinfo=timezone.utc)


class _Clock:
    now = MORNING


class _ClockDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        n = _Clock.now
        return n.astimezone(tz) if tz is not None else n.replace(tzinfo=None)


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.writes = 0

    async def get(self, key: str):
        return self.store.get(key)

    async def setex(self, key: str, ttl_seconds: int, payload: str):
        self.writes += 1
        self.store[key] = payload.encode("utf-8")


@pytest.fixture
def redis(monkeypatch):
    monkeypatch.setattr(situation_mod, "datetime", _ClockDatetime)
    monkeypatch.setattr(situation_mod, "_SITUATION_CACHE", {})
    _Clock.now = MORNING
    fake = _FakeRedis()
    # Last spoke the previous evening (2026-09-27 21:00 MDT).
    fake.store[KEY] = json.dumps(
        {"last_user_turn_at": datetime(2026, 9, 28, 3, 0, tzinfo=timezone.utc).isoformat(), "last_orion_turn_at": None}
    ).encode("utf-8")
    monkeypatch.setattr(session_turn_phase, "_BUS", SimpleNamespace(redis=fake))
    return fake


_ISOLATED_RUNTIME = SimpleNamespace(
    orion_situation_enabled=True,
    orion_situation_timezone=TZ,
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


@pytest.mark.parametrize(
    "gap, expected",
    [
        (timedelta(seconds=30), "same_breath"),
        (timedelta(minutes=5), "short_pause"),
        (timedelta(hours=1, minutes=46), "resumed_thread"),
        (timedelta(hours=5), "long_gap"),
        (timedelta(hours=60), "stale_thread"),
    ],
)
def test_classify_buckets_same_day(gap, expected):
    now = datetime(2026, 9, 28, 23, 0, tzinfo=timezone.utc)  # 17:00 MDT
    out = classify_conversation_phase(now - gap, now, TZ)
    assert out["phase_change"] == expected
    assert out["delta_user_seconds"] == int(gap.total_seconds())


def test_classify_next_day_and_no_history():
    assert classify_conversation_phase(datetime(2026, 9, 28, 3, 0, tzinfo=timezone.utc), MORNING, TZ)[
        "phase_change"
    ] == "next_day"
    blank = classify_conversation_phase(None, MORNING, TZ)
    assert blank["phase_change"] == "unknown" and blank["delta_user_seconds"] is None


def test_stamp_shape_is_the_spec_payload():
    stamp = conversation_phase_stamp(
        {"phase_change": "short_pause", "delta_user_seconds": 300, "crossed_day": False}, source="situation_build"
    )
    assert stamp == {
        "phase_change": "short_pause",
        "delta_user_seconds": 300,
        "crossed_day": False,
        "source": "situation_build",
    }


@pytest.mark.asyncio
async def test_fresh_build_fills_the_stamp(redis):
    out: dict = {}
    await situation_mod.build_situation_for_ctx(
        {"session_id": SID, "record_user_turn": True}, _ISOLATED_RUNTIME, phase_stamp_out=out
    )
    assert out["phase_change"] == "next_day"
    assert out["crossed_day"] is True
    assert out["source"] == "situation_build"


@pytest.mark.asyncio
async def test_cache_hit_four_minutes_later_is_not_next_day(redis):
    """The regression this stamp exists for: a cached brief must not re-stamp
    the first turn's phase onto the second."""
    first: dict = {}
    await situation_mod.build_situation_for_ctx(
        {"session_id": SID, "record_user_turn": True}, _ISOLATED_RUNTIME, phase_stamp_out=first
    )
    assert first["phase_change"] == "next_day"

    _Clock.now = MORNING + timedelta(minutes=4)
    second: dict = {}
    brief, _ = await situation_mod.build_situation_for_ctx(
        {"session_id": SID, "record_user_turn": True}, _ISOLATED_RUNTIME, phase_stamp_out=second
    )
    # It really was a cache hit: the brief still shows the first turn's phase.
    assert brief["conversation_phase"]["phase_change"] == "next_day"
    assert second["phase_change"] == "short_pause"
    assert second["delta_user_seconds"] == 240
    assert second["crossed_day"] is False
    assert second["source"] == "situation_cache"


@pytest.mark.asyncio
async def test_cache_hit_on_orion_authored_turn_measures_from_her_last_turn(redis):
    """A no-user-turn (outreach) cache entry keeps her real last turn."""
    out1: dict = {}
    await situation_mod.build_situation_for_ctx(
        {"session_id": SID, "record_user_turn": False}, _ISOLATED_RUNTIME, phase_stamp_out=out1
    )
    _Clock.now = MORNING + timedelta(minutes=3)
    out2: dict = {}
    await situation_mod.build_situation_for_ctx(
        {"session_id": SID, "record_user_turn": False}, _ISOLATED_RUNTIME, phase_stamp_out=out2
    )
    assert out2["source"] == "situation_cache"
    assert out2["delta_user_seconds"] == out1["delta_user_seconds"] + 180


@pytest.mark.asyncio
async def test_read_only_stamp_never_writes_the_clock(redis):
    stamp = await read_conversation_phase_stamp(SID, tz_name=TZ, now_utc=MORNING)
    assert stamp is not None
    assert stamp["phase_change"] == "next_day"
    assert stamp["source"] == "session_state_read"
    assert redis.writes == 0
