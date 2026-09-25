"""bind_situation_state_buses must bind every Redis-backed situation store.

Regression for 2026-09-25: orion-hub builds every unified turn's situation
brief in its own process but bound only the affect store, so the
conversation-phase read was always unbound and every Orion-mode turn said
"Conversation phase: unknown". The gate test below fails if a module in
orion/situational grows a ``bind_*_bus`` the shared helper does not call.
"""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import orion.situational as situational_pkg
import orion.situational.context as situation_mod
from orion.schemas.situation import SituationDiagnosticsV1
from orion.situational.context import (
    _build_conversation_phase,
    _build_time_context,
    mark_orion_turn,
    settings_from_runtime,
)
from orion.situational.state_buses import bind_situation_state_buses

_KNOWN_STORES = {
    "orion.situational.session_turn_phase",
    "orion.situational.juniper_affect_state",
    "orion.situational.identity_ask_cooldown",
}


def _modules_with_bus_bind():
    out = []
    for info in pkgutil.iter_modules(situational_pkg.__path__):
        if info.ispkg or info.name == "state_buses":
            continue
        mod = importlib.import_module(f"orion.situational.{info.name}")
        binds = [
            name
            for name, fn in inspect.getmembers(mod, inspect.isfunction)
            if name.startswith("bind_") and name.endswith("_bus") and fn.__module__ == mod.__name__
        ]
        if binds:
            out.append(mod)
    return out


def test_helper_binds_every_situation_store(monkeypatch):
    modules = _modules_with_bus_bind()
    # Discovery itself must work, or the gate below passes vacuously.
    assert _KNOWN_STORES <= {m.__name__ for m in modules}
    for mod in modules:
        assert hasattr(mod, "_BUS"), f"{mod.__name__} binds a bus but keeps no _BUS handle"
        monkeypatch.setattr(mod, "_BUS", None)

    bus = object()
    bind_situation_state_buses(bus)

    unbound = sorted(m.__name__ for m in modules if m._BUS is not bus)
    assert not unbound, f"bind_situation_state_buses does not bind: {unbound}"


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    async def get(self, key: str):
        return self.store.get(key)

    async def setex(self, key: str, ttl_seconds: int, payload: str):
        self.store[key] = payload.encode("utf-8")


NOW = datetime(2026, 9, 25, 15, 0, 0, tzinfo=timezone.utc)


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz is not None else NOW.replace(tzinfo=None)


def _phase_key(session_id: str) -> str:
    return f"orion:cortex-exec:session_turn_phase:{session_id}"


def _state(redis: _FakeRedis, session_id: str) -> dict:
    raw = redis.store.get(_phase_key(session_id))
    return json.loads(raw) if raw else {}


@pytest.mark.asyncio
async def test_unified_turn_sequence_reads_and_records_the_real_session(monkeypatch):
    """The unified turn's order: the stance step (cortex-exec) marks Orion's
    turn, then Hub builds the situation brief. With the store bound and the
    real session id on both sides, the phase reflects the gap since the user's
    previous message and both timestamps land on that session's key."""
    import orion.situational.session_turn_phase as session_turn_phase

    monkeypatch.setattr(situation_mod, "datetime", _FixedDatetime)
    monkeypatch.setattr(session_turn_phase, "_BUS", None)
    redis = _FakeRedis()
    redis.store[_phase_key("orion_sid_1")] = json.dumps(
        {"last_user_turn_at": (NOW - timedelta(hours=1)).isoformat(), "last_orion_turn_at": None}
    ).encode("utf-8")
    bind_situation_state_buses(SimpleNamespace(redis=redis))
    time_ctx = _build_time_context(settings_from_runtime(SimpleNamespace()), SituationDiagnosticsV1())

    await mark_orion_turn("orion_sid_1")
    phase = await _build_conversation_phase({"session_id": "orion_sid_1"}, time_ctx, NOW)

    assert phase.phase_change == "resumed_thread"
    state = _state(redis, "orion_sid_1")
    assert state["last_user_turn_at"] == NOW.isoformat()
    assert state["last_orion_turn_at"] == NOW.isoformat()
    assert _phase_key("global") not in redis.store


@pytest.mark.asyncio
async def test_unbound_store_reads_unknown_and_records_nothing(monkeypatch):
    """What every unified turn did before Hub bound the store."""
    import orion.situational.session_turn_phase as session_turn_phase

    monkeypatch.setattr(situation_mod, "datetime", _FixedDatetime)
    monkeypatch.setattr(session_turn_phase, "_BUS", None)
    time_ctx = _build_time_context(settings_from_runtime(SimpleNamespace()), SituationDiagnosticsV1())

    phase = await _build_conversation_phase({"session_id": "orion_sid_1"}, time_ctx, NOW)

    assert phase.phase_change == "unknown"
