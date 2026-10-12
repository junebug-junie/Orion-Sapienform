"""Hub reads Orion's arousal (Temporal Self rev 4, R3 readers 2, 3 and 7).

For each reader: each level's effect, flag off = today, stale/missing/error = today, and the
must-not cases (a level never loosens a gate, and never blocks a forced or urgent run).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from orion.schemas.regulation import (
    REGULATION_STATE_REDIS_KEY,
    ArousalReadingV1,
    RegulationStateV1,
)
from scripts.arousal_reader import ArousalReader
from scripts.curiosity_investigation import (
    LINE_INVESTIGATE,
    SchedulingGateInputs,
    _line_keys,
    scheduling_block_reason,
)
from test_curiosity_investigation import _FakeBus, _loop

LEVELS = ("engaged", "idle", "strained")


def _now():
    return datetime.now(timezone.utc)


def _state(level, *, age_sec=30.0):
    """Stamped at the moment of use (a module-import stamp would go stale in a long session)."""
    at = _now() - timedelta(seconds=age_sec)
    reading = ArousalReadingV1(arousal_level=level, since=at - timedelta(minutes=4), observed_at=at,
                               reasons=[f"test_{level}"])
    return RegulationStateV1(generated_at=at, arousal=reading).model_dump_json()


def _unknown(name):
    return {
        "absent": lambda: None,
        "unparseable": lambda: "{nope",
        "stale": lambda: _state("strained", age_sec=361),
        "future_stamped": lambda: _state("strained", age_sec=-400),
        "producer_unknown": lambda: _state("unknown"),
    }[name]()


UNKNOWN = ("absent", "unparseable", "stale", "future_stamped", "producer_unknown")


def _reader(enabled=True):
    return ArousalReader(enabled=enabled, max_age_sec=360.0, name="test")


def _bus_with(raw):
    bus = _FakeBus()
    if raw is not None:
        bus.redis.values[REGULATION_STATE_REDIS_KEY] = raw
    return bus


# --- the shared reader ------------------------------------------------------------


@pytest.mark.parametrize("level", LEVELS)
def test_reader_passes_fresh_levels_through(level):
    r = _reader()
    now = _now()
    asyncio.run(r.refresh(_bus_with(_state(level)), now))
    assert r.level(now) == level


@pytest.mark.parametrize("name", UNKNOWN)
def test_reader_unknown_is_none(name):
    r = _reader()
    now = _now()
    asyncio.run(r.refresh(_bus_with(_unknown(name)), now))
    assert r.level(now) is None


def test_reader_decays_to_unknown_when_not_refreshed():
    r = _reader()
    now = _now()
    asyncio.run(r.refresh(_bus_with(_state("strained")), now))
    assert r.level(now + timedelta(seconds=400)) is None


class _GetRaises:
    class redis:  # noqa: N801
        @staticmethod
        async def get(key):
            raise ConnectionError("down")


class _NotConnected:
    @property
    def redis(self):
        raise RuntimeError("bus not connected")


class _Hangs:
    class redis:  # noqa: N801
        @staticmethod
        async def get(key):
            await asyncio.sleep(3600)


@pytest.mark.parametrize("bus", [_GetRaises(), _NotConnected(), object()])
def test_redis_failure_is_unknown_never_an_exception(bus):
    r = _reader()
    assert asyncio.run(r.refresh(bus, _now())).level == "unknown"
    assert r.level() is None


def test_a_hung_redis_does_not_stall_the_tick(monkeypatch):
    import scripts.arousal_reader as mod

    monkeypatch.setattr(mod, "READ_TIMEOUT_SEC", 0.05)
    assert asyncio.run(_reader().refresh(_Hangs(), _now())).level == "unknown"


def test_flag_off_never_reads():
    off = _reader(enabled=False)
    bus = _bus_with(_state("strained"))
    assert asyncio.run(off.refresh(bus, _now())).level == "unknown"
    assert off.level() is None


# --- reader 2: curiosity pacing (investigation line) ----------------------------------


def _sched(**over):
    base = dict(enabled=True, seconds_since_last=3000.0, min_cooldown_sec=1800.0, done_today=0, daily_cap=7)
    base.update(over)
    return SchedulingGateInputs(**base)


def test_curiosity_gate_level_effects():
    assert scheduling_block_reason(_sched()) is None
    assert scheduling_block_reason(_sched(arousal_level="idle")) is None
    assert scheduling_block_reason(_sched(arousal_level="strained")) == "arousal_strained"
    # Never run before: strained still holds (it is not a spacing rule).
    assert scheduling_block_reason(_sched(arousal_level="strained", seconds_since_last=None)) == "arousal_strained"
    # Engaged: the stretched cooldown, and only the stretch.
    assert scheduling_block_reason(
        _sched(arousal_level="engaged", arousal_engaged_cooldown_sec=3600.0)) == "arousal_engaged_cooldown"
    assert scheduling_block_reason(
        _sched(arousal_level="engaged", arousal_engaged_cooldown_sec=3600.0, seconds_since_last=4000.0)) is None
    assert scheduling_block_reason(
        _sched(arousal_level="engaged", arousal_engaged_cooldown_sec=3600.0, seconds_since_last=None)) is None
    # Plain cooldown still names itself first.
    assert scheduling_block_reason(
        _sched(arousal_level="strained", seconds_since_last=60.0)) == "cooldown"


def _curiosity_tick(raw, *, enabled=True, since_last_sec=20000.0, force=False, multiplier=2.0):
    bus = _bus_with(raw)
    cooldown_key, _, _ = _line_keys(LINE_INVESTIGATE)
    bus.redis.values[cooldown_key] = (_now() - timedelta(seconds=since_last_sec)).isoformat()
    loop = _loop(
        bus, arousal_reader=_reader(enabled=enabled), arousal_pacing_enabled=enabled,
        arousal_engaged_cooldown_multiplier=multiplier,
    )
    return asyncio.run(loop.tick(force=force))


def test_strained_holds_the_investigation_line():
    # _loop's cooldown is 14400 s; the last run was 20000 s ago: due without arousal.
    assert _curiosity_tick(_state("strained")) == "arousal_strained"


def test_engaged_waits_out_the_doubled_cooldown():
    assert _curiosity_tick(_state("engaged")) == "arousal_engaged_cooldown"
    assert _curiosity_tick(_state("engaged"), since_last_sec=30000.0) is None
    # A multiplier of 1 changes nothing.
    assert _curiosity_tick(_state("engaged"), multiplier=1.0) is None


def test_idle_runs_exactly_as_today():
    assert _curiosity_tick(_state("idle")) is None


@pytest.mark.parametrize("name", UNKNOWN)
def test_unknown_runs_exactly_as_today(name):
    assert _curiosity_tick(_unknown(name)) is None


@pytest.mark.parametrize("level", LEVELS)
def test_flag_off_runs_exactly_as_today(level):
    assert _curiosity_tick(_state(level), enabled=False) is None


@pytest.mark.parametrize("level", ["engaged", "strained"])
def test_a_forced_investigation_overrides_arousal(level):
    """Must-not: an operator's / Juniper's forced run is never held by arousal."""
    assert _curiosity_tick(_state(level), force=True) is None


# --- reader 3: curiosity self-inquiry (runs only at idle) ---------------------------------


def _si_gate(level, *, enabled=True, **over):
    return scheduling_block_reason(_sched(
        arousal_level=level if enabled else None, arousal_idle_only=enabled, **over))


def test_self_inquiry_gate_runs_only_at_idle():
    assert _si_gate("idle") is None
    assert _si_gate("engaged") == "arousal_engaged"
    assert _si_gate("strained") == "arousal_strained"
    assert _si_gate(None) is None  # unknown = today
    for level in LEVELS:
        assert _si_gate(level, enabled=False) is None  # flag off = today


def _self_inquiry_tick(raw, *, enabled=True, force=False):
    from test_curiosity_self_inquiry import _DefinitionReader, _GrantConn, _self_loop

    bus = _bus_with(raw)
    loop = _self_loop(
        bus, reader=_DefinitionReader(), conn=_GrantConn(), kickoff_via_cortex=False,
        arousal_reader=_reader(enabled=enabled), arousal_self_inquiry_enabled=enabled,
    )
    now = _now()
    # tick() refreshes once and hands `now` to the self line; this calls the line directly.
    asyncio.run(loop.arousal_reader.refresh(bus, now))
    return asyncio.run(loop.tick_self_inquiry(force=force, now=now))


def test_self_inquiry_runs_at_idle():
    assert _self_inquiry_tick(_state("idle")) is None


@pytest.mark.parametrize("level", ["engaged", "strained"])
def test_self_inquiry_holds_when_not_idle(level):
    assert _self_inquiry_tick(_state(level)) == f"arousal_{level}"


@pytest.mark.parametrize("name", UNKNOWN)
def test_self_inquiry_unknown_runs_as_today(name):
    assert _self_inquiry_tick(_unknown(name)) is None


@pytest.mark.parametrize("level", LEVELS)
def test_self_inquiry_flag_off_runs_as_today(level):
    assert _self_inquiry_tick(_state(level), enabled=False) is None


@pytest.mark.parametrize("level", ["engaged", "strained"])
def test_a_forced_self_inquiry_overrides_arousal(level):
    assert _self_inquiry_tick(_state(level), force=True) is None


def test_self_inquiry_flag_does_not_turn_on_investigation_pacing():
    """Two flags, one shared reader: self-inquiry on must not hold the investigation line."""
    bus = _bus_with(_state("strained"))
    cooldown_key, _, _ = _line_keys(LINE_INVESTIGATE)
    bus.redis.values[cooldown_key] = (_now() - timedelta(seconds=20000)).isoformat()
    loop = _loop(bus, arousal_reader=_reader(), arousal_pacing_enabled=False, arousal_self_inquiry_enabled=True)
    assert asyncio.run(loop.tick()) is None
