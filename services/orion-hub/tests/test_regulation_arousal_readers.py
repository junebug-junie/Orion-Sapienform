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


# --- reader 7: outreach cooldown (strained refuses) ---------------------------------------


def _ogate(**over):
    from scripts.endogenous_outreach import OutreachGateInputs

    base = dict(enabled=True, turn_in_flight=False, local_hour=14, quiet_start_hour=23, quiet_end_hour=8,
                seconds_since_last_outreach=4000.0, min_cooldown_sec=2700.0, sent_today=0, daily_cap=4)
    base.update(over)
    return OutreachGateInputs(**base)


def test_outreach_gate_level_effects():
    from scripts.endogenous_outreach import outreach_block_reason

    assert outreach_block_reason(_ogate(arousal_level="strained")) == "arousal_strained"
    for level in ("engaged", "idle", None):
        assert outreach_block_reason(_ogate(arousal_level=level)) is None, level
    # Door-A (a finished curiosity run with something to say) skips it like the cooldown.
    assert outreach_block_reason(_ogate(arousal_level="strained"), skip_schedule_gates=True) is None


def _outreach_with(raw, *, enabled=True):
    import time

    from test_endogenous_outreach import _outreach

    o = _outreach(min_cooldown_sec=2700.0, arousal_reader=_reader(enabled=enabled))
    o._bus = _bus_with(raw)
    o._last_outreach_at = time.time() - 4000.0
    return o


def test_strained_outreach_tick_refuses_and_status_reports_it():
    o = _outreach_with(_state("strained"))
    result = asyncio.run(o._outreach_once(force=False))
    assert result.get("reason") == "arousal_strained"
    status = o.status()
    assert status["arousal"]["level"] == "strained" and status["block_reason"] == "arousal_strained"


@pytest.mark.parametrize("raw_name", ["engaged", "idle", *UNKNOWN])
def test_outreach_unchanged_unless_strained(raw_name):
    from scripts.endogenous_outreach import outreach_block_reason

    raw = _state(raw_name) if raw_name in LEVELS else _unknown(raw_name)
    o = _outreach_with(raw)
    asyncio.run(o.arousal_reader.refresh(o._bus))
    off = _outreach_with(raw, enabled=False)
    assert outreach_block_reason(o._gate_inputs()) == outreach_block_reason(off._gate_inputs()) is None


@pytest.mark.parametrize("level", LEVELS)
def test_outreach_flag_off_is_today(level):
    o = _outreach_with(_state(level), enabled=False)
    asyncio.run(o.arousal_reader.refresh(o._bus))
    assert o._gate_inputs().arousal_level is None


def test_outreach_tick_survives_a_disconnected_bus():
    o = _outreach_with(None)
    o._bus = _NotConnected()
    assert asyncio.run(o.arousal_reader.refresh(o._bus)).level == "unknown"
    assert o._gate_inputs().arousal_level is None


def test_outreach_re_reads_arousal_before_delivery(monkeypatch):
    """Live 10-12 14:56 / 15:45, 10-13 15:39: a compose of 6+ min (agent lane timeout then chat
    fallback) re-gated on the reading cached at the top of the tick, read it stale -> unknown,
    and sent. The pre-delivery gate must see the current reading. This pins the re-read (a cached
    idle misses strain that began mid-compose); the stale-cache form of the same gap is covered by
    test_reader_decays_to_unknown_when_not_refreshed."""
    from test_endogenous_outreach import _stub_context

    o = _outreach_with(_state("idle"))
    # Patch the class and module globals `o` actually came from: other hub tests re-import and
    # drop scripts.* from sys.modules, so a fresh import here can be a different module object.
    EndogenousOutreach = type(o)
    module_globals = EndogenousOutreach._outreach_once.__globals__
    queue: asyncio.Queue = asyncio.Queue()
    o.register_connection("c1", queue, {"correlation_id": None, "kind": None})
    _stub_context(monkeypatch)
    monkeypatch.setitem(module_globals, "_juniper_spoke_since", lambda epoch: False)
    monkeypatch.setitem(module_globals, "_seconds_since_juniper_spoke", lambda: 4000.0)

    async def generate_while_strain_begins(self, prompt, session_id, correlation_id, **_kw):
        o._bus.redis.values[REGULATION_STATE_REDIS_KEY] = _state("strained")
        return "something I was thinking about", {"stub": True}

    monkeypatch.setattr(EndogenousOutreach, "_generate", generate_while_strain_begins)
    result = asyncio.run(o._outreach_once(force=False))
    assert (result["outreach"], result["reason"]) == (False, "arousal_strained_after_generation")
    assert queue.empty()
