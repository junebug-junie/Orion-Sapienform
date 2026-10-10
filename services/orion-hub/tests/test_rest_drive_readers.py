"""Curiosity and outreach read Orion's rest drive (Temporal Self rev 4, R2).

For each reader: tired -> its cooldown stretches (and only its cooldown);
rested/building/refractory -> unchanged; unknown (absent, stale, unparseable,
no_reading, Redis down, flag off) -> exactly the pre-drive decision.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone

import pytest

from orion.regulation.rest_drive import no_rest_reading, read_rest_drive
from orion.schemas.dream_cycle import SleepPressureV1
from orion.schemas.drive_reading import REST_DRIVE_REDIS_KEY
from scripts.curiosity_investigation import (
    LINE_INVESTIGATE,
    SchedulingGateInputs,
    _line_keys,
    scheduling_block_reason,
)
from scripts.endogenous_outreach import OutreachGateInputs, outreach_block_reason
from scripts.rest_drive_reader import RestDriveReader
from test_curiosity_investigation import _FakeBus, _loop
from test_endogenous_outreach import _outreach

NOW = datetime.now(timezone.utc)


def _reading(level, *, at=None, last_end=None):
    at = at or NOW
    p = SleepPressureV1(since=at - timedelta(hours=18), computed_at=at, pressure=level, threshold=3.0,
                        idle_required_minutes=45, idle_minutes=0)
    return read_rest_drive(p, now=at, source_ref="dp-test", last_attempt_end=last_end,
                           min_interval_hours=6.0, overdue=False, has_candidates=True).model_dump_json()


TIRED = _reading(3.3)
STATES_THAT_CHANGE_NOTHING = {
    "resting": _reading(0.0),
    "building": _reading(2.0),
    "refractory": _reading(9.0, last_end=NOW - timedelta(hours=1)),
    "absent": None,
    "unparseable": "{nope",
    "stale": _reading(3.3, at=NOW - timedelta(seconds=1801)),
    "no_reading": no_rest_reading(now=NOW, source_ref="dp-x", threshold=3.0, reason="pressure_read_failed").model_dump_json(),
}


def _reader(enabled=True, multiplier=2.0):
    return RestDriveReader(enabled=enabled, multiplier=multiplier, max_age_sec=1800.0, name="test")


def _bus_with(raw):
    bus = _FakeBus()
    if raw is not None:
        bus.redis.values[REST_DRIVE_REDIS_KEY] = raw
    return bus


# --- the shared reader ------------------------------------------------------------


def test_reader_stretches_only_while_due():
    r = _reader()
    asyncio.run(r.refresh(_bus_with(TIRED), NOW))
    assert r.cooldown_sec(2700.0, NOW) == 5400.0
    for raw in STATES_THAT_CHANGE_NOTHING.values():
        asyncio.run(r.refresh(_bus_with(raw), NOW))
        assert r.cooldown_sec(2700.0, NOW) is None


def test_reader_decays_to_unknown_when_not_refreshed():
    r = _reader()
    asyncio.run(r.refresh(_bus_with(TIRED), NOW))
    assert r.cooldown_sec(2700.0, NOW + timedelta(seconds=1801)) is None


def test_redis_failure_and_flag_off_are_unknown():
    class _Broken:
        class redis:  # noqa: N801
            @staticmethod
            async def get(key):
                raise ConnectionError("down")

    r = _reader()
    assert asyncio.run(r.refresh(_Broken(), NOW)).verdict == "unknown"
    assert r.cooldown_sec(2700.0, NOW) is None
    off = _reader(enabled=False)
    assert asyncio.run(off.refresh(_bus_with(TIRED), NOW)).verdict == "unknown"
    assert off.cooldown_sec(2700.0, NOW) is None


# --- curiosity ----------------------------------------------------------------------


def _sched(**over):
    base = dict(enabled=True, seconds_since_last=3000.0, min_cooldown_sec=1800.0, done_today=0, daily_cap=7)
    base.update(over)
    return SchedulingGateInputs(**base)


def test_curiosity_gate_names_the_stretch_tiredness_added():
    assert scheduling_block_reason(_sched()) is None
    assert scheduling_block_reason(_sched(rest_drive_cooldown_sec=3600.0)) == "rest_drive_cooldown"
    assert scheduling_block_reason(_sched(seconds_since_last=4000.0, rest_drive_cooldown_sec=3600.0)) is None
    assert scheduling_block_reason(_sched(seconds_since_last=60.0, rest_drive_cooldown_sec=3600.0)) == "cooldown"
    # Caps and windows decide first, unchanged.
    assert scheduling_block_reason(_sched(done_today=7, rest_drive_cooldown_sec=3600.0)) == "daily_cap"
    # Never run: nothing to space from.
    assert scheduling_block_reason(_sched(seconds_since_last=None, rest_drive_cooldown_sec=3600.0)) is None


def _curiosity_tick(raw, *, reader=None, since_last_sec=20000.0, force=False):
    bus = _bus_with(raw)
    cooldown_key, _, _ = _line_keys(LINE_INVESTIGATE)
    bus.redis.values[cooldown_key] = (datetime.now(timezone.utc) - timedelta(seconds=since_last_sec)).isoformat()
    loop = _loop(bus, rest_drive_reader=reader if reader is not None else _reader())
    return asyncio.run(loop.tick(force=force))


def test_tired_curiosity_waits_out_the_doubled_cooldown():
    # _loop's cooldown is 14400 s; the last run was 20000 s ago: due without the drive.
    assert _curiosity_tick(TIRED) == "rest_drive_cooldown"
    assert _curiosity_tick(TIRED, since_last_sec=30000.0) != "rest_drive_cooldown"


@pytest.mark.parametrize("name", sorted(STATES_THAT_CHANGE_NOTHING))
def test_curiosity_runs_exactly_as_before_unless_tired(name):
    raw = STATES_THAT_CHANGE_NOTHING[name]
    with_drive = _curiosity_tick(raw)
    without = _curiosity_tick(raw, reader=_reader(enabled=False))
    assert with_drive == without and with_drive != "rest_drive_cooldown"


def test_a_forced_curiosity_run_overrides_tiredness():
    assert _curiosity_tick(TIRED, force=True) != "rest_drive_cooldown"


# --- outreach -----------------------------------------------------------------------


def _gate(**over):
    base = dict(enabled=True, turn_in_flight=False, local_hour=14, quiet_start_hour=23, quiet_end_hour=8,
                seconds_since_last_outreach=4000.0, min_cooldown_sec=2700.0, sent_today=0, daily_cap=4)
    base.update(over)
    return OutreachGateInputs(**base)


def test_outreach_gate_stretches_the_cooldown_only():
    assert outreach_block_reason(_gate()) is None
    assert outreach_block_reason(_gate(rest_drive_cooldown_sec=5400.0)) == "rest_drive_cooldown"
    assert outreach_block_reason(_gate(seconds_since_last_outreach=6000.0, rest_drive_cooldown_sec=5400.0)) is None
    # Door-A (a curiosity finding worth saying) skips it like every schedule gate.
    assert outreach_block_reason(_gate(rest_drive_cooldown_sec=5400.0), skip_schedule_gates=True) is None
    # Protective gates still decide first.
    assert outreach_block_reason(_gate(turn_in_flight=True, rest_drive_cooldown_sec=5400.0)) == "turn_in_flight"
    assert outreach_block_reason(_gate(local_hour=2, rest_drive_cooldown_sec=5400.0)) == "quiet_hours"
    assert outreach_block_reason(_gate(sent_today=4, rest_drive_cooldown_sec=5400.0)) == "daily_cap"


def _outreach_inputs(raw, *, reader=None):
    o = _outreach(min_cooldown_sec=2700.0, rest_drive_reader=reader if reader is not None else _reader())
    o._bus = _bus_with(raw)
    o._last_outreach_at = time.time() - 4000.0
    asyncio.run(o.rest_drive_reader.refresh(o._bus))
    return o


def test_tired_outreach_blocks_on_the_doubled_cooldown_and_reports_it():
    o = _outreach_inputs(TIRED)
    inputs = o._gate_inputs()
    assert inputs.rest_drive_cooldown_sec == 5400.0
    assert outreach_block_reason(inputs) == "rest_drive_cooldown"
    status = o.status()
    assert status["rest_drive"]["verdict"] == "tired" and status["block_reason"] == "rest_drive_cooldown"


@pytest.mark.parametrize("name", sorted(STATES_THAT_CHANGE_NOTHING))
def test_outreach_unchanged_unless_tired(name):
    raw = STATES_THAT_CHANGE_NOTHING[name]
    o = _outreach_inputs(raw)
    off = _outreach_inputs(raw, reader=_reader(enabled=False))
    assert o._gate_inputs().rest_drive_cooldown_sec is None
    assert outreach_block_reason(o._gate_inputs()) == outreach_block_reason(off._gate_inputs()) is None


def test_outreach_tick_refreshes_the_drive_before_gating(monkeypatch):
    o = _outreach(min_cooldown_sec=2700.0, rest_drive_reader=_reader())
    o._bus = _bus_with(TIRED)
    o._last_outreach_at = time.time() - 4000.0
    result = asyncio.run(o._outreach_once(force=False))
    assert result.get("reason") == "rest_drive_cooldown"
