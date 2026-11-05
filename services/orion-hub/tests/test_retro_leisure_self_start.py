"""Ladder retirement PR 1: Hub's own leisure timer replaces the dispatch RPC.

The runner is a fake with Lane B's exact interface (scripts/retro_leisure.py):
``should_start(config) -> None | reason`` and ``run_leisure_session(config, act_id=) -> LeisureOutcome``,
except the read-lag tests, which drive the REAL ``should_start`` over in-memory stores.

Pins:
- the bid is placed BEFORE the gate (the want, not the aftermath), with the act id as want_id;
- each gate refusal is recorded by name, opens no act, and starts nothing;
- a passed gate opens one `play` act keyed by the self-start act id, and when the session ends the
  export writes the memory and closes that same act (the act ledger keeps grading play);
- while a session runs, a tick does nothing (no bid, no second session);
- the switch off / no runner: no bid, never a fake session;
- a failing first tick is harmless, recorded, and retried sooner;
- read-lag ordering: the tick's clock comes first, then the reads, with a real lag. A regulation
  state written after the tick's clock still reads fresh (the #3134 painting bug, here for retro).
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from orion.schemas.grammar import GrammarEventV1
from scripts import act_emit
from scripts import retro_leisure_self_start as rs
from scripts.retro_leisure_self_start import ERROR_RETRY_SEC, RetroLeisureSelfStart, gate_verdict

CONFIG = SimpleNamespace(RETRO_LEISURE_ENABLED=True)


@dataclass
class _Ended:
    """Lane B's RetroSessionEnded fields the export reads."""
    session_id: str = "retro-sess-1"
    lineage_id: str | None = "lin-1"
    leisure_id: str = "leis-1"
    game_id: str = "the-legend-of-zelda"
    started_at: float = 1792699320.0
    ended_at: float = 1792700640.0
    end_reason: str = "session_cap"
    act_id: str | None = None
    decisions_planned: int = 31
    verdicts: dict = field(default_factory=lambda: {"met": 14, "refuted": 12, "unknown": 3})
    human_may_be_in_control: bool = False


@dataclass
class _Outcome:
    started: bool
    refusal: str | None = None
    ended: _Ended | None = None


class _Runner:
    def __init__(self, gate=None, *, outcome=None, hold: asyncio.Event | None = None, raises=False,
                 gate_raises=False, order=None):
        self.gate, self.outcome, self.hold, self.raises, self.gate_raises = gate, outcome, hold, raises, gate_raises
        self.runs: list[str] = []
        self.order = order if order is not None else []

    async def should_start(self, config):
        assert config is CONFIG
        self.order.append("gate")
        if self.gate_raises:
            raise ConnectionError("redis mid-restart")
        return self.gate

    async def run_leisure_session(self, config, *, act_id=None):
        self.runs.append(act_id)
        if self.raises:
            raise RuntimeError("retro down")
        if self.hold is not None:
            await self.hold.wait()
        return self.outcome if self.outcome is not None else _Outcome(True, ended=_Ended(act_id=act_id))


class _Bus:
    def __init__(self):
        self.sent = []

    async def publish(self, channel, envelope):
        self.sent.append((channel, envelope))

    def events(self):
        return [GrammarEventV1.model_validate(e.payload) for ch, e in self.sent if ch == "orion:grammar:event"]


class _Shadow:
    """Records ShadowBid calls in order, beside the runner's own order list."""

    def __init__(self, order):
        self.order = order
        self.bids = []

    def factory(self):
        shadow = self

        class _Bid:
            def __init__(self, **kw):
                self.kw = kw
                self.real = None
                shadow.bids.append(self)

            def place(self):
                shadow.order.append("bid")

            def settle(self, real, **_):
                self.real = real

        return _Bid


@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    from scripts.settings import settings

    monkeypatch.setattr(settings, "HUB_ACT_LEDGER_ENABLED", True, raising=False)
    monkeypatch.setattr(rs, "START_GRACE_SEC", 0.05)


@pytest.fixture
def order():
    return []


@pytest.fixture
def shadow(monkeypatch, order):
    import scripts.physiology_shadow as ps

    rec = _Shadow(order)
    monkeypatch.setattr(ps, "ShadowBid", rec.factory())
    return rec


@pytest.fixture
def exported(monkeypatch):
    from scripts import retro_memory_export as rx

    calls = []

    async def fake_export(*, pool, bus, ended, dispatch_id=None, enabled=None):
        assert dispatch_id == ended.act_id
        calls.append(ended)
        await act_emit.publish_play_closed(bus, dispatch_id=dispatch_id, session_id=ended.session_id,
                                           produced=True, note="exported")
        return rx.ExportResult("exported", ended.session_id, card_id="c1")

    monkeypatch.setattr(rx, "export_leisure_outcome", fake_export)
    return calls


def _timer(runner, config=CONFIG, **kw):
    timer = RetroLeisureSelfStart(config_provider=lambda: config, pool_provider=lambda: None,
                                  runner_provider=lambda: runner, **kw)
    timer.bus = _Bus()
    return timer


async def _settle(timer):
    for _ in range(50):
        if not timer._followers:
            return
        await asyncio.sleep(0.01)


@pytest.mark.parametrize("raw,expected", [
    (None, (True, None)), (True, (True, None)), ("cabinet_hot", (False, "cabinet_hot")),
    ("", (False, "unnamed_refusal")), (False, (False, "unnamed_refusal")),
])
def test_gate_verdict(raw, expected) -> None:
    assert gate_verdict(raw) == expected


@pytest.mark.asyncio
async def test_the_bid_is_placed_before_the_gate(shadow, order) -> None:
    timer = _timer(_Runner("cabinet_hot", order=order))
    out = await timer.tick()
    assert order == ["bid", "gate"]
    (bid,) = shadow.bids
    assert (bid.kw["family"], bid.kw["verb"], bid.kw["tier"], bid.kw["component"]) == (
        "hub.retro_leisure", "play", 2, "retro_leisure")
    assert bid.kw["want_id"] == out["act_id"] and out["act_id"].startswith("retro_self_start:")
    assert bid.kw["object_ref"] == f"retro_leisure_dispatch:{out['act_id']}"
    assert bid.real == "blocked:cabinet_hot"


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["arousal_strained", "cabinet_hot", "browser_connected", "human_recent",
                                    "cooldown", "daily_cap", "retro_session_active", "disabled"])
async def test_each_gate_refusal_is_recorded_by_name_and_starts_nothing(shadow, reason) -> None:
    runner = _Runner(reason)
    timer = _timer(runner)
    out = await timer.tick()
    assert (out["result"], out["refusal"]) == ("refused", reason)
    assert runner.runs == [] and timer.bus.events() == []
    snap = timer.snapshot()
    assert snap["last"] == out and snap["refusals"] == {reason: 1} and snap["last_started"] is None


@pytest.mark.asyncio
async def test_a_refusal_from_the_runners_own_recheck_is_a_refusal(shadow) -> None:
    runner = _Runner(None, outcome=_Outcome(False, refusal="human_recent"))
    timer = _timer(runner)
    out = await timer.tick()
    assert (out["result"], out["refusal"]) == ("refused", "human_recent")
    assert runner.runs == [out["act_id"]] and timer.bus.events() == []
    assert shadow.bids[0].real == "blocked:human_recent"


@pytest.mark.asyncio
async def test_a_session_opens_the_play_act_and_its_end_exports_and_closes_it(shadow, exported) -> None:
    hold = asyncio.Event()
    runner = _Runner(None, hold=hold)
    timer = _timer(runner)
    out = await timer.tick()
    act_id = out["act_id"]
    assert out["result"] == "started" and timer.snapshot()["last_started"] == out
    assert shadow.bids[0].real == "ran"
    (opened,) = timer.bus.events()
    assert opened.event_kind == "act_opened"
    assert (opened.act.verb, opened.act.actor, opened.act.test_before) == (
        "play", "hub.retro_leisure", f"retro_leisure_dispatch:{act_id}")
    assert opened.correlation_id == act_id
    assert opened.act.expect.horizon_sec == act_emit.PLAY_HORIZON_SEC == 2400.0
    # While it runs, a tick does nothing: no bid, no second session.
    busy = await timer.tick()
    assert busy["result"] == "session_running" and len(shadow.bids) == 1 and runner.runs == [act_id]
    hold.set()
    await _settle(timer)
    assert [e.act_id for e in exported] == [act_id]
    opened, closed = timer.bus.events()
    assert closed.event_kind == "act_closed" and closed.act.act_id == opened.act.act_id
    assert closed.act.outcome.value == 1.0


@pytest.mark.asyncio
async def test_a_session_that_ends_without_data_closes_the_act_at_zero(shadow) -> None:
    hold = asyncio.Event()
    timer = _timer(_Runner(None, hold=hold, outcome=_Outcome(True, refusal=None, ended=None)))
    assert (await timer.tick())["result"] == "started"
    hold.set()
    await _settle(timer)
    _, closed = timer.bus.events()
    assert closed.act.outcome.value == 0.0 and closed.act.outcome.note.startswith("no_session")


@pytest.mark.asyncio
async def test_switch_off_or_no_runner_places_no_bid_and_never_fakes_a_session(shadow) -> None:
    off = _timer(_Runner(None), config=SimpleNamespace(RETRO_LEISURE_ENABLED=False))
    assert (await off.tick())["result"] == "disabled"
    missing = _timer(None)
    assert (await missing.tick())["result"] == "runner_unavailable"
    assert shadow.bids == [] and off.bus.events() == [] and missing.bus.events() == []


@pytest.mark.asyncio
async def test_a_failing_first_tick_is_harmless_and_retried_sooner(shadow) -> None:
    runner = _Runner(None, gate_raises=True)
    timer = _timer(runner, tick_sec=300.0)
    out = await timer.tick()
    assert (out["result"], out["refusal"]) == ("gate_failed", "gate_failed")
    assert shadow.bids[0].real == "blocked:gate_failed" and runner.runs == []
    assert timer.next_delay(out) == ERROR_RETRY_SEC < 300.0
    runner.gate_raises = False
    runner.gate = "arousal_strained"
    again = await timer.tick()
    assert again["result"] == "refused" and timer.next_delay(again) == 300.0
    assert timer.snapshot()["results"] == {"gate_failed": 1, "refused": 1}


@pytest.mark.asyncio
async def test_a_runner_that_raises_at_start_is_start_failed(shadow) -> None:
    out = await _timer(_Runner(None, raises=True)).tick()
    assert (out["result"], out["refusal"]) == ("start_failed", "start_failed")


@pytest.mark.asyncio
async def test_an_unexpected_error_is_a_recorded_tick_not_a_dead_loop(shadow) -> None:
    def broken():
        raise RuntimeError("config gone")

    timer = RetroLeisureSelfStart(config_provider=broken, pool_provider=lambda: None,
                                  runner_provider=lambda: _Runner(None))
    out = await timer.tick()
    assert out["result"] == "tick_failed" and timer.next_delay(out) == ERROR_RETRY_SEC


@pytest.mark.asyncio
async def test_the_loop_ticks_after_its_first_delay_and_keeps_ticking(shadow, monkeypatch) -> None:
    sleeps = []
    real_sleep = asyncio.sleep

    async def fake_sleep(sec):
        sleeps.append(sec)
        if len(sleeps) >= 3:
            raise asyncio.CancelledError
        await real_sleep(0)

    monkeypatch.setattr(rs.asyncio, "sleep", fake_sleep)
    timer = _timer(_Runner("cooldown"), tick_sec=300.0, first_delay_sec=60.0)
    with pytest.raises(asyncio.CancelledError):
        await timer._run()
    assert sleeps == [60.0, 300.0, 300.0] and timer.ticks == 2


@pytest.mark.asyncio
async def test_a_disabled_timer_never_starts_its_loop() -> None:
    timer = _timer(_Runner(None), enabled=False)
    await timer.begin(_Bus())
    assert timer.task is None and timer.snapshot()["running"] is False


# --- read-lag ordering, through the REAL gate ----------------------------------------------------


def _regulation_json(at: datetime, level: str = "idle") -> str:
    from orion.schemas.regulation import ArousalReadingV1, RegulationStateV1

    reading = ArousalReadingV1(arousal_level=level, since=at - timedelta(minutes=5), observed_at=at,
                               reasons=[f"test_{level}"])
    return RegulationStateV1(generated_at=at, arousal=reading).model_dump_json()


class _Redis:
    """The regulation key, written by the producer AFTER the tick read its clock."""

    def __init__(self, value):
        self.value = value

    async def get(self, key):
        await asyncio.sleep(0.01)   # a real read lag
        return self.value


@pytest.mark.asyncio
async def test_a_state_written_after_the_ticks_clock_still_reads_fresh() -> None:
    """The live ordering: tick clock T, then the producer writes at T + 1.5 s, then the GET. The age
    is computed at the read time, so it is fresh -- never 'future'/'stale' against T."""
    from scripts import retro_leisure as L

    tick_clock = datetime.now(timezone.utc)
    written = tick_clock + timedelta(seconds=1.5)
    level, reason, reflex = await L.read_regulation(_Redis(_regulation_json(written)), max_age_sec=360.0)
    assert (level, reflex) == ("idle", None), reason


@pytest.mark.asyncio
async def test_the_real_gate_passes_with_reads_after_the_tick_clock(shadow) -> None:
    """End to end through RetroLeisureSelfStart.tick and the REAL should_start: the tick reads its
    clock, then the presence store and the regulation key are read later. Nothing refuses on age."""
    from scripts import retro_leisure as L
    from scripts.retro_presence import HUMAN_AT_KEY, PresenceStore

    class _Store(PresenceStore):
        pass

    class _KV:
        def __init__(self):
            self.kv = {}

        async def get(self, k):
            await asyncio.sleep(0.01)
            return self.kv.get(k)

        async def scan_iter(self, match=None, count=None):
            if False:
                yield None

        async def lrange(self, k, a, b):
            return []

    import time as _time

    # The real wall clock: read_regulation computes the age against datetime.now() at the GET.
    start = _time.time()
    t = {"now": start}

    def clock():
        return t["now"]

    kv = _KV()
    # Juniper's last Retro action: 900 s before the tick (outside the 600 s quiet window).
    kv.kv[HUMAN_AT_KEY] = str(t["now"] - 900.0)
    store = _Store(kv, clock=clock)

    class _Retro:
        async def call(self, method, path, body=None):
            t["now"] += 0.4   # the reads after the tick take real time
            return 200, {"session": None}

    async def regulation():
        t["now"] += 1.5
        written = datetime.fromtimestamp(t["now"] - 0.2, tz=timezone.utc)
        return await L.read_regulation(_Redis(_regulation_json(written)), max_age_sec=360.0)

    deps = L.LeisureDeps(store=store, retro=_Retro(), regulation=regulation, plan=None, clock=clock)
    config = SimpleNamespace(RETRO_LEISURE_ENABLED=True, HUB_RETRO_ENABLED=True,
                             RETRO_LEISURE_GAME_IDS="the-legend-of-zelda", RETRO_LEISURE_SESSION_MAX_SEC=1800,
                             RETRO_LEISURE_DAILY_MAX_SEC=5400, RETRO_LEISURE_MIN_GAP_SEC=10800)
    verdicts = []

    class _RealGate:
        async def should_start(self, cfg):
            verdicts.append(await L.should_start(cfg, deps))
            return verdicts[-1]

        async def run_leisure_session(self, cfg, *, act_id=None):
            return _Outcome(False, refusal="test_stop")

    timer = RetroLeisureSelfStart(config_provider=lambda: config, pool_provider=lambda: None,
                                  runner_provider=lambda: _RealGate(), clock=clock)
    timer.bus = _Bus()
    out = await timer.tick()
    # The gate passed on fresh reads; the fake runner then stops on purpose.
    assert verdicts == [None], verdicts
    assert (out["result"], out["refusal"]) == ("refused", "test_stop")
    # The tick's label is its clock BEFORE the reads, not the later read time.
    assert out["ticked_at"] == datetime.fromtimestamp(start, tz=timezone.utc).isoformat()


def test_the_real_runner_module_is_found() -> None:
    mod = rs.import_runner()
    assert mod is not None and hasattr(mod, "should_start") and hasattr(mod, "run_leisure_session")


# --- the self-monitor act gate (E1), moved from dispatch ------------------------------------------


def _monitor_state(status: str, computed_at: datetime) -> str:
    rec = {
        "faculty": "act.hub.retro_leisure.play", "sources": ["grammar_acts"], "claim_kind": "success",
        "gradeability": "graded", "window_start": (computed_at - timedelta(days=7)).isoformat(),
        "window_end": computed_at.isoformat(), "n_bets": 12, "n_graded": 12, "n_failed_to_execute": 0,
        "n_expired": 0, "weight_graded": 4.0, "candidate_status": status, "status": status,
        "status_since": (computed_at - timedelta(days=1)).isoformat(),
    }
    return json.dumps({"kind": "self_monitor.state.v1", "computed_at": computed_at.isoformat(),
                       "window_days": 7.0, "faculties": [rec]})


def _real_gate(status: str, *, ledger=None, publish_lag_sec: float = 1.5):
    """A REAL ActGate. The self-monitor state it reads was published AFTER the tick's clock."""
    from orion.self_monitor.gate import ActGate
    from orion.self_monitor.view import self_monitor_view

    def read_view(now):
        return self_monitor_view(_monitor_state(status, now + timedelta(seconds=publish_lag_sec)), now=now)

    return ActGate(enabled=True, probe_interval_sec=21600.0, read_view=read_view,
                   ledger_rows=ledger or (lambda actor, verb, secs: []))


@pytest.mark.asyncio
async def test_a_failing_faculty_is_probed_once_then_refused_without_a_bid(shadow, exported) -> None:
    hold = asyncio.Event()
    runner = _Runner(None, hold=hold)
    timer = _timer(runner, gate=_real_gate("broken"))
    first = await timer.tick()
    assert first["result"] == "started" and first["probe"] is True    # the 6 h re-test probe
    hold.set()
    await _settle(timer)
    second = await timer.tick()
    assert (second["result"], second["refusal"]) == ("refused", "self_monitor:track_record_failing")
    assert len(shadow.bids) == 1 and len(runner.runs) == 1   # refused before the bid, never ran
    assert timer.snapshot()["refusals"] == {"self_monitor": 1}


@pytest.mark.asyncio
async def test_a_refused_start_keeps_the_probe(shadow) -> None:
    runner = _Runner("cabinet_hot")
    timer = _timer(runner, gate=_real_gate("broken"))
    assert (await timer.tick())["refusal"] == "cabinet_hot"
    runner.gate = "human_recent"
    # The probe was not spent: the next tick is offered it again, and reaches the doer's gate.
    assert (await timer.tick())["refusal"] == "human_recent"


@pytest.mark.asyncio
async def test_a_view_published_after_the_ticks_clock_is_fresh_not_unknown(shadow) -> None:
    """Read-lag regression for the gate: computed_at is 1.5 s after the tick clock. A reliable
    faculty admits; the view is read as known (not future_stamped)."""
    gate = _real_gate("reliable")
    timer = _timer(_Runner("cooldown"), gate=gate)
    out = await timer.tick()
    assert out["refusal"] == "cooldown"   # admitted by the gate, refused by the doer
    assert gate._view.known and gate._view.reason == "fresh"


@pytest.mark.asyncio
async def test_build_act_gate_reads_the_ledger_on_the_loops_pool() -> None:
    from datetime import datetime as _dt

    seen = []

    class _Conn:
        async def fetch(self, sql, *args):
            seen.append((sql, args))
            return [{"object_ref": "retro_session:x", "test_before": "retro_leisure_dispatch:a",
                     "opened_at": _dt.now(timezone.utc)}]

    class _Acquire:
        async def __aenter__(self):
            return _Conn()

        async def __aexit__(self, *a):
            return False

    class _Pool:
        def acquire(self):
            return _Acquire()

    settings = SimpleNamespace(SELF_MONITOR_ACT_GATE_ENABLED=True, SELF_MONITOR_PROBE_INTERVAL_SEC=21600.0,
                               ORION_BUS_URL="redis://127.0.0.1:1/0")
    loop = asyncio.get_running_loop()
    gate = rs.build_act_gate(settings, pool_provider=lambda: _Pool(), loop=loop)
    rows = await asyncio.to_thread(gate._ledger_rows, "hub.retro_leisure", "play", 21600.0)
    assert rows[0]["object_ref"] == "retro_session:x"
    sql, args = seen[0]
    assert "$1" in sql and args == ("hub.retro_leisure", "play", 21600.0)
    none_gate = rs.build_act_gate(settings, pool_provider=lambda: None, loop=loop)
    with pytest.raises(RuntimeError):
        await asyncio.to_thread(none_gate._ledger_rows, "hub.retro_leisure", "play", 21600.0)
