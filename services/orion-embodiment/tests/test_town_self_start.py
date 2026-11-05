"""Embodiment's own AI Town session timer (app/town_self_start.py, ladder retirement PR 3, 2026-11-05).

Pins:
- a tick with every cap open bids first, then asks the controller, which starts the session and
  opens the play act (act id aitown_self_start:<uuid>, test_before aitown_session_dispatch:<act id>);
  the controller's pause closes the act with a real outcome;
- an open session, the switch off, and a held refusal place no bid (dispatch sent no RPC then);
- a controller refusal holds for its own retry_after_sec (daily cap: until local midnight; cooldown:
  the rest of the gap), then the next tick asks again;
- a dependency mid-restart is never held and the tick never raises: the ledger (Redis) and an
  exception are retried 60 s later, Convex down waits the full tick (dispatch's 300 s);
- the self-monitor gate (E1) dispatch held: a failing faculty is refused with no bid, one probe per
  6 h is offered from the controller's own session ledger and spent only on a real start;
- read-lag ordering: the clock is read first, every read happens later, and nothing reads a later
  value against the earlier clock (#3134);
- Juniper's Unfreeze still works and is never absorbed;
- /health shows each tick's result and the refusal counts.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from app import physiology_shadow
from app import town_self_start as tss
from app.town_self_start import TownSelfStart, build_act_gate
from app.town_sessions import TownConfig, TownSessionController, session_id_for
from orion.embodiment import town_sessions as ts
from orion.schemas.aitown_session import AITownSessionRowV1
from orion.self_monitor.gate import ActGate
from orion.self_monitor.view import SelfMonitorView

DEN = ZoneInfo("America/Denver")


def at(y, m, d, hh, mm=0, ss=0) -> float:
    return datetime(y, m, d, hh, mm, ss, tzinfo=DEN).timestamp()


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


class LaggyRedis:
    """Every read takes ``lag`` seconds of the shared clock: the read happens after the tick's clock."""

    def __init__(self, clock: Clock, lag: float = 0.0):
        self.h: dict = {}
        self.kv: dict = {}
        self.clock, self.lag = clock, lag
        self.fail = False

    async def hgetall(self, key):
        if self.fail:
            raise ConnectionError("redis mid-restart")
        self.clock.t += self.lag
        return dict(self.h.get(key, {}))

    async def hset(self, key, field, value):
        if self.fail:
            raise ConnectionError("redis mid-restart")
        self.h.setdefault(key, {})[field] = value

    async def hdel(self, key, *fields):
        for f in fields:
            self.h.get(key, {}).pop(f, None)

    async def set(self, key, value):
        self.kv[key] = value

    async def get(self, key):
        return self.kv.get(key)


class FakeTown:
    def __init__(self):
        self.running = False
        self.calls: list = []
        self.correlation = None
        self.down = False

    def engine_running(self, world_id):
        if self.down:
            raise TimeoutError("convex down")
        return self.running

    def resume(self):
        self.calls.append("resume")
        self.running = True

    def stop(self):
        self.calls.append("stop")
        self.running = False

    def heartbeat(self, world_id):
        self.calls.append("heartbeat")

    def set_correlation(self, value):
        self.correlation = value


@pytest.fixture
def bids(monkeypatch):
    placed: list = []

    async def abid(want):
        placed.append(want)
        return None

    async def arelease(want_id):
        return None

    monkeypatch.setattr(physiology_shadow, "abid", abid)
    monkeypatch.setattr(physiology_shadow, "arelease", arelease)
    return placed


def make(t0, *, lag=0.0, cfg=None, gate=None):
    clock = Clock(t0)
    redis, town, acts = LaggyRedis(clock, lag), FakeTown(), []

    async def publish(bus, event):
        acts.append(event)

    c = TownSessionController(cfg or TownConfig(), redis_getter=lambda: redis, world_id_getter=lambda: "w1",
                              town=town, act_publisher=publish, clock=clock)
    timer = TownSelfStart(c, tick_sec=300.0, gate=gate, clock=clock)
    asyncio.run(c.tick())   # the controller's boot (recover) tick
    return timer, c, redis, town, clock, acts


def run(coro):
    return asyncio.run(coro)


def tick(timer):
    async def go():
        out = await timer.tick()
        await asyncio.sleep(0)   # let the shadow bid task run
        return out

    return asyncio.run(go())


def ended_row(sid, start, end):
    return AITownSessionRowV1(session_id=sid, kind="orion", started_at=start, last_seen_at=end, ended_at=end,
                              start_reason="want_funded", stop_reason="session_cap", dispatch_id=f"old:{sid}")


# --- the happy path and the act ledger ------------------------------------------------------------


def test_a_tick_bids_first_then_starts_and_opens_the_play_act(bids, monkeypatch) -> None:
    timer, c, redis, town, clock, acts = make(at(2026, 11, 5, 9))
    order: list = []
    real_place, real_start = physiology_shadow.ShadowBid.place, c.start_session

    def place(self):
        order.append("bid")
        return real_place(self)

    async def start_session(act_id):
        order.append("start")
        return await real_start(act_id)

    monkeypatch.setattr(physiology_shadow.ShadowBid, "place", place)
    monkeypatch.setattr(c, "start_session", start_session)
    out = tick(timer)
    assert out["result"] == "started" and order == ["bid", "start"]
    act_id = out["act_id"]
    assert act_id.startswith("aitown_self_start:") and out["session_id"] == session_id_for(act_id)
    (want,) = bids
    assert (want.family, want.verb, want.tier, want.want_id) == ("aitown.session", "play", 2, out["session_id"])
    assert want.object_ref == f"aitown_session_dispatch:{act_id}"
    assert town.calls == ["resume"] and town.correlation == out["session_id"]
    (opened,) = acts
    assert opened.event_kind == "act_opened" and opened.correlation_id == out["session_id"]
    assert (opened.act.actor, opened.act.verb) == ("aitown.session", "play")
    assert opened.act.test_before == f"aitown_session_dispatch:{act_id}"
    assert opened.act.object_ref == f"aitown_session:{out['session_id']}"
    (r,) = run(ts.load_rows(redis))
    assert (r.kind, r.dispatch_id, r.start_reason) == ("orion", act_id, "want_funded")


def test_the_session_cap_closes_the_act_with_a_real_outcome(bids) -> None:
    timer, c, redis, town, clock, acts = make(at(2026, 11, 5, 9))
    out = tick(timer)
    c.note_utterance()
    clock.t += 1800
    run(c.tick())
    closed = acts[-1]
    assert closed.event_kind == "act_closed" and closed.correlation_id == out["session_id"]
    assert closed.act.outcome.value == 1.0 and not town.running


def test_an_open_session_holds_the_timer_with_no_bid(bids) -> None:
    timer, c, redis, town, clock, acts = make(at(2026, 11, 5, 9))
    tick(timer)
    for _ in range(5):
        clock.t += 300
        run(c.tick())
        assert tick(timer)["result"] == "session_running"
    assert len(bids) == 1 and town.calls.count("resume") == 1


def test_the_switch_off_places_no_bid(bids) -> None:
    timer, c, *_ = make(at(2026, 11, 5, 9), cfg=TownConfig(enabled=False))
    assert tick(timer)["result"] == "disabled" and bids == []


# --- dispatch's refusal hold, carried over ------------------------------------------------------


def test_a_daily_cap_refusal_holds_until_local_midnight_then_starts(bids) -> None:
    timer, c, redis, town, clock, acts = make(at(2026, 11, 5, 20), cfg=TownConfig(min_gap_sec=0.0))
    run(ts.save_row(redis, ended_row("a", at(2026, 11, 5, 9), at(2026, 11, 5, 11))))   # 2 h used today
    out = tick(timer)
    assert (out["result"], out["refusal"]) == ("refused", "daily_cap")
    assert out["retry_after_sec"] == pytest.approx(ts.next_local_midnight(clock.t) - clock.t)
    n = len(bids)
    while clock.t < at(2026, 11, 6, 0):
        clock.t += 300
        res = tick(timer)
        if clock.t < at(2026, 11, 6, 0):
            assert (res["result"], res["refusal"]) == ("held", "held:daily_cap")
    assert len(bids) == n + 1            # no bid while held; the first tick after midnight bids
    assert res["result"] == "started"
    assert timer.refusals["held"] > 40 and timer.refusals["daily_cap"] == 1


def test_a_cooldown_holds_for_the_rest_of_the_gap(bids) -> None:
    timer, c, redis, town, clock, acts = make(at(2026, 11, 5, 12))
    run(ts.save_row(redis, ended_row("a", at(2026, 11, 5, 11), at(2026, 11, 5, 11, 30))))   # gap ends 12:30
    out = tick(timer)
    assert out["refusal"] == "cooldown" and out["retry_after_sec"] == pytest.approx(1800.0)
    clock.t = at(2026, 11, 5, 12, 25)
    assert tick(timer)["result"] == "held"
    clock.t = at(2026, 11, 5, 12, 30, 1)
    assert tick(timer)["result"] == "started"


# --- failures are harmless and retried -----------------------------------------------------------


def test_redis_mid_restart_is_retried_in_60_s_never_held(bids) -> None:
    clock = Clock(at(2026, 11, 5, 9))
    redis, town = LaggyRedis(clock), FakeTown()
    redis.fail = True
    c = TownSessionController(TownConfig(), redis_getter=lambda: redis, world_id_getter=lambda: "w1",
                              town=town, act_publisher=lambda b, e: asyncio.sleep(0), clock=clock)
    timer = TownSelfStart(c, clock=clock)
    run(c.tick())                         # recover fails: not booted
    out = tick(timer)
    assert (out["result"], out["refusal"]) == ("refused", "ledger_unknown")
    assert timer.next_delay(out) == tss.ERROR_RETRY_SEC and timer._hold is None
    redis.fail = False
    clock.t += 60
    run(c.tick())                         # the controller boots
    assert tick(timer)["result"] == "started"


def test_convex_down_waits_the_full_tick_like_dispatch_never_held(bids) -> None:
    timer, c, redis, town, clock, acts = make(at(2026, 11, 5, 9))
    town.down = True
    out = tick(timer)
    assert out["refusal"] == "town_unreachable" and timer.next_delay(out) == 300.0 and timer._hold is None
    town.down = False
    clock.t += 300
    assert tick(timer)["result"] == "started"


def test_a_tick_that_raises_is_recorded_and_retried(bids, monkeypatch) -> None:
    timer, c, *_ = make(at(2026, 11, 5, 9))

    async def boom(act_id):
        raise RuntimeError("bug")

    monkeypatch.setattr(c, "start_session", boom)
    out = tick(timer)
    assert out["result"] == "tick_failed" and timer.next_delay(out) == 60.0
    assert timer.next_delay({"result": "refused", "refusal": "cooldown"}) == 300.0


# --- the self-monitor act gate (E1) --------------------------------------------------------------


class StubView:
    known = True
    reason = "fresh"

    def __init__(self, status):
        self._status = status

    def status(self, faculty):
        return self._status if faculty == "act.aitown.session.play" else None


def _gate(controller, status="broken"):
    return ActGate(enabled=True, probe_interval_sec=21600.0, read_view=lambda _now: StubView(status),
                   ledger_rows=lambda actor, verb, secs: controller.act_attempt_rows(secs))


def test_a_failing_faculty_is_refused_with_no_bid_until_its_probe_is_due(bids) -> None:
    timer, c, redis, town, clock, acts = make(at(2026, 11, 5, 9))
    timer.gate = _gate(c)
    # A session opened 2 h ago (inside the 6 h window): no probe due.
    # (A real session id is a uuid: act_faculty keeps a plain-word name as a separate faculty.)
    row = ended_row("2640e568-c741-5285-b9db-c6edf4934b5b", clock.t - 7200, clock.t - 5400)
    run(ts.save_row(redis, row))
    c._rows.append(row)
    out = tick(timer)
    assert (out["result"], out["refusal"]) == ("refused", "self_monitor:track_record_failing") and bids == []
    # 6 h after that attempt the probe is offered, and spent on the real start.
    clock.t += 4 * 3600 + 1
    out = tick(timer)
    assert out["result"] == "started" and out["probe"] is True and len(bids) == 1
    run(c.tick())
    clock.t += 1800
    run(c.tick())                         # the session closes
    clock.t += 3600
    assert tick(timer)["refusal"] == "self_monitor:track_record_failing"   # the window's probe is spent


def test_a_probe_is_spent_only_by_a_real_start(bids) -> None:
    """A probe offered to an attempt the caps refuse stays the window's probe; a started one is claimed."""
    timer, c, redis, town, clock, acts = make(at(2026, 11, 5, 12))
    timer.gate = _gate(c)
    # No play act in the last 6 h, so the probe is offered; Juniper is in town, so the caps refuse.
    town.running = True
    out = tick(timer)
    assert out["refusal"] == "manual_session_open"
    assert timer.gate._memo.get("act.aitown.session.play") is None      # the refused probe is not spent
    town.running = False
    clock.t += 300
    out = tick(timer)
    assert out["result"] == "started" and out["probe"] is True
    assert timer.gate._memo.get("act.aitown.session.play") is not None  # the started probe is claimed


def test_an_unknown_view_admits_with_no_ledger_read(bids) -> None:
    timer, c, *_ = make(at(2026, 11, 5, 9))

    def no_ledger(*a):
        raise AssertionError("no ledger read for an unknown view")

    timer.gate = ActGate(enabled=True, probe_interval_sec=21600.0,
                         read_view=lambda _n: SelfMonitorView(known=False, reason="absent"), ledger_rows=no_ledger)
    assert tick(timer)["result"] == "started"


def test_the_gate_ledger_is_the_controllers_own_session_ledger() -> None:
    timer, c, redis, town, clock, acts = make(at(2026, 11, 5, 9))
    run(c.start_session("aitown_self_start:x"))
    (row,) = c.act_attempt_rows(21600.0)
    assert row["object_ref"] == f"aitown_session:{session_id_for('aitown_self_start:x')}"
    assert row["test_before"] == "aitown_session_dispatch:aitown_self_start:x"
    assert row["opened_at"] == datetime.fromtimestamp(clock.t, tz=timezone.utc)
    fresh = TownSessionController(TownConfig(), redis_getter=lambda: redis, world_id_getter=lambda: "w1",
                                  town=town, clock=clock)
    with pytest.raises(RuntimeError):
        fresh.act_attempt_rows(21600.0)    # not loaded yet: the gate reads "unreadable", not "refuse"


def test_the_live_view_reader_ages_the_view_at_read_time(monkeypatch) -> None:
    import orion.self_monitor.gate as gate_mod

    seen: dict = {}

    def fake_read(url, **kw):
        seen.update(kw, url=url)
        return SelfMonitorView(known=False, reason="absent")

    monkeypatch.setattr(gate_mod, "read_view_sync", fake_read)
    settings = SimpleNamespace(bus_url="redis://bus:6379/0", self_monitor_act_gate_enabled=True,
                               self_monitor_probe_interval_sec=21600.0)
    timer, c, *_ = make(at(2026, 11, 5, 9))
    gate = build_act_gate(settings, c)
    gate.begin(datetime(2020, 1, 1, tzinfo=timezone.utc))   # an old clock must never reach the reader
    assert seen == {"url": "redis://bus:6379/0"}


# --- read-lag ordering (#3134) -------------------------------------------------------------------


def test_the_gap_is_aged_at_the_ledger_read_not_at_the_tick_clock(bids) -> None:
    """The live order: the tick reads its clock, then the ledger read takes 2.4 s. A gap that ends
    between the two is over at the read, so the start goes ahead (a pre-read clock would refuse it)."""
    t0 = at(2026, 11, 5, 12)
    timer, c, redis, town, clock, acts = make(t0, lag=2.4)
    clock.t = t0
    run(ts.save_row(redis, ended_row("a", t0 - 5400, t0 - 3600 + 1.0)))   # gap ends 1 s after the tick clock
    out = tick(timer)
    assert out["result"] == "started"
    (r,) = [r for r in run(ts.load_rows(redis)) if r.session_id == out["session_id"]]
    assert r.started_at > t0 + 2.0          # the session starts after the reads, not at the tick label


def test_a_session_written_after_the_tick_clock_is_never_a_negative_age(bids) -> None:
    """The reverse: a session that ended after the tick's clock (written during the read lag) must
    still read as a fresh cooldown, never as an old one."""
    t0 = at(2026, 11, 5, 12)
    timer, c, redis, town, clock, acts = make(t0, lag=2.4)
    clock.t = t0

    real = redis.hgetall

    async def hgetall_writes_first(key):
        await redis.hset(key, "late", ended_row("late", t0 - 1800, t0 + 1.0).model_dump_json())
        return await real(key)

    redis.hgetall = hgetall_writes_first
    out = tick(timer)
    assert out["refusal"] == "cooldown"
    assert out["retry_after_sec"] == pytest.approx(3600.0 - 1.4, abs=0.01)   # aged at the read (t0 + 2.4)


# --- Juniper's Unfreeze ------------------------------------------------------------------------


def test_juniper_unfreeze_still_works_and_is_never_absorbed(bids) -> None:
    timer, c, redis, town, clock, acts = make(at(2026, 11, 5, 9))
    town.running = True                    # Unfreeze
    clock.t += 10
    run(c.tick())
    out = tick(timer)
    assert (out["result"], out["refusal"]) == ("refused", "manual_session_open")
    assert timer._hold is None and timer.next_delay(out) == 300.0
    (r,) = run(ts.load_rows(redis))
    assert r.kind == "manual" and "resume" not in town.calls and acts == []


# --- the loop and /health ------------------------------------------------------------------------


def test_the_loop_waits_60_s_then_ticks_every_300_s(bids) -> None:
    timer, c, *_ = make(at(2026, 11, 5, 9), cfg=TownConfig(enabled=False))
    sleeps: list = []

    async def sleep(sec):
        sleeps.append(sec)
        if len(sleeps) >= 3:
            raise asyncio.CancelledError

    timer._sleep = sleep
    with pytest.raises(asyncio.CancelledError):
        run(timer._run())
    assert sleeps == [60.0, 300.0, 300.0] and timer.ticks == 2


def test_health_snapshot_shows_results_refusals_and_the_hold(bids) -> None:
    timer, c, redis, town, clock, acts = make(at(2026, 11, 5, 20), cfg=TownConfig(min_gap_sec=0.0))
    run(ts.save_row(redis, ended_row("a", at(2026, 11, 5, 9), at(2026, 11, 5, 11))))
    tick(timer)
    clock.t += 300
    tick(timer)
    snap = timer.snapshot()
    assert snap["ticks"] == 2 and snap["bids"] == 1
    assert snap["results"] == {"refused": 1, "held": 1} and snap["refusals"] == {"daily_cap": 1, "held": 1}
    assert snap["held_for"] == "daily_cap" and snap["held_until"].startswith("2026-11-06T07:00")   # MST midnight
    assert snap["last"]["refusal"] == "held:daily_cap" and snap["running"] is False
    assert snap["tick_sec"] == 300.0 and snap["enabled"] is True
