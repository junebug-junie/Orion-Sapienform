"""AI Town session controller (app/town_sessions.py): paused by default, Orion's sessions capped.

Pins:
- outside an Orion session the controller never keeps the town awake; a running town with no Orion
  session is Juniper's manual session (logged, not counted, never paused here);
- a start RPC that passes the gate unpauses the town, writes the ledger row, sets the NPC
  correlation id and opens the play act; a refused one changes nothing;
- the session cap and the daily cap pause the town and close the act, crossing local midnight right;
- a restart closes the session it left open at its last tick and pauses the town;
- a ledger that cannot be read never starts an uncounted session.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app import physiology_shadow
from app.town_sessions import TownConfig, TownSessionController, session_id_for
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.embodiment import town_sessions as ts
from orion.schemas.aitown_session import (
    AITOWN_SESSION_START_REPLY_PREFIX,
    AITOWN_SESSION_START_REQUEST_KIND,
    AITownSessionRowV1,
    AITownSessionStartRequestV1,
)

DEN = ZoneInfo("America/Denver")


def at(y, m, d, hh, mm=0, ss=0) -> float:
    return datetime(y, m, d, hh, mm, ss, tzinfo=DEN).timestamp()


class FakeRedis:
    def __init__(self):
        self.h: dict = {}
        self.kv: dict = {}
        self.fail = False

    async def hgetall(self, key):
        if self.fail:
            raise ConnectionError("redis down")
        return dict(self.h.get(key, {}))

    async def hset(self, key, field, value):
        if self.fail:
            raise ConnectionError("redis down")
        self.h.setdefault(key, {})[field] = value

    async def hdel(self, key, *fields):
        for f in fields:
            self.h.get(key, {}).pop(f, None)

    async def set(self, key, value):
        self.kv[key] = value

    async def get(self, key):
        return self.kv.get(key)


class FakeTown:
    def __init__(self, running=False):
        self.running = running
        self.calls: list = []
        self.correlation = "unset"

    def engine_running(self, world_id):
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


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture(autouse=True)
def _no_market(monkeypatch):
    async def abid(want):
        return None

    async def arelease(want_id):
        return None

    monkeypatch.setattr(physiology_shadow, "abid", abid)
    monkeypatch.setattr(physiology_shadow, "arelease", arelease)


def make(t0, *, running=False, cfg=None):
    redis, town, clock, acts = FakeRedis(), FakeTown(running), Clock(t0), []

    async def publish(bus, event):
        acts.append(event)

    c = TownSessionController(cfg or TownConfig(), redis_getter=lambda: redis, world_id_getter=lambda: "w1",
                              town=town, act_publisher=publish, clock=clock)
    return c, redis, town, clock, acts


def rows(redis) -> list[AITownSessionRowV1]:
    return asyncio.run(ts.load_rows(redis))


def req(d="dispatch:proposal:aitown_session:t1"):
    return AITownSessionStartRequestV1(dispatch_id=d)


def test_paused_by_default_no_heartbeat_and_no_session() -> None:
    c, redis, town, clock, acts = make(at(2026, 10, 30, 9))
    for _ in range(3):
        asyncio.run(c.tick())
        clock.t += 10
    assert town.calls == [] and not c.heartbeat_wanted
    assert rows(redis) == [] and acts == []
    assert redis.kv  # the controller published its limits for the report


def test_a_town_left_running_at_boot_is_paused_once() -> None:
    """The first deploy: the old always-on heartbeat left the town running, no session on the ledger."""
    c, redis, town, clock, _ = make(at(2026, 10, 30, 9), running=True)
    asyncio.run(c.tick())
    assert town.calls == ["stop"] and not town.running and rows(redis) == []


def test_juniper_open_session_survives_a_restart() -> None:
    c, redis, town, clock, _ = make(at(2026, 10, 30, 9))
    asyncio.run(c.tick())
    town.running = True                       # Juniper presses Unfreeze
    clock.t += 10
    asyncio.run(c.tick())
    c2 = TownSessionController(TownConfig(), redis_getter=lambda: redis, world_id_getter=lambda: "w1",
                               town=town, act_publisher=c._publish_act, clock=Clock(clock.t + 60))
    asyncio.run(c2.tick())
    assert "stop" not in town.calls and town.running
    (r,) = rows(redis)
    assert r.kind == "manual" and r.ended_at is None


def test_a_town_running_without_orion_is_a_manual_session_never_paused() -> None:
    c, redis, town, clock, _ = make(at(2026, 10, 30, 9))
    asyncio.run(c.tick())                     # boot: paused
    town.running = True                       # Juniper presses Unfreeze
    clock.t += 10
    asyncio.run(c.tick())
    clock.t += 600
    asyncio.run(c.tick())
    assert "stop" not in town.calls and "heartbeat" not in town.calls
    (r,) = rows(redis)
    assert (r.kind, r.start_reason, r.ended_at) == ("manual", "juniper_override", None)
    assert town.correlation == r.session_id
    town.running = False           # Juniper pressed Freeze / the idle stop fired
    clock.t += 30
    asyncio.run(c.tick())
    (r,) = rows(redis)
    assert r.stop_reason == "town_paused" and r.ended_at == clock.t
    assert town.correlation is None
    assert ts.seconds_today(rows(redis), clock.t, kind="orion") == 0.0


def test_start_unpauses_the_town_and_opens_the_act() -> None:
    c, redis, town, clock, acts = make(at(2026, 10, 30, 9))
    asyncio.run(c.tick())
    out = asyncio.run(c.start(req()))
    assert out.state == "started" and out.session_id == session_id_for(req().dispatch_id)
    assert town.calls == ["resume"] and town.correlation == out.session_id
    (r,) = rows(redis)
    assert (r.kind, r.start_reason, r.dispatch_id) == ("orion", "want_funded", req().dispatch_id)
    assert [e.event_kind for e in acts] == ["act_opened"]
    assert acts[0].act.actor == "aitown.session" and acts[0].act.verb == "play"
    assert acts[0].correlation_id == out.session_id
    # A replayed dispatch answers with the same session, no second resume.
    assert asyncio.run(c.start(req())).session_id == out.session_id and town.calls == ["resume"]
    # The session heartbeats the town while open.
    clock.t += 10
    asyncio.run(c.tick())
    assert town.calls[-1] == "heartbeat" and c.heartbeat_wanted


def test_session_cap_pauses_the_town_and_closes_the_act() -> None:
    c, redis, town, clock, acts = make(at(2026, 10, 30, 9))
    asyncio.run(c.tick())
    asyncio.run(c.start(req()))
    c.note_utterance()
    clock.t += 1799
    asyncio.run(c.tick())
    assert town.running
    clock.t += 1
    asyncio.run(c.tick())
    assert not town.running and town.calls[-1] == "stop" and town.correlation is None
    (r,) = rows(redis)
    assert (r.stop_reason, r.ended_at - r.started_at, r.utterances) == ("session_cap", 1800.0, 1)
    closed = acts[-1]
    assert closed.event_kind == "act_closed" and closed.act.outcome.value == 1.0
    # Paused after the session: no manual session is invented from our own stop.
    clock.t += 10
    asyncio.run(c.tick())
    assert len(rows(redis)) == 1 and not c.heartbeat_wanted


def test_a_silent_session_closes_as_not_produced() -> None:
    c, _, _, clock, acts = make(at(2026, 10, 30, 9))
    asyncio.run(c.tick())
    asyncio.run(c.start(req()))
    clock.t += 1800
    asyncio.run(c.tick())
    assert acts[-1].act.outcome.value == 0.0


def test_daily_cap_pauses_mid_session_and_refuses_until_local_midnight() -> None:
    cfg = TownConfig(min_gap_sec=0.0)
    c, redis, town, clock, _ = make(at(2026, 10, 30, 20), cfg=cfg)
    # 1 h 50 min already spent today.
    asyncio.run(ts.save_row(redis, AITownSessionRowV1(
        session_id="earlier", kind="orion", started_at=at(2026, 10, 30, 9), last_seen_at=at(2026, 10, 30, 10, 50),
        ended_at=at(2026, 10, 30, 10, 50), start_reason="want_funded", stop_reason="session_cap")))
    asyncio.run(c.tick())
    assert asyncio.run(c.start(req())).state == "started"
    clock.t += 599
    asyncio.run(c.tick())
    assert town.running
    clock.t += 1
    asyncio.run(c.tick())
    assert not town.running
    assert {r.session_id: r.stop_reason for r in rows(redis)}[session_id_for(req().dispatch_id)] == "daily_cap"
    refused = asyncio.run(c.start(req("d:2")))
    assert refused.state == "refused" and refused.refusal == "daily_cap"
    assert refused.retry_after_sec == pytest.approx(ts.next_local_midnight(clock.t) - clock.t)
    # After local midnight the new day's cap is open again.
    clock.t = at(2026, 10, 31, 0, 0, 5)
    assert asyncio.run(c.start(req("d:3"))).state == "started"


def test_a_session_crossing_midnight_is_not_cut_by_yesterdays_total() -> None:
    cfg = TownConfig(min_gap_sec=0.0)
    c, redis, town, clock, _ = make(at(2026, 10, 30, 23, 50), cfg=cfg)
    asyncio.run(ts.save_row(redis, AITownSessionRowV1(
        session_id="earlier", kind="orion", started_at=at(2026, 10, 30, 9), last_seen_at=at(2026, 10, 30, 10, 40),
        ended_at=at(2026, 10, 30, 10, 40), start_reason="want_funded", stop_reason="session_cap")))
    asyncio.run(c.tick())
    assert asyncio.run(c.start(req())).state == "started"   # 1 h 40 min used: 20 min left today
    for _ in range(9):                                       # to 00:05 on the 31st, 10 s ticks every 100 s
        clock.t += 100
        asyncio.run(c.tick())
    # 23:50 -> 00:05: 10 min yesterday (1 h 50 min total, under the cap), 5 min today.
    assert town.running and c.heartbeat_wanted


def test_juniper_freezing_an_orion_session_closes_it_town_paused() -> None:
    c, redis, town, clock, acts = make(at(2026, 10, 30, 9))
    asyncio.run(c.tick())
    asyncio.run(c.start(req()))
    town.running = False
    clock.t += 60
    asyncio.run(c.tick())
    (r,) = rows(redis)
    assert r.stop_reason == "town_paused" and "stop" not in town.calls
    assert acts[-1].event_kind == "act_closed"


def test_refusals_change_nothing() -> None:
    c, redis, town, clock, acts = make(at(2026, 10, 30, 9))
    asyncio.run(c.tick())
    town.running = True                                     # Juniper is in town
    asyncio.run(c.tick())
    out = asyncio.run(c.start(req()))
    assert (out.state, out.refusal) == ("refused", "manual_session_open")
    assert "resume" not in town.calls and acts == []
    off, _, town2, _, _ = make(at(2026, 10, 30, 9), cfg=TownConfig(enabled=False))
    asyncio.run(off.tick())
    assert asyncio.run(off.start(req())).refusal == "disabled" and town2.calls == []


def test_an_unreadable_ledger_never_starts_an_uncounted_session() -> None:
    c, redis, town, clock, _ = make(at(2026, 10, 30, 9))
    asyncio.run(c.tick())
    redis.fail = True
    out = asyncio.run(c.start(req()))
    assert (out.state, out.refusal) == ("refused", "ledger_unknown") and town.calls == []


def test_turning_the_switch_off_closes_an_open_session() -> None:
    c, redis, town, clock, _ = make(at(2026, 10, 30, 9))
    asyncio.run(c.tick())
    asyncio.run(c.start(req()))
    c.cfg.enabled = False
    clock.t += 10
    asyncio.run(c.tick())
    (r,) = rows(redis)
    assert r.stop_reason == "disabled" and not town.running


def test_restart_closes_the_open_session_at_its_last_tick_and_pauses() -> None:
    c, redis, town, clock, acts = make(at(2026, 10, 30, 9))
    asyncio.run(c.tick())
    asyncio.run(c.start(req()))
    clock.t += 100
    asyncio.run(c.tick())                                   # last_seen written (first write after start)
    last = {r.session_id: r for r in rows(redis)}[session_id_for(req().dispatch_id)].last_seen_at
    # A new process on the same ledger, the town still running.
    c2 = TownSessionController(TownConfig(), redis_getter=lambda: redis, world_id_getter=lambda: "w1",
                               town=town, act_publisher=c._publish_act, clock=Clock(clock.t + 600))
    asyncio.run(c2.tick())
    (r,) = rows(redis)
    assert (r.stop_reason, r.ended_at) == ("embodiment_restart", last)
    assert town.calls[-1] == "stop" and not town.running
    assert acts[-1].event_kind == "act_closed"


def test_rpc_handle_answers_only_on_the_reply_prefix() -> None:
    c, _, _, _, _ = make(at(2026, 10, 30, 9))
    asyncio.run(c.tick())
    sent = []

    async def publish(ch, env):
        sent.append((ch, env))

    src = ServiceRef(name="orion-embodiment")
    bad = BaseEnvelope(kind=AITOWN_SESSION_START_REQUEST_KIND, source=src, reply_to="orion:elsewhere",
                       payload=req().model_dump(mode="json"))
    asyncio.run(c.handle(bad, publish=publish, source_ref=src))
    assert sent == []
    good = bad.model_copy(update={"reply_to": f"{AITOWN_SESSION_START_REPLY_PREFIX}abc"})
    asyncio.run(c.handle(good, publish=publish, source_ref=src))
    assert sent[0][0].endswith("abc") and sent[0][1].payload["state"] == "started"


class FlakyTown(FakeTown):
    """Engine reads fail (Convex timeouts); stop can be made to fail too."""

    def __init__(self, running=False):
        super().__init__(running)
        self.read_fails = False
        self.stop_fails = False

    def engine_running(self, world_id):
        if self.read_fails:
            raise TimeoutError("convex 30 s")
        return self.running

    def stop(self):
        if self.stop_fails:
            self.calls.append("stop_failed")
            raise TimeoutError("convex 30 s")
        super().stop()


def make_flaky(t0, cfg=None):
    redis, town, clock, acts = FakeRedis(), FlakyTown(), Clock(t0), []

    async def publish(bus, event):
        acts.append(event)

    c = TownSessionController(cfg or TownConfig(), redis_getter=lambda: redis, world_id_getter=lambda: "w1",
                              town=town, act_publisher=publish, clock=clock)
    return c, redis, town, clock, acts


def test_the_cap_is_enforced_even_when_engine_reads_fail() -> None:
    c, redis, town, clock, _ = make_flaky(at(2026, 10, 30, 9))
    asyncio.run(c.tick())
    asyncio.run(c.start(req()))
    town.read_fails = True
    clock.t += 1800
    asyncio.run(c.tick())
    assert not town.running and rows(redis)[0].stop_reason == "session_cap"


def test_a_failed_stop_keeps_the_session_open_and_counting_until_it_lands() -> None:
    c, redis, town, clock, _ = make_flaky(at(2026, 10, 30, 9))
    asyncio.run(c.tick())
    asyncio.run(c.start(req()))
    town.stop_fails = True
    clock.t += 1800
    asyncio.run(c.tick())
    (r,) = rows(redis)
    assert r.ended_at is None and r.last_seen_at == clock.t and c.heartbeat_wanted
    town.stop_fails = False
    clock.t += 10
    asyncio.run(c.tick())
    (r,) = rows(redis)
    assert (r.stop_reason, r.ended_at - r.started_at) == ("session_cap", 1810.0) and not town.running


def test_a_start_never_absorbs_a_town_juniper_just_opened() -> None:
    c, redis, town, clock, acts = make(at(2026, 10, 30, 9))
    asyncio.run(c.tick())
    town.running = True                     # Unfreeze lands before the next tick
    out = asyncio.run(c.start(req()))
    assert (out.state, out.refusal) == ("refused", "manual_session_open")
    assert rows(redis) == [] and acts == [] and "resume" not in town.calls


def test_a_replayed_dispatch_after_a_restart_never_rewrites_its_row() -> None:
    c, redis, town, clock, _ = make(at(2026, 10, 30, 9))
    asyncio.run(c.tick())
    asyncio.run(c.start(req()))
    clock.t += 1800
    asyncio.run(c.tick())
    c2 = TownSessionController(TownConfig(), redis_getter=lambda: redis, world_id_getter=lambda: "w1",
                               town=town, act_publisher=c._publish_act, clock=Clock(clock.t + 7200))
    asyncio.run(c2.tick())
    out = asyncio.run(c2.start(req()))
    assert (out.state, out.refusal) == ("refused", "already_ran")
    (r,) = rows(redis)
    assert r.stop_reason == "session_cap"


def test_session_open_refusal_holds_for_the_rest_of_the_session() -> None:
    c, _, _, clock, _ = make(at(2026, 10, 30, 9))
    asyncio.run(c.tick())
    asyncio.run(c.start(req()))
    clock.t += 600
    out = asyncio.run(c.start(req("d:2")))
    assert out.refusal == "session_open" and out.retry_after_sec == 1200.0


def test_rollback_boot_does_not_pause_the_town() -> None:
    """Sessions off + legacy heartbeat: a stoppedByDeveloper world could never be woken again."""
    c, redis, town, clock, _ = make(at(2026, 10, 30, 9), running=True, cfg=TownConfig(enabled=False))
    asyncio.run(c.tick())
    assert "stop" not in town.calls and town.running
