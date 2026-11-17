"""The Family C run-cap override on AI Town starts (design 2026-11-16, Experiment 1).

Pins:
- a valid ``orion:physiology:run_cap:aitown.session`` key refuses a start with ``run_cap_override``, its
  ``retry_after_sec`` no more than the key's TTL; the town is not resumed and no ledger row is written;
- a key with no TTL, a value not allowed, a ``set_at`` older than ``max_hold_s``, a family not in the YAML,
  or a malformed value is ignored (counted), never obeyed;
- the override only gates a START: an open session is never stopped by it;
- TTL expiry releases the hold: a dead writer cannot hold the town;
- the race: the key is read under the start lock before the town is resumed; while a start is inside the
  gate ``start_in_flight`` is true (what the experiment's post-set check reads through /health);
- the self-start timer places no shadow bid while a hold is on (an experiment's hold is not hunger);
- /health's ``start_eligible_at`` is the controller's own answer: the gap's end, the next local midnight
  after the daily cap, None while a session is open or the self-monitor gate holds.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from app import physiology_shadow
from app.town_self_start import TownSelfStart
from app.town_sessions import TownConfig, TownSessionController
from orion.embodiment import town_sessions as ts
from orion.physiology.run_cap_override import load_bounds
from orion.schemas.aitown_session import AITownSessionRowV1
from orion.schemas.physiology_run_cap import RunCapOverrideV1, run_cap_key

import pytest

DEN = ZoneInfo("America/Denver")
REPO = Path(__file__).resolve().parents[3]
BOUNDS = load_bounds(REPO / "config" / "physiology" / "run_cap_overrides.yaml")
KEY = run_cap_key("aitown.session")


def at(y, m, d, hh, mm=0, ss=0) -> float:
    return datetime(y, m, d, hh, mm, ss, tzinfo=DEN).timestamp()


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


class TtlRedis:
    """Hash + string keys with Redis TTL semantics on the shared clock."""

    def __init__(self, clock: Clock):
        self.clock = clock
        self.h: dict = {}
        self.kv: dict = {}
        self.exp: dict = {}
        self.on_get = None

    def _live(self, key):
        e = self.exp.get(key)
        if e is not None and self.clock.t >= e:
            self.kv.pop(key, None)
            self.exp.pop(key, None)
        return key in self.kv

    async def hgetall(self, key):
        return dict(self.h.get(key, {}))

    async def hset(self, key, field, value):
        self.h.setdefault(key, {})[field] = value

    async def hdel(self, key, *fields):
        for f in fields:
            self.h.get(key, {}).pop(f, None)

    async def set(self, key, value, ex=None):
        self.kv[key] = value
        if ex is None:
            self.exp.pop(key, None)
        else:
            self.exp[key] = self.clock.t + ex

    async def get(self, key):
        if self.on_get is not None:
            self.on_get(key)
        return self.kv.get(key) if self._live(key) else None

    async def ttl(self, key):
        if not self._live(key):
            return -2
        e = self.exp.get(key)
        return -1 if e is None else max(1, int(e - self.clock.t))


class FakeTown:
    def __init__(self):
        self.running = False
        self.calls: list = []

    def engine_running(self, world_id):
        return self.running

    def resume(self):
        self.calls.append("resume")
        self.running = True

    def stop(self):
        self.calls.append("stop")
        self.running = False

    def heartbeat(self, world_id):
        pass

    def set_correlation(self, value):
        pass


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


def make(t0, *, bounds=BOUNDS):
    clock = Clock(t0)
    redis, town = TtlRedis(clock), FakeTown()

    async def publish(bus, event):
        pass

    c = TownSessionController(TownConfig(), redis_getter=lambda: redis, world_id_getter=lambda: "w1",
                              town=town, act_publisher=publish, clock=clock, run_cap_bounds=bounds)
    asyncio.run(c.tick())   # boot (recover)
    return c, redis, town, clock


def override(clock: Clock, *, value=True, set_ago=0.0, family="aitown.session", knob="hold_starts", phase="block"):
    set_at = datetime.fromtimestamp(clock.t - set_ago, tz=timezone.utc)
    return RunCapOverrideV1(family=family, knob=knob, value=value, experiment_id="exp-1", block_index=3,
                            phase=phase, set_at=set_at, expires_at=set_at + timedelta(seconds=90)).model_dump_json()


def put(redis: TtlRedis, value: str, ex=90):
    asyncio.run(redis.set(KEY, value, ex=ex))


def start(c: TownSessionController, act="aitown_self_start:t1"):
    return asyncio.run(c.start_session(act))


# --- the gate ---------------------------------------------------------------------------------------


def test_the_shipped_yaml_holds_town_starts_for_one_block_at_most():
    b = BOUNDS["aitown.session"]
    assert (b.knob, b.allowed, b.max_hold_s) == ("hold_starts", (True,), 2700.0)


def test_a_valid_key_refuses_the_start_with_retry_no_longer_than_its_ttl():
    c, redis, town, clock = make(at(2026, 11, 17, 1))
    put(redis, override(clock), ex=90)
    clock.t += 20
    out = start(c)
    assert out.state == "refused" and out.refusal == "run_cap_override"
    assert 0 < out.retry_after_sec <= 70
    assert town.calls == [] and asyncio.run(ts.load_rows(redis)) == []
    assert c.run_cap_counts["obeyed:gate"] == 1 and c.run_cap_last["experiment_id"] == "exp-1"


@pytest.mark.parametrize("value,ex,set_ago,bounds,reason", [
    (None, None, 0.0, BOUNDS, "refused:no_ttl"),                    # no TTL: never obeyed
    ("off", 90, 0.0, BOUNDS, "refused:not_allowed"),                # value not in allowed
    (None, 90, 2701.0, BOUNDS, "refused:stale_set_at"),             # set_at past max_hold_s
    (None, 90, 0.0, {}, None),                                      # family not in the YAML
    ("garbage", 90, 0.0, BOUNDS, "refused:malformed"),
])
def test_an_unobeyable_key_is_ignored_and_the_town_starts(value, ex, set_ago, bounds, reason):
    c, redis, town, clock = make(at(2026, 11, 17, 1), bounds=bounds)
    if value == "off":
        raw = override(clock, value=False)
    elif value == "garbage":
        raw = "{not json"
    else:
        raw = override(clock, set_ago=set_ago)
    put(redis, raw, ex=ex)
    out = start(c)
    assert out.state == "started"
    assert town.calls == ["resume"]
    if reason:
        assert c.run_cap_counts[reason] == 1
    assert not any(k.startswith("obeyed") for k in c.run_cap_counts)


def test_the_override_never_stops_an_open_session():
    c, redis, town, clock = make(at(2026, 11, 17, 1))
    assert start(c).state == "started"
    put(redis, override(clock))
    for _ in range(5):
        clock.t += 60
        asyncio.run(c.tick())
    assert town.running and "stop" not in town.calls and c.orion is not None


def test_ttl_expiry_releases_the_hold_when_the_writer_dies():
    c, redis, town, clock = make(at(2026, 11, 17, 1))
    put(redis, override(clock), ex=90)
    assert start(c, "aitown_self_start:a").refusal == "run_cap_override"
    clock.t += 91   # the runner died: nobody refreshed the key
    assert start(c, "aitown_self_start:b").state == "started"


def test_a_refreshed_key_still_lets_go_after_max_hold():
    """A runner alive but stuck refreshing the TTL cannot hold the town past one block."""
    c, redis, town, clock = make(at(2026, 11, 17, 1))
    raw = override(clock)
    for _ in range(46):   # 46 min of refreshes, set_at never moves
        put(redis, raw, ex=90)
        clock.t += 60
    put(redis, raw, ex=90)
    assert start(c).state == "started"
    assert c.run_cap_counts["refused:stale_set_at"] == 1


def test_the_key_is_read_inside_the_start_lock_with_start_in_flight_set():
    """Race: the gate reads the key under the lock, before resume, and /health sees the start in flight."""
    c, redis, town, clock = make(at(2026, 11, 17, 1))
    seen = []
    redis.on_get = lambda key: seen.append((key, c.start_in_flight, c._lock.locked())) if key == KEY else None
    out = start(c)
    assert out.state == "started"
    assert seen == [(KEY, True, True)]
    assert c.start_in_flight is False


def test_a_key_set_while_a_start_waits_for_the_lock_is_obeyed():
    """The runner sets the key while another holder has the lock: the start that runs after it is held."""
    c, redis, town, clock = make(at(2026, 11, 17, 1))

    async def go():
        await c._lock.acquire()
        task = asyncio.create_task(c.start_session("aitown_self_start:x"))
        await asyncio.sleep(0)
        await redis.set(KEY, override(clock), ex=90)
        c._lock.release()
        return await task

    out = asyncio.run(go())
    assert out.refusal == "run_cap_override" and town.calls == []


# --- the self-start timer -----------------------------------------------------------------------------


def test_the_timer_places_no_bid_while_held(bids):
    c, redis, town, clock = make(at(2026, 11, 17, 1))
    timer = TownSelfStart(c, tick_sec=300.0, clock=clock)
    put(redis, override(clock), ex=90)

    async def go():
        out = await timer.tick()
        await asyncio.sleep(0)
        return out

    out = asyncio.run(go())
    assert out["result"] == "refused" and out["refusal"] == "run_cap_override"
    assert bids == [] and town.calls == []
    assert c.run_cap_counts["obeyed:precheck"] == 1
    # The hold is never longer than the key's TTL: after it lapses the next tick bids and starts.
    clock.t += 300
    out = asyncio.run(go())
    assert out["result"] == "started" and len(bids) == 1


# --- eligibility --------------------------------------------------------------------------------------


def ended(sid, start_t, end_t):
    return AITownSessionRowV1(session_id=sid, kind="orion", started_at=start_t, last_seen_at=end_t, ended_at=end_t,
                              start_reason="want_funded", stop_reason="session_cap", dispatch_id=f"d:{sid}")


def running_timer(c, clock):
    timer = TownSelfStart(c, tick_sec=300.0, clock=clock)

    class Done:
        def done(self):
            return False

    timer.task = Done()
    return timer


def test_eligible_at_is_the_gap_end_after_a_session():
    t = at(2026, 11, 17, 1)
    c, redis, town, clock = make(t)
    c._rows = [ended("a", t - 1900, t - 100)]
    timer = running_timer(c, clock)
    assert timer.start_eligible_at() == pytest.approx(t - 100 + TownConfig().min_gap_sec)
    snap = timer.snapshot()
    assert snap["start_eligible_at"] is not None and snap["start_in_flight"] is False
    assert snap["run_cap_override"]["configured"] is True


def test_eligible_at_is_local_midnight_after_the_daily_cap():
    t = at(2026, 11, 17, 6)
    c, redis, town, clock = make(t)
    c._rows = [ended(str(i), t - 18000 + i * 4000, t - 18000 + i * 4000 + 1800) for i in range(4)]
    timer = running_timer(c, clock)
    assert timer.start_eligible_at() == at(2026, 11, 18, 0)


def test_eligible_at_is_none_while_a_session_is_open_or_the_self_monitor_holds():
    t = at(2026, 11, 17, 1)
    c, redis, town, clock = make(t)
    timer = running_timer(c, clock)
    assert timer.start_eligible_at() == t
    timer.last = {"result": "refused", "refusal": "self_monitor:track_record_failing"}
    assert timer.start_eligible_at() is None
    timer.last = None
    assert start(c).state == "started"
    assert timer.start_eligible_at() is None


def test_eligible_at_ignores_the_experiments_own_hold():
    t = at(2026, 11, 17, 1)
    c, redis, town, clock = make(t)
    timer = running_timer(c, clock)
    timer._hold = (t + 80, "run_cap_override")
    assert timer.start_eligible_at() == t
    timer._hold = (t + 600, "cooldown")
    assert timer.start_eligible_at() == t + 600


def test_health_carries_the_experiment_fields():
    import os
    os.environ.setdefault("EMBODIMENT_FCC_ENV_PATH", "/tmp/nonexistent-embodiment-fcc-test.env")
    from app import main

    block = asyncio.run(main.health())["town_self_start"]
    assert {"start_eligible_at", "start_in_flight", "run_cap_override", "manual_session_open"} <= set(block)
    json.dumps(block)
