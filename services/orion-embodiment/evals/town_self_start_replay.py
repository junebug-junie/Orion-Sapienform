"""Replay: embodiment's AI Town session timer (ladder retirement PR 3) against the real session history.

    python -P services/orion-embodiment/evals/town_self_start_replay.py            # fixture (CI)
    python -P services/orion-embodiment/evals/town_self_start_replay.py --live --days 14
    python -P services/orion-embodiment/evals/town_self_start_replay.py --live --days 14 --tick-sec 3600

What runs: the SHIPPED ``TownSelfStart.tick`` (app/town_self_start.py) every ``--tick-sec`` (300 s,
the shipped default) and the SHIPPED ``TownSessionController`` (app/town_sessions.py) tick every 10 s,
over a simulated clock, against a fake town and a fake Redis ledger. Every cap is the controller's
own: 1800 s per session, 7200 s per America/Denver day, 3600 s gap, one session at a time, Juniper's
manual session never absorbed, plus the timer's refusal hold. Shadow bids are counted, never sent.
Live ordering: the tick reads its clock first; every ledger read happens after a real read lag
(0.3-2.4 s, deterministic), and the clock moves during it.

The world (live mode) is the real record of what the controller could not decide for itself:

- Juniper's manual sessions (the ``manual`` rows in Redis ``orion:aitown:sessions``): the town runs, so
  the timer is refused ``manual_session_open`` and the ledger counts nothing for Orion;
- embodiment restarts (every ``embodiment_restart`` stop in the ledger): the process restarts at
  that instant, closes an open session, and the timer's first tick waits 60 s;
- the controller unreachable (dispatch's ``rpc_error`` replies): the town reads fail for 60 s.

The real side is the ``orion`` rows in the same ledger: every one was a dispatch-started session.

What this proves, and what it does not. The caps are the only gates the controller has, and the
timer asks every tick, so this is the timer's rate under those caps: sessions per local day against
the real range, and where in the day they fall. The real sessions were also paced by the allocator's
information floor (``allocator:below_information_floor`` on most frames), which this PR retires on
purpose; the replay shows what that changes (start hours). Not modelled: the self-monitor gate (the
faculty reads ``unknown``, which admits), Convex latency, restarts the ledger did not record. Before
the controller existed (2026-10-30) the town ran all day with no sessions, so a 14-day window only
replays from the controller's first session.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

SVC = Path(__file__).resolve().parents[1]
REPO = SVC.parents[1]
for p in (str(REPO), str(SVC)):   # SVC first: the service's own app/ package
    if p not in sys.path:
        sys.path.insert(0, p)

from app import physiology_shadow  # noqa: E402
from app.town_self_start import TownSelfStart  # noqa: E402
from app.town_sessions import TownConfig, TownSessionController  # noqa: E402
from orion.embodiment import town_sessions as ts  # noqa: E402

DEN = ZoneInfo("America/Denver")
CONTROLLER_TICK_SEC = 10.0
LAGS = (0.3, 1.1, 2.4, 0.7, 1.8)


@dataclass
class World:
    manual: list[tuple[float, float]] = field(default_factory=list)   # Juniper's town windows
    restarts: list[float] = field(default_factory=list)              # embodiment process restarts
    unreachable: list[float] = field(default_factory=list)           # controller unreachable (60 s each)

    def manual_at(self, t: float) -> bool:
        return any(a <= t < b for a, b in self.manual)

    def unreachable_at(self, t: float) -> bool:
        return any(a <= t < a + 60.0 for a in self.unreachable)


class Clock:
    def __init__(self, t: float):
        self.t = float(t)
        self._i = 0

    def __call__(self) -> float:
        return self.t

    def lag(self) -> None:
        self.t += LAGS[self._i % len(LAGS)]
        self._i += 1


class Redis:
    def __init__(self, clock: Clock):
        self.h: dict = {}
        self.kv: dict = {}
        self.clock = clock

    async def hgetall(self, key):
        self.clock.lag()   # the read happens after the caller's clock
        return dict(self.h.get(key, {}))

    async def hset(self, key, f, v):
        self.h.setdefault(key, {})[f] = v

    async def hdel(self, key, *fs):
        for f in fs:
            self.h.get(key, {}).pop(f, None)

    async def set(self, key, v):
        self.kv[key] = v

    async def get(self, key):
        return self.kv.get(key)


class Town:
    def __init__(self, world: World, clock: Clock):
        self.world, self.clock = world, clock
        self.orion_running = False
        self.resumes_during_manual = 0

    def engine_running(self, world_id):
        if self.world.unreachable_at(self.clock.t):
            raise TimeoutError("controller unreachable (replayed rpc_error)")
        return self.orion_running or self.world.manual_at(self.clock.t)

    def resume(self):
        if self.world.manual_at(self.clock.t):
            self.resumes_during_manual += 1
        self.orion_running = True

    def stop(self):
        self.orion_running = False

    def heartbeat(self, world_id):
        pass

    def set_correlation(self, value):
        pass


async def replay(world: World, start: float, end: float, *, tick_sec: float = 300.0) -> dict:
    async def abid(want):
        return None

    async def arelease(want_id):
        return None

    physiology_shadow.abid = abid
    physiology_shadow.arelease = arelease
    clock = Clock(start)
    redis, town = Redis(clock), Town(world, clock)

    async def no_publish(bus, event):
        return None

    def boot():
        c = TownSessionController(TownConfig(), redis_getter=lambda: redis, world_id_getter=lambda: "w",
                                  town=town, act_publisher=no_publish, clock=clock)
        return c, TownSelfStart(c, tick_sec=tick_sec, clock=clock)

    controller, timer = boot()
    next_ctrl, next_timer = start, start + 60.0
    restarts = sorted(r for r in world.restarts if start < r < end)
    totals: Counter = Counter()
    refusals: Counter = Counter()
    bids = 0
    while clock.t < end:
        if restarts and clock.t >= restarts[0]:
            restarts.pop(0)
            totals.update(timer.results)
            refusals.update(timer.refusals)
            bids += timer.bids
            controller, timer = boot()
            next_ctrl, next_timer = clock.t, clock.t + 60.0
        if clock.t >= next_ctrl:
            await controller.tick()
            next_ctrl = clock.t + CONTROLLER_TICK_SEC
        if clock.t >= next_timer:
            out = await timer.tick()
            next_timer = clock.t + timer.next_delay(out)
        clock.t = max(clock.t, min(next_ctrl, next_timer))
    totals.update(timer.results)
    refusals.update(timer.refusals)
    bids += timer.bids
    rows = await ts.load_rows(redis)
    orion = sorted((r for r in rows if r.kind == "orion"), key=lambda r: r.started_at)
    return {"sessions": orion, "results": dict(totals), "refusals": dict(refusals), "bids": bids,
            "resumes_during_manual": town.resumes_during_manual, "tick_sec": tick_sec}


def _day(t: float) -> str:
    return datetime.fromtimestamp(t, tz=DEN).strftime("%Y-%m-%d")


def _hm(t: float) -> str:
    return datetime.fromtimestamp(t, tz=DEN).strftime("%H:%M")


def checks(out: dict, world: World, start: float, end: float) -> list[tuple[str, bool, str]]:
    s = out["sessions"]
    per_day: dict = defaultdict(float)
    for r in s:
        per_day[_day(r.started_at)] += (r.ended_at or end) - r.started_at
    gaps = [b.started_at - (a.ended_at or end) for a, b in zip(s, s[1:])]
    lengths = [(r.ended_at or end) - r.started_at for r in s if r.ended_at]
    return [
        ("starts at least one session", bool(s), f"{len(s)} sessions"),
        ("never over the daily cap (7200 s per local day, +1 controller tick)",
         all(v <= 7200 + CONTROLLER_TICK_SEC + 3 for v in per_day.values()), f"max {max(per_day.values(), default=0):.0f} s"),
        ("never two inside the 3600 s gap", all(g >= 3600 for g in gaps), f"min gap {min(gaps, default=0):.0f} s"),
        ("no session past the 1800 s cap (+1 controller tick)",
         all(x <= 1800 + CONTROLLER_TICK_SEC + 3 for x in lengths), f"max {max(lengths, default=0):.0f} s"),
        ("never resumes the town while Juniper's session is open", out["resumes_during_manual"] == 0,
         f"{out['resumes_during_manual']}"),
        ("no session starts inside a manual window", not [r for r in s if world.manual_at(r.started_at)], ""),
        ("bids only on asks, never while held or a session runs",
         out["bids"] == sum(v for k, v in out["results"].items() if k in ("started", "refused"))
         - out["refusals"].get("self_monitor", 0), f"{out['bids']} bids"),
        ("no refusal for a read age (no unnamed or negative-age refusal)",
         not [k for k in out["refusals"] if k in ("unnamed_refusal", "future", "negative_age")], str(out["refusals"])),
    ]


def _fixture_world() -> tuple[World, float, float]:
    d0 = datetime(2026, 11, 5, 0, 0, tzinfo=DEN).timestamp()
    w = World(manual=[(d0 + 86400 + 10 * 3600, d0 + 86400 + 13 * 3600)],   # Juniper in town day 2, 10-13
              restarts=[d0 + 86400 + 3600 + 45 * 60],                        # a restart mid-session, day 2 01:45
              unreachable=[d0 + 3 * 3600 + 60 * k for k in range(10)])       # 10 min of Convex down, day 1
    return w, d0, d0 + 2 * 86400


def fixture_main() -> int:
    world, start, end = _fixture_world()
    out = replay_sync(world, start, end)
    results = checks(out, world, start, end)
    per = Counter(_day(r.started_at) for r in out["sessions"])
    results.append(("the daily cap binds on a clear day (4 x 30 min)", per[_day(start)] == 4, str(dict(per))))
    results.append(("a restart closes the open session embodiment_restart",
                    any(r.stop_reason == "embodiment_restart" for r in out["sessions"]), ""))
    results.append(("the first open window starts on the first tick (60 s after boot)",
                    out["sessions"][0].started_at - start < 120, f"{out['sessions'][0].started_at - start:.0f} s"))
    results.append(("a daily-cap refusal is held, not re-asked every tick",
                    out["refusals"].get("held", 0) > 100 and out["refusals"].get("daily_cap", 0) <= 3,
                    str(out["refusals"])))
    for name, ok, note in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {note}")
    print(f"{sum(ok for _, ok, _ in results)}/{len(results)} checks")
    return 0 if all(ok for _, ok, _ in results) else 1


def replay_sync(world, start, end, **kw) -> dict:
    return asyncio.run(replay(world, start, end, **kw))


# --- live -------------------------------------------------------------------------------------


def _psql(sql: str) -> list[list[str]]:
    out = subprocess.run(["docker", "exec", "orion-athena-sql-db", "psql", "-U", "postgres", "-d", "conjourney",
                          "-AtF", "|", "-c", sql], capture_output=True, text=True, timeout=120, check=True)
    return [line.split("|") for line in out.stdout.splitlines() if line.strip()]


def _old_dispatch_rpc_errors() -> list[float]:
    """Times the old dispatch route found the town unreachable. That route retired 2026-11-07 (ladder
    retirement PR 6) and its table is dropped by the follow-up migration: once it is gone, the replay
    runs with no unreachable windows (stated on stderr, never silently)."""
    present = _psql("select to_regclass('substrate_dispatch_results') is not null")
    if not present or present[0][0] != "t":
        print("note: substrate_dispatch_results is gone (dispatch retired); no unreachable windows modelled",
              file=sys.stderr)
        return []
    return [float(t) for (t,) in _psql(
        "select extract(epoch from created_at) from substrate_dispatch_results "
        "where result_json->'settlement'->>'kind'='aitown_session' "
        "and result_json->'settlement'->>'state'='rpc_error'")]


def live_world(days: float):
    import redis

    url = os.environ.get("ORION_BUS_URL", "redis://100.92.216.81:6379/0")
    raw = redis.Redis.from_url(url, socket_timeout=5).hgetall("orion:aitown:sessions")
    rows = ts.parse_rows(raw)
    now = datetime.now(timezone.utc).timestamp()
    orion_rows = sorted((r for r in rows if r.kind == "orion"), key=lambda r: r.started_at)
    era = orion_rows[0].started_at if orion_rows else now
    start = max(now - days * 86400, era - 60.0)
    world = World(
        manual=[(r.started_at, r.ended_at or now) for r in rows if r.kind == "manual"],
        restarts=[r.ended_at for r in rows if r.stop_reason == "embodiment_restart" and r.ended_at],
        unreachable=_old_dispatch_rpc_errors(),
    )
    real = [r for r in orion_rows if r.started_at >= start]
    return world, start, now, real


def live_main(days: float, tick_sec: float) -> int:
    world, start, end, real = live_world(days)
    out = replay_sync(world, start, end, tick_sec=tick_sec)
    new = out["sessions"]
    days_all = sorted({_day(r.started_at) for r in real} | {_day(r.started_at) for r in new})
    first, last = _day(start), _day(end)
    full = [d for d in days_all if d not in (first, last)]
    real_per = Counter(_day(r.started_at) for r in real)
    new_per = Counter(_day(r.started_at) for r in new)
    lo, hi = (min(real_per[d] for d in full), max(real_per[d] for d in full)) if full else (0, 0)
    print(f"window {datetime.fromtimestamp(start, tz=timezone.utc):%Y-%m-%d %H:%M} .. "
          f"{datetime.fromtimestamp(end, tz=timezone.utc):%Y-%m-%d %H:%M} UTC "
          f"(asked {days:g} days; the controller's first session bounds it), tick {tick_sec:.0f} s")
    print(f"sessions: new timer {len(new)}, real (dispatch) {len(real)}")
    print("local day (America/Denver)  new  real   new starts (local)                  real starts (local)")
    for d in days_all:
        tag = "" if d in full else "  (partial day)"
        ns = " ".join(_hm(r.started_at) for r in new if _day(r.started_at) == d)
        rs = " ".join(_hm(r.started_at) for r in real if _day(r.started_at) == d)
        print(f"{d}{tag:15s} {new_per[d]:>4} {real_per[d]:>5}   {ns:36s} {rs}")
    in_range = all(lo <= new_per[d] <= hi for d in full)
    sec_new = sum((r.ended_at or end) - r.started_at for r in new) / 3600
    sec_real = sum((r.ended_at or end) - r.started_at for r in real) / 3600
    print(f"full days: real range {lo}-{hi} per day; new {[new_per[d] for d in full]} -> "
          f"{'IN RANGE' if in_range else 'OUT OF RANGE'}")
    print(f"Orion town hours: new {sec_new:.1f}, real {sec_real:.1f}")
    print(f"timer results {out['results']}")
    print(f"refusals {out['refusals']}  bids {out['bids']}")
    hours_new = Counter(datetime.fromtimestamp(r.started_at, tz=DEN).hour // 6 for r in new)
    hours_real = Counter(datetime.fromtimestamp(r.started_at, tz=DEN).hour // 6 for r in real)
    bands = ["00-06", "06-12", "12-18", "18-24"]
    print("start band (local)  " + "  ".join(f"{b}" for b in bands))
    print("new                 " + "  ".join(f"{hours_new[i]:>5}" for i in range(4)))
    print("real                " + "  ".join(f"{hours_real[i]:>5}" for i in range(4)))
    results = checks(out, world, start, end)
    results.append(("sessions per full local day within the real range", in_range, f"{lo}-{hi}"))
    for name, ok, note in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {note}")
    print(json.dumps({"new": len(new), "real": len(real), "range": [lo, hi], "in_range": in_range}))
    return 0 if all(ok for _, ok, _ in results) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--days", type=float, default=14.0)
    ap.add_argument("--tick-sec", type=float, default=300.0)
    args = ap.parse_args()
    if args.live:
        return live_main(args.days, args.tick_sec)
    return fixture_main()


if __name__ == "__main__":
    raise SystemExit(main())
