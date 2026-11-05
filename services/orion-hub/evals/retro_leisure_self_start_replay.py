"""Replay: Hub's leisure timer (ladder retirement PR 1) against the real retro leisure history.

    python -P services/orion-hub/evals/retro_leisure_self_start_replay.py            # fixture (CI)
    python -P services/orion-hub/evals/retro_leisure_self_start_replay.py --live --days 14

What runs: the SHIPPED ``RetroLeisureSelfStart.tick`` (scripts/retro_leisure_self_start.py) every
300 s, calling the SHIPPED gate (``retro_leisure.should_start`` -> ``gather_gate_inputs`` ->
``leisure_block_reason``) and the runner's own re-check, over a simulated clock. The shadow bid is
counted, never sent. Live ordering: the tick reads its clock first; every gate read happens after a
real read lag (0.3-2.4 s, deterministic), and samples the world at the read time.

The world (live mode) comes from the only real record of the doer's gate over time: the old
dispatch route's start RPCs (``substrate_dispatch_results``, settlement kind ``retro_leisure``;
about one every 5-9 minutes). Each one is the runner's own verdict at that moment. The replay reads
the nearest verdict within 450 s of each read:

- ``started``, ``cooldown``, ``daily_cap`` -> every world gate clear (the ledger gates are the
  replay's own: its own starts set the cooldown and the daily cap);
- a world refusal (``arousal_strained``, ``cabinet_hot``, ``human_recent``, ``browser_connected``,
  ``retro_session_active``, ``arousal_unknown``, ...) -> that gate holds;
- no verdict within 450 s, or an ``rpc_error`` -> unobserved: the presence read fails
  (``presence_unknown``), so the replay never starts on a guess.

A session lasts the median real leisure session (``retro_play_sessions`` kind ``leisure``).

Limits, said plainly: a verdict shows only the FIRST gate that refused, so a gate hidden behind it
is not seen; the self-monitor act gate is not modelled (the faculty read ``unknown``, which admits);
a session's own effect on the world (heat) is not modelled.
"""
from __future__ import annotations

import argparse
import asyncio
import bisect
import json
import statistics
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

HUB = Path(__file__).resolve().parents[1]
REPO = HUB.parents[1]
for p in (str(REPO), str(HUB)):   # HUB first on the path: the repo root has its own scripts/ package
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts import retro_leisure as L  # noqa: E402
from scripts import retro_leisure_self_start as rs  # noqa: E402

TICK_SEC = 300.0
NEAREST_SEC = 450.0
OPEN = "open"
UNOBSERVED = "unobserved"
OPEN_VERDICTS = {"started", "cooldown", "daily_cap"}
CONFIG = SimpleNamespace(RETRO_LEISURE_ENABLED=True, HUB_RETRO_ENABLED=True,
                         RETRO_LEISURE_GAME_IDS="the-legend-of-zelda", RETRO_LEISURE_SESSION_MAX_SEC=1800,
                         RETRO_LEISURE_DAILY_MAX_SEC=5400, RETRO_LEISURE_MIN_GAP_SEC=10800)


def condition(verdict: str | None) -> str:
    if verdict is None or verdict == "rpc_error":
        return UNOBSERVED
    return OPEN if verdict in OPEN_VERDICTS else verdict


@dataclass
class World:
    """Verdicts on a timeline (epoch seconds, sorted)."""
    times: list[float]
    verdicts: list[str]
    session_sec: float = 1.0
    # Sensitivity: with carry=True a stretch with no verdict keeps the last verdict before it
    # (an rpc_error stays unobserved), instead of failing the presence read.
    carry: bool = False

    def at(self, t: float) -> str:
        if not self.times:
            return UNOBSERVED
        i = bisect.bisect_left(self.times, t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self.times) and abs(self.times[j] - t) <= NEAREST_SEC:
                if best is None or abs(self.times[j] - t) < abs(self.times[best] - t):
                    best = j
        if best is None and self.carry and i > 0:
            best = i - 1
        return condition(self.verdicts[best] if best is not None else None)


class Clock:
    def __init__(self, t: float):
        self.t = t
        self.n = 0

    def __call__(self) -> float:
        return self.t

    def lag(self) -> None:
        """A deterministic read lag of 0.3-2.4 s (the live GETs take real time)."""
        self.n += 1
        self.t += 0.3 + (self.n * 7919 % 22) / 10.0


@dataclass
class Store:
    """PresenceStore's read interface, sampling the world at the replay clock (after a read lag)."""
    world: World
    clock: Clock
    ledger_rows: list = field(default_factory=list)

    async def seconds_since_human(self):
        self.clock.lag()
        c = self.world.at(self.clock())
        if c == UNOBSERVED:
            return False, None
        return True, (60.0 if c == "human_recent" else None)

    async def browsers_connected(self):
        c = self.world.at(self.clock())
        return 1 if c == "browser_connected" else 0

    async def ledger(self):
        return [dict(r) for r in self.ledger_rows]


class Retro:
    def __init__(self, world: World, clock: Clock):
        self.world, self.clock = world, clock

    async def call(self, method, path, body=None):
        self.clock.lag()
        c = self.world.at(self.clock())
        if c == "retro_unavailable":
            return 503, None
        state = "running" if c == "retro_session_active" else None
        return 200, {"session": {"state": state} if state else None}


def _regulation(world: World, clock: Clock):
    async def read():
        clock.lag()
        c = world.at(clock())
        if c == "arousal_unknown":
            return "unknown", "stale", None
        if c == "cabinet_hot":
            return "strained", "cabinet_hot", "cabinet_hot"
        if c == "arousal_strained":
            return "strained", "strained", None
        return "idle", "idle", None
    return read


class Runner:
    """Lane B's interface over the replay world: the REAL gate, and a session of real length."""

    def __init__(self, world: World, clock: Clock, store: Store):
        self.clock, self.store, self.world = clock, store, world
        self.deps = L.LeisureDeps(store=store, retro=Retro(world, clock), regulation=_regulation(world, clock),
                                  plan=None, clock=clock)
        self.started: list[float] = []

    async def should_start(self, config):
        return await L.should_start(config, self.deps)

    async def run_leisure_session(self, config, *, act_id=None):
        reason = L.leisure_block_reason(await L.gather_gate_inputs(config, self.deps))
        if reason:
            return SimpleNamespace(started=False, refusal=reason, ended=None)
        start = self.clock()
        self.store.ledger_rows.append({"leisure_id": act_id, "started_at": start,
                                       "ended_at": start + self.world.session_sec,
                                       "seconds": self.world.session_sec})
        self.started.append(start)
        return SimpleNamespace(started=True, refusal=None, ended=None)


async def replay(world: World, start: float, end: float) -> dict:
    import scripts.physiology_shadow as ps

    bids: list[str] = []

    class _Bid:
        def __init__(self, **kw):
            bids.append(kw["want_id"])

        def place(self):
            pass

        def settle(self, real, **_):
            pass

    ps.ShadowBid = _Bid
    clock = Clock(start)
    store = Store(world, clock)
    runner = Runner(world, clock, store)
    timer = rs.RetroLeisureSelfStart(config_provider=lambda: CONFIG, pool_provider=lambda: None,
                                     runner_provider=lambda: runner, clock=clock, tick_sec=TICK_SEC)
    timer.bus = None
    rs.START_GRACE_SEC = 0.5
    results: list[dict] = []
    tick_at = start
    while tick_at < end:
        clock.t = tick_at
        out = await timer.tick()
        out["_t"] = tick_at
        results.append(out)
        # A session lasts session_sec; the follower finishes before the next tick.
        await asyncio.sleep(0)
        if timer._followers:
            await asyncio.gather(*list(timer._followers), return_exceptions=True)
        tick_at += timer.next_delay(out)
    refusals = Counter(r.get("refusal") for r in results if r["result"] == "refused")
    return {"ticks": len(results), "starts": runner.started, "bids": len(bids), "refusals": refusals,
            "results": Counter(r["result"] for r in results), "rows": results}


# --- fixture (CI) -------------------------------------------------------------------------------

def _fixture_world() -> tuple[World, float, float]:
    """Two scripted days, one verdict every 300 s:
    day 1: 00-06 strained, 06-08 open, 08-09 cabinet_hot, 09-14 open, 14-16 human_recent,
           16-20 Juniper's game (retro_session_active), 20-24 open;
    day 2: 00-02 no verdicts at all (unobserved), 02-24 open (the daily cap must bind)."""
    t0 = datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()
    plan = [(0, 6, "arousal_strained"), (6, 8, "started"), (8, 9, "cabinet_hot"), (9, 14, "cooldown"),
            (14, 16, "human_recent"), (16, 20, "retro_session_active"), (20, 24, "started"),
            (26, 48, "started")]
    times, verdicts = [], []
    for a, b, v in plan:
        t = t0 + a * 3600
        while t < t0 + b * 3600:
            times.append(t + 17.0)
            verdicts.append(v)
            t += 300
    return World(times, verdicts, session_sec=1800.0), t0, t0 + 48 * 3600


def fixture_checks(out: dict, world: World, t0: float) -> list[tuple[str, bool, str]]:
    starts = out["starts"]
    hours = [(s - t0) / 3600 for s in starts]
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    checks = []
    checks.append(("starts only where the world was open", all(world.at(s) == OPEN for s in starts), f"{hours}"))
    checks.append(("never two starts inside the 3 h cooldown", all(g >= CONFIG.RETRO_LEISURE_MIN_GAP_SEC for g in gaps),
                   f"min gap {min(gaps) if gaps else None}"))
    day2 = [s for s in starts if s >= t0 + 24 * 3600]
    used = []
    for s in day2:
        used.append(sum(min(world.session_sec, CONFIG.RETRO_LEISURE_SESSION_MAX_SEC)
                        for x in starts if s - 86400 < x < s))
    checks.append(("the rolling daily cap binds (90 min = 3 sessions of 30 min)",
                   all(u < CONFIG.RETRO_LEISURE_DAILY_MAX_SEC for u in used) and len(day2) <= 3, f"day 2 starts {len(day2)}"))
    checks.append(("no start while Juniper's game runs (16-20 h)", not any(16 <= h < 20 for h in hours), ""))
    checks.append(("no start on an unobserved stretch (day 2, 00-02 h)", not any(24 <= h < 26 for h in hours), ""))
    checks.append(("the first open window starts at once (06 h)", bool(hours) and 6 <= hours[0] < 6.1, f"{hours[:1]}"))
    checks.append(("one bid per tick that reached the gate", out["bids"] == out["ticks"] - out["results"]["session_running"],
                   f"bids {out['bids']} ticks {out['ticks']}"))
    checks.append(("refusals carry the doer's own names",
                   {"arousal_strained", "cabinet_hot", "human_recent", "retro_session_active", "cooldown",
                    "presence_unknown", "daily_cap"} <= set(out["refusals"]), f"{sorted(out['refusals'])}"))
    no_age = not any(str(r).startswith(("arousal_unknown", "stale", "future")) for r in out["refusals"])
    checks.append(("read-lag ordering: no refusal for an age (the #3134 shape)", no_age, ""))
    return checks


# --- live ---------------------------------------------------------------------------------------

def _psql(sql: str) -> list[list[str]]:
    out = subprocess.run(["docker", "exec", "orion-athena-sql-db", "psql", "-U", "postgres", "-d", "conjourney",
                          "-AtF", "\t", "-c", sql], check=True, capture_output=True, text=True).stdout
    return [line.split("\t") for line in out.splitlines() if line]


def live_world(days: float) -> tuple[World, float, float, dict]:
    end_rows = _psql("select extract(epoch from max(created_at)) from substrate_dispatch_results "
                     "where result_json->'settlement'->>'kind'='retro_leisure'")
    end = float(end_rows[0][0])
    start = end - days * 86400
    rows = _psql(
        "select extract(epoch from created_at), coalesce(result_json->'settlement'->>'refusal', "
        "result_json->'settlement'->>'state') from substrate_dispatch_results "
        f"where result_json->'settlement'->>'kind'='retro_leisure' and created_at >= to_timestamp({start}) "
        f"and created_at <= to_timestamp({end}) order by created_at")
    sessions = _psql("select extract(epoch from started_at), duration_sec from retro_play_sessions "
                     f"where kind='leisure' and started_at >= to_timestamp({start}) and started_at <= to_timestamp({end})")
    times = [float(r[0]) for r in rows]
    verdicts = [r[1] for r in rows]
    durations = [float(s[1]) for s in sessions]
    world = World(times, verdicts, session_sec=statistics.median(durations) if durations else 1.0)
    real = {"sends": len(rows), "verdicts": Counter(verdicts), "sessions": [float(s[0]) for s in sessions],
            "durations": durations}
    return world, start, end, real


def _day(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).date().isoformat()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--days", type=float, default=14.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--carry", action="store_true", help="sensitivity: an unobserved stretch keeps the last verdict")
    args = ap.parse_args()
    if not args.live:
        world, t0, t1 = _fixture_world()
        out = asyncio.run(replay(world, t0, t1))
        checks = fixture_checks(out, world, t0)
        for name, ok, note in checks:
            print(f"{'PASS' if ok else 'FAIL'}  {name}  {note}")
        passed = sum(ok for _, ok, _ in checks)
        print(f"fixture replay: {passed}/{len(checks)} checks; {len(out['starts'])} starts, {out['ticks']} ticks")
        return 0 if passed == len(checks) else 1

    world, start, end, real = live_world(args.days)
    world.carry = bool(args.carry)
    out = asyncio.run(replay(world, start, end))
    new_days = Counter(_day(s) for s in out["starts"])
    real_days = Counter(_day(s) for s in real["sessions"])
    unobserved = sum(1 for r in out["rows"] if world.at(r["_t"]) == UNOBSERVED)
    summary = {
        "window": [datetime.fromtimestamp(start, tz=timezone.utc).isoformat(),
                   datetime.fromtimestamp(end, tz=timezone.utc).isoformat()],
        "new": {"sessions": len(out["starts"]), "ticks": out["ticks"], "bids": out["bids"],
                "results": dict(out["results"]), "refusals": dict(out["refusals"].most_common()),
                "unobserved_ticks": unobserved,
                "starts": [datetime.fromtimestamp(s, tz=timezone.utc).isoformat() for s in out["starts"]]},
        "real": {"sessions": len(real["sessions"]), "sends": real["sends"],
                 "verdicts": dict(real["verdicts"].most_common()),
                 "starts": [datetime.fromtimestamp(s, tz=timezone.utc).isoformat() for s in sorted(real["sessions"])],
                 "median_session_sec": world.session_sec},
        "per_day_new_real": {d: [new_days.get(d, 0), real_days.get(d, 0)]
                             for d in sorted(set(new_days) | set(real_days))},
    }
    print(json.dumps(summary, indent=2 if args.json else None))
    real_reasons = {r for r, n in real["verdicts"].items() if r not in OPEN_VERDICTS | {"rpc_error"}}
    new_reasons = set(out["refusals"]) - {"presence_unknown", "cooldown", "daily_cap"}
    same = new_reasons <= real_reasons
    close = abs(len(out["starts"]) - len(real["sessions"])) <= max(2, round(0.5 * len(real["sessions"])))
    print(f"acceptance: sessions new {len(out['starts'])} vs real {len(real['sessions'])} "
          f"({'PASS' if close else 'FAIL'}); world refusal reasons a subset of the real ones "
          f"({'PASS' if same else 'FAIL'}: new {sorted(new_reasons)} real {sorted(real_reasons)})")
    return 0 if (close and same) else 1


if __name__ == "__main__":
    raise SystemExit(main())
