#!/usr/bin/env python3
"""Eval: Orion's own painting trigger, replayed over real history. READ-ONLY.

Runs the SHIPPED trigger (``app.painting_self_start.PaintingSelfStart``) every 60 s with the REAL
baseline schedule (``orion.reverie.baseline.schedule``, the shipped config/proposals/visual_baseline.v1.yaml)
on an in-memory checkpoint, and a model of everything downstream of the submit, built from real rows:

* arousal: ``temporal_self_event`` ``arousal_transition`` rows, piecewise constant (the schedule's own
  arousal deferral, R3 reader 4);
* the thermal gate: athena's ``cabinet_temp_c`` readings (``orion_biometrics_summary``), folded with the
  gate's own hysteresis (``orion.autonomy.thermal_gate.thermal_state``; a reading older than 300 s is
  unknown, which allows work, as live). A generate in a ``hot`` room is retried every 300 s;
* the single-flight claim: a real dream-hop attempt (``reverie_visual_attempt`` with ``dream_hop``) holds
  the claim from its start to its act close, and every attempt start (dream or waking) blocks the next
  claim for ``retry_sec`` (store.claim_visual_attempt ``retry_cooldown``);
* the diffusion hold (the queue): claim -> first generation, sampled in order from the real waking
  paintings of the last 30 days that had no thermal deferral;
* the run deadline: one interval (5,400 s) after submit; a run with no generation by then is abandoned
  and opens no act (as live: the act opens at the first real generation);
* a generation takes the real median act duration, then the painting is produced (resets the schedule).

Not modelled: the self-monitor gate (the painting faculty read ``reliable``, 100 %, for the whole window);
a generation failure after the GPU started (0 of the 14-day window's opened acts closed 0); the dream
hops' own reaction to our claims (their timings are the real ones).

``--fixture`` (default, CI): a scripted 3-day world. Gates: about 24 h / (5400 s + queue + generation) a day
when nothing defers; no submit while arousal reads strained; no generation in a hot room; never two
waking runs in flight; every painting came from a submitted run; every submit placed a hunger bid; a
dream attempt delays the claim.

``--interval-sec N`` (calibration only): the same replay with another interval than the shipped one.

``--live --days 14``: the real window. Prints paintings per day under the trigger next to the real
``thought.painting`` acts (dispatch + baseline), and the acceptance check: the 14-day total within
+-30 % of the real one.

    python -P services/orion-thought/evals/painting_self_start_replay.py [--live --days 14]
"""
from __future__ import annotations

import argparse
import asyncio
import bisect
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median

SVC = Path(__file__).resolve().parents[1]
ROOT = SVC.parents[1]
sys.path[:0] = [str(ROOT), str(SVC)]

from app.painting_self_start import PaintingSelfStart  # noqa: E402
from orion.autonomy.thermal_gate import DEFAULT_MAX_READING_AGE_SEC, thermal_state  # noqa: E402
from orion.reverie.baseline import load_baseline_policy, schedule  # noqa: E402
from orion.schemas.reverie_visual import VisualActivityV1  # noqa: E402

TICK = timedelta(seconds=60)
THERMAL_RETRY = timedelta(seconds=300)   # visual_steps._THERMAL_RETRY_AFTER_SEC
CONTAINER = "orion-athena-sql-db"
POLICY = load_baseline_policy()


@dataclass
class World:
    arousal: list[tuple[datetime, str]]                 # (since, level), sorted
    temps: list[tuple[datetime, float]]                 # (ts, cabinet C), sorted
    dream: list[tuple[datetime, datetime]]              # (attempt start, end), sorted
    queue_waits: list[float]                            # claim -> generation, seconds
    gen_sec: float
    _thermal: list[tuple[datetime, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        prev = "normal"
        for ts, c in self.temps:
            v = thermal_state(temp_c=c, age_sec=0.0, previous_state=prev)
            if v.state != "unknown":
                prev = v.state
            self._thermal.append((ts, prev))
        self._arousal_ts = [t for t, _ in self.arousal]
        self._thermal_ts = [t for t, _ in self._thermal]

    def arousal_at(self, t: datetime) -> str | None:
        i = bisect.bisect_right(self._arousal_ts, t) - 1
        return self.arousal[i][1] if i >= 0 else None

    def hot_at(self, t: datetime) -> bool:
        i = bisect.bisect_right(self._thermal_ts, t) - 1
        if i < 0 or (t - self._thermal[i][0]).total_seconds() > DEFAULT_MAX_READING_AGE_SEC:
            return False                                 # unknown allows work
        return self._thermal[i][1] == "hot"

    def dream_open(self, t: datetime) -> datetime | None:
        """End of a dream attempt open at t, else None."""
        for s, e in self.dream:
            if s <= t < e:
                return e
            if s > t:
                break
        return None

    def last_attempt_start_before(self, t: datetime) -> datetime | None:
        best = None
        for s, _ in self.dream:
            if s <= t:
                best = s
            else:
                break
        return best


@dataclass
class Run:
    run_id: str
    submitted: datetime
    deadline: datetime
    need_last_success: datetime | None
    claim_at: datetime | None = None
    gen_at: datetime | None = None
    produced_at: datetime | None = None
    ended: str | None = None
    thermal_retries: int = 0


@dataclass
class Result:
    produced: list[datetime]
    runs: list[Run]
    tick_reasons: Counter
    bids: int
    max_waking_in_flight: int


def replay(world: World, start: datetime, end: datetime, *, last_success: datetime | None) -> Result:
    state = {"last_success": last_success, "checkpoint": {}, "runs": [], "now": start,
             "last_waking_claim": None, "bids": 0, "produced": [], "qi": 0}

    def activity() -> VisualActivityV1:
        now = state["now"]
        active = any(r.claim_at and r.claim_at <= now and r.ended is None for r in state["runs"])
        active = active or world.dream_open(now) is not None
        ls = state["last_success"]
        return VisualActivityV1(history_status="ok", observed_at=now, last_success_at=ls,
                                last_success_chain_id=f"chain-{ls.isoformat()}" if ls else None,
                                last_success_sha256="a" * 64 if ls else None,
                                active_attempt_id="open" if active else None)

    def eligibility(act, *, now, policy, arousal_level=None):
        need, state["checkpoint"], reason = schedule(act, state["checkpoint"], now=now, policy=policy,
                                                     arousal_level=arousal_level)
        return need, reason

    async def submit(request):
        now = state["now"]
        state["runs"].append(Run(run_id=request.run_id, submitted=now,
                                 deadline=request.admission.deadline_at, need_last_success=state["last_success"]))
        return None

    class _Shadow:
        def settle(self, real, **_):
            pass

    def shadow(_req):
        state["bids"] += 1
        return _Shadow()

    trigger = PaintingSelfStart(load_activity=activity, baseline_eligibility=eligibility,
                                arousal_level=lambda: world.arousal_at(state["now"]), submit=submit,
                                shadow_bid=shadow, policy_loader=lambda: POLICY)

    def advance(now: datetime) -> None:
        for r in state["runs"]:
            if r.ended is not None:
                continue
            if r.claim_at is None:
                if now >= r.deadline:
                    r.ended = "deadline_unclaimed"
                    continue
                if world.dream_open(now) is not None:
                    continue
                if any(o is not r and o.claim_at and o.ended is None for o in state["runs"]):
                    continue
                starts = [s for s in (world.last_attempt_start_before(now), state["last_waking_claim"]) if s]
                if starts and now < max(starts) + timedelta(seconds=POLICY.retry_sec):
                    continue                                 # retry_cooldown
                if state["last_success"] != r.need_last_success:
                    r.ended = "already_satisfied"
                    continue
                r.claim_at = now
                state["last_waking_claim"] = now
                wait = world.queue_waits[state["qi"] % len(world.queue_waits)]
                state["qi"] += 1
                r.gen_at = now + timedelta(seconds=wait)
            if r.produced_at is None and r.gen_at is not None and now >= r.gen_at:
                if now >= r.deadline:
                    r.ended = "deadline_abandoned"
                    continue
                if world.hot_at(r.gen_at):
                    r.thermal_retries += 1
                    r.gen_at = r.gen_at + THERMAL_RETRY
                    continue
                r.produced_at = r.gen_at + timedelta(seconds=world.gen_sec)
            if r.produced_at is not None and now >= r.produced_at:
                r.ended = "produced"
                state["last_success"] = r.produced_at
                state["produced"].append(r.produced_at)

    async def run() -> Counter:
        reasons: Counter = Counter()
        max_in_flight = 0
        while state["now"] <= end:
            now = state["now"]
            advance(now)
            reasons[(await trigger.tick(now)).split(":")[0]] += 1
            max_in_flight = max(max_in_flight, sum(1 for r in state["runs"] if r.claim_at and r.ended is None))
            state["now"] = now + TICK
        for _ in range(120):                                 # let the last runs finish
            state["now"] += TICK
            advance(state["now"])
        state["max_in_flight"] = max_in_flight
        return reasons

    reasons = asyncio.run(run())
    return Result(produced=[p for p in state["produced"] if p <= end], runs=state["runs"], tick_reasons=reasons,
                  bids=state["bids"], max_waking_in_flight=state["max_in_flight"])


def per_day(times: list[datetime]) -> dict[str, int]:
    c = Counter(t.strftime("%Y-%m-%d") for t in times)
    return dict(sorted(c.items()))


def summary(res: Result, days: float) -> dict:
    ends = Counter(r.ended or "in_flight" for r in res.runs)
    return {"paintings": len(res.produced), "per_day_mean": round(len(res.produced) / days, 2),
            "per_day": per_day(res.produced), "runs_submitted": len(res.runs), "hunger_bids": res.bids,
            "run_ends": dict(ends), "thermal_retries": sum(r.thermal_retries for r in res.runs),
            "tick_results": dict(res.tick_reasons), "max_waking_runs_in_flight": res.max_waking_in_flight}


# --- fixture -----------------------------------------------------------------------------------------
T0 = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)


def fixture_world() -> World:
    temps = []
    t = T0
    while t < T0 + timedelta(days=3):
        hour = (t - T0).total_seconds() / 3600.0
        c = 33.0 if 36 <= hour < 40 else 27.0              # day 2, 12:00-16:00: hot room
        temps.append((t, c))
        t += timedelta(seconds=60)
    arousal = [(T0, "idle"), (T0 + timedelta(hours=60), "strained"), (T0 + timedelta(hours=66), "idle")]
    dream = [(T0 + timedelta(hours=3), T0 + timedelta(hours=3, minutes=20))]
    return World(arousal=arousal, temps=temps, dream=dream, queue_waits=[456.0], gen_sec=61.0)


def fixture_checks() -> list[tuple[str, bool, str]]:
    world = fixture_world()
    end = T0 + timedelta(days=3) - TICK
    res = replay(world, T0, end, last_success=T0 - timedelta(seconds=POLICY.interval_sec))
    s = summary(res, 3.0)
    checks = []
    day1 = [p for p in res.produced if p < T0 + timedelta(days=1)]
    cycle = POLICY.interval_sec + 456.0 + 61.0
    expect = 86400.0 / cycle
    checks.append(("an ungated day paints about 24 h / (interval + queue + generation)",
                   abs(len(day1) - expect) <= 1.5, f"day1={len(day1)} expected~{expect:.1f}"))
    strained = [r for r in res.runs if T0 + timedelta(hours=60) <= r.submitted < T0 + timedelta(hours=66)]
    checks.append(("no submit while arousal reads strained", not strained, f"submits={len(strained)}"))
    hot = [p for p in res.produced if T0 + timedelta(hours=36) <= p - timedelta(seconds=61) < T0 + timedelta(hours=40)]
    checks.append(("no generation in a hot room", not hot, f"in_hot={len(hot)} retries={s['thermal_retries']}"))
    checks.append(("the painting comes back after the hot spell",
                   any(T0 + timedelta(hours=40) <= p < T0 + timedelta(hours=43) for p in res.produced), ""))
    checks.append(("never two waking runs in flight", res.max_waking_in_flight <= 1, f"max={res.max_waking_in_flight}"))
    checks.append(("every painting came from a submitted run",
                   len(res.produced) == sum(1 for r in res.runs if r.ended == "produced"), ""))
    checks.append(("every submit placed one hunger bid", res.bids == len(res.runs), f"bids={res.bids} runs={len(res.runs)}"))
    in_dream = [r for r in res.runs if r.claim_at and T0 + timedelta(hours=3) <= r.claim_at < T0 + timedelta(hours=3, minutes=20)]
    checks.append(("a dream attempt holds the claim", not in_dream, ""))
    checks.append(("about 9-16 a day over the scripted world", 9 <= s["per_day_mean"] <= 16, f"mean={s['per_day_mean']}"))
    return checks


# --- live --------------------------------------------------------------------------------------------
def _psql(sql: str) -> list[list[str]]:
    out = subprocess.run(["docker", "exec", CONTAINER, "psql", "-U", "postgres", "-d", "conjourney", "-AtF", "\t",
                          "-c", sql], check=True, capture_output=True, text=True).stdout
    return [line.split("\t") for line in out.splitlines() if line]


def _t(s: str) -> datetime:
    return datetime.fromisoformat(s.replace(" ", "T")).astimezone(timezone.utc)


def live(days: float) -> dict:
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=days)
    s_, e_ = f"{start:%Y-%m-%d %H:%M:%S}+00", f"{end:%Y-%m-%d %H:%M:%S}+00"
    temps = [(_t(r[0]), float(r[1])) for r in _psql(
        "SELECT timestamp, (measurements->>'cabinet_temp_c')::float FROM orion_biometrics_summary "
        f"WHERE node = 'athena' AND timestamp >= '{start - timedelta(hours=2):%Y-%m-%d %H:%M:%S}' "
        "AND measurements ? 'cabinet_temp_c' ORDER BY timestamp") if r[1]]
    arousal = [(_t(r[0]), r[1]) for r in _psql(
        "SELECT occurred_at, payload_json->>'to' FROM temporal_self_event WHERE source_kind = 'arousal_transition' "
        f"AND occurred_at >= '{start - timedelta(days=2):%Y-%m-%d %H:%M:%S}+00' ORDER BY occurred_at")]
    dream = [(_t(r[0]), _t(r[1]) if r[1] else _t(r[0]) + timedelta(seconds=120)) for r in _psql(
        "SELECT v.started_at, a.closed_at FROM reverie_visual_attempt v LEFT JOIN grammar_acts a "
        "ON a.object_ref = 'durable_run:' || (SELECT r.run_id FROM durable_admission_runs r "
        "  WHERE r.request->'brief'->'visual_request'->>'dispatch_id' = v.dispatch_id LIMIT 1) "
        f"WHERE v.stage_json ? 'dream_hop' AND v.started_at >= '{start - timedelta(hours=3):%Y-%m-%d %H:%M:%S}+00' "
        "ORDER BY v.started_at")]
    waits = [float(r[0]) for r in _psql(
        "SELECT extract(epoch FROM a.opened_at - v.started_at) FROM grammar_acts a JOIN reverie_visual_attempt v "
        "ON a.test_before = 'visual_dispatch:' || v.dispatch_id WHERE a.actor = 'thought.painting' "
        f"AND a.opened_at >= '{end - timedelta(days=30):%Y-%m-%d %H:%M:%S}+00' AND NOT EXISTS (SELECT 1 FROM "
        "jsonb_array_elements(COALESCE(v.stage_json->'deferrals', '[]'::jsonb)) d WHERE d->>'reason' = 'thermal_refused') "
        "ORDER BY a.opened_at")]
    gen = [float(r[0]) for r in _psql(
        "SELECT extract(epoch FROM closed_at - opened_at) FROM grammar_acts WHERE actor = 'thought.painting' "
        f"AND closed_at IS NOT NULL AND opened_at >= '{end - timedelta(days=30):%Y-%m-%d %H:%M:%S}+00'")]
    real = [_t(r[0]) for r in _psql(
        f"SELECT opened_at FROM grammar_acts WHERE actor = 'thought.painting' AND opened_at >= '{s_}' "
        f"AND opened_at < '{e_}' ORDER BY opened_at")]
    real_split = _psql(
        "SELECT (v.need_id IS NOT NULL) AS baseline, count(*) FROM grammar_acts a JOIN reverie_visual_attempt v "
        "ON a.test_before = 'visual_dispatch:' || v.dispatch_id WHERE a.actor = 'thought.painting' "
        f"AND a.opened_at >= '{s_}' AND a.opened_at < '{e_}' GROUP BY 1")
    last = _psql(f"SELECT max(closed_at) FROM grammar_acts WHERE actor = 'thought.painting' AND closed_at < '{s_}'")
    last_success = _t(last[0][0]) if last and last[0] and last[0][0] else None
    world = World(arousal=arousal, temps=temps, dream=dream, queue_waits=waits or [456.0],
                  gen_sec=median(gen) if gen else 61.0)
    res = replay(world, start, end, last_success=last_success)
    new = summary(res, days)
    real_n = len(real)
    ratio = len(res.produced) / real_n if real_n else None
    strained_h = 0.0
    for (t, lvl), nxt in zip(arousal, arousal[1:] + [(end, None)]):
        a, b = max(t, start), min(nxt[0], end)
        if lvl == "strained" and b > a:
            strained_h += (b - a).total_seconds() / 3600.0
    return {
        "window": [start.isoformat(), end.isoformat()],
        "inputs": {"cabinet_readings": len(temps), "arousal_transitions": len(arousal),
                   "arousal_strained_hours_in_window": round(strained_h, 1), "dream_attempts": len(dream),
                   "queue_wait_samples": len(waits), "queue_wait_median_sec": round(median(waits), 1) if waits else None,
                   "generation_median_sec": round(world.gen_sec, 1),
                   "hot_minutes": sum(1 for t, _ in temps if t >= start and world.hot_at(t))},
        "policy": {"interval_sec": POLICY.interval_sec, "retry_sec": POLICY.retry_sec},
        "new_trigger": new,
        "real": {"painting_acts": real_n, "per_day_mean": round(real_n / days, 2), "per_day": per_day(real),
                 "by_path": {("baseline" if r[0] == "t" else "allocator"): int(r[1]) for r in real_split}},
        "acceptance": {"ratio_new_to_real": round(ratio, 3) if ratio is not None else None,
                       "within_30pct": ratio is not None and 0.7 <= ratio <= 1.3},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--days", type=float, default=14.0)
    ap.add_argument("--interval-sec", type=float, default=None,
                    help="calibration only: replay another baseline interval than the shipped one")
    args = ap.parse_args()
    if args.interval_sec is not None:
        global POLICY
        POLICY = POLICY.model_copy(update={"interval_sec": float(args.interval_sec)})
    if args.live:
        out = live(args.days)
        print(json.dumps(out, indent=2, default=str))
        return 0 if out["acceptance"]["within_30pct"] else 1
    checks = fixture_checks()
    for name, ok, why in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {why}")
    print(json.dumps({"eval": "painting_self_start_replay", "passed": sum(ok for _, ok, _ in checks),
                      "total": len(checks)}))
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
