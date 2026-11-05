#!/usr/bin/env python3
"""Eval: the cooling world act on its own loop, replayed over cabinet history. READ-ONLY.

Runs the SHIPPED loop (``app.cooling_act.CoolingAct``) tick by tick (60 s) with:

* a store backed by cabinet readings (``orion_biometrics_summary``) and an in-memory episode ledger;
* the REAL market decision (``orion.physiology.choice.decide`` on a fresh body state with no prices, which is
  what this zero-lane family meets live; the family's own event history grows as it bids);
* a fake pool that starts each shed and expires it at its TTL (the pool's own caps are the loop's admission);
* the REAL settle-time scorer (``orion.autonomy.world_settlement.score_world_episode``), run once each
  episode's window closes -- the proof that sql-writer can score what the new loop writes.

``--fixture`` (default; deterministic, no database): a scripted day (cool morning, a warm afternoon in the
29.5-32 C band, a hot spell over 32 C, a critical minute at 34 C, a sensor gap). Gates: no want outside the
warm band; no want while the reflex covers (critical / unknown); at least one episode per warm hour; the
treated share is within 0.5 +- 0.35; every episode row is scoreable (``score_world_episode`` returns a score,
never None after its window); no two episodes overlap (in flight blocks); every treated episode sent exactly
one shed RPC.

``--live``: the last ``--days`` (default 14) of REAL cabinet readings, plus the 30 days before them as the
salience reference (``docker exec <container> psql``, read-only). Prints the eligible minutes, the wants
posted, funded / deferred / no-answer counts, the episodes (treated/control) the loop would have drawn, and
the scorer's verdicts; and, for comparison, the OLD path's numbers in the same window (the real episodes in
``substrate_world_action_episodes``, and the minutes with an eligible shed proposal in
``substrate_proposal_frames``, which keeps about 10 days).

    python -P services/orion-durable-runs/evals/cooling_act_replay.py [--live --days 14]
"""
from __future__ import annotations

import argparse
import asyncio
import bisect
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

SVC = Path(__file__).resolve().parents[1]
ROOT = SVC.parents[1]
sys.path[:0] = [str(ROOT), str(SVC)]

from app.cooling_act import CoolingAct  # noqa: E402
from orion.autonomy import shed_act  # noqa: E402
from orion.autonomy.cabinet_heat import read_cabinet_heat  # noqa: E402
from orion.autonomy.self_shed import HardwareWatchView  # noqa: E402
from orion.autonomy.world_settlement import score_world_episode  # noqa: E402
from orion.hardware_watch.rules import TempPoint  # noqa: E402
from orion.physiology.choice import ReservationBook, build_context, decide  # noqa: E402
from orion.physiology.families import load_families  # noqa: E402
from orion.schemas.gpu_pool import GpuPoolShedResultV1  # noqa: E402
from orion.schemas.physiology import PhysiologyConstraintReadingV1 as R, PhysiologyStateV1  # noqa: E402

TICK = timedelta(seconds=60)
SCORER_LAG = timedelta(seconds=30)      # SQL_WRITER_WORLD_SETTLEMENT_INTERVAL_SEC
CONTAINER = "orion-athena-sql-db"
FAMS, _ = load_families(ROOT / "config/physiology/families.yaml")


def _settings() -> SimpleNamespace:
    return SimpleNamespace(cooling_act_tick_sec=60.0, cooling_act_holdback_fraction=0.5,
                           hardware_watch_health_url="http://replay", shed_rise_threshold_c=0.5,
                           gpu_pool_shed_rpc_timeout_sec=5.0, shed_ttl_sec=900.0, shed_min_gap_sec=900.0,
                           shed_max_sec_per_day=3600.0, service_name="replay",
                           action_outcome_channel="orion:autonomy:action:outcome")


class ReplayStore:
    """Cabinet readings + an in-memory episode ledger. Mirrors CoolingActStore's methods."""

    def __init__(self, points: list[TempPoint]) -> None:
        self.points = sorted(points, key=lambda p: p.ts)
        self.ts = [p.ts for p in self.points]
        self.warm = shed_act.warming_minutes(self.points)
        self.warm_ts = [t for t, _ in self.warm]
        self.episodes: dict[str, dict] = {}
        self.pool: dict[str, dict] = {}

    def load_cabinet_points(self, *, since, until):
        return self.points[bisect.bisect_left(self.ts, since):bisect.bisect_right(self.ts, until)]

    def load_warming_history(self, *, now, rise_threshold_c):
        return self.warm[bisect.bisect_left(self.warm_ts, now - timedelta(days=30)):bisect.bisect_right(self.warm_ts, now)]

    def load_background_occupancy(self):
        return 1, 0          # background work is present nearly always (live: hundreds of leases a day)

    def admission_inputs(self, *, now):
        in_flight = [e for e, r in self.episodes.items() if r.get("scored_at") is None
                     and r["decided_at"] > now - timedelta(hours=6) and r["decided_at"] <= now]
        treated = [r for r in self.episodes.values() if r["arm"] == "treated" and r["decided_at"] >= now - timedelta(days=1)]
        return in_flight, treated

    def insert_episode(self, row):
        if row["episode_id"] in self.episodes:
            return False
        self.episodes[row["episode_id"]] = {**row, "settlement_state": None, "scored_at": None}
        return True

    def episode_arm(self, episode_id):
        return (self.episodes.get(episode_id) or {}).get("arm")

    def record_settlement(self, *, episode_id, shed_id, state, settlement):
        ep = self.episodes[episode_id]
        ep["settlement_state"] = state
        ep["settlement"] = {**(ep.get("settlement") or {}), **settlement}
        ep["shed_id"] = shed_id or ep.get("shed_id")

    def load_pending_settlements(self, *, limit=20):
        return [{"episode_id": e, "dispatch_kind": r["dispatch_kind"], "settlement": r["settlement"],
                 "pool_row": self.pool.get(e)} for e, r in self.episodes.items()
                if r["arm"] == "treated" and (r.get("settlement") or {}).get("state") == "shed_pending"][:limit]

    def load_effect_posteriors(self):
        return {}

    def latest_surprise(self):
        return 0.1


class ReplayMarket:
    """The real ``decide`` on a fresh, price-free body state; the family's event history grows as it bids."""

    def __init__(self, clock) -> None:
        self.book = ReservationBook()
        self.events: dict[str, float] = {}         # want_id -> the intensity of its first decision
        self.decisions: list = []
        self.families = FAMS
        self._clock = clock

    async def bid(self, want, now=None):
        now = now or self._clock()
        state = PhysiologyStateV1(tick_id=f"physiology:{now.isoformat()}", generated_at=now - timedelta(seconds=30),
                                  constraints=[R(constraint="heat", unit="c", level=30.0, budget=31.0,
                                                 budget_kind="soft_line",
                                                 price={"value": 0.0, "ceiling": 1.0, "bind_state": "slack",
                                                        "horizon_sec": 21600.0, "why": ""})])
        ctx = build_context(FAMS, [], [], state, now, event_history={shed_act.FAMILY: list(self.events.values())})
        d = decide(want, state, ctx, self.book, now)
        self.events.setdefault(want.want_id, float(d.intensity or 0.0))
        self.decisions.append(d)
        return d

    def release(self, want_id):
        return self.book.release(want_id) if hasattr(self.book, "release") else 0


class ReplayAct(CoolingAct):
    """The shipped loop; only the bus is replaced (a fake pool that starts each shed)."""

    def __init__(self, store: ReplayStore, market: ReplayMarket, clock, incidents) -> None:
        self.clock_box = clock
        self.rpcs: list[str] = []
        super().__init__(settings=_settings(), store=store, market_getter=lambda: market, bus_getter=lambda: None,
                         hardware_watch=self._hw, clock=lambda: clock["now"])
        self._incidents = incidents

    def _hw(self, now):
        pts = self._store.load_cabinet_points(since=now - timedelta(hours=2), until=now)
        reflex = read_cabinet_heat(pts, now).reflex
        for inc in self._incidents:
            if inc["shed_requested"] and inc["opened_at"] <= now and (inc["resolved_at"] is None or inc["resolved_at"] > now):
                reflex = reflex or "cooling_incident"
        return HardwareWatchView.from_health({"enabled": True, "last_tick_ok": True,
                                              "last_tick_at": (now - timedelta(seconds=15)).isoformat(),
                                              "open_incidents": [],
                                              "reflex_shed": {"active": bool(reflex), "reason": reflex}}, now=now)

    async def _shed_rpc(self, episode_id, want, detail, ttl):
        now = self.clock_box["now"]
        self.rpcs.append(episode_id)
        self._store.pool[episode_id] = {"shed_id": f"oshed_{len(self.rpcs)}", "state": "active", "started_at": now,
                                        "ended_at": None, "drained_at": None, "grants_withheld": 3,
                                        "delayed_grant_sec": 600.0, "background_live_at_start": 10, "ttl_sec": ttl}
        return (GpuPoolShedResultV1(ok=True, shed_id=f"oshed_{len(self.rpcs)}", dispatch_id=episode_id, state="active",
                                    ttl_sec=ttl, started_at=now, valid_until=now + timedelta(seconds=ttl)), None, 40.0)

    async def _publish_outcome(self, **kw):
        return True


def advance_pool_and_scorer(store: ReplayStore, now: datetime, scores: list) -> None:
    for eid, row in store.pool.items():
        if row["state"] == "active" and now >= row["started_at"] + timedelta(seconds=row["ttl_sec"]):
            row.update(state="expired", ended_at=row["started_at"] + timedelta(seconds=row["ttl_sec"]))
    for eid, ep in store.episodes.items():
        if ep.get("scored_at") is None and now >= ep["scoring_due_at"] + SCORER_LAG:
            end = ep["scoring_due_at"]
            pts = store.load_cabinet_points(since=ep["decided_at"] - timedelta(minutes=2), until=end + timedelta(minutes=2))
            s = score_world_episode(episode=ep, points=pts, incidents=[], prior=None, control_prior=None, now=now)
            if s is None:
                if now >= ep["scoring_due_at"] + timedelta(hours=1):
                    scores.append((eid, ep["arm"], "never_scored"))
                    ep["scored_at"] = now
                continue
            ep["scored_at"] = now
            scores.append((eid, ep["arm"], s.outcome.get("excluded_reason") or "scored"))


@dataclass
class ReplayResult:
    ticks: Counter
    eligible_minutes: int
    eligible_by_hour: Counter
    wants: int
    decisions: Counter
    episodes: list
    scores: list
    rpcs: int
    no_want_while_reflex: bool


def replay(points: list[TempPoint], start: datetime, end: datetime, incidents=()) -> ReplayResult:
    store = ReplayStore(points)
    clock = {"now": start}
    market = ReplayMarket(lambda: clock["now"])
    act = ReplayAct(store, market, clock, list(incidents))
    ticks: Counter = Counter()
    eligible = 0
    by_hour: Counter = Counter()
    scores: list = []
    reflex_ok = True

    async def run():
        nonlocal eligible, reflex_ok
        while clock["now"] <= end:
            now = clock["now"]
            advance_pool_and_scorer(store, now, scores)
            before = len(market.decisions)
            result = await act.tick(now)
            ticks[result.split(":")[0] if result.startswith(("acted", "deferred", "no_answer", "event")) else result] += 1
            if result not in ("ineligible", "not_warming"):
                eligible += 1
                by_hour[now.strftime("%m-%d %H")] += 1
            if len(market.decisions) > before:
                pts = store.load_cabinet_points(since=now - timedelta(hours=2), until=now)
                r = read_cabinet_heat(pts, now)
                if r.critical or r.thermal_state == "unknown":
                    reflex_ok = False
            clock["now"] = now + TICK
        # close out the scorer for episodes decided near the end
        for _ in range(60):
            clock["now"] += TICK
            advance_pool_and_scorer(store, clock["now"], scores)

    asyncio.run(run())
    dec = Counter(f"{d.outcome}:{d.reason}" for d in market.decisions)
    return ReplayResult(ticks=ticks, eligible_minutes=eligible, eligible_by_hour=by_hour,
                        wants=len({d.want_id for d in market.decisions}), decisions=dec,
                        episodes=sorted(store.episodes.values(), key=lambda r: r["decided_at"]), scores=scores,
                        rpcs=len(act.rpcs), no_want_while_reflex=reflex_ok)


def summary(res: ReplayResult) -> dict:
    arms = Counter(e["arm"] for e in res.episodes)
    sal = sorted(e["eligibility"]["want"]["salience"] for e in res.episodes)
    return {"eligible_minutes": res.eligible_minutes, "eligible_hours": len(res.eligible_by_hour),
            "tick_results": dict(res.ticks), "wants_posted": res.wants, "decisions": dict(res.decisions),
            "episodes": len(res.episodes), "treated": arms.get("treated", 0), "control": arms.get("control", 0),
            "episode_days": sorted({e["decided_at"].strftime("%m-%d") for e in res.episodes}),
            "shed_rpcs": res.rpcs, "scorer": dict(Counter(f"{a}:{v}" for _, a, v in res.scores)),
            "episode_salience_min_median_max": [sal[0], sal[len(sal) // 2], sal[-1]] if sal else None}


# --- fixture ---------------------------------------------------------------------------------------
def fixture_points() -> tuple[list[TempPoint], datetime, datetime]:
    """35 days: 34 ordinary days as the salience history, then the scripted day. An ordinary afternoon
    (13:00-19:00) runs AC cycles: 30 minutes warming from 29.6 C by up to 1.5 C, 10 minutes cooling back. The
    scripted day: a cool morning, the same warm afternoon, a hot spell (32-33.5 C, still Orion's band), 15
    minutes at 34.5 C (critical: the reflex's), and a 30-minute sensor gap (unknown: the reflex's)."""
    import math
    pts: list[TempPoint] = []
    t0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
    for d in range(35):
        amp = 0.8 + 0.7 * ((d * 7) % 10) / 10.0          # each day's cycles warm by a different amount
        for k in range(0, 24 * 120):                      # one reading every 30 s
            t = t0 + timedelta(days=d, seconds=30 * k)
            h = t.hour + t.minute / 60.0
            base = 27.0
            if 13.0 <= h < 19.0:
                m = (t.hour * 60 + t.minute) % 40
                base = 29.6 + (amp * m / 30.0 if m < 30 else amp * (40 - m) / 10.0)
            if d == 34:
                if 15.0 <= h < 16.0:
                    base += 2.4                           # hot: 32-33.5 C, still below the 34 C reflex line
                if 16.0 <= h < 16.25:
                    base = 34.5                           # critical: the reflex covers it, Orion posts nothing
                if 17.0 <= h < 17.5:
                    continue                              # sensor gap: unknown, the reflex covers it
            pts.append(TempPoint(ts=t, value=round(base + 0.03 * math.sin(k / 7.0), 2)))
    return pts, t0 + timedelta(days=34), t0 + timedelta(days=35) - timedelta(minutes=1)


def fixture_checks() -> list[tuple[str, bool, str]]:
    pts, start, end = fixture_points()
    res = replay(pts, start, end)
    s = summary(res)
    checks = []
    warm_hours = {h for h, n in res.eligible_by_hour.items()}
    decided_hours = {e["decided_at"].strftime("%m-%d %H") for e in res.episodes}
    checks.append(("no want outside the warm band (morning)",
                   all(e["decided_at"].hour >= 12 for e in res.episodes), f"hours={sorted(decided_hours)}"))
    checks.append(("no want while the reflex covers (critical or unknown)", res.no_want_while_reflex, ""))
    checks.append(("at least 3 episodes on the warm afternoon", len(res.episodes) >= 3, f"episodes={len(res.episodes)}"))
    checks.append(("eligible hours exist", len(warm_hours) >= 4, f"eligible_hours={len(warm_hours)}"))
    tr = s["treated"] / max(1, s["episodes"])
    checks.append(("treated share within 0.5 +- 0.35", 0.15 <= tr <= 0.85, f"treated={s['treated']} control={s['control']}"))
    checks.append(("every treated episode sent exactly one shed RPC", res.rpcs == s["treated"], f"rpcs={res.rpcs}"))
    unscored = [x for x in res.scores if x[2] == "never_scored"]
    checks.append(("the sql-writer scorer scores every episode the loop writes", not unscored
                   and len(res.scores) == len(res.episodes), f"scorer={s['scorer']}"))
    eps = res.episodes
    overlap = any(b["decided_at"] < a["scoring_due_at"] for a, b in zip(eps, eps[1:]))
    checks.append(("no two episodes overlap (in flight blocks)", not overlap, ""))
    funded = sum(v for k, v in res.decisions.items() if k.startswith("funded"))
    checks.append(("only funded decisions became episodes", len(eps) <= funded, f"funded={funded}"))
    return checks


# --- live ------------------------------------------------------------------------------------------
def _psql(sql: str) -> list[list[str]]:
    out = subprocess.run(["docker", "exec", CONTAINER, "psql", "-U", "postgres", "-d", "conjourney", "-AtF", "\t", "-c",
                          sql], check=True, capture_output=True, text=True).stdout
    return [line.split("\t") for line in out.splitlines() if line]


def _t(s: str) -> datetime:
    return datetime.fromisoformat(s.replace(" ", "T")).astimezone(timezone.utc)


def live(days: float) -> dict:
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=days)
    since = start - timedelta(days=30, hours=2)
    rows = _psql("SELECT timestamp, (measurements->>'cabinet_temp_c')::float FROM orion_biometrics_summary "
                 f"WHERE node = 'athena' AND timestamp >= '{since:%Y-%m-%d %H:%M:%S}' AND measurements ? 'cabinet_temp_c' "
                 "ORDER BY timestamp")
    pts = [TempPoint(ts=_t(r[0]), value=float(r[1])) for r in rows if r[1]]
    inc = [{"opened_at": _t(r[0]), "resolved_at": _t(r[1]) if r[1] else None, "shed_requested": r[2] == "t"}
           for r in _psql("SELECT opened_at, resolved_at, shed_requested FROM hardware_watch_incident WHERE rule = 'cooling' "
                          f"AND (resolved_at IS NULL OR resolved_at >= '{since:%Y-%m-%d %H:%M:%S}')")]
    res = replay(pts, start, end, inc)
    new = summary(res)
    warm_band = [p for p in pts if p.ts >= start]
    hot = sum(1 for p in warm_band if p.value >= 32.0)
    warm = sum(1 for p in warm_band if p.value >= 29.5)
    old_eps = _psql("SELECT arm, count(*) FROM substrate_world_action_episodes WHERE template = 'shed_background_gpu' "
                    f"AND decided_at >= '{start:%Y-%m-%d %H:%M:%S}' GROUP BY arm")
    old_prop = _psql("SELECT count(DISTINCT date_trunc('minute', f.created_at)), min(f.created_at)::date, count(*) "
                     "FROM substrate_proposal_frames f, jsonb_array_elements(f.proposal_frame_json->'candidates') c "
                     f"WHERE f.created_at >= '{start:%Y-%m-%d %H:%M:%S}' "
                     "AND c->'execution_intent'->>'template' = 'shed_background_gpu'")
    frames_since = _psql("SELECT min(created_at)::date FROM substrate_proposal_frames")
    # The same replay counted only from the day proposal frames are still kept, so the two compare like for like.
    retained = _t(frames_since[0][0] + " 00:00:00+00:00") if frames_since and frames_since[0][0] else start
    new_since_retained = sum(n for h, n in res.eligible_by_hour.items()
                             if datetime.strptime(f"{start.year}-{h}", "%Y-%m-%d %H").replace(tzinfo=timezone.utc) >= retained)
    return {
        "window": [start.isoformat(), end.isoformat()], "cabinet_readings": len(warm_band),
        "share_readings_ge_29_5C": round(warm / max(1, len(warm_band)), 3),
        "share_of_warm_readings_ge_32C (pressure would tie at 1.0)": round(hot / max(1, warm), 3),
        "new_loop": new,
        "old_path": {
            "episodes_by_arm": {a: int(n) for a, n in old_eps},
            "eligible_proposal_minutes": int(old_prop[0][0]) if old_prop else 0,
            "eligible_proposal_candidates": int(old_prop[0][2]) if old_prop else 0,
            "proposal_frames_kept_since": frames_since[0][0] if frames_since else None,
            "new_loop_eligible_minutes_same_span": new_since_retained,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--days", type=float, default=14.0)
    args = ap.parse_args()
    if args.live:
        print(json.dumps(live(args.days), indent=2, default=str))
        return 0
    checks = fixture_checks()
    for name, ok, why in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {why}")
    print(json.dumps({"eval": "cooling_act_replay", "passed": sum(ok for _, ok, _ in checks), "total": len(checks)}))
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
