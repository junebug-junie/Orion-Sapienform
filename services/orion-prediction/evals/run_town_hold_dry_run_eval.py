"""Family C Experiment 1, dry run against LIVE read-only data: what would tonight's schedule be?

Reads, never writes: embodiment /health (the town's own start gate), the stop rules as the runner reads them
now (tier-0/1 leases queued on fast, the cabinet reading, the regulation key), the town's sessions from the act
ledger. Places no bid and sets no key: the projected timeline runs the REAL runner's decision loop against an
in-memory copy of the town (``tests/test_town_hold.SimTown``: 5-min ticks, 30-min sessions, 60-min gap, 2 h
cap) seeded from the live state, with in-memory Redis and store.

Run (inside app-net, read-only):
  POSTGRES_URI=... ORION_BUS_URL=... PYTHONPATH=../..:. python evals/run_town_hold_dry_run_eval.py \
      --embodiment http://orion-athena-embodiment:8130 --seed 0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[3])]

from app.series import load_registry  # noqa: E402
from app.town_hold import (  # noqa: E402
    EmbodimentProbe,
    TownHoldConfig,
    TownHoldRunner,
    first_night_after,
    night_window,
    plan_night,
)

DEN = ZoneInfo("America/Denver")
REGISTRY = HERE.parents[3] / "config" / "prediction" / "series.v1.yaml"


class _LiveEffects:
    """Just what stop_reasons reads: the clock, the store, the cabinet series spec."""

    def __init__(self, store, now):
        self.store = store
        self.engine = SimpleNamespace(clock=lambda: now)
        self._reg = load_registry(REGISTRY)

    def _spec(self, series_id, role=None):
        return self._reg.get(series_id)


class _ReadOnlyRedis:
    def __init__(self, url):
        import redis.asyncio as aioredis

        self.r = aioredis.from_url(url, decode_responses=True, socket_timeout=3.0)

    async def get(self, key):
        return await self.r.get(key)

    async def set(self, *a, **k):   # pragma: no cover -- a dry run never writes
        raise RuntimeError("dry run: refusing a live Redis write")

    delete = eval = set


def project(night: date, seed: int, eligible_at: datetime | None, sessions_today: int) -> list[str]:
    """The real decision loop against an in-memory town seeded from the live state."""
    sys.path.insert(0, str(HERE.parents[1]))
    from tests.test_town_hold import SimTown, regulation, world

    logging.disable(logging.WARNING)
    w0, w1 = night_window(night, TownHoldConfig())
    w = world(w0 - timedelta(minutes=1))
    w.sched = {"run_id": "dry-run", "first_night": night.isoformat(), "nights": 1, "seed": seed}
    town = SimTown(w.clock, w.redis, w.store, daily_cap_sessions=4)
    town.eligible_at = max(eligible_at or w0, w0 - timedelta(minutes=1))
    town.next_tick = w0 - timedelta(minutes=1)
    w.runner.probe = town

    async def sleep(sec):
        w.clock.t += timedelta(seconds=sec)
        await w.redis.set("orion:regulation:latest", regulation(w.clock))

    w.runner._sleep = sleep
    asyncio.run(w.runner._run_night(w.sched, night))
    lines = []
    for r in sorted(w.store.ivs, key=lambda r: r["started_at"]):
        n = json.loads(r["notes"])
        if r.get("arm"):
            lines.append(f"  {r['started_at'].astimezone(DEN):%H:%M} decision #{n['block_index']}: "
                         f"{'ON  (town plays)' if r['arm'] == 'on' else 'OFF (hold starts 40 min)'}")
    for a in w.store.acts:
        lines.append(f"  {a['opened_at'].astimezone(DEN):%H:%M} town session starts (30 min)")
    lines.sort()
    lines.append(f"  -> {sum(1 for r in w.store.ivs if r.get('arm') == 'on')} ON, "
                 f"{sum(1 for r in w.store.ivs if r.get('arm') == 'off')} OFF, {len(w.store.acts)} sessions "
                 f"(held town ticks: {town.held})")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embodiment", default=os.environ.get("PREDICTION_EMBODIMENT_URL", "http://orion-athena-embodiment:8130"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--night", type=date.fromisoformat, default=None)
    args = ap.parse_args()
    uri, bus = os.environ.get("POSTGRES_URI"), os.environ.get("ORION_BUS_URL")
    if not uri or not bus:
        print("POSTGRES_URI and ORION_BUS_URL are required (both read-only here)")
        return 2
    from app.store import PostgresStore

    now = datetime.now(timezone.utc)
    cfg = TownHoldConfig()
    night = args.night or first_night_after(now, cfg)
    store = PostgresStore(uri)
    runner = TownHoldRunner(_LiveEffects(store, now), _ReadOnlyRedis(bus), EmbodimentProbe(args.embodiment), cfg=cfg)
    view = runner.probe.view()
    async def live_stop_rules():
        try:
            return await runner.stop_reasons(now)
        finally:
            await runner.redis.r.aclose()

    stops = asyncio.run(live_stop_rules())
    plan = plan_night(args.seed, night, cfg)
    w0, w1 = night_window(night, cfg)
    print(f"now {now.astimezone(DEN):%Y-%m-%d %H:%M %Z}; next night {night} = "
          f"{w0.astimezone(DEN):%H:%M}..{w1.astimezone(DEN):%H:%M} {DEN.key} ({w0:%H:%M}..{w1:%H:%M} UTC)")
    print(f"\nthe town now (embodiment /health): {view}")
    if view is not None and not view.override_configured:
        print("  WARNING: embodiment does not obey the override yet (image without this PR): a run would be refused")
    print(f"stop rules read now: {stops or 'none (a decision now would go to the bid)'}")
    print(f"\nnight seed {plan['seed']} (run seed {args.seed}); arm list (taken in order, refusals take none):")
    print("  " + " ".join(a.upper() for a in plan["arms"]))
    acts = store.town_acts(w0 - timedelta(days=1), now)
    print(f"\nlast town sessions: " + ", ".join(f"{pd_local(a['opened_at'])}" for a in acts[-4:]))
    elig = view.eligible_at if view is not None else None
    print(f"\nprojected timeline for {night} (real decision loop, simulated town; no live write):")
    for line in project(night, args.seed, elig if elig and elig > w0 else None, 0):
        print(line)
    return 0


def pd_local(t) -> str:
    return t.astimezone(DEN).strftime("%m-%d %H:%M")


if __name__ == "__main__":
    sys.exit(main())
