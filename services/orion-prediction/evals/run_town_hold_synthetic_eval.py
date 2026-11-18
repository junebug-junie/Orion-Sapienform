"""Family C Experiment 1, synthetic eval: whole runs through the REAL runner against a simulated town.

What it does. For each replicate: run ``NIGHTS`` nights through ``TownHoldRunner._run_night`` (the real
decision loop: pending hold, race read, stop rules, bid, arms, key writes, block checks) against a town that
follows the controller's own rules (5-min ticks, 30-min sessions, 60-min gap, refuses to start while the key
is live). Then plant effects that come from the SESSIONS that really ran (not from the arm labels): GPU3 watts
+EFFECT_W while a session is open, tier-1 hold +EFFECT_HOLD s for calls granted during a session. Analyse with
the real effect table (``EffectTable.rows_for_run``: stratified estimate, restricted permutation).

Truth for one replicate = the same estimator on the noise-free effect signal alone (the intention-to-treat
effect of the arm on the window, which is what the experiment can measure).

Gates (exit 1 on a miss):
  * no session ever starts inside an OFF hold, and no night ever holds more than 2 times in a row;
  * 95 % intervals cover the truth in >= 80 % of replicates (watts and tier-1 hold);
  * with no planted effect, p <= 0.05 in <= 20 % of replicates (the restricted permutation is calibrated);
  * with the planted effect, p <= 0.05 in >= 70 % of replicates (the design has power at this size).

Run from services/orion-prediction:  PYTHONPATH=../..:. python evals/run_town_hold_synthetic_eval.py
"""
from __future__ import annotations

import asyncio
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# The service dir first: the repo root also has a ``tests`` package, and the simulated town lives in this
# service's tests (one town for the unit tests and this eval).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.causal import max_off_run  # noqa: E402
from app.family_c import EffectTable  # noqa: E402
from app.town_hold import TownHoldConfig, night_window  # noqa: E402
from tests.test_town_hold import SimTown, regulation, world  # noqa: E402

logging.disable(logging.WARNING)

NIGHTS = 12
REPS = 16
EFFECT_W = 12.0
EFFECT_HOLD = 6.0
N_PERM = 300
FIRST = date(2026, 11, 17)


def run_nights(seed: int):
    """One replicate: every night through the real runner. Returns the store (experiments, blocks, acts)."""
    w = world(night_window(FIRST, TownHoldConfig())[0] - timedelta(minutes=1))
    w.sched = {"run_id": f"rep-{seed}", "first_night": FIRST.isoformat(), "nights": NIGHTS, "seed": seed}
    rng = np.random.default_rng(seed)

    async def sleep(sec):
        w.clock.t += timedelta(seconds=sec)
        await w.redis.set("orion:regulation:latest", regulation(w.clock))

    w.runner._sleep = sleep
    for n in range(NIGHTS):
        day = FIRST + timedelta(days=n)
        w0, _ = night_window(day, w.runner.cfg)
        w.clock.t = w0 - timedelta(minutes=1) + timedelta(seconds=int(rng.integers(0, 3600)))
        town = SimTown(w.clock, w.redis, w.store, daily_cap_sessions=4)   # the 2 h daily cap
        # The town's own phase: eligible some time in the first hour, ticks on its own 5-min clock.
        town.eligible_at = w.clock() + timedelta(minutes=int(rng.integers(0, 60)))
        town.next_tick = w.clock() + timedelta(seconds=int(rng.integers(0, 300)))
        w.runner.probe = town
        asyncio.run(w.runner._run_night(w.sched, day))
    return w


def plant(w, effect_w: float, effect_hold: float, seed: int, noise: bool = True):
    """Readings driven by the sessions that ran: GPU3 watts on a 5-min grid, tier-1 leases, Kev latencies."""
    rng = np.random.default_rng(seed + 10_000)
    t0 = night_window(FIRST, TownHoldConfig())[0] - timedelta(days=15)
    idx = pd.date_range(t0, t0 + timedelta(days=NIGHTS + 16), freq="5min", tz="UTC")
    tod = (idx.hour * 60 + idx.minute).to_numpy()
    base = 100 + 10 * np.sin(2 * np.pi * tod / 1440)
    ar = np.zeros(len(idx))
    if noise:
        e = rng.normal(0, 6, len(idx))
        for i in range(1, len(idx)):
            ar[i] = 0.6 * ar[i - 1] + e[i]
    on = np.zeros(len(idx))
    for a in w.store.acts:
        on[(idx >= pd.Timestamp(a["opened_at"])) & (idx < pd.Timestamp(a["closed_at"]))] = 1.0
    gpu = pd.Series(base * (1 if noise else 0) + ar + effect_w * on, index=idx)
    leases = []
    for t in pd.date_range(t0 + timedelta(days=15), idx[-1], freq="7min", tz="UTC"):
        in_s = bool(on[idx.searchsorted(t, side="right") - 1])
        hold = (4.0 if noise else 0.0) + (effect_hold if in_s else 0.0) + (rng.normal(0, 2.0) if noise else 0.0)
        leases.append({"holder": "cortex-exec", "work_class": "metacog", "priority": "system", "status": "released",
                       "queued_since": t, "granted_at": t, "updated_at": t + pd.Timedelta(seconds=max(0.1, hold))})
    w.store.leases = leases

    def loader(a, b):
        s = gpu[(gpu.index >= pd.Timestamp(a)) & (gpu.index < pd.Timestamp(b))]
        return pd.DataFrame({"circe.gpu3_watts": s}), {}

    return EffectTable(w.store, loader, units={"circe.gpu3_watts": "W"}, grain_min=5, n_perm=N_PERM)


def rows(w, table):
    nights = [e for e in w.store.experiments(status="analyzed")]
    rs, extra = table.rows_for_run(nights)
    return {r.target: r for r in rs}, extra, len(nights)


def main() -> int:
    leaks = floor = 0
    cover = {"circe.gpu3_watts": [], "lane.fast.tier1_hold_s": []}
    sig_effect = {"circe.gpu3_watts": [], "lane.fast.tier1_hold_s": []}
    sig_null = {"circe.gpu3_watts": [], "lane.fast.tier1_hold_s": []}
    per_arm = []
    for rep in range(REPS):
        w = run_nights(rep)
        for e in w.store.exps.values():
            ivs = sorted((r for r in w.store.ivs if r["experiment_id"] == e["experiment_id"] and r.get("arm")),
                         key=lambda r: r["started_at"])
            arms = [r["arm"] for r in ivs]
            floor += max_off_run(arms) > 2
            per_arm.append((arms.count("on"), arms.count("off")))
            for r in ivs:
                if r["arm"] == "off":
                    leaks += sum(1 for a in w.store.acts if r["started_at"] <= a["opened_at"] < r["ended_at"])
        by, extra, _n = rows(w, plant(w, EFFECT_W, EFFECT_HOLD, rep))
        truth, _, _ = rows(w, plant(w, EFFECT_W, EFFECT_HOLD, rep, noise=False))
        null, _, _ = rows(w, plant(w, 0.0, 0.0, rep))
        for t in cover:
            r, tr, nu = by[t], truth[t], null[t]
            if r.ci_lo is not None and tr.estimate is not None:
                cover[t].append(r.ci_lo <= tr.estimate <= r.ci_hi)
            sig_effect[t].append(r.p_value is not None and r.p_value <= 0.05)
            sig_null[t].append(nu.p_value is not None and nu.p_value <= 0.05)
        print(f"rep {rep:2d}: nights={len(w.store.exps)} manip={extra['manipulation']['verdict']} "
              f"watts {by['circe.gpu3_watts'].estimate:+.1f} (truth {truth['circe.gpu3_watts'].estimate:+.1f}) "
              f"hold {by['lane.fast.tier1_hold_s'].estimate:+.2f} (truth {truth['lane.fast.tier1_hold_s'].estimate:+.2f})")
    on = np.mean([a for a, _ in per_arm])
    off = np.mean([b for _, b in per_arm])
    print(f"\nblocks per night: on {on:.2f} off {off:.2f} (design: ~3.5 each, upper bound)")
    print(f"sessions started inside an OFF hold: {leaks}; nights with > 2 holds in a row: {floor}")
    ok = leaks == 0 and floor == 0
    for t in cover:
        c, pe, pn = np.mean(cover[t]), np.mean(sig_effect[t]), np.mean(sig_null[t])
        print(f"{t}: coverage {c:.2f} (n={len(cover[t])}), power {pe:.2f}, null p<=0.05 {pn:.2f}")
        ok &= c >= 0.80 and pe >= 0.70 and pn <= 0.20
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
