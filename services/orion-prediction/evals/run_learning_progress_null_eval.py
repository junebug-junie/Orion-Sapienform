"""How often does the learning-progress rule call a move when nothing moved? Read-only (nothing is stored). Issue
#3691; app/learning_progress.py "Weather spells are not independent days".

Three nulls, each run through the real ``progress()`` twice: as shipped (rho estimated from history) and with
independent days (``rho=0``, the plain Welch rule it replaced). A "false move" is a ``better``/``worse`` reading.

1. Live, shuffled labels: each series' live daily skill since its last roster change, in a seeded random order.
   Destroys spells and any real change, keeps the real spread and outliers (the 2026-11-14 dip).
2. Live, circular shifts: the same days rotated, every offset. Keeps the spells; a real change in the data counts
   against the rule, so this rate is an upper bound.
3. Simulated AR(1) at the live day-to-day spread, at rho 0 / the live estimate / 0.3 / 0.5, seeded. Per daily look,
   and per week (any of 7 consecutive daily looks on overlapping windows): Prime reads each look, the 5% target is
   per look; the per-week number is printed so the re-check cost stays visible.

    cd services/orion-prediction && POSTGRES_URI=... PYTHONPATH=../..:. python evals/run_learning_progress_null_eval.py

Exit 1 if the shipped rule's simulated per-look rate at the live rho (14 or 56 days of history) is over 5% (+1
point of seed noise).

Baseline 2026-11-18 (live: cabinet and office, 14 usable days each, 11-04..11-17; shipped / Welch):
  lag-1 raw: cabinet +0.070, office +0.377 (office's 0.0 days 11-04..05 before the champion served are a step;
  over 11-06..11-17 alone it is +0.21). Moment-matched rho (detrended 7-day blocks, 12 pairs): cabinet 0.37,
  office 0.00; used after shrinking toward the 0.3 prior: 0.32 and 0.20 (mean 0.26 = "live" below).
  Shuffled labels, per look: cabinet 0.000 / 0.010, office 0.016 / 0.041 (1000 shuffles each).
  Circular shifts: cabinet 0/14 / 0/14, office 0/14 / 8/14 (office's real 11-06 step is in the data).
  Simulated AR(1), sd 0.156, per look, 14 days | 56 days of history:
    rho 0    0.017 / 0.042 | 0.020 / 0.040
    rho 0.2  0.039 / 0.089 | 0.041 / 0.092
    rho 0.26 0.038 / 0.104 | 0.041 / 0.106   (live)
    rho 0.3  0.037 / 0.122 | 0.051 / 0.133
    rho 0.5  0.094 / 0.209 | 0.069 / 0.201
  Per week (any of 7 daily looks, 56 days): rho 0 0.120 / 0.193; live 0.193 / 0.417; rho 0.5 0.303 / 0.557.
  PASS (0.041). Runtime about 3 min.
"""
from __future__ import annotations

import math
import os
import random
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SERVICE), str(SERVICE.parents[1])]

from app import learning_progress as lp  # noqa: E402

SEED = 3691
TODAY = date(2026, 11, 20)  # synthetic frame; only relative days matter


def as_days(skills: list[float], today: date = TODAY) -> list[lp.DayStat]:
    n = len(skills)
    return [lp.DayStat(today - timedelta(days=n - i), 288, 0.2 * (1 - s), 0.2) for i, s in enumerate(skills)]


def moved(skills: list[float], rho: float | None, today: date = TODAY) -> bool:
    return lp.progress("s", as_days(skills, today), {}, today, rho=rho).direction in ("better", "worse")


def lag1(xs: list[float]) -> float:
    m = sum(xs) / len(xs)
    c0 = sum((x - m) ** 2 for x in xs)
    return sum((a - m) * (b - m) for a, b in zip(xs, xs[1:])) / c0 if c0 else 0.0


def ar1(rng: random.Random, rho: float, n: int, sd: float, level: float = 0.3) -> list[float]:
    e, out = sd * rng.gauss(0, 1), []
    for _ in range(n + 20):
        e = rho * e + sd * math.sqrt(1 - rho * rho) * rng.gauss(0, 1)
        out.append(level + e)
    return out[-n:]


def live_segments() -> dict[str, list[tuple[int, float]]]:
    from app.store import PostgresStore

    store = PostgresStore(os.environ["POSTGRES_URI"])
    since = datetime(2026, 10, 1, tzinfo=timezone.utc)
    today = datetime.now(timezone.utc).date()
    pairs = store.daily_skill_pairs(lp.SERIES, lp.HORIZON_MIN, since)
    rosters = store.daily_rosters(lp.SERIES, since)
    out = {}
    for sid in lp.SERIES:
        roster = {r["day"]: frozenset(r["models"]) for r in rosters if r["series_id"] == sid and r["day"] < today}
        change = lp.last_roster_change(roster)
        days = sorted((r for r in pairs if r["series_id"] == sid and r["day"] < today
                       and (change is None or r["day"] > change)), key=lambda r: r["day"])
        stats = [lp.DayStat(r["day"], int(r["pairs"]), float(r["mae_champion"]), float(r["mae_baseline"]))
                 for r in days]
        out[sid] = [(d.day.toordinal(), d.skill) for d in stats if d.skill is not None]  # real days: gaps stay gaps
        print(f"{sid}: {len(out[sid])} usable days after the {change} roster change")
    return out


def main() -> int:
    rng = random.Random(SEED)
    segs = live_segments()
    rhos, sds = [], []
    print("\n== live day-to-day correlation")
    for sid, pts in segs.items():
        xs, end = [v for _, v in pts], pts[-1][0] + 1
        raw, npairs = lp.moment_rho(pts, end)
        rho, _, source = lp.estimate_rho(pts, end)
        sd = (sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
        rhos.append(rho)
        sds.append(sd)
        print(f"{sid}: n {len(xs)}  lag-1 raw {lag1(xs):+.3f}  moment-matched {raw if raw is None else round(raw, 3)} "
              f"({npairs} pairs)  used {rho:.3f} ({source})  sd {sd:.3f}")
    segs = {sid: [v for _, v in pts] for sid, pts in segs.items()}

    print("\n== live, shuffled labels (false-move rate per look: shipped / Welch)")
    for sid, xs in segs.items():
        if len(xs) < 2 * lp.WINDOW_DAYS:
            print(f"{sid}: too few days")
            continue
        n, a, b = 1000, 0, 0
        for _ in range(n):
            ys = xs[:]
            rng.shuffle(ys)
            a += moved(ys, None)
            b += moved(ys, 0.0)
        print(f"{sid}: {a / n:.3f} / {b / n:.3f}  ({n} shuffles)")

    print("\n== live, circular shifts (upper bound: real change counts against the rule)")
    for sid, xs in segs.items():
        if len(xs) < 2 * lp.WINDOW_DAYS:
            continue
        a = sum(moved(xs[k:] + xs[:k], None) for k in range(len(xs)))
        b = sum(moved(xs[k:] + xs[:k], 0.0) for k in range(len(xs)))
        print(f"{sid}: {a}/{len(xs)} / {b}/{len(xs)}")

    live_rho = sum(rhos) / len(rhos)
    sd = sum(sds) / len(sds)
    print(f"\n== simulated AR(1), sd {sd:.3f} (shipped / Welch); 14 days = the live history today")
    worst = 0.0
    for rho in sorted({0.0, round(live_rho, 2), 0.2, 0.3, 0.5}):
        for hist in (2 * lp.WINDOW_DAYS,):
            n, a, b = 1000, 0, 0
            for _ in range(n):
                xs = ar1(rng, rho, hist, sd)
                a += moved(xs, None)
                b += moved(xs, 0.0)
            print(f"rho {rho:.2f}: {hist} days, per look {a / n:.3f} / {b / n:.3f}")
            if rho == round(live_rho, 2):
                worst = max(worst, a / n)
        n, a, b = 1500, 0, 0
        for _ in range(n):
            xs = ar1(rng, rho, lp.RHO_HISTORY_DAYS, sd)
            a += moved(xs, None)
            b += moved(xs, 0.0)
        wn, wa, wb = 300, 0, 0
        for _ in range(wn):  # 7 daily looks on overlapping windows
            xs = ar1(rng, rho, lp.RHO_HISTORY_DAYS + 6, sd)
            looks = [(xs[:lp.RHO_HISTORY_DAYS + k], TODAY + timedelta(days=k)) for k in range(7)]
            wa += any(moved(x, None, d) for x, d in looks)
            wb += any(moved(x, 0.0, d) for x, d in looks)
        tag = "  <- live" if rho == round(live_rho, 2) else ""
        if rho == round(live_rho, 2):
            worst = max(worst, a / n)
        print(f"rho {rho:.2f}: {lp.RHO_HISTORY_DAYS} days, per look {a / n:.3f} / {b / n:.3f}   per week {wa / wn:.3f} / {wb / wn:.3f}{tag}")
    ok = worst <= 0.06
    print(f"\n{'PASS' if ok else 'FAIL'}: shipped per-look false-move at live rho {live_rho:.2f}: {worst:.3f} (target 0.05)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
