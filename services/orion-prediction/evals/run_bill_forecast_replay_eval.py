"""Replay the power-bill bet over the stored meter history (read-only, dry run: nothing is written).

Two checks, same code the service runs (app/bill_forecast.py):

1. Bills. Every past day is replayed as of 00:05 local, knowing only the meter hours the feed would
   have delivered by then (its measured lag) and the bills already retrieved. Each bet is settled
   against the real RMP bill for its period. Printed per bet: days of meter data in the cycle, the
   projection, the 80 % range, the bill, and whether the range held.
2. Metered windows. Every 30-day window the meter covers end to end is a pseudo-bill (truth = metered
   kWh). Projections at each day-into-window (no lag) are scored against it. Windows overlap heavily,
   so the coverage here is a sanity check on the range's width, not an independent sample: the
   independent count is about (days of continuous data / 30).

    PYTHONPATH=../..:. POSTGRES_URI=... python evals/run_bill_forecast_replay_eval.py [--json]
Exit 0 always: a report, not a gate.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

_HERE = Path(__file__).resolve()
# Repo checkout: service dir + repo root. Image: /app holds both app/ and orion/.
REPO = _HERE.parents[3] if len(_HERE.parents) > 3 else _HERE.parents[1]
sys.path[:0] = [str(_HERE.parents[1]), str(REPO)]

from app import bill_forecast as B  # noqa: E402
from app.settings import get_settings  # noqa: E402
from app.store import PostgresStore  # noqa: E402
from orion.energy.ledger import UsageLedger  # noqa: E402
from orion.energy.reconcile import project_period  # noqa: E402
from orion.energy.tariff import load_tariff  # noqa: E402
from orion.schemas.energy import EnergyUsageIntervalV1  # noqa: E402



def metered_windows(intervals, tariff, tz, window_days: int = 30) -> dict:
    """Coverage of the 80 % range against metered 30-day totals, by whole days into the window."""
    point = intervals[0].usage_point_id
    ledger = UsageLedger(tariff, tz=tz, cycle_start_day=1)
    for iv in intervals:
        ledger.upsert(iv)
    days = sorted({iv.interval_start.astimezone(tz).date() for iv in intervals})
    by_d: dict[int, list] = {}
    for first in days:
        start = datetime.combine(first, datetime.min.time(), tzinfo=tz)
        end = start + timedelta(days=window_days)
        prefix, covered = ledger.window_prefix(point, start, end)
        if covered is None or covered < end:
            continue
        truth = sum(iv.energy_kwh for iv in prefix)
        for d in range(2, window_days):
            cut = start + timedelta(days=d)
            known = [iv for iv in intervals if iv.interval_end <= cut]
            led = UsageLedger(tariff, tz=tz, cycle_start_day=1)
            for iv in known:
                if iv.interval_start >= start - timedelta(days=B.TRAILING_DAYS + 2):
                    led.upsert(iv)
            proj, _, _ = project_period(led, point, start, end)
            sp = B.spread(B.daily_totals(known, tz, until=cut))
            if proj is None or sp is None:
                continue
            lo, hi, _ = B.kwh_range(proj.projected_kwh, proj.observed_kwh, remaining_days=window_days - d,
                                    observed_days=d, sp=sp)
            by_d.setdefault(d, []).append((lo <= truth <= hi, abs(proj.projected_kwh - truth) / truth))
    out = {}
    for d, v in sorted(by_d.items()):
        out[d] = {"n_windows": len(v), "coverage80": sum(x for x, _ in v) / len(v),
                  "mean_abs_pct_miss": statistics.fmean(e for _, e in v)}
    allv = [x for v in by_d.values() for x, _ in v]
    return {"by_day": out, "overall_coverage80": (sum(allv) / len(allv)) if allv else None, "n_scores": len(allv)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    s = get_settings()
    tz = ZoneInfo(s.energy_timezone)
    tariff_path = s.energy_tariff_path
    if not Path(tariff_path).exists():
        tariff_path = str(REPO / "config" / "energy" / Path(tariff_path).name)
    tariff = load_tariff(tariff_path)
    now = datetime.now(timezone.utc)
    pg = PostgresStore(os.environ["POSTGRES_URI"])
    rows, bills = pg.energy_intervals(now - timedelta(days=B.HISTORY_DAYS)), pg.energy_bills()
    pg.close()
    mem = B.MemoryBillStore(rows, bills)
    bf = B.BillForecast(mem, tariff, tz=tz, cycle_start_day=s.energy_billing_cycle_start_day,
                        usage_point_id=s.energy_usage_point_id)
    bf.replayed = False
    tick = bf.tick(now)
    bets = mem.bill_bet_rows(now - timedelta(days=B.HISTORY_DAYS))
    replayed = [b for b in bets if b["origin"] == "replay"]
    intervals = sorted((EnergyUsageIntervalV1(**r) for r in rows), key=lambda iv: iv.interval_start)
    report = {"tick": {k: v for k, v in tick.items() if k != "at"},
              "by_period": B.grade_by_period(replayed),
              "bets": [{k: b.get(k) for k in ("bet_date", "period_start", "period_end", "days_into_cycle",
                                              "projected_kwh", "kwh_lo80", "kwh_hi80", "projected_usd", "usd_lo80",
                                              "usd_hi80", "billed_kwh", "billed_usd", "kwh_in_80", "usd_in_80",
                                              "range_reason")} for b in replayed],
              "metered_windows": metered_windows(intervals, tariff, tz) if intervals else None}
    if a.json:
        print(json.dumps(report, default=str, indent=1))
        return
    print(f"replay: {tick['replay']}  live-today: issued={tick['issued']} gap={tick['gap']} settled={tick['settled']}")
    print("\nday  period      days_in  proj_kWh  80%_kWh          bill_kWh  in  proj_$   80%_$             bill_$   in")
    for b in report["bets"]:
        def f(x, w=8, p=0):
            return f"{x:{w}.{p}f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else f"{str(x):>{w}}"
        print(f"{b['bet_date']} {b['period_start']} {f(b['days_into_cycle'], 6, 1)} {f(b['projected_kwh'])} "
              f"{f(b['kwh_lo80'], 7)}..{f(b['kwh_hi80'], 6)} {f(b['billed_kwh'])} {f(b['kwh_in_80'], 3)} "
              f"{f(b['projected_usd'], 7, 2)} {f(b['usd_lo80'], 7, 2)}..{f(b['usd_hi80'], 7, 2)} "
              f"{f(b['billed_usd'], 7, 2)} {f(b['usd_in_80'], 3)}")
    print("\nper period (one bill = one answer):")
    for p in report["by_period"]:
        print(" ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()})
    mw = report["metered_windows"]
    if mw:
        print(f"\nmetered 30-day windows: overall 80% coverage {mw['overall_coverage80']} over {mw['n_scores']} "
              "scores (overlapping windows; not independent)")
        for d, v in mw["by_day"].items():
            print(f"  day {d:2d}: windows {v['n_windows']:2d}  coverage {v['coverage80']:.2f}  "
                  f"mean |miss| {v['mean_abs_pct_miss']:.1%}")


if __name__ == "__main__":
    main()
