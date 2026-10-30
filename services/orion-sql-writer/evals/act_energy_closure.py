#!/usr/bin/env python3
"""Eval: closure and coverage of the whole-machine energy meter, per host, over one UTC day.

Two code paths split the same measured quantity, the host's chassis watts above its pinned rest
(config/physiology/rest_body.yaml):

* the per-act fill (orion/grammar/act_energy_fill.act_energy_cost): one act's ``wh_by_host``;
* the hourly attribution (``attribute_host_hour``, the writer of orion_energy_attribution_hourly):
  above_rest = sponsored + metered + unattributed, per hour.

This replays both read-only for every hour of the day and reports per host:

* the hourly rows summed (rest, above rest, sponsored, metered, leased-without-act, unattributed),
  and COVERAGE = metered / (above_rest - sponsored), the figure the allowance's coverage gate reads;
* CLOSURE: above_rest == sponsored + metered + unattributed, and unattributed never below 0;
* AGREEMENT: the sum of the per-act fill over every act whose lease ended that day vs the hourly
  ``metered_wh``. They differ only at day boundaries (an act's lease is counted whole on the day
  it ends; the hourly rows clip at midnight) and where an act's lease falls outside its own
  open..close window.

GATES: unattributed >= -0.01 Wh every hour; |acts - metered| <= AGREEMENT_TOLERANCE x metered + 1 Wh.

    python3 services/orion-sql-writer/evals/act_energy_closure.py --day 2026-10-30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sqlalchemy import create_engine, text  # noqa: E402

from orion.grammar import act_energy_fill as fill  # noqa: E402

DEFAULT_URI = "postgresql://postgres:postgres@localhost:55432/conjourney"
AGREEMENT_TOLERANCE = 0.10
NEGATIVE_TOLERANCE_WH = 0.01

_ACT_CORRS_SQL = text(
    """
    SELECT DISTINCT g.turn_correlation_id
      FROM gpu_pool_events g
     WHERE g.generated_at >= :since AND g.generated_at < :until
       AND g.event IN ('released', 'expired', 'aborted') AND g.held_ms IS NOT NULL
       AND EXISTS (SELECT 1 FROM grammar_acts a WHERE a.correlation_id = g.turn_correlation_id)
    """
)


def closure(conn, day_start: datetime, day_end: datetime, hosts: list[str]) -> dict:
    out: dict[str, dict] = {}
    for h in hosts:
        tot = {"rest_wh": 0.0, "above_rest_wh": 0.0, "sponsored_wh": 0.0, "metered_wh": 0.0,
               "leased_unmetered_wh": 0.0, "unattributed_wh": 0.0, "covered_h": 0.0, "split_unknown_h": 0.0}
        negative_hours: list[str] = []
        why = None
        t = day_start
        while t < day_end:
            got, rest_w, source = fill.attribute_host_hour(conn, h, t, min(t + timedelta(hours=1), day_end),
                                                           now=day_end)
            if got is None:
                why = source
                break
            for k in ("rest_wh", "above_rest_wh", "sponsored_wh", "metered_wh", "leased_unmetered_wh"):
                tot[k] += getattr(got, k)
            tot["unattributed_wh"] += got.unattributed_wh
            tot["covered_h"] += got.covered_s / 3600.0
            tot["split_unknown_h"] += got.split_unknown_s / 3600.0
            if got.unattributed_wh < -NEGATIVE_TOLERANCE_WH:
                negative_hours.append(f"{t.isoformat()}: {got.unattributed_wh:.3f}")
            why = source
            t += timedelta(hours=1)
        o = {k: round(v, 2) for k, v in tot.items()}
        o["rest"] = why
        o["negative_hours"] = negative_hours
        activity = tot["above_rest_wh"] - tot["sponsored_wh"]
        o["coverage"] = round(tot["metered_wh"] / activity, 4) if activity > 0 else None
        o["closes"] = abs(tot["above_rest_wh"] - tot["sponsored_wh"] - tot["metered_wh"] - tot["unattributed_wh"]) < 1e-6
        out[h] = o
    # The per-act path: every act whose GPU lease ended in the day.
    acts = {h: 0.0 for h in hosts}
    acts_gpu = {h: 0.0 for h in hosts}
    n_acts = 0
    unmeasured: dict[str, int] = {}
    for (corr,) in conn.execute(_ACT_CORRS_SQL, {"since": day_start, "until": day_end}).fetchall():
        got = fill.act_energy_cost(conn, correlation_id=str(corr), since=day_start, until=day_end, now=day_end)
        if "wh_by_host" not in got:
            reason = (got.get("basis") or {}).get("wh", "no lease in the window")
            unmeasured[reason] = unmeasured.get(reason, 0) + 1
            continue
        n_acts += 1
        for h, wh in got["wh_by_host"].items():
            if h in acts:
                acts[h] += wh
                acts_gpu[h] += got["wh_gpu_by_host"].get(h, 0.0)
    for h in hosts:
        o = out[h]
        o["acts_wh"] = round(acts[h], 2)
        o["acts_gpu_wh"] = round(acts_gpu[h], 2)
        o["agrees"] = abs(acts[h] - o["metered_wh"]) <= AGREEMENT_TOLERANCE * o["metered_wh"] + 1.0
    return {"day_start": day_start.isoformat(), "day_end": day_end.isoformat(), "acts_charged": n_acts,
            "acts_unmeasured_by_reason": unmeasured, "hosts": out}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--day", type=date.fromisoformat,
                    default=(datetime.now(timezone.utc) - timedelta(days=1)).date())
    ap.add_argument("--hosts", default="circe,athena")
    args = ap.parse_args(argv)
    start = datetime(args.day.year, args.day.month, args.day.day, tzinfo=timezone.utc)
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    end = min(start + timedelta(days=1), now)
    engine = create_engine(os.environ.get("ORION_ANALYSIS_PG_URI", DEFAULT_URI))
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        out = closure(conn, start, end, [h.strip() for h in args.hosts.split(",") if h.strip()])
        conn.rollback()
    print(json.dumps(out, indent=2, default=str))
    measured = [o for o in out["hosts"].values() if o["covered_h"] > 0]
    if not measured:
        print("UNVERIFIED: no host has a pinned rest and chassis readings for the day", file=sys.stderr)
        return 2
    ok = all(o["closes"] and not o["negative_hours"] and o["agrees"] for o in measured)
    print(f"gates (closure, no negative unattributed, acts ~ metered within {AGREEMENT_TOLERANCE:.0%} + 1 Wh): "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
