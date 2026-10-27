#!/usr/bin/env python3
"""Eval wrapper for the shadow-vs-real disagreement report (body economy Patch C3a). READ-ONLY.

Two lanes:

* ``--fixture`` (default; deterministic, CI): a small hand-built window -- one want per cell, plus
  each out-of-table case -- through the report's own ``classify`` / ``summarize``. Fails on any
  misfiled row.
* ``--live``: runs ``body_economy_shadow_vs_real.py`` on the live database over ``--days`` and
  checks its invariants: every decision lands in exactly one place (a cell or one named exclusion),
  the cells add up to ``in_table``, no "not run" is claimed for a family with no act producer, and
  (acceptance check A3, once bid sites are live) the share of in-table funded LEDGER wants that
  joined an act is printed. A3 is reported, not enforced, until the bid sites have run a week.

    .venv/bin/python scripts/analysis/eval_body_economy_shadow_vs_real.py [--live --days 7]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))

import body_economy_shadow_vs_real as svr  # noqa: E402

PLACES = (*svr.CELLS, "no_market", "no_act_producer", "before_act_producer", "synthetic_probe")


def check_invariants(report: dict) -> list[str]:
    fails = []
    for name, sec in report["sources"].items():
        for r in sec["rows"]:
            placed = [p for p in PLACES if r.get("cell") == p or r.get("excluded") == p]
            if len(placed) != 1:
                fails.append(f"{name}: {r['decision_id']} placed {placed}")
            if r.get("excluded") in ("no_act_producer", "before_act_producer") and r.get("ran") is False:
                fails.append(f"{name}: {r['decision_id']} claims 'not run' with no act producer")
        for fam, x in sec["summary"]["families"].items():
            if sum(x[c] for c in svr.CELLS) != x["in_table"]:
                fails.append(f"{name}: {fam} cells do not add up to in_table")
            if x["bids"] != x["funded"] + x["deferred"]:
                fails.append(f"{name}: {fam} bids != funded + deferred")
    return fails


def fixture_report() -> dict:
    t0 = datetime(2026, 10, 25, 12, tzinfo=timezone.utc)
    fam = "durable.self_inquiry"

    def d(i, outcome, reason="net_positive", producer="orion-durable-runs.admission", family=fam, at=t0):
        return svr.Decision(f"d{i}", family, producer, at, outcome, reason, "lane:agent" if outcome == "deferred" else None,
                            f"c{i}", f"durable_run:r{i}", at + timedelta(hours=6))
    acts = [svr.Act(fam, t0 + timedelta(hours=1), "c1", None), svr.Act(fam, t0 + timedelta(hours=2), "c3", None),
            svr.Act(fam, t0 - timedelta(days=1), "old", None)]
    decs = [d(1, "funded"), d(2, "funded"), d(3, "deferred", "capacity_reserved"), d(4, "deferred", "price"),
            d(5, "deferred", "physiology_unavailable"), d(6, "funded", producer="probe"),
            d(7, "funded", family="concept_induction"), d(8, "funded", at=t0 - timedelta(days=3))]
    rows = svr.classify(decs, acts)
    return {"start": "", "end": "", "sources": {"fixture": {"rows": rows, "summary": svr.summarize(rows)}}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--days", type=float, default=7.0)
    a = ap.parse_args(argv)
    if not a.live:
        rep = fixture_report()
        rows = rep["sources"]["fixture"]["rows"]
        want = ["funded_ran", "funded_not_run", "deferred_ran", "deferred_not_run",
                "no_market", "synthetic_probe", "no_act_producer", "before_act_producer"]
        got = [r["cell"] or r["excluded"] for r in rows]
        fails = check_invariants(rep) + ([f"fixture placed {got}, expected {want}"] if got != want else [])
        print("fixture:", "PASS" if not fails else "FAIL", *fails, sep="\n  ")
        return 1 if fails else 0
    with tempfile.TemporaryDirectory() as td:
        svr.main(["--days", str(a.days), "--out-json", f"{td}/r.json", "--out-md", f"{td}/r.md"])
        import json

        rep = json.loads(Path(f"{td}/r.json").read_text())
    fails = check_invariants(rep)
    led = rep["sources"].get("ledger", {}).get("rows", [])
    funded = [r for r in led if r["outcome"] == "funded" and r["cell"]]
    joined = sum(1 for r in funded if r["ran"])
    print(f"A3 (reported): {joined}/{len(funded)} in-table funded ledger wants joined an act")
    print("live invariants:", "PASS" if not fails else "FAIL", *fails, sep="\n  ")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
