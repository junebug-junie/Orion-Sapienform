#!/usr/bin/env python3
"""Eval: Hub outreach obeys its body-economy bid (Patch D for hub.outreach).

Two lanes:

* fixture (default, CI): scripted market answers through the replay loop
  (scripts/analysis/outreach_binding_replay.py ``replay_attempt``, which obeys with the same pure rule Hub
  uses, ``orion.physiology.binding.bind_verdict``). Gates: funded and lane-queued wants are never held; a
  price hold re-bids at the market's relief time and is funded; a never-funded want goes at the 1 h
  window's end (held, never dropped) with a bounded number of bids; no answer never holds; a sated hold
  with no relief time named re-bids after the 600 s default. Also: only the doors that bind are replayed
  (Door-A, the strain report and forced sends are not).
* ``--live``: the last ``--days`` of real outreach attempts re-bid with the live market code against the
  stored body state (read-only, ``docker exec orion-athena-sql-db psql``). Gates: nothing held past the
  window; every attempt decided; at most ``MAX_BIDS`` bids per attempt. Prints the replay numbers.

    PYTHONPATH=. python services/orion-hub/evals/run_outreach_binding_replay.py [--live --days 14]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(_ROOT))

import outreach_binding_replay as rep  # noqa: E402

T0 = datetime(2026, 11, 1, 18, 0, tzinfo=timezone.utc)


def _dec(outcome, reason, net, retry):
    return SimpleNamespace(outcome=outcome, reason=reason, net=net, retry_after_sec=retry, binding_constraint=None,
                           decision_id="d", why=f"test {reason}")


def _attempt(door="endogenous_tick"):
    return {"decision_id": "x", "door": door, "at": T0, "reason": "sent", "correlation_id": None,
            "cap_counted": True}


def fixture() -> list[str]:
    fails: list[str] = []

    def run(fn):
        return rep.replay_attempt(_attempt(), lambda a, t, n: fn(n, t))

    def expect(name, r, verdict, held, bids=None):
        if r["verdict"] != verdict or r["held_sec"] != held or (bids is not None and r["n_bids"] != bids):
            fails.append(f"{name}: got verdict={r['verdict']} held={r['held_sec']} bids={r['n_bids']}")

    expect("funded", run(lambda n, t: _dec("funded", "net_positive", 0.3, None)), "funded", 0.0, 1)
    expect("lane full but worth it", run(lambda n, t: _dec("deferred", "capacity_reserved", 0.3, 200.0)),
           "queued_at_pool", 0.0, 1)
    expect("price 3 x 900 s then funded",
           run(lambda n, t: _dec("funded", "net_positive", 0.4, None) if n >= 3 else _dec("deferred", "price", -0.1, 900.0)),
           "funded", 2700.0, 4)
    never = run(lambda n, t: _dec("deferred", "price", -0.5, 120.0))
    expect("never funded -> the window's end", never, "window_end", 3600.0)
    if never["n_bids"] > rep.MAX_BIDS:
        fails.append(f"never funded: {never['n_bids']} bids > {rep.MAX_BIDS}")
    for name, fn in (("no answer", lambda n, t: None),
                     ("market unavailable", lambda n, t: _dec("deferred", "physiology_unavailable", None, 120.0))):
        r = run(fn)
        if r["held_sec"] != 0.0 or not r["verdict"].startswith("no_answer"):
            fails.append(f"{name} must never hold: {r['verdict']} {r['held_sec']}")
    expect("sated, no relief time -> 600 s default",
           run(lambda n, t: _dec("deferred", "sated", 0.0, None) if n == 0 else _dec("funded", "net_positive", 0.3, None)),
           "funded", 600.0, 2)
    expect("net 0 on a full lane holds one re-bid",
           run(lambda n, t: _dec("deferred", "capacity_reserved", 0.0, 1200.0) if n == 0
               else _dec("deferred", "capacity_reserved", 0.2, 1200.0)),
           "queued_at_pool", 1200.0, 2)

    rows = [
        {"decision_id": "a", "decided_at": T0.isoformat(), "reason": "sent", "forced": False, "source": None,
         "correlation_id": "c1"},
        {"decision_id": "b", "decided_at": T0.isoformat(), "reason": "sent", "forced": False, "source": "curiosity_outreach"},
        {"decision_id": "c", "decided_at": T0.isoformat(), "reason": "sent", "forced": False, "source": "strain_report"},
        {"decision_id": "d", "decided_at": T0.isoformat(), "reason": "sent", "forced": True, "source": None},
        {"decision_id": "e", "decided_at": T0.isoformat(), "reason": "quiet_hours", "forced": False, "source": None},
        {"decision_id": "f", "decided_at": T0.isoformat(), "reason": "orion_passed", "forced": False,
         "source": "reading_finding_outreach"},
        {"decision_id": "g", "decided_at": T0.isoformat(), "reason": "juniper_spoke_after_generation", "forced": False,
         "source": None},
        {"decision_id": "h", "decided_at": T0.isoformat(), "reason": "no_question", "forced": False,
         "source": "office_reach_out"},
    ]
    got = rep.attempts_from_rows(rows, {"c1": T0 - timedelta(seconds=90)})
    if [a["decision_id"] for a in got] != ["a", "f", "g", "h"]:
        fails.append(f"door filter: {[a['decision_id'] for a in got]}")
    elif got[0]["at"] != T0 - timedelta(seconds=90) or not got[0]["cap_counted"] or not got[1]["cap_counted"]:
        fails.append(f"attempt time / cap-counted wrong: {got[0]} {got[1]}")
    print(f"fixture: {'PASS' if not fails else 'FAIL'} ({len(fails)} failure(s))")
    return fails


def live(days: float) -> list[str]:
    import json

    with tempfile.TemporaryDirectory() as d:
        rep.main(["--days", str(days), "--out-json", f"{d}/r.json", "--out-md", f"{d}/r.md"])
        report = json.loads(Path(f"{d}/r.json").read_text())
    fails = []
    rows = report["rows"]
    past = [r["decision_id"] for r in rows if (r["held_sec"] or 0) > rep.WINDOW_SEC + 1]
    if past:
        fails.append(f"held past the 1 h window: {past}")
    never = [r["decision_id"] for r in rows if r["verdict"] == "never_decided"]
    if never:
        fails.append(f"never decided: {never}")
    many = [r["decision_id"] for r in rows if r["n_bids"] > rep.MAX_BIDS]
    if many:
        fails.append(f"more than {rep.MAX_BIDS} bids: {many}")
    s = report["summary"].get("all", {})
    print(f"live: {s.get('attempts', 0)} attempts; let through at the first bid {s.get('let_through_at_first_bid')}; "
          f"held {s.get('held_then_admitted')} (median {s.get('held_sec_median')} s, max {s.get('held_sec_max')} s)")
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--days", type=float, default=14.0)
    a = ap.parse_args(argv)
    fails = fixture()
    if a.live:
        fails += live(a.days)
    for f in fails:
        print("FAIL:", f)
    print("PASS" if not fails else "FAIL")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
