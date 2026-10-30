#!/usr/bin/env python3
"""Eval: does curiosity investigate's BINDING bid queue, never silence? (body economy Patch D). READ-ONLY.

Two lanes:

* ``--fixture`` (default; deterministic, no database): synthetic bid streams through the shipped
  verdict rule (``app.run_bids.bind_verdict``), re-bidding at each verdict's ``retry_at`` exactly as
  admission does. Checks: a lane that is ALWAYS full (capacity_reserved, net > 0) holds no run -- the
  pool's queue is the queue; a price deferral holds, re-bids and is admitted once funded; a want the
  market never funds is admitted at the end of its deferrable window (held, never dropped); a market
  with no answer never holds. The not-shipped rule (every capacity deferral holds) is run beside it
  on the saturated stream to show what it would have done (every run held to the window's end).
* ``--live``: ``scripts/analysis/body_economy_binding_replay.py`` over the last ``--days`` of REAL
  curiosity admissions (live Choice code, live physiology_state, read-only). Gate (the "does not
  silence curiosity" check): under the shipped rule at most ``--max-held-share`` of runs are held
  longer than one hour, and no run is held past its window. The not-shipped rule's numbers are printed.

    PYTHONPATH=. .venv/bin/python services/orion-durable-runs/evals/curiosity_binding_replay.py [--live --days 7]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

SVC = Path(__file__).resolve().parents[1]
ROOT = SVC.parents[1]
sys.path[:0] = [str(ROOT), str(SVC), str(ROOT / "scripts" / "analysis")]

from app.run_bids import bind_verdict  # noqa: E402

T0 = datetime(2026, 10, 30, 6, 0, tzinfo=timezone.utc)
WINDOW = timedelta(seconds=21600)        # families.yaml durable.curiosity_investigate deferrable_window_s


def _dec(outcome, reason, net, retry):
    return SimpleNamespace(decision_id=f"d@{reason}", outcome=outcome, reason=reason, net=net, charge=0.1,
                           retry_after_sec=retry, binding_constraint="lane:agent", lane="agent", cost_frac={},
                           why=reason, state_tick_id="t")


def simulate(market, *, capacity_queues_at_pool=True, max_bids=500) -> dict:
    """One run: bid at T0, re-bid at each retry_at until admitted. ``market(n, t)`` -> decision or None."""
    t, end = T0, T0 + WINDOW
    for n in range(max_bids):
        v = bind_verdict(market(n, t), t, end, capacity_queues_at_pool=capacity_queues_at_pool)
        if v.admit:
            return {"verdict": v.why, "held_sec": (t - T0).total_seconds(), "bids": n + 1}
        assert v.retry_at > t and v.retry_at <= end, (v.retry_at, t, end)
        t = v.retry_at
    raise AssertionError("never admitted")


def fixture() -> list[str]:
    fails = []
    full = lambda n, t: _dec("deferred", "capacity_reserved", 0.8, 3600.0)          # noqa: E731
    r = simulate(full)
    if (r["verdict"], r["held_sec"]) != ("queued_at_pool", 0.0):
        fails.append(f"saturated lane held a run: {r}")
    old = simulate(full, capacity_queues_at_pool=False)
    if (old["verdict"], old["held_sec"]) != ("window_end", WINDOW.total_seconds()):
        fails.append(f"not-shipped rule on a saturated lane should hold to the window's end: {old}")
    price = lambda n, t: _dec("funded", "net_positive", 0.4, None) if n >= 3 else _dec("deferred", "price", -0.1, 900.0)  # noqa: E731
    r = simulate(price)
    if (r["verdict"], r["held_sec"], r["bids"]) != ("funded", 2700.0, 4):
        fails.append(f"price deferral should hold 3 x 900 s then fund: {r}")
    never = lambda n, t: _dec("deferred", "price", -0.5, None)                       # noqa: E731
    r = simulate(never)
    if (r["verdict"], r["held_sec"]) != ("window_end", WINDOW.total_seconds()):
        fails.append(f"a never-funded want must be admitted at the window's end, not dropped: {r}")
    for none in (lambda n, t: None, lambda n, t: _dec("deferred", "physiology_unavailable", None, 120.0)):
        r = simulate(none)
        if r["held_sec"] != 0.0 or not r["verdict"].startswith("no_answer"):
            fails.append(f"no market answer must never hold: {r}")
    no_hunger_full = lambda n, t: _dec("deferred", "capacity_reserved", 0.0, 3600.0) if n == 0 else full(n, t)  # noqa: E731
    r = simulate(no_hunger_full)
    if (r["verdict"], r["held_sec"]) != ("queued_at_pool", 3600.0):
        fails.append(f"no hunger (net 0) on a full lane should hold one re-bid, then queue: {r}")
    sated = lambda n, t: _dec("deferred", "sated", 0.0, None) if n == 0 else _dec("funded", "net_positive", 0.3, None)  # noqa: E731
    r = simulate(sated)
    if (r["verdict"], r["held_sec"]) != ("funded", 600.0):
        fails.append(f"sated (no relief time named) should re-bid after the 600 s default: {r}")
    print(f"fixture: {7 - len(fails)}/7 streams as expected; saturated lane: shipped holds 0 s, "
          f"not-shipped holds {old['held_sec']:.0f} s")
    return fails


def live(days: float, max_held_share: float) -> list[str]:
    import json

    import body_economy_binding_replay as rep

    with tempfile.TemporaryDirectory() as d:
        rep.main(["--days", str(days), "--out-json", f"{d}/r.json", "--out-md", f"{d}/r.md"])
        report = json.loads(Path(f"{d}/r.json").read_text())
    fails = []
    shipped = report["bounds"]["shipped"]
    rows = shipped["rows"]
    long_held = [r for r in rows if (r["held_sec"] or 0) > 3600]
    share = len(long_held) / len(rows) if rows else 0.0
    if share > max_held_share:
        fails.append(f"shipped rule holds {len(long_held)}/{len(rows)} runs > 1 h ({share:.0%} > {max_held_share:.0%})")
    past = [r["run_id"] for r in rows if (r["held_sec"] or 0) > WINDOW.total_seconds() + 1]
    if past:
        fails.append(f"held past the deferrable window: {past}")
    print(f"live: {len(rows)} runs; shipped rule held {sum(1 for r in rows if (r['held_sec'] or 0) > 0)} "
          f"({len(long_held)} longer than 1 h)")
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--max-held-share", type=float, default=0.10)
    a = ap.parse_args(argv)
    fails = live(a.days, a.max_held_share) if a.live else fixture()
    for f in fails:
        print("FAIL", f)
    print("PASS" if not fails else f"FAIL ({len(fails)})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
