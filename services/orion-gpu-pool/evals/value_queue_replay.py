#!/usr/bin/env python3
"""Eval (Replay D): the GPU pool's background agent hold queue served by value instead of arrival. READ-ONLY.

Attention patch 3, docs/superpowers/specs/2026-11-07-attention-patch3-market-decides-design.md
("Replay D", acceptance check 3). Patch 3a commits it; patch 3b builds the queue order it measures.

The rule (decided 2026-11-10, defaults 2 and 5): inside the background class a waiting hold is scored
``net + aging x hours waited``; a later hold is served before an earlier one only when its score is
at least MARGIN (0.10) higher; two holds of the same family keep arrival order; a hold with no bid
(upkeep: compactors, orion-day, curiosity compaction, ...) keeps its arrival place ahead of every hold
that arrived after it. ``pick`` is that rule.

Two lanes:

* ``--fixture`` (default; deterministic, no database): synthetic queues: a 0.3-lower hold wins after
  3 h at aging 0.10/h and not before; a gap under the margin keeps arrival order; the same family
  keeps arrival order; upkeep keeps its place; with aging 0 and no margin gap nothing reorders.
* ``--live --days 14 --aging 0.10``: at each real grant of a background agent hold (gpu_pool_events
  ``admitted`` -> ``granted`` of ``durable-runs:*`` holders, priority background, work_class agent;
  gpu_pool_leases is pruned to about 7 days, the events are not), the holds waiting then are picked by
  ``pick``. Reading's holds carry the net Replay C gives the read
  (services/orion-durable-runs/evals/reading_binding_replay.py); a bidding run carries its last
  want_ledger net before the grant; anything else is upkeep. A rank-order approximation: service
  times stay as they really were, so it counts changes; it does not simulate new waits.
  Gate: the granted family changes on more than 0 grants (the queue really chooses).

    PYTHONPATH=. .venv/bin/python services/orion-gpu-pool/evals/value_queue_replay.py [--live --days 14 --aging 0.10]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "services" / "orion-durable-runs" / "evals"), str(ROOT / "scripts" / "analysis")]

MARGIN = 0.10
DEFAULT_AGING = 0.10
DEFAULT_URI = "postgresql://postgres:postgres@localhost:55432/conjourney"


@dataclass
class Hold:
    lease_id: str
    family: str
    net: Optional[float]          # None: upkeep (no bid)
    queued_at: datetime
    granted_at: Optional[datetime] = None


def score(h: Hold, now: datetime, aging: float) -> float:
    return float(h.net) + aging * max(0.0, (now - h.queued_at).total_seconds()) / 3600.0


def pick(waiting: Sequence[Hold], now: datetime, *, aging: float = DEFAULT_AGING, margin: float = MARGIN) -> Hold:
    """The hold the value-ordered queue serves at ``now`` (spec section 1 rule)."""
    order = sorted(waiting, key=lambda h: h.queued_at)
    best = order[0]
    for o in order[1:]:
        if best.net is None or o.net is None:
            break                      # upkeep keeps its place: nothing after it may jump it
        if o.family == best.family:
            continue                   # same family: arrival order
        if score(o, now, aging) >= score(best, now, aging) + margin:
            best = o
    return best


# --- fixture ------------------------------------------------------------------------------------

T0 = datetime(2026, 11, 7, 6, 0, tzinfo=timezone.utc)


def fixture() -> list[str]:
    fails: list[str] = []
    h = lambda lid, fam, net, mins: Hold(lid, fam, net, T0 + timedelta(minutes=mins))  # noqa: E731
    # The higher-value later hold wins at once (gap 0.3 >= margin).
    low, high = h("a", "reading", 0.5, 0), h("b", "curiosity", 0.8, 1)
    if pick([low, high], T0 + timedelta(minutes=2)).lease_id != "b":
        fails.append("a later hold 0.3 better must be served first")
    # Aging: the earlier 0.3-lower hold wins once it has waited 3 h longer than the other.
    old, new = h("a", "reading", 0.5, 0), h("b", "curiosity", 0.8, 180)
    if pick([old, new], T0 + timedelta(minutes=181)).lease_id != "a":
        fails.append("after 3 h of extra wait a 0.3-lower hold must win (aging 0.10/h)")
    old2, new2 = h("a", "reading", 0.5, 0), h("b", "curiosity", 0.8, 60)
    if pick([old2, new2], T0 + timedelta(minutes=61)).lease_id != "b":
        fails.append("after 1 h of extra wait the 0.3 gap still decides")
    # Margin: a gap under 0.10 keeps arrival order.
    if pick([h("a", "reading", 0.70, 0), h("b", "curiosity", 0.75, 1)], T0 + timedelta(minutes=2)).lease_id != "a":
        fails.append("a gap under the margin must keep arrival order")
    # Same family keeps arrival order, however large the gap.
    if pick([h("a", "reading", 0.1, 0), h("b", "reading", 0.9, 1)], T0 + timedelta(minutes=2)).lease_id != "a":
        fails.append("the same family must keep arrival order")
    # Upkeep keeps its place: a later bid never jumps it, and it never jumps an earlier bid.
    if pick([h("u", "upkeep", None, 0), h("b", "curiosity", 0.9, 1)], T0 + timedelta(minutes=2)).lease_id != "u":
        fails.append("upkeep must keep its place ahead of a later bid")
    q = [h("a", "reading", 0.2, 0), h("u", "upkeep", None, 1), h("b", "curiosity", 0.9, 2)]
    if pick(q, T0 + timedelta(minutes=3)).lease_id != "a":
        fails.append("a bid after upkeep must not be served before the upkeep (nor before what came first)")
    # Aging 0, margin 0: pure value; with aging only time can reorder.
    if pick([low, high], T0 + timedelta(minutes=2), aging=0.0, margin=0.0).lease_id != "b":
        fails.append("aging 0, margin 0: the higher net wins")
    print(f"fixture: {8 - len(fails)}/8 queues as expected (margin {MARGIN}, aging {DEFAULT_AGING}/h)")
    return fails


# --- live ---------------------------------------------------------------------------------------

_HOLDS_SQL = """
    SELECT lease_id, max(holder) AS holder,
           min(generated_at) FILTER (WHERE event = 'admitted') AS q,
           min(generated_at) FILTER (WHERE event = 'granted') AS g
      FROM gpu_pool_events
     WHERE holder LIKE 'durable-runs:%' AND priority = 'background' AND work_class = 'agent'
       AND generated_at >= :since AND generated_at < :until
     GROUP BY lease_id
"""
_READ_RUNS_SQL = """
    SELECT r.run_id, r.stage, a.act_id
      FROM reading_durable_turn r
      JOIN world_pulse_read_seed s ON s.seed_id = r.seed_id
      LEFT JOIN grammar_acts a ON a.actor = 'hub.world_pulse_read' AND a.correlation_id = s.request_id::text
     WHERE r.run_id = ANY(:runs)
"""
_LEDGER_SQL = """
    SELECT want_id, family, net, decided_at FROM want_ledger
     WHERE want_id = ANY(:wids) AND net IS NOT NULL ORDER BY decided_at
"""


def _aware(t: datetime) -> datetime:
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def load_holds(conn, start: datetime, end: datetime, read_nets: dict[str, float]) -> list[Hold]:
    from sqlalchemy import text

    rows = [r for r in conn.execute(text(_HOLDS_SQL), {"since": start - timedelta(days=1), "until": end}).mappings()
            if r["g"] is not None and start <= _aware(r["g"]) < end]
    wids = sorted({str(r["holder"]).split(":", 1)[1] for r in rows})
    reads = {str(x["run_id"]): (int(x["stage"]), x["act_id"]) for x in
             conn.execute(text(_READ_RUNS_SQL), {"runs": [w for w in wids if w.startswith("reading-")]}).mappings()}
    ledger: dict[str, list] = defaultdict(list)
    for x in conn.execute(text(_LEDGER_SQL), {"wids": [w for w in wids if w not in reads]}).mappings():
        ledger[str(x["want_id"])].append((_aware(x["decided_at"]), str(x["family"]), float(x["net"])))
    holds = []
    for r in rows:
        wid = str(r["holder"]).split(":", 1)[1]
        g = _aware(r["g"])
        q = _aware(r["q"]) if r["q"] is not None else g
        if wid in reads:
            stage, act_id = reads[wid]
            net = read_nets.get(str(act_id)) if act_id else None
            # A read with no replayed net (no act: refused before work) bids nothing: treat as upkeep.
            fam = f"reading(stage{stage})" if net is not None else "no_bid:reading"
        else:
            before = [d for d in ledger.get(wid, []) if d[0] <= g]
            net, fam = (before[-1][2], before[-1][1]) if before else (None, "no_bid:" + wid.split("-")[0].split(":")[0])
        holds.append(Hold(str(r["lease_id"]), fam, net, q, g))
    holds.sort(key=lambda x: x.granted_at)
    return holds


def replay(holds: list[Hold], *, aging: float, margin: float = MARGIN) -> dict:
    base = lambda f: f.split("(")[0]  # noqa: E731 -- reading stage 1 and 2 are one family
    contested = changed = 0
    pair: Counter = Counter()
    gain: Counter = Counter()
    loss: Counter = Counter()
    for h in holds:
        waiting = [o for o in holds if o is not h and o.queued_at <= h.granted_at < o.granted_at
                   and (o.granted_at - h.granted_at).total_seconds() > 5]
        if not waiting:
            continue
        contested += 1
        best = pick([h] + waiting, h.granted_at, aging=aging, margin=margin)
        if best is not h and base(best.family) != base(h.family):
            changed += 1
            pair[(best.family, h.family)] += 1
            gain[best.family] += 1
            loss[h.family] += 1
    return {"holds": len(holds), "by_family": dict(Counter(h.family for h in holds).most_common()),
            "contested": contested, "changed": changed,
            "changed_of_contested": changed / contested if contested else 0.0,
            "changed_of_all": changed / len(holds) if holds else 0.0,
            "gain": dict(gain.most_common()), "loss": dict(loss.most_common()), "pairs": pair.most_common(8)}


def live(days: float, aging: float) -> list[str]:
    import body_economy_shadow_vs_real as svr
    from sqlalchemy import create_engine, text

    import reading_binding_replay as c

    end = datetime.now(timezone.utc).replace(microsecond=0)
    start = end - timedelta(days=days)
    ex = svr.fetch(start, end, history_days=30)
    rc = c.run_live(ex, c._families(), c.measured_read_costs(start - timedelta(days=30), end))
    engine = create_engine(os.environ.get("ORION_ANALYSIS_PG_URI", DEFAULT_URI))
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        holds = load_holds(conn, start, end, rc["net_by_act"])
        conn.rollback()
    engine.dispose()
    out = {a: replay(holds, aging=a) for a in sorted({aging, 0.0})}
    for a, r in out.items():
        print(f"aging={a}/h margin={MARGIN}: holds={r['holds']} by family={r['by_family']}")
        print(f"  grants with another hold waiting={r['contested']}; granted family changes={r['changed']} "
              f"({r['changed_of_contested']:.1%} of contested, {r['changed_of_all']:.1%} of all)")
        print(f"  would be served sooner: {r['gain']}")
        print(f"  would wait longer: {r['loss']}")
        for (w, l), n in r["pairs"]:
            print(f"    {w} over {l}: {n}")
    fails = []
    if not out[aging]["holds"]:
        fails.append("no background agent holds in the window (UNVERIFIED)")
    elif out[aging]["changed"] == 0:
        fails.append("the value order never changes the granted family: the queue does not choose")
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--fixture", action="store_true", help="the default lane")
    ap.add_argument("--days", type=float, default=14.0)
    ap.add_argument("--aging", type=float, default=DEFAULT_AGING)
    a = ap.parse_args(argv)
    fails = live(a.days, a.aging) if a.live else fixture()
    for f in fails:
        print("FAIL", f)
    print("PASS" if not fails else f"FAIL ({len(fails)})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
