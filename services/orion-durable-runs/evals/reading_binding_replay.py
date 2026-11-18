#!/usr/bin/env python3
"""Eval (Replay C): reading as a BINDING family, charged its measured lane cost. READ-ONLY.

Attention patch 3, docs/superpowers/specs/2026-11-07-attention-patch3-market-decides-design.md
("Replay C", acceptance check 5). Patch 3a commits it; patch 3c (reading binds) is gated on it.

Two lanes:

* ``--fixture`` (default; deterministic, no database): families.yaml declares ``hub.world_pulse_read``
  on lanes [agent, agent-gpu2] (one_of); a read act with a measured ``gpu_slot_s_by_lane`` gives
  appetite a lane cost (a read with none gives none); a synthetic reading want through the real
  ``orion.physiology.choice.decide`` and ``orion.physiology.binding.bind_verdict``: a full agent
  lane queues at the pool (never held), and a price deferral is held then re-bid.
* ``--live --days 14``: every real read start in the window bids at its act's real ``opened_at``
  against the ``physiology_state`` row in force, the real act history (reading's acts carrying the
  lane cost the 3a fill computes: ``orion.grammar.act_ledger.late_read_cost``) and one reservation
  book whose exploration floor counts acts started, as the live market does since patch 3b (other
  families at their real opens, reading at its replayed admissions). A held read re-bids at ``bind_verdict``'s ``retry_at``
  until admitted or its hold end: min(family window, 2,900 s) (decided default 6). Pessimistic: the
  lane levels include reading's own real use.

Gate (acceptance check 5): reads held > 1 h <= 5%, reads held to the 2,900 s hold end (admitted only
by ``window_end``) <= 5% (a hold cannot pass 1 h here, so this is the gate that can fail), and
reading's share of Tier 2 ACTS STARTED over the last 7 days >= 5%, with reading's starts taken at the
replayed admission time and the other families' at their real starts. Also printed: verdicts, delays, reading's replayed net, and the floor's share for
reading as the book counts it. Writes nothing (psql in a read-only transaction; SQLAlchemy
connection with SET TRANSACTION READ ONLY).

    PYTHONPATH=. .venv/bin/python services/orion-durable-runs/evals/reading_binding_replay.py [--live --days 14]
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

SVC = Path(__file__).resolve().parents[1]
ROOT = SVC.parents[1]
sys.path[:0] = [str(ROOT), str(SVC), str(ROOT / "scripts" / "analysis")]

from orion.physiology.appetite import ActRecord, build_family_appetite  # noqa: E402
from orion.physiology.binding import bind_verdict  # noqa: E402
from orion.physiology.choice import FundedWant, ReservationBook, acts_started, build_context, decide  # noqa: E402
from orion.physiology.families import load_families  # noqa: E402
from orion.schemas.physiology import PhysiologyStateV1  # noqa: E402
from orion.schemas.physiology_want import WantV1, WantWindowV1  # noqa: E402

FAMILY = "hub.world_pulse_read"
PRODUCER = "orion-durable-runs.admission.reading.binding"
HOLD_END_SEC = 2900.0          # decided default 6: Hub's 3,500 s reclaim minus 600 s
MAX_HELD_OVER_1H = 0.05
# A hold ends at HOLD_END_SEC (0.81 h), so "held > 1 h" cannot happen in this replay: the gate is the
# share of reads held to the hold's end (admitted only by ``window_end``), the binding equivalent of
# "held too long".
MAX_HELD_TO_END = 0.05
MIN_ACTS_SHARE = 0.05
DEFAULT_URI = "postgresql://postgres:postgres@localhost:55432/conjourney"


def _families():
    fams, err = load_families(ROOT / "config" / "physiology" / "families.yaml")
    if err:
        raise SystemExit(f"families.yaml: {err}")
    return fams


def _ts(v: Any) -> Optional[datetime]:
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _q(xs: list[float], q: float) -> Optional[float]:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))] if xs else None


def _want(n: int, act_id: str, corr: Optional[str], opened: datetime, end: datetime) -> WantV1:
    return WantV1(want_id=f"replay-read-{act_id}-{n}", family=FAMILY, producer=PRODUCER, verb="look",
                  object_ref=corr, tier=2, opened_at=opened, window=WantWindowV1(latest=end))


# --- fixture ------------------------------------------------------------------------------------

T0 = datetime(2026, 11, 7, 6, 0, tzinfo=timezone.utc)


def _state(agent_level: float, *, price: Optional[float] = None) -> PhysiologyStateV1:
    lanes = []
    for lane, slots, level in (("agent", 1, agent_level), ("agent-gpu2", 2, 2 * agent_level)):
        r = {"constraint": f"lane:{lane}", "unit": "slot_s", "level": level, "budget": 3600.0 * slots,
             "budget_kind": "capacity", "detail": {"slots": slots, "window_sec": 3600.0}}
        if price is not None:
            r["price"] = {"value": price, "ceiling": max(1.0, price), "bind_state": "binding"}
        lanes.append(r)
    return PhysiologyStateV1.model_validate({"tick_id": f"physiology:{T0.isoformat()}", "generated_at": T0,
                                             "constraints": lanes})


def _read_acts(cost: Optional[dict]) -> list[ActRecord]:
    out = []
    for i in range(40):
        opened = T0 - timedelta(hours=3 * (i + 1))
        out.append(ActRecord(act_id=f"r{i}", family=FAMILY, opened_at=opened, closed_at=opened + timedelta(minutes=20),
                             outcome_value=1.0 if i % 3 else 0.0, outcome_source="producer",
                             signal_id="artifact_produced", cost=dict(cost) if cost else {}))
    return out


def fixture() -> list[str]:
    fails: list[str] = []
    fams = _families()
    cfg = fams.get(FAMILY)
    if cfg is None or list(cfg.lanes) != ["agent", "agent-gpu2"] or cfg.placement != "one_of":
        fails.append(f"families.yaml must declare {FAMILY} lanes [agent, agent-gpu2] one_of, got "
                     f"{getattr(cfg, 'lanes', None)} {getattr(cfg, 'placement', None)}")
    measured = _read_acts({"seconds": 1200.0, "gpu_slot_s_by_lane": {"agent-gpu2": 900.0, "agent": 150.0}})
    app = build_family_appetite(cfg, measured, T0)
    if not app.total_slot_s_q or app.total_slot_s_q <= 0:
        fails.append(f"a read with measured lane time must give appetite a lane cost: {app.total_slot_s_q}")
    blind = build_family_appetite(cfg, _read_acts({"seconds": 1200.0}), T0)
    if blind.total_slot_s_q is not None:
        fails.append(f"a read with no lane time must give no lane cost (never 0): {blind.total_slot_s_q}")

    def bid(state: PhysiologyStateV1) -> tuple[Any, Any]:
        ctx = build_context(fams, measured, [], state, T0)
        w = _want(1, "fx", "corr-fx", T0, T0 + timedelta(seconds=HOLD_END_SEC))
        d = decide(w, state, ctx, ReservationBook(), T0)
        return d, bind_verdict(d, T0, T0 + timedelta(seconds=HOLD_END_SEC))

    d, v = bid(_state(3600.0))
    if (d.net or 0) <= 0:
        fails.append(f"a hungry read with a measured cost should have a positive net: {d.net} ({d.why})")
    if d.reason == "capacity_reserved" and not (v.admit and v.why == "queued_at_pool"):
        fails.append(f"a full agent lane must queue at the pool, never hold: {d.reason} {v.why}")
    if d.reason != "capacity_reserved":
        fails.append(f"a full agent lane should defer on capacity: {d.reason} net={d.net} ({d.why})")
    d2, v2 = bid(_state(0.0))
    if not v2.admit:
        fails.append(f"an empty lane should admit a hungry read: {d2.outcome}/{d2.reason} {v2.why}")
    d3, v3 = bid(_state(1800.0, price=10.0))
    if d3.reason != "price" or v3.admit or v3.retry_at is None or not (T0 < v3.retry_at <= T0 + timedelta(seconds=HOLD_END_SEC)):
        fails.append(f"an expensive lane should defer on price and hold until a re-bid inside the hold end: "
                     f"{d3.reason} net={d3.net} {v3.why} retry_at={v3.retry_at}")
    print(f"fixture: lanes {list(cfg.lanes)} {cfg.placement}; appetite p90 lane cost {app.total_slot_s_q:.0f} slot-s "
          f"(none without measurement); full lane -> {d.reason} -> {v.why}; empty lane -> {d2.outcome} -> {v2.why}; "
          f"priced lane -> {d3.reason} (net {d3.net}) -> {v3.why}")
    return fails


# --- live ---------------------------------------------------------------------------------------

_READ_TARGETS_SQL = """
    SELECT a.act_id, a.actor, a.correlation_id, a.opened_at, a.closed_at, a.cost_json,
           s.status, s.completed_at, s.stage2_status, s.stage2_completed_at, s.created_at AS seed_created_at
      FROM grammar_acts a JOIN world_pulse_read_seed s ON s.request_id::text = a.correlation_id
     WHERE a.actor = :actor AND a.closed_at IS NOT NULL AND a.opened_at >= :since AND a.opened_at < :until
"""


def measured_read_costs(since: datetime, until: datetime, *, uri: Optional[str] = None) -> dict[str, dict]:
    """act_id -> gpu_slot_s_by_lane as the 3a fill computes it (late_read_cost), read-only."""
    from sqlalchemy import create_engine, text

    from orion.grammar.act_ledger import late_read_cost

    engine = create_engine(uri or os.environ.get("ORION_ANALYSIS_PG_URI", DEFAULT_URI))
    out: dict[str, dict] = {}
    now = datetime.now(timezone.utc)
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        for t in conn.execute(text(_READ_TARGETS_SQL), {"actor": FAMILY, "since": since, "until": until}).mappings().all():
            late = late_read_cost(conn, dict(t), now=now)
            if late.lanes:
                out[str(t["act_id"])] = late.lanes
        conn.rollback()
    engine.dispose()
    return out


def run_live(ex: dict, fams, costs: dict[str, dict]) -> dict:
    """Replay every real read start in [start, end) as a binding bid. Pure over the extract."""
    start, end = _ts(ex["start"]), _ts(ex["end"])
    acts, reads = [], []
    for r in ex["acts"]:
        cost = r.get("cost_json") if isinstance(r.get("cost_json"), dict) else {}
        if r["actor"] == FAMILY and str(r["act_id"]) in costs:
            cost = dict(cost or {}, gpu_slot_s_by_lane=costs[str(r["act_id"])])
        a = ActRecord(act_id=str(r["act_id"]), family=str(r["actor"]), opened_at=_ts(r.get("opened_at")),
                      closed_at=_ts(r.get("closed_at")), outcome_value=r.get("outcome_value"),
                      outcome_source=r.get("outcome_source"), signal_id=r.get("signal_id"), cost=cost or {})
        acts.append(a)
        if a.family == FAMILY and a.opened_at is not None and start <= a.opened_at < end:
            reads.append((a.opened_at, a.act_id, r.get("correlation_id")))
    reads.sort()
    from orion.physiology.weights import UseRecord

    uses = [UseRecord(act_id=str(u["act_id"]), family=str(u["actor"]), started_at=_ts(u.get("started_at")),
                      link_status=str(u["link_status"]), used=u.get("used"), window_closed=bool(u.get("window_closed")))
            for u in ex["uses"]]
    states = sorted((PhysiologyStateV1.model_validate(s) for s in ex["states"]), key=lambda s: s.generated_at)
    # The floor counts acts started (patch 3b, as the live market does): other families' real opens,
    # and reading's starts at the times this replay admitted them.
    others = [a for a in acts if a.family != FAMILY]
    cfg = fams.get(FAMILY)
    hold = timedelta(seconds=min(float(cfg.deferrable_window_s), HOLD_END_SEC))
    import bisect

    st_t = [s.generated_at for s in states]

    def state_at(t):
        i = bisect.bisect_right(st_t, t) - 1
        return states[i] if i >= 0 else None

    book = ReservationBook()
    verdicts, reasons, delays, nets = Counter(), Counter(), [], []
    admitted_at: list[datetime] = []
    net_by_act: dict[str, float] = {}
    ctx_key, ctx = None, None
    for t_open, act_id, corr in reads:
        t, stop, n = t_open, t_open + hold, 0
        while True:
            n += 1
            key = t.replace(minute=0, second=0, microsecond=0)
            if key != ctx_key:
                hist = [a for a in acts if (a.closed_at or a.opened_at) is not None and (a.closed_at or a.opened_at) < t]
                ctx = build_context(fams, hist, [u for u in uses if u.started_at and u.started_at < t], state_at(t), t)
                ctx_key = key
            book.seed_started(acts_started(others, fams, t)
                              + [FundedWant(f"replay:{i}", at, FAMILY, 2) for i, at in enumerate(admitted_at)], t)
            w = _want(n, act_id, corr, t_open, stop)
            d = decide(w, state_at(t), ctx, book, t)
            if n == 1 and d.net is not None:
                nets.append(float(d.net))
                net_by_act[act_id] = float(d.net)
            reasons[d.reason] += 1
            v = bind_verdict(d, t, stop)
            if v.admit:
                verdicts[v.why.split(":")[0]] += 1
                delays.append((t - t_open).total_seconds() / 3600.0)
                admitted_at.append(t)
                if d.outcome == "funded":
                    book.note_funded(d.decision_id, FAMILY, 2, t, floor=d.reason == "exploration_floor")
                book.release(w.want_id)
                break
            t = v.retry_at
    tier2 = {f for f, c in fams.families.items() if c.tier == 2}
    week = end - timedelta(days=7)
    # Other Tier 2 families start when they really did; reading starts when the replay admitted it.
    started = Counter(a.family for a in acts if a.family in tier2 and a.family != FAMILY and a.opened_at is not None
                      and week <= a.opened_at < end)
    started[FAMILY] = sum(1 for t in admitted_at if week <= t < end)
    n = len(delays)
    return {
        "reads": n, "costed": sum(1 for _, a, _ in reads if a in costs), "verdicts": dict(verdicts),
        "bid_reasons": dict(reasons), "held": sum(1 for d in delays if d > 0),
        "held_over_1h": sum(1 for d in delays if d > 1.0),
        "held_to_end": verdicts.get("window_end", 0),
        "delay_h_p50": _q(delays, 0.5), "delay_h_p90": _q(delays, 0.9),
        "net_p10": _q(nets, 0.1), "net_p50": statistics.median(nets) if nets else None, "net_p90": _q(nets, 0.9),
        "floor_share_7d": book.funded_share(FAMILY, 2, end),
        "acts_started_share_7d": (started[FAMILY] / sum(started.values())) if started else None,
        "acts_started_7d": dict(started.most_common()),
        "net_by_act": net_by_act,
    }


def live(days: float) -> list[str]:
    import body_economy_shadow_vs_real as svr

    end = datetime.now(timezone.utc).replace(microsecond=0)
    start = end - timedelta(days=days)
    ex = svr.fetch(start, end, history_days=30)
    costs = measured_read_costs(start - timedelta(days=30), end)
    r = run_live(ex, _families(), costs)
    held_share = r["held_over_1h"] / r["reads"] if r["reads"] else 0.0
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    print(f"live {start.isoformat()} .. {end.isoformat()}: {r['reads']} read starts ({r['costed']} with a measured lane cost)")
    print(f"  verdicts={r['verdicts']} bid reasons={r['bid_reasons']}")
    end_share = r["held_to_end"] / r["reads"] if r["reads"] else 0.0
    print(f"  held at all={r['held']} held to the {HOLD_END_SEC:.0f} s hold end={r['held_to_end']} ({end_share:.1%}) "
          f"held > 1 h={r['held_over_1h']} ({held_share:.1%}); delay p50={fmt(r['delay_h_p50'])} h "
          f"p90={fmt(r['delay_h_p90'])} h")
    print(f"  reading net p10={fmt(r['net_p10'])} p50={fmt(r['net_p50'])} p90={fmt(r['net_p90'])}")
    print(f"  floor's share for reading (acts started, 7 d)={fmt(r['floor_share_7d'])}; "
          f"of Tier 2 acts started (7 d)={fmt(r['acts_started_share_7d'])}")
    print(f"  Tier 2 acts started (7 d): {r['acts_started_7d']}")
    fails = []
    if not r["reads"]:
        fails.append("no read starts in the window (UNVERIFIED)")
    if held_share > MAX_HELD_OVER_1H:
        fails.append(f"reads held > 1 h {held_share:.1%} > {MAX_HELD_OVER_1H:.0%}")
    if end_share > MAX_HELD_TO_END:
        fails.append(f"reads held to the hold end {end_share:.1%} > {MAX_HELD_TO_END:.0%}")
    if r["acts_started_share_7d"] is None or r["acts_started_share_7d"] < MIN_ACTS_SHARE:
        fails.append(f"reading's Tier 2 acts-started share {r['acts_started_share_7d']} < {MIN_ACTS_SHARE:.0%}")
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--fixture", action="store_true", help="the default lane")
    ap.add_argument("--days", type=float, default=14.0)
    a = ap.parse_args(argv)
    fails = live(a.days) if a.live else fixture()
    for f in fails:
        print("FAIL", f)
    print("PASS" if not fails else f"FAIL ({len(fails)})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
