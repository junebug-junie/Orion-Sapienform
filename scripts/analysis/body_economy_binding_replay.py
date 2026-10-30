#!/usr/bin/env python3
"""Binding replay (READ-ONLY): what Patch D's binding bid would have done to each real run.

Body economy Patch D (curiosity investigate first). Design:
docs/superpowers/specs/2026-10-23-organism-body-economy-design.md, Patch D row.

The shadow-vs-real report (``body_economy_shadow_vs_real.py``) asks "did the ONE shadow decision at
admission agree with what happened?". That is the wrong question for a binding bid that QUEUES: a
capacity deferral now means "held, bid again later", not "never ran". This replay answers the right
one, per run of a binding family over the window:

* held at all? funded at its first bid, funded on a later re-bid, or admitted at the end of its
  family's deferrable window (``window_end``: never dropped), or admitted because the market had no
  answer;
* how long it was held (admission -> the verdict that let it ask the pool);
* whether that is LATER than the run really started (its act's real open): the extra delay binding
  would have added.

Every want-family admission in the window is replayed in time order against the ``physiology_state``
row in force and one reservation book, exactly as ``replay_admissions`` does. Non-binding families bid
once at admission (shadow, as live). A binding family's run re-bids at ``bind_verdict``'s
``retry_at`` -- the same pure function admission uses (``app.run_bids``) -- until it is admitted.

Pessimistic by construction: the lane levels are the REAL ones, which include the held runs' own
real use at their real times. A held run would not have been using the lane then, so real relief
comes sooner than replayed.

Writes nothing to any database (reads the ``--cache`` extract or ``body_economy_shadow_vs_real.fetch``).

    .venv/bin/python scripts/analysis/body_economy_binding_replay.py --days 7 \
        --out-json /tmp/x.json --out-md /tmp/x.md [--cache /tmp/x.cache.json] [--family durable.curiosity_investigate]
"""
from __future__ import annotations

import argparse
import heapq
import itertools
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "services" / "orion-durable-runs"))

import body_economy_shadow_vs_real as svr  # noqa: E402


def _q(xs: list[float], q: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))], 1)


def without_own_use(states: list[dict], acts: list[dict], binding: frozenset[str], window_sec: float = 3600.0) -> list[dict]:
    """The optimistic bound: each state's lane levels minus the binding families' OWN real use in the
    trailing window (each act's per-lane slot-seconds spread evenly over its open..close). A held run
    was not using the lane then; this removes that use but does not add it back at the later time the
    run would really have run, so relief comes sooner than it would have. Pure."""
    from orion.schemas.physiology import LANE_PREFIX

    spans = []
    for a in acts:
        if a.get("actor") not in binding:
            continue
        o, c = svr._ts(a.get("opened_at")), svr._ts(a.get("closed_at"))
        by_lane = (a.get("cost_json") or {}).get("gpu_slot_s_by_lane") if isinstance(a.get("cost_json"), dict) else None
        if o is None or c is None or c <= o or not isinstance(by_lane, dict):
            continue
        spans.append((o, c, {str(k): float(v) for k, v in by_lane.items() if isinstance(v, (int, float))}))
    out = []
    for s in states:
        t = svr._ts(s["generated_at"])
        lo = t - timedelta(seconds=window_sec)
        own: dict[str, float] = {}
        for o, c, by_lane in spans:
            ov = (min(c, t) - max(o, lo)).total_seconds()
            if ov <= 0:
                continue
            frac = ov / (c - o).total_seconds()
            for ln, v in by_lane.items():
                own[ln] = own.get(ln, 0.0) + v * frac
        cons = []
        for c in s["constraints"]:
            k = c.get("constraint", "")
            if k.startswith(LANE_PREFIX) and c.get("level") is not None and own.get(k[len(LANE_PREFIX):]):
                c = {**c, "level": max(0.0, float(c["level"]) - own[k[len(LANE_PREFIX):]])}
            cons.append(c)
        out.append({**s, "constraints": cons})
    return out


def replay(ex: dict, fams, binding: frozenset[str], *, subtract_own_use: bool = False,
           capacity_queues_at_pool: bool = True, prime: Optional[Any] = None) -> list[dict]:
    """One row per run of a ``binding`` family: its bids, verdict, hold and extra delay.

    ``prime``: optional ``(family, t) -> PrimeRegulationV1 | None`` -- a Prime regulate charge in force at
    ``t`` (Orion Prime step 3), applied to the binding family's bids exactly as the market applies it.
    Each bid then also records the charge's effect (``orion.physiology.prime_charge.effect_of``)."""
    from app.run_acts import run_act_family
    from app.run_bids import bind_verdict, hold_end, want_for_run

    from orion.physiology.appetite import ActRecord
    from orion.physiology.choice import FundedWant, ReservationBook, SYNTHETIC_PRODUCERS, build_context, decide
    from orion.physiology.weights import UseRecord, weight_series
    from orion.schemas.physiology import PhysiologyStateV1

    ts = svr._ts
    acts = [ActRecord(act_id=str(r["act_id"]), family=str(r["actor"]), opened_at=ts(r.get("opened_at")),
                      closed_at=ts(r.get("closed_at")), outcome_value=r.get("outcome_value"),
                      outcome_source=r.get("outcome_source"), signal_id=r.get("signal_id"),
                      cost=r.get("cost_json") if isinstance(r.get("cost_json"), dict) else {}) for r in ex["acts"]]
    uses = [UseRecord(act_id=str(r["act_id"]), family=str(r["actor"]), started_at=ts(r.get("started_at")),
                      link_status=str(r["link_status"]), used=r.get("used"), window_closed=bool(r.get("window_closed")))
            for r in ex["uses"]]
    raw_states = without_own_use(ex["states"], ex["acts"], binding) if subtract_own_use else ex["states"]
    states = [PhysiologyStateV1.model_validate(s) for s in raw_states]
    start, end = ts(ex["start"]), ts(ex["end"])
    series = weight_series(fams, acts, uses, start=(start - timedelta(days=1)).date(), end=end.date())
    book = ReservationBook()
    book.seed_funded([FundedWant(str(r["decision_id"]), ts(r["decided_at"]), str(r["family"]), int(r["tier"]))
                      for r in ex.get("ledger_funded_before") or [] if r.get("producer") not in SYNTHETIC_PRODUCERS],
                     start)
    act_by_ref: dict[str, dict] = {}
    for r in ex["acts"]:
        if r.get("object_ref") and r.get("opened_at"):
            act_by_ref.setdefault(r["object_ref"], r)

    seq = itertools.count()
    heap: list = []        # (t, order, seq, kind, payload): bids before opens before closes

    def push(t, order, kind, payload):
        heapq.heappush(heap, (t, order, next(seq), kind, payload))

    runs: dict[str, dict] = {}
    for adm in ex["admissions"]:
        req = adm["request"] if isinstance(adm["request"], dict) else json.loads(adm["request"])
        fam = run_act_family(req)
        if fam is None:
            continue
        t = ts(adm["created_at"])
        act = act_by_ref.get(f"durable_run:{adm['run_id']}")
        real_open = ts(act["opened_at"]) if act is not None else None
        bound = fam.actor in binding
        if bound:
            runs[adm["run_id"]] = {"run_id": adm["run_id"], "family": fam.actor, "admitted_at": t,
                                   "real_open": real_open, "bids": [], "verdict": None, "released_at": None}
        push(t, 0, "bid", {"req": req, "family": fam, "bound": bound})
        if real_open is not None:
            # A binding run that is held past its real open opens when it is released (below).
            if not bound:
                push(real_open, 1, "open", adm["run_id"])
        if adm.get("terminal") and adm.get("updated_at"):
            push(ts(adm["updated_at"]), 2, "close", adm["run_id"])

    si, current = 0, None
    ctx_cache: dict = {}
    while heap:
        t, _o, _s, kind, p = heapq.heappop(heap)
        if kind == "open":
            book.mark_opened(p, t)
            continue
        if kind == "close":
            book.release(p)
            continue
        while si < len(states) and states[si].generated_at <= t:
            current = states[si]
            si += 1
        cfg = fams.get(p["family"].actor)
        want = want_for_run(p["req"], cfg.tier if cfg is not None else 0)
        if want is None:
            continue
        key = (current.tick_id if current is not None else None, t.replace(second=0, microsecond=0))
        if key not in ctx_cache:
            before = [a for a in acts if (a.closed_at or a.opened_at) is not None and (a.closed_at or a.opened_at) < t]
            ctx_cache.clear()
            ctx_cache[key] = build_context(fams, before, [u for u in uses if u.started_at and u.started_at < t],
                                           current, t, weights=series.get(t.date(), {}))
        if p["bound"] and t > end:
            # Past the extract: no physiology state to bid against. Censored, not "no answer"; never
            # decided, so the book is not touched.
            runs[want.want_id]["verdict"] = "censored_at_data_end"
            continue
        sig = prime(want.family, t) if (prime is not None and p["bound"]) else None
        d = decide(want, current, ctx_cache[key], book, t, reserve_at_open=True, prime=sig)
        if not p["bound"]:
            continue
        run = runs[want.want_id]
        effect = None
        if sig is not None and d.net is not None and d.family_weight is not None:
            from orion.physiology.prime_charge import amount, effect_of

            effect = effect_of(sig.direction, net_with=d.net, net_without=d.net + amount(sig, d.family_weight),
                               floor_funded=d.reason == "exploration_floor")
        v = bind_verdict(d, t, hold_end(want, cfg.deferrable_window_s if cfg is not None else 0.0),
                         capacity_queues_at_pool=capacity_queues_at_pool)
        run["bids"].append({"at": t.isoformat(), "outcome": d.outcome, "reason": d.reason,
                            "binding": d.binding_constraint, "net": d.net, "retry_after_sec": d.retry_after_sec,
                            "verdict": v.why, "prime_effect": effect})
        if v.admit:
            run["verdict"], run["released_at"] = v.why, t
            if d.outcome != "funded":
                book.release(want.want_id)   # nothing reserved, but keep the book honest
            if run["real_open"] is not None:
                push(max(run["real_open"], t), 1, "open", want.want_id)
        else:
            push(v.retry_at, 0, "bid", p)
    rows = []
    for r in runs.values():
        held = (r["released_at"] - r["admitted_at"]).total_seconds() if r["released_at"] else None
        extra = None
        if r["real_open"] is not None and r["released_at"] is not None:
            extra = max(0.0, (r["released_at"] - r["real_open"]).total_seconds())
        rows.append({**r, "admitted_at": r["admitted_at"].isoformat(),
                     "real_open": r["real_open"].isoformat() if r["real_open"] else None,
                     "released_at": r["released_at"].isoformat() if r["released_at"] else None,
                     "held_sec": round(held, 1) if held is not None else None,
                     "extra_delay_sec": round(extra, 1) if extra is not None else None,
                     "n_bids": len(r["bids"])})
    return rows


def summarize(rows: list[dict]) -> dict:
    verdicts = Counter(r["verdict"] or "never_decided" for r in rows)
    first = Counter(f"{r['bids'][0]['outcome']}:{r['bids'][0]['reason']}" for r in rows if r["bids"])
    defer_reasons = Counter(b["reason"] for r in rows for b in r["bids"] if b["outcome"] == "deferred")
    binds = Counter(b["binding"] for r in rows for b in r["bids"] if b["outcome"] == "deferred" and b["binding"])
    held = [r["held_sec"] for r in rows if r["held_sec"] is not None and r["held_sec"] > 0]
    extra = [r["extra_delay_sec"] for r in rows if r["extra_delay_sec"]]
    n = len(rows)
    funded_first = sum(1 for r in rows if r["bids"] and r["bids"][0]["outcome"] == "funded")
    return {
        "runs": n, "verdicts": dict(verdicts), "first_bid": dict(first),
        "funded_first_bid": funded_first,
        "funded_after_hold": sum(1 for r in rows if r["verdict"] == "funded" and r["n_bids"] > 1),
        "admitted_at_window_end": verdicts.get("window_end", 0),
        "queued_at_pool": verdicts.get("queued_at_pool", 0), "censored": verdicts.get("censored_at_data_end", 0),
        "no_answer": sum(v for k, v in verdicts.items() if k.startswith("no_answer")),
        "held_runs": len(held), "held_sec_median": _q(held, 0.5), "held_sec_p90": _q(held, 0.9),
        "held_sec_max": max(held) if held else None,
        "runs_delayed_past_real_open": len(extra), "extra_delay_sec_median": _q(extra, 0.5),
        "extra_delay_sec_p90": _q(extra, 0.9), "bids_total": sum(r["n_bids"] for r in rows),
        "deferral_reasons_all_bids": dict(defer_reasons), "binding_constraints_all_bids": binds.most_common(5),
    }


# (name, subtract the binding family's own use, capacity deferral queues at the pool)
POLICIES = (
    ("shipped", False, True),
    ("hold_on_capacity_pessimistic", False, False),
    ("hold_on_capacity_optimistic", True, False),
)
BOUND_NOTE = {
    "shipped": ("the shipped policy: a capacity deferral with net > 0 joins the GPU pool's queue; price, sated "
                "and net <= 0 deferrals hold and re-bid. Real lane levels"),
    "hold_on_capacity_pessimistic": ("NOT shipped: every deferral holds, capacity included. Real lane levels "
                                     "(they include the held runs' own real use at their real times)"),
    "hold_on_capacity_optimistic": ("NOT shipped: every deferral holds, capacity included. Lane levels minus the "
                                    "binding family's own real use (never added back later). Only lane LEVELS are lowered: the "
                                    "stored prices are unchanged, so price deferrals do not move"),
}


def render_md(report: dict) -> str:
    L = [f"# Binding replay ({', '.join(report['families'])}), {report['start'][:16]}Z .. {report['end'][:16]}Z", ""]
    for name, b in report["bounds"].items():
        L += [f"## {name}: {BOUND_NOTE[name]}", ""] + _render_bound(b)
    return "\n".join(L) + "\n"


def _render_bound(b: dict) -> list[str]:
    s = b["summary"]
    L = [f"{s['runs']} real runs. First bid: {s['first_bid']}.",
         f"Verdicts: {s['verdicts']} -- funded at the first bid {s['funded_first_bid']}, funded after a hold "
         f"{s['funded_after_hold']}, admitted at the end of the deferrable window {s['admitted_at_window_end']} "
         f"(held, never dropped), joined the pool queue on a capacity deferral {s['queued_at_pool']}, no market "
         f"answer {s['no_answer']}, censored at the end of the data {s['censored']}.",
         f"Held: {s['held_runs']} runs, median {s['held_sec_median']} s, p90 {s['held_sec_p90']} s, max {s['held_sec_max']} s.",
         f"Later than the run really started: {s['runs_delayed_past_real_open']} runs, median {s['extra_delay_sec_median']} s, "
         f"p90 {s['extra_delay_sec_p90']} s.",
         f"{s['bids_total']} bids in all (re-bids included). Deferral reasons: {s['deferral_reasons_all_bids']}; "
         f"binding constraints: {s['binding_constraints_all_bids']}.", "",
         "| run | admitted | bids | verdict | held (s) | real open | extra delay (s) |", "|---|---|---|---|---|---|---|"]
    for r in sorted(b["rows"], key=lambda r: r["admitted_at"]):
        L.append(f"| {r['run_id']} | {r['admitted_at'][:16]} | {r['n_bids']} | {r['verdict']} | {r['held_sec']} | "
                 f"{(r['real_open'] or '-')[:16]} | {r['extra_delay_sec']} |")
    return L + [""]


def main(argv: Optional[list[str]] = None) -> int:
    from app.run_bids import BINDING_FAMILIES

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--end", default=None, help="ISO time; default now")
    ap.add_argument("--cache", default=None, help="read/write the extract as JSON (shared with shadow_vs_real)")
    ap.add_argument("--family", action="append", default=None, help="binding family (default: BINDING_FAMILIES)")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args(argv)
    end = svr._ts(a.end) if a.end else datetime.now(timezone.utc).replace(microsecond=0)
    if a.cache and Path(a.cache).exists():
        ex = json.loads(Path(a.cache).read_text())
    else:
        ex = svr.fetch(end - timedelta(days=a.days), end)
        if a.cache:
            Path(a.cache).write_text(json.dumps(ex, default=str))
    families = frozenset(a.family) if a.family else BINDING_FAMILIES
    fams = svr._families()
    report = {"start": ex["start"], "end": ex["end"], "families": sorted(families), "bounds": {}}
    for name, sub, cq in POLICIES:
        rows = replay(ex, fams, families, subtract_own_use=sub, capacity_queues_at_pool=cq)
        report["bounds"][name] = {"summary": summarize(rows), "rows": rows}
    Path(a.out_json).write_text(json.dumps(report, indent=1, default=str))
    md = render_md(report)
    Path(a.out_md).write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
