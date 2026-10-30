#!/usr/bin/env python3
"""Eval: what does the dream cycle's binding bid do to Orion's sleeps, with and without Prime's slow charge?
READ-ONLY.

Body economy Patch D for ``dream.cycle`` and dream's ``slow`` receptor (Orion Prime step 4). The rule is the
shared one (``orion.physiology.binding.bind_verdict``); the dream's bounds are ``app.physiology_bind``
(``hold_end``: the family's 6 h window or the overdue line, whichever is first; an overdue sleep never bids).

Two lanes:

* ``--fixture`` (default; deterministic, no database): one sleep that clears the dream's gates 30 min
  after the last one closed, against the REAL Choice code and the REAL binding rule and bounds, re-bidding
  at the dream's own check cadence. Checks: with no charge it sleeps at once (a Tier 1 sleep with hunger is
  funded); under the largest slow charge (0.5) it is held, released by its OWN growing hunger well inside the
  6 h window (a price, never a veto); and a sleep that is overdue is never held, whatever the charge (the
  floor).
* ``--live``: every REAL dream sleep (``grammar_acts`` actor ``dream.cycle``) over the last ``--days``,
  re-decided at its real start against the ``physiology_state`` row in force and the act history before it
  (its own act and later sleeps excluded), re-bidding every check until admitted. Scenarios: no charge (what
  binding alone would have done); a 0.5 slow charge in force the whole time (an upper bound Prime cannot
  reach); 6 h on / 6 h off in both phases (the most Prime can do: each signal is followed by an equal rest).
  Also prints the live shadow ledger's agreement with what really ran. Gates: with no charge, no sleep priced
  on dream's own rhythm is held longer than one hour (binding alone must not silence or badly delay sleep;
  cold-start holds are printed, not gated); under the most Prime can do, at most 5% of those sleeps run only
  because their window ran out, and some bids outbid the charge (it is a price, not a veto); the regulate
  grade is not stuck at 0 or 1.

Each want starts at the real sleep's start: the first check that cleared the dream's own gates, arousal
included, which is where the binding bid first happens (the economy's window starts at the first bid,
app/physiology_bind.py). Optimistic about those gates afterwards: a held sleep is assumed to still clear
pressure, idle and arousal at each later check (a real one might not, which only makes a hold matter less).

    PYTHONPATH=. .venv/bin/python services/orion-dream/evals/dream_binding_replay.py [--live --days 7]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

SVC = Path(__file__).resolve().parents[1]
ROOT = SVC.parents[1]
sys.path[:0] = [str(ROOT), str(SVC), str(ROOT / "scripts" / "analysis")]

from app.physiology_bind import FAMILY, hold_end, next_check_at, overdue_line  # noqa: E402
from orion.physiology import prime_charge as pc  # noqa: E402
from orion.physiology.binding import bind_verdict  # noqa: E402

CHECK_SEC = 600.0          # DREAM_CYCLE_CHECK_INTERVAL_SEC
LOOKBACK_H = 48.0          # DREAM_LOOKBACK_HOURS
T0 = datetime(2026, 10, 30, 7, 0, tzinfo=timezone.utc)


def _signal(*, start: datetime, hours: float, magnitude: float = pc.MAX_MAGNITUDE, direction: str = "slow"):
    return pc.PrimeRegulationV1(signal_id="eval", act_id="act_eval", family=FAMILY, direction=direction,
                                arm="treated", magnitude=magnitude, why="eval", opened_at=start,
                                expires_at=start + timedelta(hours=hours))


def simulate(*, want_at: datetime, last_start: Optional[datetime], decide_at: Callable[[datetime, Any], Any],
             prime: Callable[[datetime], Any], max_bids: int = 500) -> dict:
    """One sleep from ``want_at``: bid, obey, re-bid at the next check, until admitted. ``decide_at(t, sig)``
    returns ``(decision_with_sig, decision_without)``. Pure apart from the callbacks."""
    end = hold_end(want_at, last_start, lookback_hours=LOOKBACK_H)
    line = overdue_line(last_start, LOOKBACK_H)
    t, bids = want_at, []
    for _ in range(max_bids):
        if line is not None and t >= line:
            return {"verdict": "floor:overdue", "held_sec": (t - want_at).total_seconds(), "bids": bids, "released_at": t}
        sig = prime(t)
        d, without = decide_at(t, sig)
        eff = None
        if sig is not None and d.net is not None and without is not None and without.net is not None:
            eff = pc.effect_between(sig.direction, d, without)
        v = bind_verdict(d, t, end)
        bids.append({"at": t.isoformat(), "outcome": d.outcome, "reason": d.reason, "net": d.net,
                     "binding": d.binding_constraint, "verdict": v.why, "prime_effect": eff, "why": d.why})
        if v.admit:
            return {"verdict": v.why, "held_sec": (t - want_at).total_seconds(), "bids": bids, "released_at": t}
        t = next_check_at(t, v.retry_at, CHECK_SEC)
    raise AssertionError("never admitted")


# --- fixture ----------------------------------------------------------------------------------------------

def _fixture():
    from orion.physiology.appetite import ActRecord
    from orion.physiology.choice import ReservationBook, build_context, decide
    from orion.physiology.families import load_families
    from orion.schemas.physiology import PhysiologyConstraintReadingV1 as R, PhysiologyStateV1
    from orion.schemas.physiology_want import WantV1

    fams, err = load_families(ROOT / "config/physiology/families.yaml")
    assert fams is not None, err
    # Real-looking spacing: dream slept at uneven gaps (6 .. 20 h), the last close at T0.
    acts, t = [], T0
    for i, g in enumerate([9, 14, 7, 20, 11, 6, 16, 8, 13, 10, 18, 7, 12, 9, 15]):
        acts.append(ActRecord(act_id=f"d{i}", family=FAMILY, opened_at=t - timedelta(seconds=10), closed_at=t,
                              outcome_value=1.0, outcome_source="producer",
                              cost={"seconds": 10.0, "gpu_slot_s_by_lane": {"metacog": 10.0}}))
        t -= timedelta(hours=g)

    def state(at):
        price = {"value": 0.0, "ceiling": 1.0, "bind_state": "slack", "horizon_sec": 3600.0, "why": ""}
        return PhysiologyStateV1(tick_id=f"p:{at.isoformat()}", generated_at=at, constraints=[
            R(constraint="lane:metacog", unit="slot_s", level=0.0, budget=3600.0, budget_kind="capacity",
              detail={"slots": 1, "window_sec": 3600.0}, price=price)])

    def decide_at_for(want_at):
        def f(at, sig):
            ctx = build_context(fams, acts, [], state(at), at, weights={})
            want = WantV1(want_id="sleep", family=FAMILY, producer="eval", verb="make", tier=1, opened_at=want_at)
            d = decide(want, state(at), ctx, ReservationBook(), at, prime=sig)
            wo = decide(want, state(at), ctx, ReservationBook(), at) if sig is not None else None
            return d, wo
        return f

    return decide_at_for


def fixture() -> list[str]:
    fails = []
    decide_at_for = _fixture()
    last_start = T0 - timedelta(seconds=10)
    always = _signal(start=T0 - timedelta(hours=1), hours=60)
    on = lambda t: always if pc.active(always, t) else None  # noqa: E731

    def run(hours_after_close: float, prime):
        at = T0 + timedelta(hours=hours_after_close)
        return simulate(want_at=at, last_start=last_start, decide_at=decide_at_for(at), prime=prime)

    # 8 h after the last close: just past the dream's 6 h refractory, a below-median gap (the fixture's gaps
    # run 6 .. 20 h). 16 h: a long gap, a hungry sleep. 48 h: overdue.
    base, slow = run(8, lambda t: None), run(8, on)
    hungry = run(16, on)
    overdue = run(LOOKBACK_H, on)
    if base["held_sec"] != 0 or base["verdict"] != "funded":
        fails.append(f"with no charge a Tier 1 sleep past its refractory should go at once: {base['verdict']} "
                     f"{base['held_sec']} s")
    if not (0 < slow["held_sec"] < 6 * 3600) or slow["verdict"] != "funded":
        fails.append(f"slow 0.5 should hold the 8 h sleep and its own hunger should release it (funded) inside the "
                     f"6 h window: {slow['verdict']} after {slow['held_sec']} s")
    if not all("Prime's slow charge" in b["why"] for b in slow["bids"]):
        fails.append("every decision under the charge must name it in why")
    if hungry["held_sec"] != 0 or hungry["bids"][0]["prime_effect"] != "outbid":
        fails.append(f"a hungry sleep (16 h) must outbid the charge at once: {hungry['verdict']} {hungry['held_sec']} s")
    if overdue["verdict"] != "floor:overdue" or overdue["held_sec"] != 0:
        fails.append(f"an overdue sleep must never be held: {overdue['verdict']} {overdue['held_sec']} s")
    print(f"fixture: 8 h after the last sleep, no charge: {base['verdict']} at once; slow 0.5: held "
          f"{slow['held_sec'] / 60:.0f} min ({len(slow['bids'])} bids), then {slow['verdict']} by its own hunger; "
          f"16 h after: {hungry['bids'][0]['prime_effect']} the charge at once; overdue (48 h): {overdue['verdict']}")
    print(f"  first slow why: ...{slow['bids'][0]['why'][-150:]}")
    print(f"  last slow why:  ...{slow['bids'][-1]['why'][-150:]}")
    return fails


# --- live -------------------------------------------------------------------------------------------------

def _live_context(ex: dict):
    import body_economy_shadow_vs_real as svr

    from orion.physiology.appetite import ActRecord
    from orion.physiology.choice import ReservationBook, build_context, decide
    from orion.physiology.weights import UseRecord, weight_series
    from orion.schemas.physiology import PhysiologyStateV1
    from orion.schemas.physiology_want import WantV1

    ts = svr._ts
    acts = [ActRecord(act_id=str(r["act_id"]), family=str(r["actor"]), opened_at=ts(r.get("opened_at")),
                      closed_at=ts(r.get("closed_at")), outcome_value=r.get("outcome_value"),
                      outcome_source=r.get("outcome_source"), signal_id=r.get("signal_id"),
                      cost=r.get("cost_json") if isinstance(r.get("cost_json"), dict) else {}) for r in ex["acts"]]
    uses = [UseRecord(act_id=str(r["act_id"]), family=str(r["actor"]), started_at=ts(r.get("started_at")),
                      link_status=str(r["link_status"]), used=r.get("used"), window_closed=bool(r.get("window_closed")))
            for r in ex["uses"]]
    states = [PhysiologyStateV1.model_validate(s) for s in ex["states"]]
    start, end = ts(ex["start"]), ts(ex["end"])
    fams = svr._families()
    series = weight_series(fams, acts, uses, start=(start - timedelta(days=1)).date(), end=end.date())
    sleeps = sorted((a for a in acts if a.family == FAMILY and a.opened_at is not None and start <= a.opened_at < end),
                    key=lambda a: a.opened_at)
    all_sleeps = sorted(a.opened_at for a in acts if a.family == FAMILY and a.opened_at is not None)

    def state_at(t):
        cur = None
        for s in states:
            if s.generated_at <= t:
                cur = s
            else:
                break
        return cur

    def decide_at_for(sleep):
        # History before t, with this sleep's own act and every later sleep removed (they had not happened).
        def f(t, sig):
            before = [a for a in acts if (a.closed_at or a.opened_at) is not None and (a.closed_at or a.opened_at) < t
                      and not (a.family == FAMILY and a.opened_at is not None and a.opened_at >= sleep.opened_at)]
            st = state_at(t)
            ctx = build_context(fams, before, [u for u in uses if u.started_at and u.started_at < t], st, t,
                                weights=series.get(t.date(), {}))
            want = WantV1(want_id=f"replay:{sleep.act_id}", family=FAMILY, producer="replay", verb="make", tier=1,
                          opened_at=sleep.opened_at)
            book = ReservationBook()
            d = decide(want, st, ctx, book, t, prime=sig)
            wo = decide(want, st, ctx, copy.deepcopy(book), t) if sig is not None else None
            return d, wo
        return f

    def last_start_for(sleep):
        prev = [o for o in all_sleeps if o < sleep.opened_at]
        return prev[-1] if prev else None

    from orion.physiology.appetite import TAU_MIN_GAPS, WINDOW

    closes = sorted(a.closed_at for a in acts if a.family == FAMILY and a.closed_at is not None)

    def warm(sleep) -> bool:
        """The market prices this sleep on dream's own rhythm (>= TAU_MIN_GAPS gaps between closes in 30 days),
        not the cold-start fallback (tau = the 6 h window, where a long gap can rank 0 against 2-3 values)."""
        n = sum(1 for c in closes if sleep.opened_at - WINDOW <= c < sleep.opened_at)
        return n - 1 >= TAU_MIN_GAPS

    return sleeps, decide_at_for, last_start_for, start, end, warm


def _scenario(sleeps, decide_at_for, last_start_for, prime, warm=lambda s: True) -> dict:
    rows = []
    for s in sleeps:
        r = simulate(want_at=s.opened_at, last_start=last_start_for(s), decide_at=decide_at_for(s), prime=lambda t: prime(t))
        rows.append({"sleep": s.act_id, "at": s.opened_at.isoformat(), "warm": warm(s), **r})
    held = [r for r in rows if r["held_sec"] > 0]
    return {"rows": rows, "n": len(rows), "held": len(held),
            "held_min": sorted(r["held_sec"] / 60 for r in held),
            "window_end": sum(1 for r in rows if r["verdict"] == "window_end"),
            "floor": sum(1 for r in rows if r["verdict"] == "floor:overdue"),
            "charged_bids": sum(1 for r in rows for b in r["bids"] if b["prime_effect"] is not None),
            "outbid_bids": sum(1 for r in rows for b in r["bids"] if b["prime_effect"] == "outbid"),
            "held_by_charge": sum(1 for r in rows if any(b["prime_effect"] == "held" for b in r["bids"])),
            "reasons": sorted({b["reason"] for r in rows for b in r["bids"] if b["outcome"] == "deferred"}),
            "warm_n": sum(1 for r in rows if r["warm"]),
            "warm_window_end": sum(1 for r in rows if r["warm"] and r["verdict"] == "window_end"),
            "warm_held_min": sorted(r["held_sec"] / 60 for r in rows if r["warm"] and r["held_sec"] > 0),
            "bid_effects": sorted((b["at"], b["prime_effect"]) for r in rows for b in r["bids"] if b["prime_effect"]),
            "nets_first": [round(r["bids"][0]["net"], 3) for r in rows if r["bids"] and r["bids"][0]["net"] is not None]}


def _line(name: str, s: dict) -> str:
    hm = s["held_min"]
    med = hm[len(hm) // 2] if hm else 0.0
    return (f"  {name}: sleeps held {s['held']}/{s['n']} (median {med:.0f} min, max {max(hm) if hm else 0:.0f} min); "
            f"held to the window end {s['window_end']}; floor (overdue) {s['floor']}; bids that met the charge "
            f"{s['charged_bids']}, outbid it {s['outbid_bids']}; sleeps the charge held at least once "
            f"{s['held_by_charge']}; deferral reasons {s['reasons']}\n"
            f"    warm-history sleeps only: held {len(s['warm_held_min'])}/{s['warm_n']}"
            f"{_med_max(s['warm_held_min'])}, held to the window end {s['warm_window_end']}")


def _med_max(xs: list) -> str:
    return f" (median {xs[len(xs) // 2]:.0f} min, max {max(xs):.0f} min)" if xs else ""


def ledger_agreement(ex: dict) -> dict:
    """The live shadow ledger for dream.cycle next to what really ran (an act with the want's id opened)."""
    import body_economy_shadow_vs_real as svr

    opened = {r.get("correlation_id") for r in ex["acts"] if r.get("actor") == FAMILY and r.get("opened_at")}
    rows = [r for r in ex["ledger"] if r.get("family") == FAMILY]
    out = {"bids": len(rows), "wants": len({r["want_id"] for r in rows}), "cells": {}, "reasons": {}}
    for r in rows:
        ran = r["want_id"] in opened or (r.get("correlation_id") in opened)
        cell = f"{r['outcome']}_{'ran' if ran else 'not_run'}"
        out["cells"][cell] = out["cells"].get(cell, 0) + 1
        if r["outcome"] == "deferred":
            out["reasons"][r["reason"]] = out["reasons"].get(r["reason"], 0) + 1
    out["first"] = min((r["decided_at"] for r in rows), default=None)
    _ = svr
    return out


def live(days: float, magnitude: float, cache: Optional[str], max_unbound_hold_min: float,
         max_window_end_share: float = 0.05) -> list[str]:
    import body_economy_shadow_vs_real as svr

    end = datetime.now(timezone.utc).replace(microsecond=0)
    if cache and Path(cache).exists():
        ex = json.loads(Path(cache).read_text())
    else:
        ex = svr.fetch(end - timedelta(days=days), end)
        if cache:
            Path(cache).write_text(json.dumps(ex, default=str))
    sleeps, decide_at_for, last_start_for, start, stop, warm = _live_context(ex)
    fails = []
    led = ledger_agreement(ex)
    print(f"live: {len(sleeps)} real dream sleeps, {ex['start'][:16]}Z .. {ex['end'][:16]}Z")
    print(f"  shadow ledger (bids since {str(led['first'])[:16]}): {led['bids']} bids for {led['wants']} wants; "
          f"cells {led['cells']}; deferral reasons {led['reasons'] or 'none'}")
    base = _scenario(sleeps, decide_at_for, last_start_for, lambda t: None, warm)
    print(_line("binding, no charge", base))
    print(f"    first-bid nets: {base['nets_first']}")
    wh = base["warm_held_min"]
    print(f"    priced on dream's own rhythm (warm history): {base['warm_n']} sleeps, held {len(wh)}"
          f"{f' (max {max(wh):.0f} min)' if wh else ''}; cold-start sleeps (fewer than 6 closes in 30 days, tau = the "
          f"6 h window): {base['n'] - base['warm_n']}, held {base['held'] - len(wh)} (NOT gated: a cold-start "
          f"artifact, bounded by the window; it recurs only if dream has fewer than 6 sleeps in 30 days)")
    if not base["warm_n"]:
        fails.append("no sleep in the window was priced on dream's own rhythm: nothing to judge binding on")
    if wh and max(wh) > max_unbound_hold_min:
        fails.append(f"binding alone held a warm-history sleep {max(wh):.0f} min (> {max_unbound_hold_min:.0f}): "
                     "it would badly delay sleep")
    t0 = start - timedelta(days=1)
    always = _signal(start=t0, hours=24 * (days + 2), magnitude=magnitude)
    up = _scenario(sleeps, decide_at_for, last_start_for, lambda t: always, warm)
    print(_line(f"slow {magnitude} in force the whole time (upper bound, not reachable)", up))

    def duty(phase: int):
        def f(t):
            k = int((t - t0).total_seconds() // (6 * 3600))
            return _signal(start=t0 + timedelta(hours=6 * k), hours=6.0, magnitude=magnitude) if k % 2 == phase else None
        return f

    for phase in (0, 1):
        s = _scenario(sleeps, decide_at_for, last_start_for, duty(phase), warm)
        print(_line(f"6 h on / 6 h off, phase {phase} (the most Prime can do)", s))
        if s["charged_bids"] and not s["outbid_bids"]:
            fails.append(f"phase {phase}: no bid outbid the charge: it would act as a veto")
        # The market, not the timer, must be what releases a sleep Prime slowed: at most max_window_end_share of
        # the warm-history sleeps may run only because their window ran out (#3011's gate for curiosity, 5%).
        if s["warm_n"] and s["warm_window_end"] / s["warm_n"] > max_window_end_share:
            fails.append(f"phase {phase}: {s['warm_window_end']}/{s['warm_n']} warm-history sleeps held to the end of "
                         f"their window (> {max_window_end_share:.0%}): the charge would act as a timer, not a price")
    # The regulate act's grade (artifact_produced = the charge changed a decision) for 3 h signals opened every
    # 30 min, from the always-on bids (a held sleep re-bids every check, so a signal sees every bid in force).
    eff = [(datetime.fromisoformat(a), e) for a, e in up["bid_effects"]]
    t, seen, produced, total = start, 0, 0, 0
    while t < stop - timedelta(hours=3):
        inside = [e for (bt, e) in eff if t <= bt < t + timedelta(hours=3)]
        total += 1
        if inside:
            seen += 1
            produced += any(e == "held" for e in inside)
        t += timedelta(minutes=30)
    rate = produced / seen if seen else 0.0
    print(f"  3 h signals opened every 30 min: {seen}/{total} windows saw a dream bid; of those {produced} "
          f"({rate:.0%}) would grade 1 (the charge held a sleep)")
    if seen and not (0.0 < rate < 1.0):
        fails.append(f"the regulate grade is saturated at {rate:.0%} among windows with a bid")
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--magnitude", type=float, default=pc.DEFAULT_MAGNITUDE)
    ap.add_argument("--cache", default=None, help="read/write the live extract as JSON")
    ap.add_argument("--max-unbound-hold-min", type=float, default=60.0)
    ap.add_argument("--max-window-end-share", type=float, default=0.05)
    a = ap.parse_args(argv)
    fails = (live(a.days, a.magnitude, a.cache, a.max_unbound_hold_min, a.max_window_end_share) if a.live
             else fixture())
    for f in fails:
        print("FAIL", f)
    print("PASS" if not fails else f"FAIL ({len(fails)})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
