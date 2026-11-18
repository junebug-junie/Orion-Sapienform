#!/usr/bin/env python3
"""Eval: what does Prime's "slow" charge do to curiosity investigate? (Orion Prime step 3). READ-ONLY.

Two lanes:

* ``--fixture`` (default; deterministic, no database): one curiosity run that bids right after the
  family's last close, against the REAL Choice code (``orion.physiology.choice.decide``) and the REAL
  binding rule (``app.run_bids.bind_verdict``), re-bidding at each verdict's ``retry_at`` as admission
  does. Its hunger grows between bids. Checks: with no charge it goes ahead at the first bid that has any
  hunger; with the largest slow charge (0.5) it is held, and released by its OWN growing hunger, long
  before the family's 6 h window (never a veto); a quicken credit does NOT fund it while the market values it
  at zero (never a mandate), so it goes exactly when it would have; and the decision's ``why`` names the
  charge each time.
* ``--live``: ``scripts/analysis/body_economy_binding_replay.py`` over the last ``--days`` of REAL
  curiosity admissions (live Choice code, live physiology_state, read-only): no charge; a slow charge of
  ``--magnitude`` in force the WHOLE time (an upper bound Prime cannot reach: after each signal the organ
  rests as long as the signal lasted); the most Prime CAN do, 6 h on / 6 h off, in both phases; and
  windowed signals of ``--hours``. Gates: under the most Prime can do, at most ``--max-window-end-share``
  of runs are held to the end of their 6 h window (where they run anyway: held, never dropped) and some
  bids outbid the charge (it is a price, not a veto); the windowed grade (the regulate act's
  ``artifact_produced``) is neither always 0 nor always 1 among windows that saw a bid.

    PYTHONPATH=. .venv/bin/python services/orion-durable-runs/evals/prime_charge_replay.py [--live --days 7]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

SVC = Path(__file__).resolve().parents[1]
ROOT = SVC.parents[1]
sys.path[:0] = [str(ROOT), str(SVC), str(ROOT / "scripts" / "analysis")]

from app.run_bids import bind_verdict  # noqa: E402
from orion.physiology import prime_charge as pc  # noqa: E402

C = "durable.curiosity_investigate"
T0 = datetime(2026, 10, 30, 7, 0, tzinfo=timezone.utc)
WINDOW = timedelta(seconds=21600)        # families.yaml durable.curiosity_investigate deferrable_window_s


def _signal(direction="slow", magnitude=pc.MAX_MAGNITUDE, start=T0 - timedelta(hours=1), hours=6.0):
    return pc.PrimeRegulationV1(signal_id="eval", act_id="act_eval", family=C, direction=direction, arm="treated",
                                magnitude=magnitude, why="eval", opened_at=start,
                                expires_at=start + timedelta(hours=hours))


def _fixture_market():
    """A curiosity family whose closes came at uneven gaps (its value history spreads), the last one at T0."""
    from orion.physiology.appetite import ActRecord
    from orion.physiology.choice import ReservationBook, build_context
    from orion.physiology.families import load_families
    from orion.schemas.physiology import PhysiologyConstraintReadingV1 as R, PhysiologyStateV1

    fams, err = load_families(ROOT / "config/physiology/families.yaml")
    assert fams is not None, err
    acts, t = [], T0
    for i, g in enumerate([0.2, 0.5, 1.0, 2.0, 3.0, 0.3, 0.7, 1.5, 4.0, 0.4, 2.5, 0.8, 1.2, 0.6, 2.2]):
        acts.append(ActRecord(act_id=f"c{i}", family=C, opened_at=t - timedelta(minutes=30), closed_at=t,
                              outcome_value=1.0, outcome_source="producer",
                              cost={"seconds": 1800.0, "gpu_slot_s_by_lane": {"agent": 1800.0}}))
        t -= timedelta(hours=g)

    def state(at):
        price = {"value": 0.0, "ceiling": 1.0, "bind_state": "slack", "horizon_sec": 3600.0, "why": ""}
        return PhysiologyStateV1(tick_id=f"p:{at.isoformat()}", generated_at=at, constraints=[
            R(constraint="lane:agent", unit="slot_s", level=0.0, budget=3600.0, budget_kind="capacity",
              detail={"slots": 1, "window_sec": 3600.0}, price=price)])

    def ctx(at):
        return build_context(fams, acts, [], state(at), at, weights={})

    return fams, ctx, state, ReservationBook


def simulate(sig_fn) -> dict:
    """One run admitted at T0 + 2 min; re-bid at each retry_at until admitted."""
    from orion.physiology.choice import decide
    from orion.schemas.physiology_want import WantV1

    fams, ctx, state, Book = _fixture_market()
    t = T0 + timedelta(minutes=2)
    end = t + WINDOW
    whys = []
    for n in range(200):
        want = WantV1(want_id="run", family=C, producer="eval", verb="infer", tier=2, opened_at=T0 + timedelta(minutes=2))
        d = decide(want, state(t), ctx(t), Book(), t, reserve_at_open=True, prime=sig_fn(t))
        whys.append(d.why)
        v = bind_verdict(d, t, end)
        if v.admit:
            return {"verdict": v.why, "held_sec": (t - T0 - timedelta(minutes=2)).total_seconds(), "bids": n + 1,
                    "whys": whys}
        t = v.retry_at
    raise AssertionError("never admitted")


def fixture() -> list[str]:
    fails = []
    base = simulate(lambda t: None)
    slow = simulate(lambda t: _signal() if pc.active(_signal(), t) else None)
    quick = simulate(lambda t: _signal("quicken") if pc.active(_signal("quicken"), t) else None)
    if base["held_sec"] > 1200:
        fails.append(f"with no charge the run should go within one re-bid of having hunger: {base}")
    if not (slow["held_sec"] > base["held_sec"] and slow["held_sec"] < WINDOW.total_seconds() / 2):
        fails.append(f"slow 0.5 should hold it longer than no charge, released by its own hunger well inside "
                     f"the window: base {base['held_sec']} s, slow {slow['held_sec']} s")
    if slow["verdict"] != "funded":
        fails.append(f"slow: the run should be released by the market (funded), not the window: {slow['verdict']}")
    if not all("Prime's slow charge" in w for w in slow["whys"]):
        fails.append("every decision under the charge must name it in why")
    if "did not beat" not in slow["whys"][0] or " beat Prime's slow charge" not in slow["whys"][-1]:
        fails.append(f"why should read held then outbid: {slow['whys'][0][-120:]} / {slow['whys'][-1][-120:]}")
    if quick["held_sec"] != base["held_sec"] or "quicken credit does not apply" not in quick["whys"][0]:
        fails.append(f"quicken must not fund a want valued at zero (never a mandate): {quick['held_sec']} s, "
                     f"{quick['whys'][0][-120:]}")
    print(f"fixture: no charge held {base['held_sec']:.0f} s ({base['bids']} bids); slow 0.5 held "
          f"{slow['held_sec']:.0f} s ({slow['bids']} bids, {slow['verdict']}); quicken held {quick['held_sec']:.0f} s "
          "(a credit never funds a want valued at zero)")
    print(f"  first slow why: ...{slow['whys'][0][-150:]}")
    print(f"  last slow why:  ...{slow['whys'][-1][-150:]}")
    return fails


def _replay_summary(rep, svr, ex, fams, binding, base_by_run, prime) -> dict:
    rows = rep.replay(ex, fams, binding, prime=prime)
    held = [r for r in rows if any(b["prime_effect"] == "held" for b in r["bids"])]
    extra = sorted(max(0.0, (svr._ts(r["released_at"]) - svr._ts(base_by_run[r["run_id"]]["released_at"])).total_seconds())
                   for r in rows if r["released_at"] and base_by_run.get(r["run_id"], {}).get("released_at"))
    delayed = [x for x in extra if x > 0]
    return {"rows": rows, "n": len(rows), "held": len(held),
            "outbid_bids": sum(1 for r in rows for b in r["bids"] if b["prime_effect"] == "outbid"),
            "charged_bids": sum(1 for r in rows for b in r["bids"] if b["prime_effect"] is not None),
            "window_end": sum(1 for r in held if r["verdict"] == "window_end"),
            "delayed": len(delayed), "delay_median_min": delayed[len(delayed) // 2] / 60 if delayed else 0.0,
            "delay_max_min": max(delayed) / 60 if delayed else 0.0}


def _line(name: str, s: dict) -> str:
    return (f"  {name}: {s['charged_bids']} bids met the charge, {s['outbid_bids']} outbid it; runs held by it at "
            f"least once {s['held']}/{s['n']}; held to the end of their window {s['window_end']}; started later "
            f"than with no charge {s['delayed']} runs (median {s['delay_median_min']:.0f} min, max "
            f"{s['delay_max_min']:.0f} min)")


def live(days: float, magnitude: float, hours: float, cache: str | None, max_window_end_share: float) -> list[str]:
    import body_economy_binding_replay as rep
    import body_economy_shadow_vs_real as svr

    end = datetime.now(timezone.utc).replace(microsecond=0)
    if cache and Path(cache).exists():
        ex = json.loads(Path(cache).read_text())
    else:
        ex = svr.fetch(end - timedelta(days=days), end)
        if cache:
            Path(cache).write_text(json.dumps(ex, default=str))
    fams = svr._families()
    binding = frozenset({C})
    t0 = svr._ts(ex["start"]) - timedelta(days=1)
    always = _signal(magnitude=magnitude, start=t0, hours=24 * (days + 2))
    base = rep.replay(ex, fams, binding)
    by_run = {r["run_id"]: r for r in base}
    fails = []

    def duty(phase: int):
        """6 h on, 6 h off (the most Prime can do: each signal is followed by an equal rest)."""
        def f(fam, t):
            k = int((t - t0).total_seconds() // (6 * 3600))
            if fam != C or k % 2 != phase:
                return None
            return _signal(magnitude=magnitude, start=t0 + timedelta(hours=6 * k), hours=6.0)
        return f

    up = _replay_summary(rep, svr, ex, fams, binding, by_run, lambda f, t: always if f == C else None)
    print(f"live: {up['n']} real curiosity runs, {ex['start'][:16]}Z .. {ex['end'][:16]}Z; slow charge {magnitude}")
    print(_line("in force the whole time (upper bound, not reachable)", up))
    for phase in (0, 1):
        s = _replay_summary(rep, svr, ex, fams, binding, by_run, duty(phase))
        print(_line(f"6 h on / 6 h off, phase {phase} (the most Prime can do)", s))
        if s["n"] and s["window_end"] / s["n"] > max_window_end_share:
            fails.append(f"phase {phase}: {s['window_end']}/{s['n']} runs held to the end of their window "
                         f"(> {max_window_end_share:.0%})")
        if s["charged_bids"] and not s["outbid_bids"]:
            fails.append(f"phase {phase}: no bid outbid the charge: it would act as a veto")
    slow = up["rows"]
    # The windowed grade: a signal of `hours` opened at t is produced when a run's FIRST bid inside
    # [t, t+hours) was held by it. Approximation: first-bid effects from the always-on replay.
    firsts = sorted((svr._ts(r["bids"][0]["at"]), r["bids"][0]["prime_effect"]) for r in slow if r["bids"])
    start, stop = svr._ts(ex["start"]), svr._ts(ex["end"]) - timedelta(hours=hours)
    t, seen, produced, total = start, 0, 0, 0
    while t < stop:
        inside = [e for (bt, e) in firsts if t <= bt < t + timedelta(hours=hours)]
        total += 1
        if inside:
            seen += 1
            produced += any(e == "held" for e in inside)
        t += timedelta(minutes=30)
    rate = produced / seen if seen else 0.0
    print(f"  windowed {hours:g} h signals (one opened every 30 min): {seen}/{total} windows saw a curiosity bid; "
          f"of those, {produced} ({rate:.0%}) would grade 1 (a run held by the charge)")
    if seen and not (0.0 < rate < 1.0):
        fails.append(f"the regulate grade is saturated at {rate:.0%} among windows with a bid: it cannot tell signals apart")
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--magnitude", type=float, default=pc.DEFAULT_MAGNITUDE)
    ap.add_argument("--hours", type=float, default=pc.DEFAULT_HOURS)
    ap.add_argument("--cache", default=None, help="read/write the live extract as JSON")
    ap.add_argument("--max-window-end-share", type=float, default=0.05)
    a = ap.parse_args(argv)
    fails = live(a.days, a.magnitude, a.hours, a.cache, a.max_window_end_share) if a.live else fixture()
    for f in fails:
        print("FAIL", f)
    print("PASS" if not fails else f"FAIL ({len(fails)})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
