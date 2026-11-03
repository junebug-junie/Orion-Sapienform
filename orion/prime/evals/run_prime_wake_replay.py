"""Eval: how often Prime wakes once attention moves no longer count (2026-11-03), replayed on live data.

The fixture is the real trigger timeline for 2026-10-30 08:35 .. 2026-11-03 03:20 UTC: every change in the
set of BINDING body constraints (physiology_state, one row per body tick) and every Prime act that settled
(grammar_acts: question/regulate closes and faded/lapsed concerns). Attention moves are deliberately absent.

The replay runs the real tick rule (orion/prime/state.py ``observe`` + ``wants_to_think``) every 15 s with
the live refractory interval (300 s) and treats every eligible tick as a thought (an upper bound: the bid
can still defer). Checks:

* rate_drops   -- thoughts per day fall well below the live rate in the same window (257 tries in 90.9 h,
                  when attention moves were a trigger) -- the churn this change exists to stop;
* still_wakes  -- Prime still thinks every day of the window (no day with zero thoughts);
* floor_bounds -- with the quiet floor, no gap between thoughts exceeds QUIET_FLOOR + one tick + the
                  refractory interval; without the floor the longest gap is reported, so the floor's
                  reason to exist stays visible.

Known bias: the settled-act closes come from the OLD regime, which thought ~2.4x as often and so had more acts
to settle. The replay therefore over-counts wake triggers: ``rate_drops`` is conservative (the real rate should
be lower still), and ``still_wakes`` is easier to pass than it will be live -- the quiet floor is what guarantees
it, and ``floor_bounds`` checks the floor directly.

Run: ``PYTHONPATH=. python orion/prime/evals/run_prime_wake_replay.py`` (exit 1 on any failed check).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from orion.prime import state as ps
from orion.prime.state import PrimeSnapshotV1, PrimeStateV1, note_outcome, observe, wants_to_think

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "prime_wake_triggers_live_2026-11-03.json"
TICK = timedelta(seconds=15)
REFRACTORY = timedelta(seconds=300)


def replay(data: dict, *, floor: bool = True) -> tuple[list[datetime], int]:
    """(thought times, how many of them the quiet floor woke)."""
    start = datetime.fromisoformat(data["window"]["start"])
    end = datetime.fromisoformat(data["window"]["end"])
    binding = [(datetime.fromisoformat(r["at"]), r["binding"]) for r in data["binding_changes"]]
    closes = sorted(datetime.fromisoformat(c["at"]) for c in data["own_act_closes"])
    saved_floor = ps.QUIET_FLOOR
    if not floor:
        ps.QUIET_FLOOR = timedelta(days=365)
    try:
        state = PrimeStateV1()
        thoughts: list[datetime] = []
        quiet_woken = 0
        bi = ci = 0
        current = binding[0][1]
        now = start
        while now <= end:
            while bi < len(binding) and binding[bi][0] <= now:
                current = binding[bi][1]
                bi += 1
            while ci < len(closes) and closes[ci] <= now:
                state = note_outcome(state, "an act of mine settled")
                ci += 1
            state = observe(state, PrimeSnapshotV1(binding=list(current), body_ok=True), now)
            if wants_to_think(state, now, refractory=REFRACTORY):
                thoughts.append(now)
                quiet_woken += all(c.startswith("quiet:") for c in state.pending_changes)
                state = state.model_copy(update={"pending_changes": [], "last_thought_at": now})
            now += TICK
    finally:
        ps.QUIET_FLOOR = saved_floor
    return thoughts, quiet_woken


def _max_gap(thoughts: list[datetime]) -> timedelta:
    return max((b - a for a, b in zip(thoughts, thoughts[1:])), default=timedelta(0))


def run() -> list[tuple[str, bool, str]]:
    data = json.loads(FIXTURE.read_text())
    start = datetime.fromisoformat(data["window"]["start"])
    end = datetime.fromisoformat(data["window"]["end"])
    days = (end - start).total_seconds() / 86400.0
    live_per_day = data["live_thought_tries_in_window"] / days
    thoughts, quiet_woken = replay(data)
    per_day = len(thoughts) / days
    results = [("rate_drops", per_day <= 0.6 * live_per_day,
                f"{len(thoughts)} thoughts in {days * 24:.1f} h = {per_day:.1f}/day (live with attention: "
                f"{live_per_day:.1f}/day)")]
    by_day = Counter(t.date() for t in thoughts)
    every = [(start + timedelta(days=i)).date() for i in range(int(days) + 1)]
    results.append(("still_wakes", all(by_day.get(d, 0) > 0 for d in every),
                    "thoughts per UTC day: " + ", ".join(f"{d:%m-%d} {by_day.get(d, 0)}" for d in every)))
    bound = ps.QUIET_FLOOR + TICK + REFRACTORY
    no_floor, _ = replay(data, floor=False)
    gap, gap_no_floor = _max_gap(thoughts), _max_gap(no_floor)
    results.append(("floor_bounds", gap <= bound,
                    f"longest gap {gap.total_seconds() / 3600:.1f} h with the floor (bound {bound.total_seconds() / 3600:.2f} h), "
                    f"{gap_no_floor.total_seconds() / 3600:.1f} h without; {quiet_woken} thoughts woken by the floor alone"))
    return results


def main() -> int:
    results = run()
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
