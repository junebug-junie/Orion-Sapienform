"""Eval: voluntary attention override dynamics (spec Step 2).

Replays synthetic candidate sets through the real TopDownBiasCombiner and asserts:
  (a) override rate RISES with goal priority,
  (b) override rate FALLS as effort budget shrinks,
  (c) a strongly-salient loop still wins over weak bias (bottom-up reasserts),
  (d) no goal -> zero overrides (pure bottom-up).

No bus, no Docker. Run: python orion/substrate/attention/evals/run_topdown_eval.py
"""
from __future__ import annotations

import sys

from orion.schemas.attention_frame import OpenLoopV1
from orion.substrate.attention.top_down import (
    GoalContext,
    TopDownBiasCombiner,
    TopDownConfig,
)


def _candidates(n: int) -> list[OpenLoopV1]:
    # One dominant bottom-up loop + n aligned low-salience loops (predictive).
    loops = [OpenLoopV1(id="dom", description="dominant", salience=0.75, predictive_value=0.0)]
    for i in range(n):
        loops.append(OpenLoopV1(id=f"g{i}", description=f"goal-aligned {i}",
                                salience=0.30, predictive_value=0.9))
    return loops


def _override_rate(*, priority: float, effort_max: float, trials: int = 20) -> float:
    combiner = TopDownBiasCombiner(TopDownConfig(gain=0.6, effort_max=effort_max))
    goal = GoalContext(drive_origin="predictive", priority=priority, goal_artifact_id="g")
    overrides = 0
    for _ in range(trials):
        loops = _candidates(3)
        bottom_up = {l.id: l.salience for l in loops}
        res = combiner.apply(goal=goal, loops=loops, bottom_up=bottom_up, agency_readiness=1.0)
        if res.override is not None:
            overrides += 1
    return overrides / trials


def run() -> int:
    rate_hi = _override_rate(priority=0.9, effort_max=1.0)
    rate_lo = _override_rate(priority=0.2, effort_max=1.0)
    rate_starved = _override_rate(priority=0.9, effort_max=0.05)

    # (c) strong bottom-up beats weak bias
    combiner = TopDownBiasCombiner(TopDownConfig())
    strong = [OpenLoopV1(id="dom", description="d", salience=0.98, predictive_value=0.0),
              OpenLoopV1(id="g0", description="g", salience=0.30, predictive_value=0.9)]
    strong_res = combiner.apply(goal=GoalContext("predictive", 0.3), loops=strong,
                                bottom_up={"dom": 0.98, "g0": 0.30})
    # (d) no goal -> no override
    none_res = combiner.apply(goal=None, loops=strong, bottom_up={"dom": 0.98, "g0": 0.30})

    checks = [
        ("(a) override rate rises with priority", rate_hi > rate_lo, f"hi={rate_hi} lo={rate_lo}"),
        ("(b) override rate falls under effort scarcity", rate_starved < rate_hi, f"starved={rate_starved} hi={rate_hi}"),
        ("(c) strong bottom-up beats weak bias", strong_res.override is None, f"override={strong_res.override is not None}"),
        ("(d) no goal -> no override", none_res.override is None, "override=None"),
    ]

    print("\n=== Voluntary attention override eval ===")
    print(f"override rate  priority=0.9: {rate_hi:.2f}")
    print(f"override rate  priority=0.2: {rate_lo:.2f}")
    print(f"override rate  effort=0.05 : {rate_starved:.2f}")
    print("checks:")
    ok = True
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  ({detail})")
        ok = ok and passed
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
