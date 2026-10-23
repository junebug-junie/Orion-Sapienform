"""Fixture eval for heat archaeology: does the method recover a known step, and stay quiet
when nothing changed? Pure Python, no database, so CI runs it.

Positive: 25 seeds of the 2026-10-14 reproduction (orion/physiology/tests/heat_fixture.py):
a crash loop (+144 W, load ~1.5 -> ~14) and a vision-council rise start at STEP_AT under
realistic hourly noise (harness swings ~0-285 W hour to hour). Scored: the step is found, how
far its hour lands from the truth, and how far the regime / vision-council / unexplained watts
land from the planted truth.

Negative control: the same seeds with nothing changing. A step found there is a false alarm.

  python services/orion-hub/evals/run_power_attribution_eval.py

Exit 1 if any gate below fails; prints one JSON line of scores either way.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.physiology import heat_archaeology as ha  # noqa: E402
from orion.physiology.tests import heat_fixture as fx  # noqa: E402

SEEDS = range(25)
# Gates. Set from the method's purpose (name the cause to within tens of watts, the time to
# within a few hours), not from what it happened to score.
MIN_RECALL = 0.95
MAX_MEDIAN_HOUR_ERROR = 3
MAX_MEDIAN_REGIME_ERROR_W = 15.0
MAX_MEDIAN_VISION_ERROR_W = 10.0
MAX_FALSE_ALARM_RATE = 0.05


def _regime(contrib: dict[str, float]) -> float:
    return next((v for k, v in contrib.items() if k.startswith("CPU-load regime")), 0.0)


def run() -> dict:
    found = 0
    hour_err, regime_err, vision_err, unexplained = [], [], [], []
    for seed in SEEDS:
        readings, busy, truth = fx.build(seed)
        [circe] = ha.analyze(readings, busy, window_start=fx.START, window_end=fx.END, nodes=["circe"])
        near = [s for s in circe.steps if abs((s.at - fx.STEP_AT).total_seconds()) <= 6 * 3600]
        if not near:
            continue
        found += 1
        step = near[0]
        hour_err.append(abs((step.at - fx.STEP_AT).total_seconds()) / 3600)
        attr = next(a for a in circe.attributions if a.step_at == step.at)
        regime_err.append(abs(_regime(attr.contributions_w) - truth["crash_w"]))
        vision_err.append(abs(attr.contributions_w.get("vision-council", 0.0) - truth["vision_w"]))
        unexplained.append(abs(attr.unexplained_w))
    false_alarms = 0
    for seed in SEEDS:
        readings, busy, _ = fx.build(seed, step=False)
        [circe] = ha.analyze(readings, busy, window_start=fx.START, window_end=fx.END, nodes=["circe"])
        false_alarms += bool(circe.steps)
    n = len(SEEDS)
    scores = {
        "seeds": n,
        "recall": found / n,
        "median_hour_error": statistics.median(hour_err) if hour_err else None,
        "max_hour_error": max(hour_err) if hour_err else None,
        "median_regime_error_w": round(statistics.median(regime_err), 1) if regime_err else None,
        "median_vision_error_w": round(statistics.median(vision_err), 1) if vision_err else None,
        "median_abs_unexplained_w": round(statistics.median(unexplained), 1) if unexplained else None,
        "false_alarm_rate": false_alarms / n,
    }
    gates = {
        "recall": scores["recall"] >= MIN_RECALL,
        "hour": scores["median_hour_error"] is not None and scores["median_hour_error"] <= MAX_MEDIAN_HOUR_ERROR,
        "regime": scores["median_regime_error_w"] is not None and scores["median_regime_error_w"] <= MAX_MEDIAN_REGIME_ERROR_W,
        "vision": scores["median_vision_error_w"] is not None and scores["median_vision_error_w"] <= MAX_MEDIAN_VISION_ERROR_W,
        "false_alarms": scores["false_alarm_rate"] <= MAX_FALSE_ALARM_RATE,
    }
    scores["gates"] = gates
    scores["pass"] = all(gates.values())
    return scores


def main() -> int:
    scores = run()
    print(json.dumps(scores))
    return 0 if scores["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
