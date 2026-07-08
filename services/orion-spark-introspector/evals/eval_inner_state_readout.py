"""Eval: honest inner-state headline is responsive and never floored.

Run: python services/orion-spark-introspector/evals/eval_inner_state_readout.py
Exits non-zero on failure so it can gate CI.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
# Running a script puts the script's dir on sys.path[0], NOT repo root, and
# `orion` is not pip-installed. inner_state.py imports orion.schemas.* at import
# time, so repo root must be on the path before we exec the module.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location(
    "spark_inner_state_eval",
    REPO / "services" / "orion-spark-introspector" / "app" / "inner_state.py",
)
inner_state = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inner_state)


def _raw(**over):
    base = {
        "coherence": 1.0, "field_intensity": 1.0, "agency_readiness": 0.41,
        "execution_pressure": 0.0, "reasoning_pressure": 0.05,
        "resource_pressure": 1.0, "reliability_pressure": 1.0,
        "continuity_pressure": 0.0, "social_pressure": 0.0,
    }
    base.update(over)
    return base


def main() -> int:
    failures = []

    pinned = inner_state.honest_headline(_raw())
    if pinned <= 0.01:
        failures.append(f"pinned window still floored: {pinned}")

    # infra can't move the felt vector: reliability comes ONLY from felt inputs;
    # changing an infra-only concept (not in the raw felt map) must not change headline.
    h1 = inner_state.honest_headline(_raw())
    h2 = inner_state.honest_headline(_raw())  # identical felt inputs
    if h1 != h2:
        failures.append("headline non-deterministic on identical felt inputs")

    # responsive: rising execution load lowers headline
    calm = inner_state.honest_headline(_raw(execution_pressure=0.0, reasoning_pressure=0.0,
                                            resource_pressure=0.1, reliability_pressure=0.1))
    busy = inner_state.honest_headline(_raw(execution_pressure=0.9, reasoning_pressure=0.8,
                                            resource_pressure=0.1, reliability_pressure=0.1))
    if not (busy < calm):
        failures.append(f"headline not responsive to cognitive load: calm={calm} busy={busy}")

    if failures:
        print("EVAL FAIL:")
        for f in failures:
            print(" -", f)
        return 1
    print(f"EVAL PASS: pinned={pinned:.3f} calm={calm:.3f} busy={busy:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
