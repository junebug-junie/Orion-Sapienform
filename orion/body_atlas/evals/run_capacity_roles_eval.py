"""Do the rung-4 role rules find exactly the parts a reviewer expects, and can the eval fail?

Runs ``orion/body_atlas/capacities.py`` over two frozen whole bodies: 2026-10-29 (held out: no
rule knob was set on it) and 2026-10-30 (with stores named in code). For each fixture:

* **whole output**: every hit of R-matchmaker, R-census-reader and R-boundary-in is in the
  reviewed expected set below, and every expected part is hit (precision and recall 1.0);
* **near misses stay out of S2**: the shared servers and the collectors;
* **census**: the same set on both fixtures, and ``orion:grammar:event`` (the named boundary
  case) stays out;
* **shrink**: the printout fits 40 lines and 4,000 chars.

Then the **mutation check**: drop R-matchmaker's disjoint-second-crowd clause and the eval must
FAIL (the shared servers become S2). An eval that cannot fail is not a gate.

The expected sets are the only hand-written truth here; the rules never read them.

    python orion/body_atlas/evals/run_capacity_roles_eval.py
    python orion/body_atlas/evals/run_capacity_roles_eval.py --live   # needs $FALKORDB_URI
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from orion.body_atlas import capacities as K  # noqa: E402
from orion.body_atlas.build import build_atlas  # noqa: E402

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
import run_organ_structure_eval as _structure  # noqa: E402

HELD_OUT = _HERE / "fixtures" / "whole_body_2026-10-29.json"
TODAY = _HERE / "fixtures" / "whole_body_2026-10-30_stores.json"


def _p(*names: str) -> set[str]:
    return {f"part:{n}" for n in names}


# Reviewed by hand against the code of each part (design doc, "Acceptance checks"). On 10-29 the
# atlas had no live camera source yet (cameras were declared_only), so no camera is live there.
EXPECTED = {
    HELD_OUT.name: {
        "R-matchmaker": _p("orion-gpu-pool"),
        "R-census-reader": _p("orion-equilibrium-service"),
        "R-boundary-in": _p("http:anthropic", "http:openai"),
    },
    TODAY.name: {
        "R-matchmaker": _p("orion-gpu-pool"),
        "R-census-reader": _p("orion-equilibrium-service"),
        "R-boundary-in": _p("camera:cam0", "camera:walkway", "http:anthropic", "http:openai"),
    },
}
NOT_S2 = _p("orion-llm-gateway", "orion-cortex-orch", "orion-vector-host", "orion-heartbeat",
            "orion-equilibrium-service")
EXPECTED_CENSUS = {"orion:system:health"}
CENSUS_BOUNDARY = "orion:grammar:event"


def score(got: set[str], want: set[str]) -> tuple[float, float]:
    tp = len(got & want)
    return (tp / len(got) if got else 1.0, tp / len(want) if want else 1.0)


def evaluate(reading: K.CapacityReadingV1, expected: dict[str, set[str]] | None) -> list[str]:
    failures = []
    for rule, want in (expected or {}).items():
        got = K.hits(reading, rule)
        prec, rec = score(got, want)
        if prec < 1.0 or rec < 1.0:
            failures.append(f"{rule}: precision {prec:.2f} recall {rec:.2f} "
                            f"(extra {sorted(got - want)}, missed {sorted(want - got)})")
    s2 = {r.part_id for r in reading.roles if r.role == "S2"}
    if s2 & NOT_S2:
        failures.append(f"near misses read as S2: {sorted(s2 & NOT_S2)}")
    census = {c.channel for c in reading.census if c.census}
    if census != EXPECTED_CENSUS:
        failures.append(f"census {sorted(census)} != {sorted(EXPECTED_CENSUS)}")
    if CENSUS_BOUNDARY in census:
        failures.append(f"{CENSUS_BOUNDARY} entered the census")
    text = "\n".join(K.render(reading))
    if len(text.splitlines()) > 39 or len(text) > 3900:  # room for the dig tail line
        failures.append(f"printout too big: {len(text.splitlines())} lines, {len(text)} chars")
    return failures


def run(fixtures: list[Path], *, second_set: bool = True, verbose: bool = True) -> int:
    failed = 0
    for path in fixtures:
        catalog, observed, now, body = _structure._fixture_inputs(path)
        reading = K.read_capacities(build_atlas(catalog, observed, now=now, body=body), second_set=second_set)
        failures = evaluate(reading, EXPECTED.get(path.name))
        if verbose:
            print(f"== {path.name}" + (" (held out)" if path == HELD_OUT else ""))
            print("\n".join(K.render(reading)))
            for rule, want in EXPECTED.get(path.name, {}).items():
                prec, rec = score(K.hits(reading, rule), want)
                print(f"{rule}: precision {prec:.2f} recall {rec:.2f} ({len(want)} expected)")
            for f in failures:
                print(f"FAIL {f}")
            print(f"capacity roles ({path.name}): " + ("PASS" if not failures else f"FAIL ({len(failures)})"))
        failed += bool(failures)
    return failed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="also print the live reading (no expected set)")
    args = ap.parse_args(argv)
    fixtures = [HELD_OUT, TODAY]
    failed = run(fixtures)
    mutant_failed = run(fixtures, second_set=False, verbose=False)
    print(f"mutation (drop the disjoint second crowd): {mutant_failed} of {len(fixtures)} fixtures fail -> "
          + ("CAUGHT" if mutant_failed == len(fixtures) else "NOT CAUGHT"))
    if args.live:
        catalog, observed, now, body = _structure._live_inputs(_REPO)
        reading = K.read_capacities(build_atlas(catalog, observed, now=now, body=body))
        print("== live\n" + "\n".join(K.render(reading)))
        for f in evaluate(reading, None):
            print(f"FAIL {f}")
    return 1 if failed or mutant_failed != len(fixtures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
