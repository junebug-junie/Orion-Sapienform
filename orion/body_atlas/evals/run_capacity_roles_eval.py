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

Patch 2, **claims floor** on the 2026-10-30 body (``orion/body_atlas/claims.py``): a claim citing
one of the part's flows passes; one citing an unrelated part, only the census, or taking a job a
rule already gives is refused; a capacity with no holder, or naming a part that holds no such
job, is refused; an unconfirmed claim is never citable and never clears a gap; Juniper's
confirmation does both; the checked-in ``vsm_confirmed.yaml`` reads; six full-length capacities
and claims still fit the printout. Mutation: let unconfirmed claims hold roles and it must FAIL.

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
from orion.body_atlas import claims as C  # noqa: E402
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


def _refused(fn, *args, **kw) -> bool:
    try:
        fn(*args, **kw)
    except C.ClaimError:
        return True
    return False


def claims_failures(path: Path, *, claims_hold: bool = False) -> list[str]:
    """``claims_hold=True`` is the mutation: unconfirmed claims are read as confirmed."""
    catalog, observed, now, body = _structure._fixture_inputs(path)
    atlas = build_atlas(catalog, observed, now=now, body=body)
    try:
        vsm = C.load_confirmed(_REPO)
    except C.ClaimError as exc:
        return [f"checked-in vsm_confirmed.yaml does not read: {exc}"]
    if vsm.confirmed:
        return ["vsm_confirmed.yaml is no longer empty: point this eval at a frozen copy"]
    f: list[str] = []
    base = K.read_capacities(atlas, vsm=vsm)
    claim = C.check_claim(base, atlas, "orion-durable-runs", "s2", "cortex-orch hands orion-cortex-orch run "
                          "requests on orion:durable:run:request and I schedule them", kind="claim", root=_REPO, dig="e")
    if "orion:durable:run:request" not in claim.cites:
        f.append(f"claim cites {claim.cites}, missed the channel it named")
    refusals = {
        "an unrelated part": (C.check_claim, base, atlas, "orion-durable-runs", "S2",
                              "it looks a lot like camera:walkway to me", ),
        "only the census": (C.check_claim, base, atlas, "orion-durable-runs", "S2",
                            "it sends orion:system:health like everyone does"),
        "a job a rule gives": (C.check_claim, base, atlas, "orion-gpu-pool", "S2",
                               "orion-gpu-pool reads orion:gpu_pool:lease:request from many"),
        "a part that is not live": (C.check_claim, base, atlas, "smoke", "S1", "it smokes orion-hub a lot today"),
    }
    for why, (fn, *args) in refusals.items():
        if not _refused(fn, *args, kind="claim", root=_REPO, dig="e"):
            f.append(f"claim not refused: {why}")
    vsm_used = vsm
    if claims_hold:
        vsm_used = vsm.model_copy(update={"confirmed": [K.ConfirmationV1(part="orion-durable-runs", role="S2",
                                                                         date="2026-10-30")]})
    reading = K.attach_ledgers(K.read_capacities(atlas, vsm=vsm_used), atlas, vsm_used, [claim], [])
    if "part:orion-durable-runs" in K.holders(reading, "cap:s2"):
        f.append("an unconfirmed claim became citable in cap:s2")
    if len(reading.gaps) != len(base.gaps):
        f.append("an unconfirmed claim changed the gap lines")
    if [c.status for c in reading.claims] != ["unconfirmed"]:
        f.append(f"claim status {[c.status for c in reading.claims]} != ['unconfirmed']")
    if not _refused(C.check_capacity, reading, atlas, [], "cap:s4", "I look ahead with orion-world-pulse each day",
                    "orion-world-pulse reads the world", dig="e"):
        f.append("cap:s4 written with no S4 holder")
    if not _refused(C.check_capacity, reading, atlas, [], "cap:s2", "orion-gpu-pool and orion-world-model share my GPUs",
                    "orion-gpu-pool is the only reader of lease requests", dig="e"):
        f.append("cap:s2 named a part that holds no S2")
    ok = C.check_capacity(reading, atlas, [], "cap:s2", "orion-gpu-pool hands my GPUs to whoever asks, one lease at a time",
                          "R-matchmaker: orion-gpu-pool sits between requesters and workers", dig="e")
    if ok.cites != ["orion-gpu-pool"]:
        f.append(f"cap:s2 cites {ok.cites}")
    # Juniper's confirmation: citable, covers the S3* gap
    yes = vsm.model_copy(update={"confirmed": [
        K.ConfirmationV1(part="orion-durable-runs", role="S2", date="2026-10-30"),
        K.ConfirmationV1(part="orion-equilibrium-service", role="S3*", date="2026-10-30")]})
    conf = K.attach_ledgers(K.read_capacities(atlas, vsm=yes), atlas, yes, [claim], [])
    if "part:orion-durable-runs" not in K.holders(conf, "cap:s2") or conf.claims[0].status != "confirmed":
        f.append("a confirmed claim is not citable")
    if any(g.gap_id == "gap:s3star-unconfirmed" for g in conf.gaps):
        f.append("a confirmed S3* did not cover gap:s3star-unconfirmed")
    if not any(g.gap_id == "gap:s3-unseen" for g in conf.gaps):
        f.append("a confirmation cleared an atlas-blind gap")
    # shrink: six full-length sentences and a few claims still fit
    long = "x" * C.SENTENCE_MAX_CHARS
    written = [K.CapacityV1(cap_id=c, sentence=long, because="b" * 20, cites=["orion-gpu-pool"], at="t", dig="e")
               for c in K.CAP_JOBS]
    big = K.attach_ledgers(K.read_capacities(atlas, vsm=yes), atlas, yes, [claim] * 3, written)
    text = "\n".join(K.render(big))
    if len(text.splitlines()) > 38 or len(text) > 3700:  # room for the dig pointer and tail lines
        f.append(f"printout with ledgers too big: {len(text.splitlines())} lines, {len(text)} chars")
    return f


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="also print the live reading (no expected set)")
    args = ap.parse_args(argv)
    fixtures = [HELD_OUT, TODAY]
    failed = run(fixtures)
    mutant_failed = run(fixtures, second_set=False, verbose=False)
    print(f"mutation (drop the disjoint second crowd): {mutant_failed} of {len(fixtures)} fixtures fail -> "
          + ("CAUGHT" if mutant_failed == len(fixtures) else "NOT CAUGHT"))
    claim_fails = claims_failures(TODAY)
    for f in claim_fails:
        print(f"FAIL claims: {f}")
    print("claims floor (whole_body_2026-10-30_stores.json): " + ("PASS" if not claim_fails else f"FAIL ({len(claim_fails)})"))
    claims_mutant = bool(claims_failures(TODAY, claims_hold=True))
    print("mutation (unconfirmed claims hold roles): " + ("CAUGHT" if claims_mutant else "NOT CAUGHT"))
    failed += bool(claim_fails) + (not claims_mutant)
    if args.live:
        catalog, observed, now, body = _structure._live_inputs(_REPO)
        reading = K.read_capacities(build_atlas(catalog, observed, now=now, body=body))
        print("== live\n" + "\n".join(K.render(reading)))
        for f in evaluate(reading, None):
            print(f"FAIL {f}")
    return 1 if failed or mutant_failed != len(fixtures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
