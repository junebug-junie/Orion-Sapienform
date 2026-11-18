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
job, is refused; a claim whose evidence check passes holds its role (EVIDENCE-CHECKED, citable,
#3107 + the evidence check, 2026-11-04) and covers its job's "no part is shown" gap, never an
atlas-blind one; it stops holding when its evidence does not hold, its part dies, Juniper rejects
it or a rule says no (a near miss is "not S2"); an S5 claim never holds; Juniper's confirmation
outranks a claim; the checked-in ``vsm_confirmed.yaml`` reads; six full-length capacities, claims
in every status, disputes and notes still fit the printout. Mutations (code changes, patched in),
each must be CAUGHT by its own check: a claim whose stored cites are trusted, a rejected claim, a
claim on a dead part, a claim a rule says no to, a claim Orion later disputed, and an S5 claim
each hold a role; and a held claim clears an atlas-blind gap line.

**Evidence check** (2026-11-04): Orion's first five live claims (frozen in
``fixtures/capacity_claims_2026-11-04.jsonl``) checked against the 2026-10-31 body. Each must pass or
fail as reviewed by hand, with the reason named; the same claims relabelled S5 must all fail; and
the printout with them plus the dig tail must fit uncut. Mutation: a check that passes a claim when
*any* reference holds, instead of every one (the shape gate kept); the eval must FAIL.

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
# patch 3: the live body with HTTP calls and shared Redis keys. The v1 rules read only the bus,
# so the wires must not move them (camera:doorbell is new and live since 10-30)
WIRES = _HERE / "fixtures" / "whole_body_2026-10-31_wires.json"


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
    WIRES.name: {
        "R-matchmaker": _p("orion-gpu-pool"),
        "R-census-reader": _p("orion-equilibrium-service"),
        "R-boundary-in": _p("camera:cam0", "camera:doorbell", "camera:walkway", "http:anthropic", "http:openai"),
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


def _with_floor_off(cut: str):
    """Install one mutant (a code change, patched in), each the bug a hold check guards against:
    ``evidence`` trusts the cites stored at write time instead of checking them again; ``rejected``
    drops Juniper's no; ``dead`` counts every part as live; ``rule-no`` ignores a near miss's "not S2";
    ``disputed`` ignores Orion's later dispute; ``s5`` lets an S5 claim hold; ``gap`` lets a held
    claim clear every gap line of its job, atlas-blind ones too. Returns the undo."""
    real_status, real_recheck, real_attach = K._claim_status, C.recheck, K.attach_ledgers
    if cut == "evidence":
        C.recheck = lambda reading, atlas, root: (lambda c: K.EvidenceCheckV1(
            part_id=c.part_id, role=c.role, at=c.at, holds=bool(c.cites), held=list(c.cites)))
    elif cut == "gap":
        def attach(*a, **kw):
            out = real_attach(*a, **kw)
            jobs = {r.role for r in out.claimed}
            out.gaps = [g for g in out.gaps if g.job not in jobs]
            return out
        K.attach_ledgers = attach
    else:
        def mutant(c, **kw):
            if cut == "rejected":
                kw["rejected"] = set()
            elif cut == "dead":
                kw["live"] = kw["live"] | {c.part_id}
            elif cut == "rule-no":
                kw["not_s2"] = set()
            elif cut == "disputed":
                kw["latest"] = {}
            elif cut == "s5" and c.role == "S5":
                return "evidence-checked", ["x"], []
            return real_status(c, **kw)
        K._claim_status = mutant

    def undo():
        K._claim_status, C.recheck, K.attach_ledgers = real_status, real_recheck, real_attach
    return undo


# each mutant is CAUGHT only when the floor's own check for it fails (not by chance)
MUTANTS = {"evidence": "a claim with gone evidence holds a role", "rejected": "a rejected claim holds a role",
           "dead": "a claim on a dead part holds a role", "rule-no": "a claim a rule says no to holds a role",
           "disputed": "a claim Orion later disputed holds a role", "s5": "an S5 claim holds a role",
           "gap": "a checked claim cleared an atlas-blind gap"}


def claims_failures(path: Path, *, mutant: str | None = None) -> list[str]:
    """``mutant``: run the floor with one of ``MUTANTS`` patched in."""
    undo = _with_floor_off(mutant) if mutant else (lambda: None)
    try:
        return _claims_failures(path)
    finally:
        undo()


def _attach(atlas, vsm, claims, written=()):
    r = K.read_capacities(atlas, vsm=vsm)
    return K.attach_ledgers(r, atlas, vsm, list(claims), list(written), recheck=C.recheck(r, atlas, _REPO))


def _because_for(reading, atlas, part: str) -> str:
    """A because citing the first evidence id the floor offers for this part (a holding claim)."""
    census = {c.channel for c in reading.census if c.census}
    ev = sorted(C.evidence_ids(atlas, f"part:{part}", census))[0]
    return f"{part} works with {ev} on the flows the atlas shows me"


def _claims_failures(path: Path) -> list[str]:
    catalog, observed, now, body = _structure._fixture_inputs(path)
    atlas = build_atlas(catalog, observed, now=now, body=body)
    try:
        vsm = C.load_confirmed(_REPO)
    except C.ClaimError as exc:
        return [f"checked-in vsm_confirmed.yaml does not read: {exc}"]
    if vsm.confirmed or vsm.rejected:
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
    # 2026-11-04: a claim whose evidence check passes on this read holds its role (EVIDENCE-CHECKED)
    reading = _attach(atlas, vsm, [claim])
    if [c.status for c in reading.claims] != ["evidence-checked"]:
        f.append(f"a claim whose evidence checks reads {[c.status for c in reading.claims]}, not ['evidence-checked']")
    if "part:orion-durable-runs" not in K.holders(reading, "cap:s2"):
        f.append("a claim whose evidence checks is not a cap:s2 holder")
    if [g.gap_id for g in reading.gaps] != [g.gap_id for g in base.gaps]:
        f.append("a claim changed the gap lines")
    audit = C.check_claim(base, atlas, "orion-equilibrium-service", "S3*",
                          _because_for(base, atlas, "orion-equilibrium-service"), kind="claim", root=_REPO, dig="e")
    a = _attach(atlas, vsm, [audit])
    if next(j.status for j in a.jobs if j.job == "S3*") != "EVIDENCE-CHECKED":
        f.append("a job held only by a claim does not print EVIDENCE-CHECKED")
    if any(g.gap_id == "gap:s3star-unconfirmed" for g in a.gaps):  # covered like a confirmation
        f.append("a checked S3* claim did not cover gap:s3star-unconfirmed")
    # Orion's own later dispute of the same part and job (the rule's S3* candidate) stops the claim
    late = C.check_claim(base, atlas, "orion-equilibrium-service", "S3*", _because_for(base, atlas,
                         "orion-equilibrium-service"), kind="dispute", root=_REPO, dig="e")
    a = _attach(atlas, vsm, [audit.model_copy(update={"at": "2026-11-04T00:00:00+00:00"}),
                             late.model_copy(update={"at": "2026-11-04T01:00:00+00:00"})])
    if a.claimed or "disputed" not in {c.status for c in a.claims}:
        f.append(MUTANTS["disputed"])
    # each way a claim must stop holding
    names = {p.name: p.part_id for p in atlas.parts}
    dr = names["orion-durable-runs"]
    cut = atlas.model_copy(update={"flows": [x for x in atlas.flows if dr not in (x.source, x.target)]})
    dead = atlas.model_copy(update={"parts": [p.model_copy(update={"status": "transient"}) if p.part_id == dr else p
                                              for p in atlas.parts]})
    no = vsm.model_copy(update={"rejected": [K.ConfirmationV1(part="orion-durable-runs", role="S2", date="2026-11-04")]})
    gw = C.evidence_ids(atlas, names["orion-llm-gateway"], {c.channel for c in base.census if c.census})
    near = C.check_claim(base, atlas, "orion-llm-gateway", "S2", f"orion-llm-gateway serves {sorted(gw)[0]} "
                         "and many more callers", kind="claim", root=_REPO, dig="e")
    for why, want, view, part in (
            (MUTANTS["evidence"], "evidence does not hold", _attach(cut, vsm, [claim]), dr),
            (MUTANTS["rejected"], "rejected", _attach(atlas, no, [claim]), dr),
            (MUTANTS["dead"], "part not live now", _attach(dead, vsm, [claim]), dr),
            (MUTANTS["rule-no"], "rule says no", _attach(atlas, vsm, [near]), names["orion-llm-gateway"])):
        if part in K.holders(view, "cap:s2") or view.claimed:
            f.append(why)
        elif [c.status for c in view.claims] != [want]:  # it stopped, but for the wrong reason
            f.append(f"{why.split(' holds')[0]} reads {[c.status for c in view.claims]}, not [{want!r}]")
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
    # Juniper's confirmation: citable, outranks a claim, covers the S3* gap
    yes = vsm.model_copy(update={"confirmed": [
        K.ConfirmationV1(part="orion-durable-runs", role="S2", date="2026-10-30"),
        K.ConfirmationV1(part="orion-equilibrium-service", role="S3*", date="2026-10-30")]})
    conf = _attach(atlas, yes, [claim, audit])
    if "part:orion-durable-runs" not in K.holders(conf, "cap:s2") or conf.claims[0].status != "confirmed":
        f.append("a confirmed claim is not citable")
    if next(j.status for j in conf.jobs if j.job == "S3*") != "CONFIRMED" or conf.claimed:
        f.append("a confirmation does not outrank a claim")
    if any(g.gap_id == "gap:s3star-unconfirmed" for g in conf.gaps):
        f.append("a confirmed S3* did not cover gap:s3star-unconfirmed")
    if not any(g.gap_id == "gap:s3-unseen" for g in conf.gaps):
        f.append("a confirmation cleared an atlas-blind gap")
    # shrink: six full-length sentences and a few claims still fit
    long = "x" * C.SENTENCE_MAX_CHARS
    written = [K.CapacityV1(cap_id=c, sentence=long, because="b" * 20, cites=["orion-gpu-pool"], at="t", dig="e")
               for c in K.CAP_JOBS]
    # holding claims (real evidence), one whose evidence is gone, one a rule says no to, disputes
    many = [claim.model_copy(update={"part_id": f"part:{p}", "role": r, "kind": k, "because": _because_for(base, atlas, p),
                                     "cites": []})
            for p, r, k in (("orion-world-pulse", "S4", "claim"), ("orion-prediction", "S4", "claim"),
                            ("orion-hub", "S3", "claim"), ("orion-hardware-watch", "algedonic", "claim"),
                            ("pg:self_concept_history", "S5", "claim"), ("orion-gpu-pool", "S2", "dispute"),
                            ("camera:cam0", "S1", "dispute"), ("http:openai", "S1", "dispute"))]
    many += [claim.model_copy(update={"part_id": "part:orion-hub", "role": "metabolism", "because": "x" * 400}), near]
    big = _attach(atlas, yes, [claim, *many], written)
    required = {"evidence-checked", "evidence does not hold", "rule says no", "confirmed", "unconfirmed"}
    if not required <= {c.status for c in big.claims}:
        f.append(f"the full-ledger case lacks a status: {sorted(required - {c.status for c in big.claims})}")
    if any(r.role == "S5" for r in big.claimed):
        f.append(MUTANTS["s5"])
    # a held S4 (orion-world-pulse) must leave every atlas-blind gap standing (on this bus-only body,
    # gap:s4-to-s3 is the bus line no claim can answer; with wires it is rechecked, not a blind gap)
    blind = {g.gap_id for g in base.gaps if g.gap_id not in K.COVERED_BY_CONFIRMATION
             and not (big.wires is not None and g.gap_id == "gap:s4-to-s3")}
    if not any(r.role == "S4" for r in big.claimed) or not blind <= {g.gap_id for g in big.gaps}:
        f.append(MUTANTS["gap"])
    big.notes += [f"note {i}: " + "n" * 120 for i in range(3)]
    text = "\n".join(K.render(big))
    if len(text.splitlines()) > 38 or len(text) > 3700:  # room for the dig pointer and tail lines
        f.append(f"printout with ledgers too big: {len(text.splitlines())} lines, {len(text)} chars")
    # What Orion reads: render plus the dig tail (claims to write, claims whose evidence is gone, path),
    # which `capped` never cuts. A usual reading (two claims waiting) must fit whole. In this worst
    # case (every ledger full, long notes) a cut is allowed, but the tail stays small and every job
    # line still shows.
    from orion.dig.__main__ import capped, claim_block

    path_line = "full result: /mnt/orion-fcc/dig/2026-11-04/capacities-99.json"
    tail = "\n".join(claim_block(big, atlas) + [path_line])
    if len(tail.splitlines()) > 12 or len(tail) > 900:
        f.append(f"dig tail with full ledgers too big: {len(tail.splitlines())} lines, {len(tail)} chars")
    shown = capped(K.render(big), tail=tail)
    if not all(ln in shown for ln in K.render(big)[: 1 + len(K.JOBS)]):
        f.append("dig capacities with full ledgers cut a job line")
    usual = _attach(atlas, yes, [claim, *many[:2]])
    shown = capped(K.render(usual), tail="\n".join(claim_block(usual, atlas) + [path_line]))
    if "the rest is in the file" in shown:
        f.append(f"dig capacities with two claims waiting is cut: {len(shown)} chars")
    return f


LIVE_CLAIMS = _HERE / "fixtures" / "capacity_claims_2026-11-04.jsonl"
# Reviewed by hand against the 2026-10-31 body (fixture WIRES) on 2026-11-04. The reason for each
# failure is a real atlas fact on that day, not a rule tuned to get it.
EVIDENCE_EXPECTED = {
    ("part:orion-gpu-lane-controller", "S3"): (True, None),  # sends actuate:result to gpu-pool, mesh-guardian
    ("part:orion-equilibrium-service", "S3*"): (True, None),  # R-census-reader hit; sends equilibrium:snapshot
    # on 10-31 the atlas had no redis:orion:self_monitor:latest part (it is live by 11-04)
    ("part:orion-durable-runs", "algedonic"): (False, "redis:orion:self_monitor:latest: in no part or channel"),
    ("part:orion-energy", "metabolism"): (False, "pg:energy_bill_forecast: dormant"),  # 0 rows
    ("part:orion-world-pulse", "S4"): (True, None),  # sends world_context:daily_capsule to cortex-exec
}


def _any_ref_holds(real):
    """The mutant: a claim passes when any reference holds, not every one (the shape gate stays)."""
    def mutant(atlas, claim, root, **kw):
        got = real(atlas, claim, root, **kw)
        shape = [x for x in got.failed if x.startswith("shape:")]
        return got.model_copy(update={"holds": bool(got.held) and not shape, "failed": shape})
    return mutant


def evidence_failures(path: Path, *, any_ref: bool = False) -> list[str]:
    real = C.check_evidence
    if any_ref:
        C.check_evidence = _any_ref_holds(real)
    try:
        return _evidence_failures(path)
    finally:
        C.check_evidence = real


def _evidence_failures(path: Path) -> list[str]:
    catalog, observed, now, body = _structure._fixture_inputs(path)
    atlas = build_atlas(catalog, observed, now=now, body=body)
    recs, skipped = C.load_ledger(LIVE_CLAIMS, K.CapacityClaimV1)
    f: list[str] = [f"{skipped} unreadable claim lines"] if skipped else []
    checks = C.check_claims(atlas, recs, _REPO)
    got = {(c.part_id, c.role): c for c in checks}
    for key, (holds, why) in EVIDENCE_EXPECTED.items():
        c = got.get(key)
        if c is None:
            f.append(f"evidence: {key} not checked")
        elif c.holds != holds:
            f.append(f"evidence: {key} holds={c.holds}, expected {holds} ({c.failed})")
        elif why and not any(x.startswith(why) for x in c.failed):
            f.append(f"evidence: {key} failed for {c.failed}, expected {why!r}")
    s5 = C.check_claims(atlas, [r.model_copy(update={"role": "S5"}) for r in recs], _REPO)
    if any(c.holds for c in s5):
        f.append("evidence: an S5 claim passed the code check")
    # Juniper's real S5 line, but none of her confirmations: any "confirmed by Juniper" is then a leak
    vsm = C.load_confirmed(_REPO).model_copy(update={"confirmed": [], "rejected": []})
    reading = _attach(atlas, vsm, recs)
    if {(c.part_id, c.role) for c in reading.claims if c.status == "evidence-checked"} != {
            k for k, (h, _) in EVIDENCE_EXPECTED.items() if h}:
        f.append("evidence: the reading's held claims differ from the check's passes")
    if any("confirmed by Juniper:" in j.text for j in reading.jobs):
        f.append("evidence: a checked claim printed as Juniper's confirmation")
    from orion.dig.__main__ import capped, claim_block

    shown = capped(K.render(reading), tail="\n".join(claim_block(reading, atlas)
                                                    + ["full result: /mnt/orion-fcc/dig/2026-11-04/capacities-99.json"]))
    if "the rest is in the file" in shown:
        f.append(f"evidence: dig capacities with the five live claims is cut: {len(shown)} chars")
    return f


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="also print the live reading (no expected set)")
    args = ap.parse_args(argv)
    fixtures = [HELD_OUT, TODAY, WIRES]
    failed = run(fixtures)
    mutant_failed = run(fixtures, second_set=False, verbose=False)
    print(f"mutation (drop the disjoint second crowd): {mutant_failed} of {len(fixtures)} fixtures fail -> "
          + ("CAUGHT" if mutant_failed == len(fixtures) else "NOT CAUGHT"))
    claim_fails = claims_failures(TODAY)
    for f in claim_fails:
        print(f"FAIL claims: {f}")
    print("claims floor (whole_body_2026-10-30_stores.json): " + ("PASS" if not claim_fails else f"FAIL ({len(claim_fails)})"))
    failed += bool(claim_fails)
    for cut, why in MUTANTS.items():
        caught = why in claims_failures(TODAY, mutant=cut)  # caught by the floor's own check, not by chance
        print(f"mutation ({why}): " + ("CAUGHT" if caught else "NOT CAUGHT"))
        failed += not caught
    ev_fails = evidence_failures(WIRES)
    for f in ev_fails:
        print(f"FAIL {f}")
    print(f"evidence check (5 live claims on {WIRES.name}): " + ("PASS" if not ev_fails else f"FAIL ({len(ev_fails)})"))
    ev_mutant = any(f.startswith("evidence: ") for f in evidence_failures(WIRES, any_ref=True))
    print("mutation (any reference holds, not every one): " + ("CAUGHT" if ev_mutant else "NOT CAUGHT"))
    failed += bool(ev_fails) + (not ev_mutant)
    if args.live:
        catalog, observed, now, body = _structure._live_inputs(_REPO)
        reading = K.read_capacities(build_atlas(catalog, observed, now=now, body=body))
        print("== live\n" + "\n".join(K.render(reading)))
        for f in evaluate(reading, None):
            print(f"FAIL {f}")
    return 1 if failed or mutant_failed != len(fixtures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
