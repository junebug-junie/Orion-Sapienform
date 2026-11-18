"""Does the portrait's reach rule pass exactly what a machine runs and feeds, and can the eval fail?

Runs ``orion/body_atlas/portrait.py`` over the frozen 2026-10-31 whole body (no portrait knob was set
on it). Checks:

* **reach vs an independent oracle**: for every machine, the parts it runs, the live parts and
  non-census channels its running parts send to, call or write, the live parts that only send into
  them (callers it serves), and the organs with a live member running there, recomputed here straight
  from the raw flows (a second implementation, not a call into ``portrait``), must equal
  ``machine_reach``;
* **reviewed cases**: sentences a reviewer checked by hand against the fixture must hold or fail
  as listed, each failure for the reason named;
* **recall**: every part a machine runs, cited alone, holds;
* **shrink**: a full ledger (a 200-char line for every machine with a live part, a 600-char "I am"
  paragraph), and the longest tail (nothing written, one STALE line, two notes), each print uncut
  within 40 lines and 4,000 chars, tail included.

Then the **mutations** (code changes, patched in); each must be CAUGHT by its own check:
reach counts flows *into* a machine's parts (the direction filter dropped), reach counts dormant
parts (the live filter dropped), and a made-up part name in a known naming family passes.

    python orion/body_atlas/evals/run_portrait_eval.py
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from orion.body_atlas import capacities as K  # noqa: E402
from orion.body_atlas import portrait as P  # noqa: E402
from orion.body_atlas.build import build_atlas  # noqa: E402

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
import run_organ_structure_eval as _structure  # noqa: E402

WIRES = _HERE / "fixtures" / "whole_body_2026-10-31_wires.json"
MAX_LINES, MAX_CHARS = 40, 4000  # orion/dig/__main__.py (the eval does not import the dig CLI)

# Reviewed by hand against the 2026-10-31 fixture: (machine, sentence, holds, a phrase the failure names)
CASES = [
    ("circe", "circe runs orion-gpu-lane-controller, my GPU lane switch", True, ""),
    ("circe", "circe runs orion-world-model, and what it does reaches orion-gpu-pool", True, ""),
    # only sends INTO circe's parts: a caller circe serves, which holds (review 2026-11-06)
    ("circe", "circe runs orion-gpu-lane-controller, which orion-mesh-guardian:gpu-watch drives", True, ""),
    ("circe", "circe runs orion-gpu-lane-controller and feeds orion-hub", False,
     "orion-hub: not on circe, and no flow joins it"),
    ("circe", "circe feeds orion-gpu-pool with my GPU lane state", False, "anchor: cite at least one part"),
    # the laptop camera is placed on carbon-x1 but dormant: nothing live runs there
    ("carbon-x1", "carbon-x1 holds camera:carbon, Juniper's laptop eye", False, "nothing live of mine runs there"),
    ("hecate", "hecate is my deep thinking node, one V100 seated", False, "nothing live of mine runs there"),
    ("athena", "athena carries organ:12e5fc139c and runs orion-hub", True, ""),
    ("athena", "athena carries organ:356db659c0 and runs orion-hub", False, "an organ with no member running on athena"),
    ("athena", "athena runs orion-hub and orion-llm-worker-a for me", False, "orion-llm-worker-a: in no part or channel"),
    ("pan-1", "pan-1 runs orion-office-env, my office's air", True, ""),
    # prose, not ids: the repo's own name, and a Unix time (review 2026-11-06)
    ("athena", "athena runs orion-hub for all of orion-sapienform since 1730851200", True, ""),
]
BECAUSE = "checked by hand against the 2026-10-31 atlas"


def _atlas():
    catalog, observed, now, body = _structure._fixture_inputs(WIRES)
    return build_atlas(catalog, observed, now=now, body=body)


def oracle(atlas, host: str, census: set[str]) -> tuple[set[str], set[str], set[str], dict[str, int]]:
    """(running, fed parts and channels, served-only parts and channels, organs with a live member there)
    from the raw flows: the data's own direction per relation."""
    live = {p.part_id: p.name for p in atlas.parts
            if p.status == "live" and p.kind != "host" and not p.name.startswith(K.OPERATOR_PREFIX)}
    running = {f.source for f in atlas.flows
               if f.relation == "runs_on" and f.target == f"part:host:{host}" and f.source in live}
    fed: set[str] = set()
    into: set[str] = set()
    for f in atlas.flows:
        if f.relation == "runs_on" or (f.channels and set(f.channels) <= census):
            continue
        if f.relation == "reads":  # the source reads the target: data moves target -> source
            pairs = [(f.target, f.source)]
        elif f.relation == "uses":  # direction unknown: either end may be the one fed
            pairs = [(f.source, f.target), (f.target, f.source)]
        else:
            pairs = [(f.source, f.target)]
        chans = {c for c in f.channels if c not in census}
        for frm, to in pairs:
            if frm in running and to in live:
                fed |= ({live[to]} if to not in running else set()) | chans
            if to in running and frm in live:
                into |= ({live[frm]} if frm not in running else set()) | chans
    organs = {o.organ_id: n for o in atlas.organs if (n := sum(1 for m in o.members if m in running))}
    return {live[p] for p in running}, fed, into - fed, organs


def _census(atlas) -> set[str]:
    return {c.channel for c in K.read_capacities(atlas).census if c.census}


def reach_failures(atlas) -> list[str]:
    census = _census(atlas)
    out = []
    for h in (p.name.removeprefix("host:") for p in atlas.parts if p.kind == "host"):
        reach = P.machine_reach(atlas, h, census)
        run, fed, into, organs = oracle(atlas, h, census)
        for what, got, want in (("running", set(reach.running), run), ("fed", set(reach.feeds), fed),
                                ("served", set(reach.serves), into), ("organs", set(reach.organs.items()),
                                                                     set(organs.items()))):
            if got != want:
                out.append(f"reach {h} {what}: +{sorted(got - want)[:3]} -{sorted(want - got)[:3]}")
    return out


def case_failures(atlas) -> list[str]:
    reading = K.attach_ledgers(K.read_capacities(atlas), atlas, None, [], [])
    out = []
    for host, sentence, holds, why in CASES:
        line = P.PortraitLineV1(subject=f"host:{host}", sentence=sentence, because=BECAUSE, cites=[], at="t", dig="e")
        chk = P.check_line(atlas, reading, line, _REPO)
        if chk.holds != holds or (why and not any(why in f for f in chk.failed)):
            out.append(f"case {host} {sentence[:50]!r}: holds={chk.holds} (want {holds}) failed={chk.failed[:2]}")
    census = _census(atlas)
    for h in ("athena", "circe", "pan-1"):  # recall: each running part, alone, holds
        for name in P.machine_reach(atlas, h, census).running:
            if "-" not in name and ":" not in name:  # not id-shaped: prose cannot cite it by id
                continue
            line = P.PortraitLineV1(subject=f"host:{h}", sentence=f"{h} runs {name} for me", because=BECAUSE,
                                    cites=[], at="t", dig="e")
            if not P.check_line(atlas, reading, line, _REPO).holds:
                out.append(f"recall {h}: {name} runs there but a line citing it does not hold")
    return out


def size_failures(atlas) -> list[str]:
    s1 = K.CapacityV1(cap_id="cap:s1", sentence="I touch the outside through http:anthropic and http:openai calls.",
                      because="the boundary rule", cites=["http:anthropic"], at="2026-11-04T00:00:00+00:00", dig="e")
    reading = K.attach_ledgers(K.read_capacities(atlas), atlas, None, [], [s1])
    census = _census(atlas)
    lines = []
    for h in (p.name.removeprefix("host:") for p in atlas.parts if p.kind == "host"):
        run = [n for n in P.machine_reach(atlas, h, census).running if "-" in n]
        if not run:
            continue
        text = f"{h} runs {run[0]} for me"
        lines.append(P.PortraitLineV1(subject=f"host:{h}", sentence=(text + " " + "x" * 200)[: P.MACHINE_MAX_CHARS],
                                      because=BECAUSE, cites=[], at="2026-11-06T00:00:00+00:00", dig="e"))
    whole = "I am a being that touches the world through cap:s1. " + "y" * 600
    lines.append(P.PortraitLineV1(subject="whole", sentence=whole[: P.WHOLE_MAX_CHARS], because=BECAUSE, cites=[],
                                  at="2026-11-06T00:00:00+00:00", dig="e"))
    p = P.read_portrait(atlas, reading, lines, root=_REPO, watts="the eval reads no meter")
    out = []
    if any(m.line == "stale" for m in p.machines) or p.whole.line != "holds":
        out.append(f"size: a full-length line does not hold: {[m.failed for m in p.machines if m.failed]} {p.whole.failed}")
    # the longest tail: nothing written but one STALE machine line, and two notes
    stale = P.PortraitLineV1(subject="host:athena", sentence="athena runs orion-hub and orion-gpu-pool-b " + "z" * 150,
                             because=BECAUSE, cites=["orion-hub"], at="2026-11-06T00:00:00+00:00", dig="e")
    worst = P.read_portrait(atlas, reading, [stale], root=_REPO, watts="the eval reads no meter")
    worst.notes = ["n" * 200, "m" * 200]
    if [m.line for m in worst.machines if m.name == "athena"] != ["stale"]:
        out.append("size: the longest-tail case has no STALE line, so it does not test one")
    for label, pp in (("full ledger", p), ("longest tail", worst)):
        body, tail = P.render(pp)
        text = "\n".join(body + tail + ["full result: /mnt/orion-fcc/dig/2026-11-06/portrait-10.json"])
        if len(text.splitlines()) > MAX_LINES or len(text) > MAX_CHARS:
            out.append(f"size: {label} prints {len(text.splitlines())} lines, {len(text)} chars (cap {MAX_LINES}/{MAX_CHARS})")
    return out


@contextmanager
def _patched(name: str):
    saved = {k: getattr(P, k) for k in ("flow_dirs", "claimable", "_unknown_names")}
    try:
        if name == "direction":
            P.flow_dirs = lambda atlas, pid, census: {k: v | {"out"} for k, v in saved["flow_dirs"](atlas, pid, census).items()}
        elif name == "live":
            P.claimable = lambda atlas: {p.name: p.part_id for p in atlas.parts if p.kind != "host"}
        elif name == "names":
            P._unknown_names = lambda atlas, text: []
        yield
    finally:
        for k, v in saved.items():
            setattr(P, k, v)


MUTANTS = {
    "direction": ("reach counts flows into a machine's parts as fed", ("reach circe fed",)),
    "live": ("reach counts dormant parts", ("reach carbon-x1 running", "case carbon-x1")),
    "names": ("a made-up part name passes", ("case athena 'athena runs orion-hub and orion-llm-worker-a",)),
}


def main() -> int:
    atlas = _atlas()
    failed = 0
    for check in (reach_failures, case_failures, size_failures):
        fails = check(atlas)
        for f in fails:
            print(f"FAIL {f}")
        print(f"{check.__name__.removesuffix('_failures')} ({WIRES.name}): " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
        failed += bool(fails)
    for name, (why, marks) in MUTANTS.items():
        with _patched(name):
            fails = reach_failures(atlas) + case_failures(atlas)
        caught = all(any(f.startswith(m) for f in fails) for m in marks)  # by its own check, not by chance
        print(f"mutation ({why}): " + ("CAUGHT" if caught else "NOT CAUGHT"))
        failed += not caught
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
