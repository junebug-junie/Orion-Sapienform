"""Does Orion's circulation converge into organs, or into mush?

Runs the atlas over the frozen exports in ``fixtures/`` (the real bus graph of 2026-10-23, the
whole body of 2026-10-29: hosts, cameras, stores and processes on top of the bus, and the whole
body of 2026-10-30 with each process linked to the services doing its work) or, with
``--live``, the live whole body, and checks the structure:

* modularity Q >= 0.3 (Newman: below that, community structure is weak);
* 5..25 organs, and no organ holding more than 35% of the parts (a blob is not an organ);
* every non-transient part in exactly one organ, and no host in any (hosts are placement);
* pairs anyone would call one organ land together (eyes with eyes, cortex with cortex);
* a store with any writer or reader flow is never a lone organ (only an unattributed one is).

    python orion/body_atlas/evals/run_organ_structure_eval.py              # frozen fixture
    python orion/body_atlas/evals/run_organ_structure_eval.py --live       # needs $FALKORDB_URI
    python orion/body_atlas/evals/run_organ_structure_eval.py --export F   # refresh a fixture

Live and export need redis-py and the mirror graph; run them in orion-harness-governor.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from orion.body_atlas.__main__ import DEFAULT_GRAPH, _RoGraphReader, summarize  # noqa: E402
from orion.body_atlas.build import ObservedBus, build_atlas, fetch_observed, load_catalog  # noqa: E402
from orion.body_atlas.sources import BodyInputs, default_live_readers, gather_body  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "bus_graph_2026-10-23.json"
# The whole body (hosts, cameras, stores, processes) on top of the live bus, 2026-10-29: the new
# kinds must not collapse the organs either.
WHOLE_BODY_FIXTURE = Path(__file__).parent / "fixtures" / "whole_body_2026-10-29.json"
# 2026-10-30: processes linked to the services doing their work (reducer sources, harness turn
# records, child runs), not only to their runner.
DOERS_FIXTURE = Path(__file__).parent / "fixtures" / "whole_body_2026-10-30.json"
# 2026-10-30, later: Chroma/FalkorDB stores linked to the services whose code names them
# (store_code.py), so a store named only in code is no longer a lone organ.
STORES_FIXTURE = Path(__file__).parent / "fixtures" / "whole_body_2026-10-30_stores.json"
FIXTURES = (FIXTURE, WHOLE_BODY_FIXTURE, DOERS_FIXTURE, STORES_FIXTURE)
MIN_Q = 0.3
ORGAN_RANGE = (5, 25)
MAX_ORGAN_SHARE = 0.35
# (a, b): parts that are one organ by any reading of the code (both live on the export day).
SAME_ORGAN = [
    ("orion-vision-host", "orion-vision-retina"),
    ("orion-vision-frame-router", "orion-vision-council"),
    ("orion-cortex-orch", "orion-cortex-exec"),
    ("orion-gpu-lane-controller", "orion-mesh-guardian"),
]


def _live_inputs(root: Path):
    uri = os.environ.get("FALKORDB_URI")
    if not uri:
        raise SystemExit("--live/--export need $FALKORDB_URI")
    reader = _RoGraphReader(uri, os.environ.get("FALKORDB_BUS_GRAPH", DEFAULT_GRAPH))
    try:
        catalog, observed = load_catalog(root), fetch_observed(reader)
    finally:
        reader.close()
    body = gather_body(root, catalog, live=default_live_readers(uri))
    return catalog, observed, datetime.now(timezone.utc), body


def _export(path: Path, root: Path) -> None:
    catalog, observed, now, body = _live_inputs(root)
    keep = ("name", "producer_services", "consumer_services")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "exported_at": now.isoformat(),
        "catalog": [{k: c.get(k) for k in keep} for c in catalog],
        "observed": {"publishes": observed.publishes, "causal": observed.causal},
        "body": body.model_dump(mode="json"),
    }, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {path}: {len(catalog)} channels, {len(observed.publishes)} publish + "
          f"{len(observed.causal)} causal edges, whole body with {len(body.reads)} source reads")


def _fixture_inputs(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    observed = ObservedBus(publishes=[tuple(r) for r in data["observed"]["publishes"]],
                           causal=[tuple(r) for r in data["observed"]["causal"]])
    body = BodyInputs.model_validate(data["body"]) if data.get("body") else None
    return data["catalog"], observed, datetime.fromisoformat(data["exported_at"]), body


def evaluate(atlas) -> list[str]:
    failures = []
    s = atlas.stats
    if s.modularity < MIN_Q:
        failures.append(f"modularity {s.modularity} < {MIN_Q}")
    if not ORGAN_RANGE[0] <= s.organs <= ORGAN_RANGE[1]:
        failures.append(f"{s.organs} organs outside {ORGAN_RANGE}")
    if s.largest_organ_share > MAX_ORGAN_SHARE:
        failures.append(f"largest organ holds {s.largest_organ_share:.0%} of parts (> {MAX_ORGAN_SHARE:.0%})")
    placed: dict[str, int] = {}
    for organ in atlas.organs:
        for member in organ.members:
            placed[member] = placed.get(member, 0) + 1
    body = [p.part_id for p in atlas.parts if p.status != "transient" and p.kind != "host"]
    bad = [p for p in body if placed.get(p) != 1]
    if bad:
        failures.append(f"{len(bad)} parts not in exactly one organ, e.g. {bad[:3]}")
    hosts_in = [p.part_id for p in atlas.parts if p.kind == "host" and p.part_id in placed]
    if hosts_in:
        failures.append(f"hosts are placement, not function, yet {hosts_in[:3]} sit in organs")
    organ_of = {m: o.organ_id for o in atlas.organs for m in o.members}
    size = {o.organ_id: len(o.members) for o in atlas.organs}
    linked = {f.target for f in atlas.flows if f.relation in ("writes", "reads", "uses")}
    alone = [p.part_id for p in atlas.parts if p.kind == "store" and p.part_id in linked
             and size.get(organ_of.get(p.part_id, ""), 0) == 1]
    if alone:
        failures.append(f"{len(alone)} stores with a writer or reader sit alone, e.g. {alone[:3]}")
    for a, b in SAME_ORGAN:
        if organ_of.get(f"part:{a}") is None or organ_of.get(f"part:{a}") != organ_of.get(f"part:{b}"):
            failures.append(f"{a} and {b} are in different organs")
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--export", type=Path)
    ap.add_argument("--fixture", type=Path, help="one fixture (default: every frozen fixture)")
    ap.add_argument("--root", type=Path, default=_REPO)
    args = ap.parse_args(argv)
    if args.export:
        _export(args.export, args.root)
        return 0
    if args.live:
        runs = [("live", _live_inputs(args.root))]
    else:
        runs = [(p.name, _fixture_inputs(p)) for p in ([args.fixture] if args.fixture else FIXTURES)]
    failed = 0
    for label, (catalog, observed, now, body) in runs:
        atlas = build_atlas(catalog, observed, now=now, body=body)
        print(f"== {label}")
        print(summarize(atlas, out=None))
        failures = evaluate(atlas)
        for f in failures:
            print(f"FAIL {f}")
        print(f"organ structure ({label}): " + ("PASS" if not failures else f"FAIL ({len(failures)})"))
        failed += bool(failures)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
