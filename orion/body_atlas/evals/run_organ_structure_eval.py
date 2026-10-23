"""Does Orion's circulation converge into organs, or into mush?

Runs the atlas over a frozen export of the real bus graph (``fixtures/``) or, with ``--live``,
the live one, and checks the structure:

* modularity Q >= 0.3 (Newman: below that, community structure is weak);
* 5..25 organs, and no organ holding more than 35% of the parts (a blob is not an organ);
* every non-transient part in exactly one organ;
* pairs anyone would call one organ land together (eyes with eyes, cortex with cortex).

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

FIXTURE = Path(__file__).parent / "fixtures" / "bus_graph_2026-10-23.json"
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
        return load_catalog(root), fetch_observed(reader), datetime.now(timezone.utc)
    finally:
        reader.close()


def _export(path: Path, root: Path) -> None:
    catalog, observed, now = _live_inputs(root)
    keep = ("name", "producer_services", "consumer_services")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "exported_at": now.isoformat(),
        "catalog": [{k: c.get(k) for k in keep} for c in catalog],
        "observed": {"publishes": observed.publishes, "causal": observed.causal},
    }, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {path}: {len(catalog)} channels, {len(observed.publishes)} publish + "
          f"{len(observed.causal)} causal edges")


def _fixture_inputs(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    observed = ObservedBus(publishes=[tuple(r) for r in data["observed"]["publishes"]],
                           causal=[tuple(r) for r in data["observed"]["causal"]])
    return data["catalog"], observed, datetime.fromisoformat(data["exported_at"])


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
    body = [p.part_id for p in atlas.parts if p.status != "transient"]
    bad = [p for p in body if placed.get(p) != 1]
    if bad:
        failures.append(f"{len(bad)} parts not in exactly one organ, e.g. {bad[:3]}")
    organ_of = {m: o.organ_id for o in atlas.organs for m in o.members}
    for a, b in SAME_ORGAN:
        if organ_of.get(f"part:{a}") is None or organ_of.get(f"part:{a}") != organ_of.get(f"part:{b}"):
            failures.append(f"{a} and {b} are in different organs")
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--export", type=Path)
    ap.add_argument("--fixture", type=Path, default=FIXTURE)
    ap.add_argument("--root", type=Path, default=_REPO)
    args = ap.parse_args(argv)
    if args.export:
        _export(args.export, args.root)
        return 0
    catalog, observed, now = _live_inputs(args.root) if args.live else _fixture_inputs(args.fixture)
    atlas = build_atlas(catalog, observed, now=now)
    print(summarize(atlas, out=None))
    failures = evaluate(atlas)
    for f in failures:
        print(f"FAIL {f}")
    print("organ structure: " + ("PASS" if not failures else f"FAIL ({len(failures)})"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
