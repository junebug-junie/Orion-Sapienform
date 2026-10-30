"""Are the atlas's ``writes`` / ``reads`` labels on Chroma/FalkorDB stores true?

The organ eval only measures structure, and structure does not depend on direction. This eval
runs ``store_code.code_store_refs`` over a real checkout and checks it against hand-checked
facts (each with the line that proves it, read 2026-10-30):

* ``TRUE_DIRECTIONS``: (service, store, relation) the code really does. Every one that the scan
  still labels must carry that label; one that drops to ``uses`` counts as a miss.
* ``FALSE_DIRECTIONS``: (service, store, relation) that an earlier version of the scan claimed
  and that are wrong (review of PR fix/atlas-store-writers). None may come back.

Gate: precision of every directional claim the scan makes over the known facts is 1.0, and no
false claim comes back. Recall is printed, not gated: ``uses`` is the honest fallback.

    python orion/body_atlas/evals/run_store_direction_eval.py [--root CHECKOUT]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from orion.body_atlas.store_code import code_store_refs  # noqa: E402

TRUE_DIRECTIONS = [
    # services/orion-vector-host/app/main.py: _publish_semantic_upsert(collection_name=SOCIAL_ROOM_COLLECTION)
    ("orion-vector-host", "orion_social_room", "writes"),
    # services/orion-rag/app/vector_store.py: self.collection.query(...)
    ("orion-rag", "orion_main_store", "reads"),
    # services/orion-vector-writer/app/collection_browse.py: coll.get(...), coll.count()
    ("orion-vector-writer", "orion_main_store", "reads"),
    # services/orion-hub/scripts/concept_atlas_routes.py: RedisGraphQueryClient(..., read_only=True)
    ("orion-hub", "orion_substrate_aitown", "reads"),
    # services/orion-hub/scripts/dream_routes.py: WorldviewReader(...).query("MATCH (p:Prior) ...")
    ("orion-hub", "orion_worldview", "reads"),
]
FALSE_DIRECTIONS = [
    ("orion-hub", "orion_curiosity", "writes"),  # asyncio.create_task(urgent_request_loop(...))
    ("orion-hub", "orion_substrate", "writes"),  # same create_task
    ("orion-cocreation-signals", "doc_semantic_drift", "writes"),  # asyncio.create_task(loop(...))
    ("orion-recall", "orion_substrate", "writes"),  # parser.add_argument(default=...) in a CLI script
    ("orion-recall", "orion_memory_crystallizations", "writes"),  # insert_retrieval_event: Postgres
    ("orion-graphiti-adapter", "graphiti_temporal", "reads"),  # result.get("skipped"): a dict
    ("orion-hub", "orion_worldview", "writes"),  # queue.put_nowait(dict(payload))
    ("orion-hub", "orion_memory_crystallizations", "writes"),  # Postgres helpers
    ("orion-bus-mirror", "orion_bus_synapse", "writes"),  # a one-off migration script
]


def evaluate(root: Path) -> tuple[list[str], str]:
    stores = {s for _, s, _ in TRUE_DIRECTIONS + FALSE_DIRECTIONS}
    refs = code_store_refs(root, stores).refs
    got = {(r.service, s, r.relation) for s, rs in refs.items() for r in rs}
    directional = {g for g in got if g[2] != "uses"}
    known = set(TRUE_DIRECTIONS) | set(FALSE_DIRECTIONS)
    judged = directional & known
    right = judged & set(TRUE_DIRECTIONS)
    failures = [f"false direction claimed: {g}" for g in sorted(judged - set(TRUE_DIRECTIONS))]
    # a true fact labeled the opposite way is wrong too
    for svc, store, rel in TRUE_DIRECTIONS:
        other = {g for g in directional if g[:2] == (svc, store) and g[2] != rel}
        failures += [f"wrong direction: {g} (true: {rel})" for g in sorted(other)]
    precision = len(right) / len(judged) if judged else 1.0
    recall = len(right) / len(TRUE_DIRECTIONS)
    unjudged = sorted(directional - known)
    summary = (f"directional claims {len(directional)}, judged {len(judged)}, precision {precision:.2f}, "
               f"recall {recall:.2f} ({len(right)}/{len(TRUE_DIRECTIONS)} true facts labeled)"
               + (f"; not yet hand-checked: {unjudged}" if unjudged else ""))
    return failures, summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=_REPO)
    args = ap.parse_args(argv)
    failures, summary = evaluate(args.root)
    print(summary)
    for f in failures:
        print(f"FAIL {f}")
    print("store direction: " + ("PASS" if not failures else f"FAIL ({len(failures)})"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
