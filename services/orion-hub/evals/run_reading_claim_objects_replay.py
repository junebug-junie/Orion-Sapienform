#!/usr/bin/env python3
"""Live replay, read-only: which existing concepts would recent reads be offered as claim
objects, by label (before) and by meaning (orion/world_pulse_read/concept_search.py)?

    PYTHONPATH=. .venv/bin/python services/orion-hub/evals/run_reading_claim_objects_replay.py \
        [--limit 7] [--min-similarity 0.65] [--cache /tmp/x/concept_vecs.json]

Reads the last --limit done readings that retained text (world_pulse_read_seed +
reading_document_snapshot, SELECT only), hydrates the Concept Atlas from Falkor (reads
only; no index bootstrap), and embeds every atlas concept and each read's subjects with
vector-host (/embedding persists nothing). It does NOT read or write the Chroma index, so
it works before Hub has built it; similarities are the same bge cosine the index serves.

Prints, per read: subjects, exact-label objects, meaning objects with similarity, and how
many offered objects a claim could be ACCEPTED to under the strict quote rule (the quote
must name the object by its stored label, so its label must be in the retained text).
Then the pooled max-over-subjects similarity distribution, the data the floor was set from.
Exit 0 always on a completed replay; 2 when Postgres, Falkor or the embedder is unavailable.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path

import httpx

from orion.schemas.world_pulse_read import WorldPulseReadHandoffV1
from orion.substrate.reconcile import is_identity_fenced
from orion.world_pulse_read.assertions import MAX_OBJECTS, build_claim_context
from orion.world_pulse_read.concept_search import OBJECT_MIN_SIMILARITY, concept_doc
from orion.world_pulse_read.fetch_text import load_retained_texts

SEEDS_SQL = """
SELECT seed_id, handoff_json FROM world_pulse_read_seed
WHERE stage2_status = 'done' AND handoff_json IS NOT NULL
  AND EXISTS (SELECT 1 FROM jsonb_array_elements(handoff_json->'read_evidence') e
              WHERE e->>'content_sha256' IS NOT NULL)
ORDER BY stage2_completed_at DESC LIMIT $1
"""


def _cos(a, b) -> float:
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=7)
    parser.add_argument("--min-similarity", type=float, default=OBJECT_MIN_SIMILARITY)
    parser.add_argument("--dsn", default=os.environ.get(
        "RECALL_PG_DSN", "postgresql://postgres:postgres@127.0.0.1:55432/conjourney"))
    parser.add_argument("--falkor", default=os.environ.get("FALKORDB_URI", "redis://127.0.0.1:6380"))
    parser.add_argument("--embed-url", default=os.environ.get(
        "HUB_READING_SEARCH_EMBED_URL", "http://127.0.0.1:8320/embedding"))
    parser.add_argument("--cache", help="JSON file of concept-doc vectors, reused across runs")
    args = parser.parse_args()

    import asyncpg

    from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig

    try:
        conn = await asyncpg.connect(args.dsn)
        store = FalkorSubstrateStore(FalkorSubstrateStoreConfig(uri=args.falkor, ensure_indexes=False))
    except Exception as exc:  # noqa: BLE001
        print(f"UNKNOWN: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    nodes = [n for n in store.read_concept_region(limit_nodes=5000, limit_edges=1).nodes
             if not is_identity_fenced(n)]
    cache_path = Path(args.cache) if args.cache else None
    cache: dict[str, list[float]] = json.loads(cache_path.read_text()) if cache_path and cache_path.exists() else {}
    async with httpx.AsyncClient(timeout=10.0) as client:
        async def emb(text: str) -> list[float]:
            if text not in cache:
                resp = await client.post(args.embed_url, json={"doc_id": "replay-probe", "text": text})
                resp.raise_for_status()
                cache[text] = resp.json()["embedding"]
            return cache[text]

        try:
            started = time.perf_counter()
            for n in nodes:
                await emb(concept_doc(n))
            print(f"atlas concepts={len(nodes)} embedded_in={time.perf_counter() - started:.1f}s")
            if cache_path:
                cache_path.write_text(json.dumps(cache))
            pooled: list[float] = []
            totals = {"reads": 0, "usable_before": 0, "usable_after": 0, "acceptable_before": 0,
                      "acceptable_after": 0}
            for row in await conn.fetch(SEEDS_SQL, args.limit):
                handoff = WorldPulseReadHandoffV1.model_validate(json.loads(row["handoff_json"]))
                texts = await load_retained_texts(conn, handoff.read_evidence)
                ctx = build_claim_context(store, handoff, texts)
                retained = "\n".join(t.text for t in texts).lower()
                subjects = {s.node_id for s in ctx.subjects}
                exact = [o.label for o in ctx.objects]
                meaning: list[tuple[float, str]] = []
                if ctx.subjects:
                    qv = [await emb(concept_doc(store.get_node_by_id(s)) or s) for s in subjects]
                    offered = subjects | {o.node_id for o in ctx.objects}
                    scored = sorted(((max(_cos(q, cache[concept_doc(n)]) for q in qv), n.label)
                                     for n in nodes if n.node_id not in subjects), reverse=True)
                    pooled += [s for s, _ in scored]
                    by_label = {n.label: n.node_id for n in nodes}
                    meaning = [(s, lbl) for s, lbl in scored if s >= args.min_similarity
                               and by_label.get(lbl) not in offered][: MAX_OBJECTS - len(exact)]
                ok_before = sum(1 for lbl in exact if lbl.lower() in retained)
                ok_after = ok_before + sum(1 for _, lbl in meaning if lbl.lower() in retained)
                totals["reads"] += 1
                totals["usable_before"] += bool(ctx.subjects and exact and texts)
                totals["usable_after"] += bool(ctx.subjects and (exact or meaning) and texts)
                totals["acceptable_before"] += ok_before
                totals["acceptable_after"] += ok_after
                print(f"\n{row['seed_id'][:60]}  texts={[len(t.text) for t in texts]} "
                      f"subjects={[s.label for s in ctx.subjects]}")
                print(f"  by label ({len(exact)}): {exact}")
                print(f"  by meaning ({len(meaning)}): " + ", ".join(f"{lbl!r}={s:.3f}" for s, lbl in meaning))
                print(f"  acceptable under the quote rule: before={ok_before} after={ok_after}")
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            print(f"UNKNOWN: embedder: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        finally:
            await conn.close()
    pooled.sort()
    if pooled:
        q = lambda p: pooled[min(len(pooled) - 1, int(len(pooled) * p))]  # noqa: E731
        print(f"\npooled max-over-subjects similarity n={len(pooled)} p50={q(.5):.3f} p90={q(.9):.3f} "
              f"p95={q(.95):.3f} p99={q(.99):.3f} max={pooled[-1]:.3f} floor={args.min_similarity:.2f}")
    print("totals " + " ".join(f"{k}={v}" for k, v in totals.items()))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
