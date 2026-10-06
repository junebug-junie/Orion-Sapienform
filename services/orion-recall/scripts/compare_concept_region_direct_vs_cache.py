"""Read-only live check: concept_region via direct Falkor reads vs the old hydrated cache.

For each turn text, runs the real collector (``fetch_concept_region_fragment``,
no reinforcement) twice -- once against a fully hydrated ``FalkorSubstrateStore``
(what recall used before 2026-10-06) and once against
``FalkorDirectConceptStore`` -- and reports fragment-id overlap plus timings.

Everything is read-only: both stores read through a ``GRAPH.RO_QUERY`` client,
the hydrated store skips its legacy rewrite on a read-only client, and the
direct store's writer refuses any write.

Usage (from repo root, Falkor published on the host at 6380):

    docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc \
      "select query from recall_telemetry where (backend_counts->>'concept_region')::int > 0
       order by created_at desc limit 20" > /tmp/queries.txt
    python services/orion-recall/scripts/compare_concept_region_direct_vs_cache.py \
      --uri redis://127.0.0.1:6380 --queries-file /tmp/queries.txt
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "services" / "orion-recall"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.collectors.concept_region import fetch_concept_region_fragment  # noqa: E402
from orion.graph.falkor_client import RedisGraphQueryClient  # noqa: E402
from orion.substrate.falkor_direct import FalkorDirectConceptStore  # noqa: E402
from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig  # noqa: E402


class _RefuseWrites:
    read_only = True

    def graph_query(self, cypher, params=None):
        raise RuntimeError("read-only comparison: write refused")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="redis://127.0.0.1:6380")
    parser.add_argument("--graph", default="orion_substrate")
    parser.add_argument("--queries-file", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    queries = [line.strip() for line in Path(args.queries_file).read_text().splitlines() if line.strip()]
    cfg = FalkorSubstrateStoreConfig(uri=args.uri, graph_name=args.graph)

    started = time.perf_counter()
    cache_store = FalkorSubstrateStore(
        cfg, client=RedisGraphQueryClient(uri=args.uri, graph_name=args.graph, read_only=True)
    )
    hydrate_s = time.perf_counter() - started
    receipt = cache_store.last_scan_receipt
    direct = FalkorDirectConceptStore(
        read_client=RedisGraphQueryClient(uri=args.uri, graph_name=args.graph, read_only=True),
        writer=FalkorSubstrateStore(cfg, client=_RefuseWrites(), hydrate=False),
    )

    rows = []
    for text in queries:
        old = fetch_concept_region_fragment(text, store=cache_store)
        samples = []
        new = []
        for _ in range(max(1, args.repeats)):
            t0 = time.perf_counter()
            new = fetch_concept_region_fragment(text, store=direct)
            samples.append((time.perf_counter() - t0) * 1000)
        old_ids = [f["id"] for f in old]
        new_ids = [f["id"] for f in new]
        union = set(old_ids) | set(new_ids)
        rows.append(
            {
                "query": text[:70],
                "old": len(old_ids),
                "new": len(new_ids),
                "jaccard": (len(set(old_ids) & set(new_ids)) / len(union)) if union else 1.0,
                "identical_ordered": old == new,
                "direct_ms_median": round(statistics.median(samples), 1),
            }
        )

    print(json.dumps(
        {
            "hydrate_s": round(hydrate_s, 2),
            "hydrate_complete": bool(receipt and receipt.complete),
            "nodes": receipt.node_count if receipt else None,
            "edges": receipt.edge_count if receipt else None,
            "queries": len(rows),
            "identical": sum(1 for r in rows if r["identical_ordered"]),
            "mean_jaccard": round(statistics.mean(r["jaccard"] for r in rows), 4) if rows else None,
            "direct_ms_median_all": round(statistics.median(r["direct_ms_median"] for r in rows), 1) if rows else None,
            "direct_ms_max": max((r["direct_ms_median"] for r in rows), default=None),
            "rows": rows,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
