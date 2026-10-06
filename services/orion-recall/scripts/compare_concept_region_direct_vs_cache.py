"""Live check: concept_region via direct Falkor reads vs the old hydrated cache.

For each turn text, runs the real collector twice -- once against a fully
hydrated ``FalkorSubstrateStore`` (what recall used before 2026-10-06) and once
against ``FalkorDirectConceptStore`` -- and reports fragment equality plus
timings, split into turns where a concept matched and turns where nothing did.

The comparison only counts if the cache hydrate actually completed: the script
exits 2 before comparing anything if ``last_hydrate_ok`` is not True or the
scan receipt is not complete (an empty or partial cache would "agree" with an
empty direct read for the wrong reason).

Two modes:

* default (read-only, safe against production): times
  ``fetch_concept_region_fragment`` (read only, no reinforcement). Both stores
  read through ``GRAPH.RO_QUERY``; the direct store's writer refuses writes.
* ``--reinforce`` (WRITES): times the real live unit,
  ``fetch_concept_region_fragment_and_reinforce`` (worker.py), including the
  activation-bump reads and ``MERGE`` writes. Only for a throwaway copy of the
  graph (e.g. ``redis-cli DUMP``/``RESTORE`` into a scratch FalkorDB); refuses
  to run without ``--confirm-throwaway`` and refuses anything that looks like
  production: host ``orion-athena-falkordb``, port 6380 (production's published
  port on this host), or the same resolved address as the configured
  ``FALKORDB_URI``.

Timings are per query, every sample: ``max`` is the true maximum over all
samples, not the max of per-query medians.

Usage:

    docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc \
      "select query from recall_telemetry where (backend_counts->>'concept_region')::int > 0
       order by created_at desc limit 20" > /tmp/queries.txt
    python services/orion-recall/scripts/compare_concept_region_direct_vs_cache.py \
      --uri redis://127.0.0.1:6380 --queries-file /tmp/queries.txt
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "services" / "orion-recall"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.collectors.concept_region import (  # noqa: E402
    fetch_concept_region_fragment,
    fetch_concept_region_fragment_and_reinforce,
)
from orion.graph.falkor_client import RedisGraphQueryClient  # noqa: E402
from orion.substrate.falkor_direct import FalkorDirectConceptStore  # noqa: E402
from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig  # noqa: E402

PRODUCTION_HOST_PORT = 6380
PRODUCTION_HOSTNAME = "orion-athena-falkordb"
PRODUCTION_URI_DEFAULT = f"redis://{PRODUCTION_HOSTNAME}:6379"


class _RefuseWrites:
    read_only = True

    def graph_query(self, cypher, params=None):
        raise RuntimeError("read-only comparison: write refused")


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 1)


def summarize(rows: list[dict]) -> dict:
    samples = [s for r in rows for s in r["samples_ms"]]
    return {
        "queries": len(rows),
        "identical": sum(1 for r in rows if r["identical_ordered"]),
        "fragments_cache": sum(r["cache"] for r in rows),
        "fragments_direct": sum(r["direct"] for r in rows),
        "samples": len(samples),
        "ms_median": round(statistics.median(samples), 1) if samples else None,
        "ms_p90": _pct(samples, 0.9),
        "ms_p99": _pct(samples, 0.99),
        "ms_max": round(max(samples), 1) if samples else None,
    }


def hydrate_is_complete(store: FalkorSubstrateStore) -> bool:
    receipt = store.last_scan_receipt
    return store.last_hydrate_ok is True and receipt is not None and bool(receipt.complete)


def _endpoints(uri: str) -> set[tuple[str, int]]:
    parsed = urlparse(uri)
    host = parsed.hostname or "localhost"
    port = int(parsed.port or 6379)
    try:
        addrs = {info[4][0] for info in socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)}
    except OSError:
        addrs = set()
    return {(addr, port) for addr in addrs} | {(host.lower(), port)}


def _production_uris() -> list[str]:
    uris = [PRODUCTION_URI_DEFAULT]
    configured = str(os.getenv("FALKORDB_URI", "")).strip()
    if configured:
        uris.append(configured)
    return uris


def is_production_uri(uri: str) -> bool:
    """True if ``uri`` names production FalkorDB: its container hostname, its
    published host port, or the same resolved address+port as the default
    production URI or the configured FALKORDB_URI."""
    parsed = urlparse(uri)
    if (parsed.hostname or "").lower() == PRODUCTION_HOSTNAME or parsed.port == PRODUCTION_HOST_PORT:
        return True
    target = _endpoints(uri)
    return any(target & _endpoints(prod) for prod in _production_uris())


def reinforce_allowed(uri: str, confirm_throwaway: bool) -> bool:
    return bool(confirm_throwaway) and not is_production_uri(uri)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="redis://127.0.0.1:6380")
    parser.add_argument("--graph", default="orion_substrate")
    parser.add_argument("--queries-file", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reinforce", action="store_true", help="time the reinforcing unit (WRITES)")
    parser.add_argument("--confirm-throwaway", action="store_true")
    args = parser.parse_args(argv)

    if args.reinforce and not reinforce_allowed(args.uri, args.confirm_throwaway):
        print("refusing --reinforce: it writes; point --uri at a throwaway copy and pass --confirm-throwaway",
              file=sys.stderr)
        return 2

    queries = [line.strip() for line in Path(args.queries_file).read_text().splitlines() if line.strip()]
    cfg = FalkorSubstrateStoreConfig(uri=args.uri, graph_name=args.graph)

    started = time.perf_counter()
    cache_store = FalkorSubstrateStore(
        cfg, client=RedisGraphQueryClient(uri=args.uri, graph_name=args.graph, read_only=True)
    )
    hydrate_s = time.perf_counter() - started
    receipt = cache_store.last_scan_receipt
    if not hydrate_is_complete(cache_store):
        print(json.dumps({
            "error": "cache hydrate incomplete; comparison would be meaningless",
            "last_hydrate_ok": cache_store.last_hydrate_ok,
            "receipt_complete": bool(receipt and receipt.complete),
            "reason": getattr(receipt, "reason", None),
        }), file=sys.stderr)
        return 2

    writer_client = (
        RedisGraphQueryClient(uri=args.uri, graph_name=args.graph) if args.reinforce else _RefuseWrites()
    )
    direct = FalkorDirectConceptStore(
        read_client=RedisGraphQueryClient(uri=args.uri, graph_name=args.graph, read_only=True),
        writer=FalkorSubstrateStore(cfg, client=writer_client, hydrate=False),
    )
    timed = fetch_concept_region_fragment_and_reinforce if args.reinforce else fetch_concept_region_fragment

    rows = []
    for text in queries:
        cached = fetch_concept_region_fragment(text, store=cache_store)
        samples = []
        new: list = []
        for _ in range(max(1, args.repeats)):
            t0 = time.perf_counter()
            new = timed(text, store=direct)
            samples.append((time.perf_counter() - t0) * 1000)
        rows.append({
            "query": text[:70],
            "cache": len(cached),
            "direct": len(new),
            "identical_ordered": cached == new,
            "samples_ms": [round(s, 1) for s in samples],
        })

    matched = [r for r in rows if r["cache"] or r["direct"]]
    empty = [r for r in rows if not (r["cache"] or r["direct"])]
    print(json.dumps(
        {
            "mode": "reinforce (writes)" if args.reinforce else "read-only",
            "hydrate_s": round(hydrate_s, 2),
            "hydrate_ok": cache_store.last_hydrate_ok,
            "hydrate_complete": receipt.complete,
            "nodes": receipt.node_count,
            "edges": receipt.edge_count,
            "queries": len(rows),
            "identical": sum(1 for r in rows if r["identical_ordered"]),
            "matched": summarize(matched),
            "empty": summarize(empty),
            "rows": rows,
        },
        indent=2,
    ))
    return 0 if all(r["identical_ordered"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
