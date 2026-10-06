#!/usr/bin/env python3
"""Read-only replay of an explicit neighborhood request; no hydration or writes.

Run from the repo root: python -m scripts.replay_substrate_neighborhood \
    --request /tmp/request.json --uri redis://localhost:6380
The JSON file follows NeighborhoodRequestV1. Output contains IDs, not source text.
"""
import argparse
import json
from time import perf_counter

from orion.graph.falkor_client import RedisGraphQueryClient
from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig
from orion.substrate.neighborhood import NeighborhoodRequestV1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--graph", default="orion_substrate")
    args = parser.parse_args()
    with open(args.request) as source:
        request = NeighborhoodRequestV1.model_validate_json(source.read())
    client = RedisGraphQueryClient(uri=args.uri, graph_name=args.graph, read_only=True,
                                  socket_timeout=10, socket_connect_timeout=5)
    store = FalkorSubstrateStore(FalkorSubstrateStoreConfig(uri=args.uri, graph_name=args.graph),
                                client=client, hydrate=False)
    started = perf_counter()
    result = store.read_neighborhood(request)
    nodes = {n.node_id for n in result.focal_nodes + result.neighbor_nodes}
    intact = all(e.source.node_id in nodes and e.target.node_id in nodes
                 for e in result.internal_edges + result.boundary_edges)
    print(json.dumps({
        "request": request.model_dump(mode="json"), "read_only": client.read_only,
        "read_started_at": result.read_started_at, "read_finished_at": result.read_finished_at,
        "elapsed_ms": round((perf_counter() - started) * 1000, 2),
        "source_kind": result.source_kind, "consistency": result.consistency,
        "complete_for_request": result.complete_for_request, "truncated": result.truncated,
        "degraded": result.degraded, "reason": result.reason,
        "missing_focal_node_ids": result.missing_focal_node_ids,
        "focal_nodes": [n.node_id for n in result.focal_nodes],
        "neighbor_nodes": [n.node_id for n in result.neighbor_nodes],
        "internal_edges": [e.edge_id for e in result.internal_edges],
        "boundary_edges": [{"edge_id": e.edge_id, "source": e.source.node_id,
                            "target": e.target.node_id, "predicate": e.predicate}
                           for e in result.boundary_edges],
        "endpoint_integrity": intact,
    }, indent=2))
    return 1 if result.degraded or not intact else 0


if __name__ == "__main__":
    raise SystemExit(main())
