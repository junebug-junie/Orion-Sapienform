#!/usr/bin/env python3
"""Read-only replay of an explicit neighborhood request; no hydration or writes.

Run from the repo root: python -m scripts.replay_substrate_neighborhood \
    --request /tmp/request.json --uri redis://localhost:6380
The JSON file follows NeighborhoodRequestV1. Output contains IDs, not source text.

--evidence N also reads evidence handles (read_evidence_handles, N per node)
for the focal nodes plus the returned neighbors (at most 16 ids), the shape
recall-by-referent uses. Handles are content_refs; no text is hydrated.
"""
import argparse
import json
from time import perf_counter

from orion.graph.falkor_client import RedisGraphQueryClient
from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig
from orion.substrate.evidence_handles import EvidenceHandleRequestV1
from orion.substrate.neighborhood import NeighborhoodRequestV1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--graph", default="orion_substrate")
    parser.add_argument("--evidence", type=int, default=0, metavar="N",
                        help="also read up to N evidence handles per focal/neighbor node")
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
    evidence = None
    if args.evidence:
        ids = [n.node_id for n in result.focal_nodes + result.neighbor_nodes][:16]
        started = perf_counter()
        handles = store.read_evidence_handles(EvidenceHandleRequestV1(
            node_ids=tuple(ids), per_node_limit=args.evidence)) if ids else None
        evidence = None if handles is None else {
            "elapsed_ms": round((perf_counter() - started) * 1000, 2),
            "complete_for_request": handles.complete_for_request, "truncated": handles.truncated,
            "degraded": handles.degraded, "reason": handles.reason,
            "missing_node_ids": handles.missing_node_ids, "truncated_node_ids": handles.truncated_node_ids,
            "handles": [{"node_id": h.node_id, "edge_id": h.edge_id, "predicate": h.predicate,
                         "evidence_type": h.evidence_type, "content_ref": h.content_ref,
                         "valid_from": (h.valid_from or h.observed_at).isoformat()} for h in handles.handles],
        }
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
        "evidence": evidence,
    }, indent=2))
    return 1 if result.degraded or not intact or (evidence and evidence["degraded"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
