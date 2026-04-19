#!/usr/bin/env python3
"""One-shot demo seed: writes canonical substrate nodes/edges into GraphDB via the same store stack as the hub.

Run inside the hub container (host-network + GraphDB on localhost), for example:
  docker exec orion-athena-hub sh -c 'cd /tmp && env PYTHONPATH=/repo python3 /repo/services/orion-hub/scripts/seed_substrate_graphdb.py'

Use ``cd /tmp`` so Python does not pick ``/app/orion`` (WORKDIR) ahead of ``PYTHONPATH``; the repo mount must supply the current ``orion`` package.

Requires the same env as the hub: SUBSTRATE_STORE_BACKEND=graphdb and SUBSTRATE_GRAPHDB_* / GRAPHDB_*.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    ContradictionNodeV1,
    NodeRefV1,
    SubstrateEdgeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
    SubstrateTemporalWindowV1,
)
from orion.substrate import build_substrate_store_from_env
from orion.substrate.graphdb_store import GraphDBSubstrateStore
from orion.substrate.materializer import SubstrateGraphMaterializer


def _demo_record() -> SubstrateGraphRecordV1:
    now = datetime.now(timezone.utc)
    temporal = SubstrateTemporalWindowV1(observed_at=now)
    provenance = SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="seed_script",
        source_channel="seed_substrate_graphdb",
        producer="orion-hub.scripts.seed_substrate_graphdb",
        evidence_refs=["seed:evidence:1", "seed:evidence:2"],
    )
    node_a = ConceptNodeV1(
        node_id="live-seed-node-a",
        anchor_scope="orion",
        subject_ref="orion",
        temporal=temporal,
        provenance=provenance,
        signals=SubstrateSignalBundleV1(confidence=0.82, salience=0.78),
        label="Live substrate seed — concept A",
        definition="Operator-seeded concept for review bootstrap.",
        metadata={"seed": "true", "concept_id": "seed-a"},
    )
    node_b = ConceptNodeV1(
        node_id="live-seed-node-b",
        anchor_scope="orion",
        subject_ref="orion",
        temporal=temporal,
        provenance=provenance,
        signals=SubstrateSignalBundleV1(confidence=0.7, salience=0.65),
        label="Live substrate seed — concept B",
        definition="Operator-seeded concept for review bootstrap.",
        metadata={"seed": "true", "concept_id": "seed-b"},
    )
    contradiction = ContradictionNodeV1(
        node_id="live-seed-contradiction",
        anchor_scope="orion",
        subject_ref="orion",
        temporal=temporal,
        provenance=provenance,
        signals=SubstrateSignalBundleV1(confidence=0.85, salience=0.91),
        summary="Seeded contradiction for live substrate / review queue.",
        involved_node_ids=["live-seed-node-a", "live-seed-node-b"],
        metadata={"seed": "true", "dynamic_pressure": 0.72, "resolved": False},
    )
    edge = SubstrateEdgeV1(
        edge_id="live-seed-edge-a-b",
        source=NodeRefV1(node_id="live-seed-node-a", node_kind="concept"),
        target=NodeRefV1(node_id="live-seed-node-b", node_kind="concept"),
        predicate="associated_with",
        temporal=temporal,
        confidence=0.88,
        salience=0.7,
        provenance=provenance,
    )
    return SubstrateGraphRecordV1(
        graph_id="graph-operator-seed",
        anchor_scope="orion",
        subject_ref="orion",
        nodes=[node_a, node_b, contradiction],
        edges=[edge],
    )


def main() -> int:
    store = build_substrate_store_from_env()
    if not isinstance(store, GraphDBSubstrateStore):
        print(
            "ERROR: semantic store is not GraphDBSubstrateStore (got "
            f"{type(store).__name__}). Set SUBSTRATE_STORE_BACKEND=graphdb and endpoint env.",
            file=sys.stderr,
        )
        return 2

    record = _demo_record()
    result = SubstrateGraphMaterializer(store=store).apply_record(record)
    out = {
        "store": type(store).__name__,
        "graph_id": record.graph_id,
        "nodes_created": result.nodes_created,
        "nodes_merged": result.nodes_merged,
        "edges_created": result.edges_created,
        "edges_merged": result.edges_merged,
    }
    print(json.dumps(out, indent=2))

    hot = store.query_hotspot_region(limit_nodes=8, limit_edges=16)
    con = store.query_contradiction_region(limit_nodes=8, limit_edges=16)
    cpt = store.query_concept_region(limit_nodes=8, limit_edges=16)
    probe = {
        "hotspot_nodes": len(hot.slice.nodes),
        "contradiction_nodes": len(con.slice.nodes),
        "concept_nodes": len(cpt.slice.nodes),
        "hotspot_source_kind": hot.source_kind,
    }
    print(json.dumps(probe, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
