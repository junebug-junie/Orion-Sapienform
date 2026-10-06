from datetime import datetime, timezone


from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1, EvidenceNodeV1, NodeRefV1, SubstrateEdgeV1,
    SubstrateProvenanceV1, SubstrateTemporalWindowV1,
)
from orion.substrate.store import InMemorySubstrateGraphStore

NOW = datetime(2026, 10, 6, tzinfo=timezone.utc)
PROVENANCE = SubstrateProvenanceV1(authority="local_inferred", source_kind="test",
    source_channel="test", producer="neighborhood_fixture")
TEMPORAL = SubstrateTemporalWindowV1(observed_at=NOW)


def concept(node_id, **kwargs):
    return ConceptNodeV1(node_id=node_id, label=node_id, anchor_scope="world",
        promotion_state="canonical", temporal=TEMPORAL, provenance=PROVENANCE, **kwargs)


def edge(edge_id, source, target, predicate="associated_with", **kwargs):
    return SubstrateEdgeV1(edge_id=edge_id, source=NodeRefV1(node_id=source, node_kind="concept"),
        target=NodeRefV1(node_id=target, node_kind="concept"), predicate=predicate,
        temporal=TEMPORAL, provenance=PROVENANCE, **kwargs)


def graph(nodes, edges):
    store = InMemorySubstrateGraphStore()
    for node in nodes:
        store.upsert_node(identity_key=node.node_id, node=node)
    for item in edges:
        store.upsert_edge(identity_key=item.edge_id, edge=item)
    return store


def hub_fixture(size=1500):
    focal = [concept(f"focal-{i}") for i in range(8)]
    neighbors = [concept(f"neighbor-{i:04}") for i in range(size)]
    internal = [edge(f"internal-{i}", f"focal-{i}", f"focal-{i+1}") for i in range(4)]
    boundary = [edge(f"boundary-{i:04}", n.node_id, "focal-0", salience=1) for i, n in enumerate(neighbors)]
    boundary += [edge("small-in", "neighbor-0000", "focal-7", "supports"),
                 edge("small-out", "focal-7", "neighbor-0001", "causes")]
    return graph(focal + neighbors, internal + boundary), [n.node_id for n in focal]


