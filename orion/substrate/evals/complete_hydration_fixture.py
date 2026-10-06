"""Deterministic capped durable rows shared by scan regressions and evals."""
from datetime import datetime, timezone

from orion.graph.falkor_client import RecordingFalkorClient
from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1, NodeRefV1, SubstrateEdgeV1, SubstrateProvenanceV1, SubstrateTemporalWindowV1,
)
from orion.substrate.falkor_codec import encode_node_properties, encode_edge_properties
from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig


def _concept(*, node_id="fixture-node"):
    return ConceptNodeV1(
        node_id=node_id, label=node_id, anchor_scope="orion",
        temporal=SubstrateTemporalWindowV1(observed_at=datetime(2026, 10, 6, tzinfo=timezone.utc)),
        provenance=SubstrateProvenanceV1(authority="local_inferred", source_kind="eval",
                                        source_channel="fixture", producer="complete_hydration_eval"),
    )


def _hydrated_node_row(node_id, identity_key):
    return encode_node_properties(_concept(node_id=node_id), identity_key=identity_key)


def edge_row(i, source="node0", target="node1"):
    node = _concept()
    edge = SubstrateEdgeV1(
        edge_id=f"edge{i}", source=NodeRefV1(node_id=source, node_kind="concept"),
        target=NodeRefV1(node_id=target, node_kind="concept"), predicate="associated_with",
        temporal=node.temporal, provenance=node.provenance,
    )
    return dict(encode_edge_properties(edge, identity_key=f"edge:{i}"), source_id=source, target_id=target)


class CappedClient(RecordingFalkorClient):
    def __init__(self, nodes=7, edges=11, cap=2):
        super().__init__(hydrate_node_rows=[_hydrated_node_row(f"node{i}", f"node:{i}") for i in range(nodes)],
                         hydrate_edge_rows=[edge_row(i) for i in range(edges)])
        self.cap = cap
        self.fail = False

    def graph_query(self, cypher, params=None):
        if self.fail and params and params.get("after_id", -1) >= self.cap - 1:
            raise ConnectionError("mid-page outage")
        result = super().graph_query(cypher, params)
        return result[:self.cap] if params and "after_id" in params else result


def build(client, **kw):
    return FalkorSubstrateStore(FalkorSubstrateStoreConfig(uri="redis://unused", **kw), client=client)

