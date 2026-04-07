from __future__ import annotations

from dataclasses import dataclass

from orion.core.schemas.cognitive_substrate import BaseSubstrateNodeV1, SubstrateEdgeV1


@dataclass(frozen=True)
class MaterializedSubstrateGraphState:
    nodes: dict[str, BaseSubstrateNodeV1]
    edges: dict[str, SubstrateEdgeV1]
    node_identity_index: dict[str, str]
    edge_identity_index: dict[str, str]


class InMemorySubstrateGraphStore:
    """Bounded persistent in-memory materialized substrate graph state."""

    def __init__(self) -> None:
        self._nodes: dict[str, BaseSubstrateNodeV1] = {}
        self._edges: dict[str, SubstrateEdgeV1] = {}
        self._node_identity_index: dict[str, str] = {}
        self._edge_identity_index: dict[str, str] = {}

    def get_node_by_id(self, node_id: str) -> BaseSubstrateNodeV1 | None:
        return self._nodes.get(node_id)

    def get_edge_by_id(self, edge_id: str) -> SubstrateEdgeV1 | None:
        return self._edges.get(edge_id)

    def get_node_id_by_identity(self, identity_key: str) -> str | None:
        return self._node_identity_index.get(identity_key)

    def get_edge_id_by_identity(self, identity_key: str) -> str | None:
        return self._edge_identity_index.get(identity_key)

    def upsert_node(self, *, identity_key: str | None, node: BaseSubstrateNodeV1) -> None:
        self._nodes[node.node_id] = node
        if identity_key:
            self._node_identity_index[identity_key] = node.node_id

    def upsert_edge(self, *, identity_key: str, edge: SubstrateEdgeV1) -> None:
        self._edges[edge.edge_id] = edge
        self._edge_identity_index[identity_key] = edge.edge_id

    def snapshot(self) -> MaterializedSubstrateGraphState:
        return MaterializedSubstrateGraphState(
            nodes=dict(self._nodes),
            edges=dict(self._edges),
            node_identity_index=dict(self._node_identity_index),
            edge_identity_index=dict(self._edge_identity_index),
        )
