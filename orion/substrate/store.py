from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from orion.core.schemas.cognitive_substrate import BaseSubstrateNodeV1, SubstrateEdgeV1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class MaterializedSubstrateGraphState:
    nodes: dict[str, BaseSubstrateNodeV1]
    edges: dict[str, SubstrateEdgeV1]
    node_identity_index: dict[str, str]
    edge_identity_index: dict[str, str]


@dataclass(frozen=True)
class SubstrateNeighborhoodSliceV1:
    nodes: list[BaseSubstrateNodeV1]
    edges: list[SubstrateEdgeV1]


@dataclass(frozen=True)
class SubstrateQueryResultV1:
    query_kind: str
    slice: SubstrateNeighborhoodSliceV1
    source_kind: str
    degraded: bool = False
    error: str | None = None
    truncated: bool = False
    limits: dict[str, int] = field(default_factory=dict)
    generated_at: str = field(default_factory=_utc_now_iso)
    details: dict[str, Any] = field(default_factory=dict)


class SubstrateGraphStore(Protocol):
    def get_node_by_id(self, node_id: str) -> BaseSubstrateNodeV1 | None: ...
    def get_edge_by_id(self, edge_id: str) -> SubstrateEdgeV1 | None: ...
    def get_node_id_by_identity(self, identity_key: str) -> str | None: ...
    def get_edge_id_by_identity(self, identity_key: str) -> str | None: ...
    def upsert_node(self, *, identity_key: str | None, node: BaseSubstrateNodeV1) -> None: ...
    def upsert_edge(self, *, identity_key: str, edge: SubstrateEdgeV1) -> None: ...
    def snapshot(self) -> MaterializedSubstrateGraphState: ...

    def query_focal_slice(self, *, node_ids: list[str], max_edges: int = 64) -> SubstrateQueryResultV1: ...
    def query_hotspot_region(self, *, min_salience: float = 0.6, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1: ...
    def query_contradiction_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1: ...
    def query_concept_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1: ...
    def query_provenance_neighborhood(self, *, evidence_ref: str, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1: ...

    def read_focal_slice(self, *, node_ids: list[str], max_edges: int = 64) -> SubstrateNeighborhoodSliceV1: ...
    def read_hotspot_region(self, *, min_salience: float = 0.6, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateNeighborhoodSliceV1: ...
    def read_contradiction_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateNeighborhoodSliceV1: ...
    def read_concept_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateNeighborhoodSliceV1: ...
    def read_provenance_neighborhood(self, *, evidence_ref: str, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateNeighborhoodSliceV1: ...


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

    def query_focal_slice(self, *, node_ids: list[str], max_edges: int = 64) -> SubstrateQueryResultV1:
        edges_limit = max(1, int(max_edges))
        slice_value = self.read_focal_slice(node_ids=node_ids, max_edges=edges_limit)
        return SubstrateQueryResultV1(
            query_kind="focal_slice",
            slice=slice_value,
            source_kind="cache",
            limits={"max_edges": edges_limit, "node_ids": len(node_ids)},
        )

    def query_hotspot_region(self, *, min_salience: float = 0.6, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1:
        nodes_limit = max(1, int(limit_nodes))
        edges_limit = max(1, int(limit_edges))
        slice_value = self.read_hotspot_region(min_salience=min_salience, limit_nodes=nodes_limit, limit_edges=edges_limit)
        return SubstrateQueryResultV1(
            query_kind="hotspot_region",
            slice=slice_value,
            source_kind="cache",
            limits={"limit_nodes": nodes_limit, "limit_edges": edges_limit},
            details={"min_salience": float(min_salience)},
            truncated=len(slice_value.nodes) >= nodes_limit or len(slice_value.edges) >= edges_limit,
        )

    def query_contradiction_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1:
        nodes_limit = max(1, int(limit_nodes))
        edges_limit = max(1, int(limit_edges))
        slice_value = self.read_contradiction_region(limit_nodes=nodes_limit, limit_edges=edges_limit)
        return SubstrateQueryResultV1(
            query_kind="contradiction_region",
            slice=slice_value,
            source_kind="cache",
            limits={"limit_nodes": nodes_limit, "limit_edges": edges_limit},
            truncated=len(slice_value.nodes) >= nodes_limit or len(slice_value.edges) >= edges_limit,
        )

    def query_concept_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1:
        nodes_limit = max(1, int(limit_nodes))
        edges_limit = max(1, int(limit_edges))
        slice_value = self.read_concept_region(limit_nodes=nodes_limit, limit_edges=edges_limit)
        return SubstrateQueryResultV1(
            query_kind="concept_region",
            slice=slice_value,
            source_kind="cache",
            limits={"limit_nodes": nodes_limit, "limit_edges": edges_limit},
            truncated=len(slice_value.nodes) >= nodes_limit or len(slice_value.edges) >= edges_limit,
        )

    def query_provenance_neighborhood(self, *, evidence_ref: str, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1:
        nodes_limit = max(1, int(limit_nodes))
        edges_limit = max(1, int(limit_edges))
        slice_value = self.read_provenance_neighborhood(evidence_ref=evidence_ref, limit_nodes=nodes_limit, limit_edges=edges_limit)
        return SubstrateQueryResultV1(
            query_kind="provenance_neighborhood",
            slice=slice_value,
            source_kind="cache",
            limits={"limit_nodes": nodes_limit, "limit_edges": edges_limit},
            details={"evidence_ref": str(evidence_ref or "")},
            truncated=len(slice_value.nodes) >= nodes_limit or len(slice_value.edges) >= edges_limit,
        )

    def read_focal_slice(self, *, node_ids: list[str], max_edges: int = 64) -> SubstrateNeighborhoodSliceV1:
        node_set = {str(node_id) for node_id in node_ids if str(node_id)}
        nodes = [self._nodes[node_id] for node_id in node_set if node_id in self._nodes]
        edge_candidates = [
            edge
            for edge in self._edges.values()
            if edge.source.node_id in node_set or edge.target.node_id in node_set
        ]
        edge_candidates.sort(key=lambda edge: (edge.salience, edge.confidence), reverse=True)
        return SubstrateNeighborhoodSliceV1(nodes=nodes, edges=edge_candidates[: max(1, int(max_edges))])

    def _read_by_node_predicate(self, *, node_predicate, limit_nodes: int, limit_edges: int) -> SubstrateNeighborhoodSliceV1:
        bounded_nodes = max(1, int(limit_nodes))
        bounded_edges = max(1, int(limit_edges))
        nodes = [node for node in self._nodes.values() if node_predicate(node)]
        nodes.sort(key=lambda node: (node.signals.salience, node.signals.confidence), reverse=True)
        selected_nodes = nodes[:bounded_nodes]
        node_ids = {node.node_id for node in selected_nodes}
        edges = [
            edge
            for edge in self._edges.values()
            if edge.source.node_id in node_ids or edge.target.node_id in node_ids
        ]
        edges.sort(key=lambda edge: (edge.salience, edge.confidence), reverse=True)
        return SubstrateNeighborhoodSliceV1(nodes=selected_nodes, edges=edges[:bounded_edges])

    def read_hotspot_region(self, *, min_salience: float = 0.6, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateNeighborhoodSliceV1:
        threshold = max(0.0, min(1.0, float(min_salience)))
        return self._read_by_node_predicate(
            node_predicate=lambda node: float(node.signals.salience) >= threshold,
            limit_nodes=limit_nodes,
            limit_edges=limit_edges,
        )

    def read_contradiction_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateNeighborhoodSliceV1:
        return self._read_by_node_predicate(
            node_predicate=lambda node: node.node_kind == "contradiction",
            limit_nodes=limit_nodes,
            limit_edges=limit_edges,
        )

    def read_concept_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateNeighborhoodSliceV1:
        return self._read_by_node_predicate(
            node_predicate=lambda node: node.node_kind == "concept",
            limit_nodes=limit_nodes,
            limit_edges=limit_edges,
        )

    def read_provenance_neighborhood(self, *, evidence_ref: str, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateNeighborhoodSliceV1:
        needle = str(evidence_ref or "").strip()
        if not needle:
            return SubstrateNeighborhoodSliceV1(nodes=[], edges=[])
        return self._read_by_node_predicate(
            node_predicate=lambda node: needle in (node.provenance.evidence_refs or []),
            limit_nodes=limit_nodes,
            limit_edges=limit_edges,
        )
