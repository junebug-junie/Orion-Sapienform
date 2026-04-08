from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from orion.core.schemas.cognitive_substrate import BaseSubstrateNodeV1, SubstrateEdgeV1, SubstrateGraphRecordV1

from .reconcile import EdgeMergeDecision, NodeMergeDecision, SubstrateIdentityResolver, merge_edge, merge_node
from .graphdb_store import build_substrate_store_from_env
from .store import SubstrateGraphStore


@dataclass(frozen=True)
class MaterializationResultV1:
    source_graph_id: str
    nodes_seen: int
    edges_seen: int
    nodes_created: int
    nodes_merged: int
    edges_created: int
    edges_merged: int
    node_decisions: list[NodeMergeDecision]
    edge_decisions: list[EdgeMergeDecision]


class SubstrateGraphMaterializer:
    def __init__(
        self,
        *,
        store: SubstrateGraphStore | None = None,
        identity_resolver: SubstrateIdentityResolver | None = None,
    ) -> None:
        self._store = store or build_substrate_store_from_env()
        self._identity_resolver = identity_resolver or SubstrateIdentityResolver()

    @property
    def store(self) -> SubstrateGraphStore:
        return self._store

    def apply_record(self, record: SubstrateGraphRecordV1) -> MaterializationResultV1:
        canonical_id_by_input_id: dict[str, str] = {}
        node_decisions: list[NodeMergeDecision] = []
        edge_decisions: list[EdgeMergeDecision] = []
        nodes_created = 0
        nodes_merged = 0
        edges_created = 0
        edges_merged = 0

        for node in record.nodes:
            identity_key = self._identity_resolver.canonical_node_key(node)
            existing_id = self._store.get_node_id_by_identity(identity_key) if identity_key else self._store.get_node_by_id(node.node_id)
            existing_node = self._store.get_node_by_id(existing_id) if isinstance(existing_id, str) else None
            if existing_node is None and identity_key is None:
                existing_node = self._store.get_node_by_id(node.node_id)

            if existing_node is None:
                canonical_node = node.model_copy(update={"metadata": {**node.metadata, "materialized_from_graph_id": record.graph_id}})
                self._store.upsert_node(identity_key=identity_key, node=canonical_node)
                nodes_created += 1
                node_decisions.append(NodeMergeDecision(canonical_node_id=canonical_node.node_id, merged=False, reason="created"))
            else:
                canonical_node = merge_node(existing_node, node, source_graph_id=record.graph_id)
                self._store.upsert_node(identity_key=identity_key, node=canonical_node)
                nodes_merged += 1
                node_decisions.append(NodeMergeDecision(canonical_node_id=canonical_node.node_id, merged=True, reason="identity_match" if identity_key else "node_id_match"))

            canonical_id_by_input_id[node.node_id] = canonical_node.node_id

        for edge in record.edges:
            canonical_source = canonical_id_by_input_id.get(edge.source.node_id, edge.source.node_id)
            canonical_target = canonical_id_by_input_id.get(edge.target.node_id, edge.target.node_id)
            canonical_edge = edge.model_copy(
                update={
                    "source": edge.source.model_copy(update={"node_id": canonical_source}),
                    "target": edge.target.model_copy(update={"node_id": canonical_target}),
                    "metadata": {**edge.metadata, "materialized_from_graph_id": record.graph_id},
                }
            )
            identity_key = self._identity_resolver.canonical_edge_key(canonical_edge)
            existing_edge_id = self._store.get_edge_id_by_identity(identity_key)
            existing_edge = self._store.get_edge_by_id(existing_edge_id) if existing_edge_id else None
            if existing_edge is None:
                self._store.upsert_edge(identity_key=identity_key, edge=canonical_edge)
                edges_created += 1
                edge_decisions.append(EdgeMergeDecision(canonical_edge_id=canonical_edge.edge_id, merged=False, reason="created"))
            else:
                merged_edge = merge_edge(existing_edge, canonical_edge, source_graph_id=record.graph_id)
                self._store.upsert_edge(identity_key=identity_key, edge=merged_edge)
                edges_merged += 1
                edge_decisions.append(EdgeMergeDecision(canonical_edge_id=merged_edge.edge_id, merged=True, reason="identity_match"))

        return MaterializationResultV1(
            source_graph_id=record.graph_id,
            nodes_seen=len(record.nodes),
            edges_seen=len(record.edges),
            nodes_created=nodes_created,
            nodes_merged=nodes_merged,
            edges_created=edges_created,
            edges_merged=edges_merged,
            node_decisions=node_decisions,
            edge_decisions=edge_decisions,
        )

    def apply_records(self, records: Iterable[SubstrateGraphRecordV1]) -> list[MaterializationResultV1]:
        return [self.apply_record(record) for record in records]
