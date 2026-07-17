"""Post-save Concept Atlas materialization via Cypher-native FalkorSubstrateStore.

Filters ``map_concept_profile_to_substrate`` output to Concept nodes and
concept↔concept edges only (PR #1120 durable-write contract). Evidence /
hypothesis / contradiction nodes are skipped intentionally — Option A thin cut.

Note: the profile mapper currently emits NO concept↔concept edges (its edges
are concept→hypothesis and evidence→concept), so the live outcome today is
concept NODES only. The edge path below is exercised by tests and becomes
live if/when the mapper emits concept↔concept relations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from orion.core.schemas.concept_induction import ConceptProfile
from orion.core.schemas.cognitive_substrate import ConceptNodeV1, SubstrateEdgeV1
from orion.substrate.adapters.concept_induction import map_concept_profile_to_substrate
from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig

logger = logging.getLogger("orion.spark.concept.falkor_materialization")


class SubstrateWriteStore(Protocol):
    def upsert_node(self, *, identity_key: str | None, node: Any) -> None: ...

    def upsert_edge(self, *, identity_key: str, edge: SubstrateEdgeV1) -> None: ...


@dataclass(frozen=True)
class FalkorMaterializationResult:
    concept_nodes: int
    concept_edges: int
    skipped_nodes: int
    skipped_edges: int


def filter_concept_atlas_record(record) -> tuple[list[ConceptNodeV1], list[SubstrateEdgeV1], int, int]:
    """Keep ConceptNodeV1 + edges whose both endpoints are concepts."""
    concepts = [n for n in record.nodes if isinstance(n, ConceptNodeV1)]
    concept_ids = {n.node_id for n in concepts}
    skipped_nodes = len(record.nodes) - len(concepts)
    edges: list[SubstrateEdgeV1] = []
    skipped_edges = 0
    for edge in record.edges:
        src_kind = getattr(edge.source, "node_kind", None)
        tgt_kind = getattr(edge.target, "node_kind", None)
        if (
            src_kind == "concept"
            and tgt_kind == "concept"
            and edge.source.node_id in concept_ids
            and edge.target.node_id in concept_ids
        ):
            edges.append(edge)
        else:
            skipped_edges += 1
    return concepts, edges, skipped_nodes, skipped_edges


def edge_identity_key(edge: SubstrateEdgeV1) -> str:
    """Canonical edge identity — must match FalkorSubstrateStore._edge_identity
    (``src|pred|tgt``) so identity caches agree with Hub and re-materialized
    revisions MERGE instead of duplicating relationships."""
    return f"{edge.source.node_id}|{edge.predicate}|{edge.target.node_id}"


def materialize_concept_profile_to_falkor(
    *,
    profile: ConceptProfile,
    store: SubstrateWriteStore,
    anchor_scope: str = "orion",
) -> FalkorMaterializationResult:
    record = map_concept_profile_to_substrate(profile=profile, anchor_scope=anchor_scope)
    concepts, edges, skipped_nodes, skipped_edges = filter_concept_atlas_record(record)
    for node in concepts:
        store.upsert_node(identity_key=node.node_id, node=node)
    for edge in edges:
        store.upsert_edge(identity_key=edge_identity_key(edge), edge=edge)
    logger.info(
        "concept_profile_falkor_materialization subject=%s revision=%s "
        "concept_nodes=%d concept_edges=%d skipped_nodes=%d skipped_edges=%d",
        profile.subject,
        profile.revision,
        len(concepts),
        len(edges),
        skipped_nodes,
        skipped_edges,
    )
    return FalkorMaterializationResult(
        concept_nodes=len(concepts),
        concept_edges=len(edges),
        skipped_nodes=skipped_nodes,
        skipped_edges=skipped_edges,
    )


def build_falkor_substrate_store(
    *,
    uri: str,
    graph_name: str,
    client: Any | None = None,
    hydrate: bool = False,
) -> FalkorSubstrateStore:
    """Construct a FalkorSubstrateStore. hydrate=False for write-only worker path."""
    return FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri=uri, graph_name=graph_name),
        client=client,
        hydrate=hydrate,
    )
