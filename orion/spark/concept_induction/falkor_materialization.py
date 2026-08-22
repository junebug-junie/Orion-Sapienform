"""Post-save Concept Atlas materialization via Cypher-native FalkorSubstrateStore.

Filters ``map_concept_profile_to_substrate`` output to Concept nodes and
concept↔concept edges only (PR #1120 durable-write contract). Evidence /
hypothesis / contradiction nodes are skipped intentionally — Option A thin cut.

As of 2026-08-22 the profile mapper emits real concept↔concept edges: each
induced concept gets an ``associated_with`` edge from the golden subject
anchor node it was induced from (Orion/Juniper/the relationship — see
``orion.substrate.adapters.concept_induction._GOLDEN_SUBJECT_ANCHOR_NODE_IDS``).
That anchor node is NOT part of the profile's own ``record.nodes`` (it's a
pre-existing node seeded separately at Hub startup), so
``filter_concept_atlas_record`` below only requires ONE endpoint of a
concept↔concept edge to belong to this record — not both — or every such
edge would be silently dropped as "referencing an unknown node."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from orion.core.schemas.concept_induction import ConceptProfile
from orion.core.schemas.cognitive_substrate import ConceptNodeV1, SubstrateEdgeV1
from orion.substrate.adapters.concept_induction import map_concept_profile_to_substrate
from orion.substrate.falkor_codec import EXTERNALLY_OWNED_METADATA_KEYS
from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig

logger = logging.getLogger("orion.spark.concept.falkor_materialization")


class SubstrateWriteStore(Protocol):
    def upsert_node(
        self,
        *,
        identity_key: str | None,
        node: Any,
        skip_metadata_keys: frozenset[str] | None = None,
    ) -> None: ...

    def upsert_edge(self, *, identity_key: str, edge: SubstrateEdgeV1) -> None: ...


@dataclass(frozen=True)
class FalkorMaterializationResult:
    concept_nodes: int
    concept_edges: int
    skipped_nodes: int
    skipped_edges: int


def filter_concept_atlas_record(record) -> tuple[list[ConceptNodeV1], list[SubstrateEdgeV1], int, int]:
    """Keep ConceptNodeV1 + edges whose both endpoints are concept-kind.

    An edge only needs ONE endpoint present in this record's own concept set,
    not both -- the subject-anchor edges the profile mapper now emits
    (concept_induction.py's ``_GOLDEN_SUBJECT_ANCHOR_NODE_IDS``) reference a
    pre-existing golden node (Orion/Juniper/the relationship) that was seeded
    into the store separately, so it never appears in ``record.nodes``
    itself. Both endpoints must still be ``node_kind="concept"`` -- this
    never lets an evidence/hypothesis/contradiction endpoint through.
    """
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
            and (edge.source.node_id in concept_ids or edge.target.node_id in concept_ids)
        ):
            edges.append(edge)
        else:
            skipped_edges += 1
    return concepts, edges, skipped_nodes, skipped_edges


def edge_identity_key(edge: SubstrateEdgeV1) -> str:
    """Canonical edge identity for the in-process identity cache.

    Must match ``FalkorSubstrateStore._edge_identity`` (``src|pred|tgt``) so
    identity lookups agree with Hub. Durable Cypher still MERGEs relationships
    on ``edge_id`` today — deterministic ``edge_id`` (or identity-based MERGE)
    is required before concept↔concept edge writes go live.
    """
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
        # skip_metadata_keys: this is a blind upsert with no existing-node read
        # at all -- unlike SubstrateGraphMaterializer.apply_record()'s merge
        # branch, there's no merge_node() step to even reason about here. A
        # freshly-mapped concept's own metadata never carries prediction_error/
        # contributing_turn_ids, so without this, falkor_codec.py's
        # encode_node_properties() emits prediction_error=None (its documented
        # "absent" default) and set_assignments() never filters None out of the
        # Cypher SET clause -- meaning an unprotected write here would NULL
        # (not just freeze) those fields on any node this induced concept's own
        # node_id happens to collide with, worse than the freeze bug this same
        # protection was built for in materializer.py/dynamics.py. Confirmed
        # live 2026-07-30 as a real, currently-unpatched instance of that same
        # bug class on this service's default backend
        # (CONCEPT_PROFILE_GRAPH_BACKEND=falkor) -- see
        # falkor_codec.EXTERNALLY_OWNED_METADATA_KEYS's docstring for the
        # original incident.
        store.upsert_node(identity_key=node.node_id, node=node, skip_metadata_keys=EXTERNALLY_OWNED_METADATA_KEYS)
    for edge in edges:
        # A subject-anchor edge's source is a golden node seeded by a
        # *different* process (Hub's startup seed, orion/substrate/seed.py)
        # -- Falkor's own upsert_edge Cypher is `MERGE (source:SubstrateNode
        # {node_id: $source_id})`, which creates a bare, label-less stub if
        # that node doesn't durably exist yet (e.g. this service starts
        # before Hub ever has). Not a correctness bug: decode_concept_node()
        # requires node_kind="concept" to even construct a ConceptNodeV1, so
        # a bare stub never surfaces as a node anywhere -- it just self-heals
        # into a real node the moment Hub's idempotent seed step runs (every
        # startup), at which point the edge that was already waiting resolves
        # normally. Worth knowing when debugging "an edge exists but its
        # source node doesn't render yet" during a fresh bring-up.
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
