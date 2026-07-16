"""Regression tests for embedding-aware concept identity resolution.

Covers the Phase 3 fix to SubstrateIdentityResolver.canonical_node_key:
paraphrased concept labels ("surface encodings" vs "surface-level
representations") with similar embeddings must resolve to the same canonical
node identity, while the pre-existing exact-string fallback (for nodes with no
embedding) must remain unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    SubstrateProvenanceV1,
    SubstrateTemporalWindowV1,
)
from orion.substrate.reconcile import SubstrateIdentityResolver
from orion.substrate.store import InMemorySubstrateGraphStore


def _concept_node(
    *,
    node_id: str,
    label: str,
    anchor_scope: str = "orion",
    subject_ref: str = "project:atlas",
    embedding: list[float] | None = None,
) -> ConceptNodeV1:
    metadata: dict = {}
    if embedding is not None:
        metadata["concept_embedding"] = embedding
    return ConceptNodeV1(
        node_id=node_id,
        anchor_scope=anchor_scope,
        subject_ref=subject_ref,
        label=label,
        temporal=SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc)),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="test",
            source_channel="test",
            producer="test_reconcile",
        ),
        metadata=metadata,
    )


def test_similar_embeddings_resolve_to_same_identity_across_paraphrased_labels() -> None:
    store = InMemorySubstrateGraphStore()
    resolver = SubstrateIdentityResolver(store=store)

    # Cosine similarity of these two vectors is ~0.9986, comfortably >= the 0.8
    # threshold reused from ConceptClusterer.
    existing = _concept_node(
        node_id="node-a",
        label="surface encodings",
        embedding=[1.0, 1.0, 0.0],
    )
    incoming = _concept_node(
        node_id="node-b",
        label="surface-level representations",
        embedding=[1.0, 0.9, 0.0],
    )

    existing_key = resolver.canonical_node_key(existing)
    assert existing_key is not None
    store.upsert_node(identity_key=existing_key, node=existing)

    incoming_key = resolver.canonical_node_key(incoming)

    assert incoming_key == existing_key, (
        "paraphrased concept with a similar embedding must resolve to the "
        "existing node's identity key so materialization merges them"
    )


def test_dissimilar_embeddings_do_not_merge() -> None:
    store = InMemorySubstrateGraphStore()
    resolver = SubstrateIdentityResolver(store=store)

    existing = _concept_node(
        node_id="node-a",
        label="surface encodings",
        embedding=[1.0, 0.0, 0.0],
    )
    incoming = _concept_node(
        node_id="node-b",
        label="totally unrelated topic",
        embedding=[0.0, 1.0, 0.0],
    )

    existing_key = resolver.canonical_node_key(existing)
    assert existing_key is not None
    store.upsert_node(identity_key=existing_key, node=existing)

    incoming_key = resolver.canonical_node_key(incoming)

    assert incoming_key != existing_key


def test_no_embedding_falls_back_to_exact_label_match_unchanged() -> None:
    """Regression guard: two nodes with different labels and NO embeddings must
    still resolve to different identities -- the pre-existing exact-string
    behavior must not be broken by the embedding-match addition."""
    store = InMemorySubstrateGraphStore()
    resolver = SubstrateIdentityResolver(store=store)

    existing = _concept_node(node_id="node-a", label="surface encodings")
    incoming = _concept_node(node_id="node-b", label="surface-level representations")

    existing_key = resolver.canonical_node_key(existing)
    assert existing_key is not None
    store.upsert_node(identity_key=existing_key, node=existing)

    incoming_key = resolver.canonical_node_key(incoming)

    assert incoming_key != existing_key
    assert incoming_key == f"concept|orion|project:atlas|label:surface-level representations"


def test_no_embedding_same_label_still_merges() -> None:
    """Same-label nodes with no embeddings keep matching via the legacy path."""
    store = InMemorySubstrateGraphStore()
    resolver = SubstrateIdentityResolver(store=store)

    existing = _concept_node(node_id="node-a", label="coherence")
    incoming = _concept_node(node_id="node-b", label="Coherence")

    existing_key = resolver.canonical_node_key(existing)
    assert existing_key is not None
    store.upsert_node(identity_key=existing_key, node=existing)

    incoming_key = resolver.canonical_node_key(incoming)

    assert incoming_key == existing_key


def test_resolver_without_store_behaves_exactly_like_legacy_resolver() -> None:
    """Default construction (no store) -- what every existing caller does today
    (e.g. SubstrateGraphMaterializer's ``identity_resolver or
    SubstrateIdentityResolver()``) -- must be unaffected even when nodes DO carry
    embeddings, since there is no store to compare against."""
    resolver = SubstrateIdentityResolver()

    node_a = _concept_node(node_id="node-a", label="surface encodings", embedding=[1.0, 1.0, 0.0])
    node_b = _concept_node(node_id="node-b", label="surface-level representations", embedding=[1.0, 0.9, 0.0])

    key_a = resolver.canonical_node_key(node_a)
    key_b = resolver.canonical_node_key(node_b)

    assert key_a != key_b
    assert key_a == "concept|orion|project:atlas|label:surface encodings"
    assert key_b == "concept|orion|project:atlas|label:surface-level representations"


def test_malformed_embedding_degrades_to_label_match_without_raising() -> None:
    store = InMemorySubstrateGraphStore()
    resolver = SubstrateIdentityResolver(store=store)

    existing = _concept_node(node_id="node-a", label="surface encodings")
    store.upsert_node(
        identity_key="concept|orion|project:atlas|label:surface encodings",
        node=existing,
    )

    incoming = _concept_node(node_id="node-b", label="surface-level representations")
    # Malformed embedding: not a list of numbers.
    incoming = incoming.model_copy(update={"metadata": {"concept_embedding": "not-a-vector"}})

    # Must not raise, and must fall back to the legacy label-based key.
    incoming_key = resolver.canonical_node_key(incoming)

    assert incoming_key == "concept|orion|project:atlas|label:surface-level representations"


def test_store_read_error_degrades_to_label_match_without_raising() -> None:
    """If the store's read_concept_region raises (e.g. a graphdb backend that's
    temporarily unreachable), the resolver must degrade to the legacy label match
    rather than propagate the error."""

    class _ExplodingStore(InMemorySubstrateGraphStore):
        def read_concept_region(self, *, limit_nodes: int = 32, limit_edges: int = 64):  # noqa: D401
            raise RuntimeError("store unreachable")

    store = _ExplodingStore()
    resolver = SubstrateIdentityResolver(store=store)

    incoming = _concept_node(
        node_id="node-b",
        label="surface-level representations",
        embedding=[1.0, 0.9, 0.0],
    )

    incoming_key = resolver.canonical_node_key(incoming)

    assert incoming_key == "concept|orion|project:atlas|label:surface-level representations"


def test_one_sided_embedding_presence_falls_back_to_label_match() -> None:
    """An existing node with no embedding cannot be embedding-matched even when the
    incoming node has one -- there's nothing to compare cosine similarity against."""
    store = InMemorySubstrateGraphStore()
    resolver = SubstrateIdentityResolver(store=store)

    existing = _concept_node(node_id="node-a", label="surface encodings")  # no embedding
    existing_key = resolver.canonical_node_key(existing)
    assert existing_key is not None
    store.upsert_node(identity_key=existing_key, node=existing)

    incoming = _concept_node(
        node_id="node-b",
        label="surface-level representations",
        embedding=[1.0, 0.9, 0.0],
    )
    incoming_key = resolver.canonical_node_key(incoming)

    assert incoming_key != existing_key
    assert incoming_key == "concept|orion|project:atlas|label:surface-level representations"


def test_embedding_match_only_considers_same_scope_and_subject() -> None:
    store = InMemorySubstrateGraphStore()
    resolver = SubstrateIdentityResolver(store=store)

    existing = _concept_node(
        node_id="node-a",
        label="surface encodings",
        anchor_scope="orion",
        subject_ref="project:atlas",
        embedding=[1.0, 1.0, 0.0],
    )
    existing_key = resolver.canonical_node_key(existing)
    assert existing_key is not None
    store.upsert_node(identity_key=existing_key, node=existing)

    # Same embedding, but a different subject_ref -- must NOT merge across subjects.
    incoming = _concept_node(
        node_id="node-b",
        label="surface-level representations",
        anchor_scope="orion",
        subject_ref="project:other",
        embedding=[1.0, 0.9, 0.0],
    )
    incoming_key = resolver.canonical_node_key(incoming)

    assert incoming_key != existing_key
