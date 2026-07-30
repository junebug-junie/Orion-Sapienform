"""Regression guard (2026-07-30): SubstrateGraphMaterializer.apply_record()'s
existing-node lookup must catch a raw node_id collision even when the
incoming node's own identity_key fails to resolve (no identity-index entry
found for that key) -- not just when identity_key is None outright.

Found during review of the sibling merge-branch metadata-clobber fix (see
falkor_codec.EXTERNALLY_OWNED_METADATA_KEYS's docstring for the original
incident). Before this fix, a node_id collision with an existing node
indexed under a *different* identity_key took the "created" branch: a full
wholesale overwrite via node.model_copy(...), no merge_node() call, and no
skip_metadata_keys protection at all -- strictly worse than a metadata-only
clobber. This applies even though a reducer's own direct write (e.g.
_write_prediction_error_node) DOES register a real, non-None identity_key
(services/orion-substrate-runtime/app/worker.py:
f"substrate_prediction_error|{node_id}") -- that string is never what
SubstrateIdentityResolver.canonical_node_key() independently computes for
an incoming concept-induction node sharing the same raw node_id, so the
identity-index lookup still misses and the raw-node_id fallback is what
has to catch it.
"""

from __future__ import annotations

from datetime import datetime, timezone

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
    SubstrateTemporalWindowV1,
)
from orion.substrate import InMemorySubstrateGraphStore, SubstrateGraphMaterializer


def _reducer_owned_node(*, node_id: str, prediction_error: float) -> ConceptNodeV1:
    """Shaped like a real reducer's own direct write (_write_prediction_error_node):
    fixed label/anchor_scope/subject_ref. Registered under the reducer's own
    identity_key convention (f"substrate_prediction_error|{node_id}"), not the
    canonical identity_key SubstrateIdentityResolver.canonical_node_key()
    would independently compute from anchor_scope/subject_ref/label -- these
    two identity schemes never agree, which is exactly why a colliding
    incoming node's identity-index lookup misses even though a real,
    non-None identity_key is registered for the existing node."""
    return ConceptNodeV1(
        node_id=node_id,
        anchor_scope="orion",
        subject_ref="entity:orion",
        label=f"substrate:{node_id}",
        temporal=SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc)),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="substrate_prediction_error",
            source_channel="substrate.runtime",
            producer="substrate_runtime_worker",
        ),
        signals=SubstrateSignalBundleV1(confidence=1.0, salience=1.0),
        metadata={"prediction_error": prediction_error},
    )


def _colliding_incoming_record_node(*, node_id: str) -> ConceptNodeV1:
    """A differently-shaped incoming node that happens to share node_id with
    the reducer-owned node above -- its own canonical identity_key (derived
    by SubstrateIdentityResolver.canonical_node_key() from this node's own
    anchor_scope/subject_ref/label) won't match anything indexed, because it
    is a structurally different string from the reducer's own
    "substrate_prediction_error|..." identity_key convention -- the two
    schemes are independent, not because nothing was registered."""
    from orion.core.schemas.cognitive_substrate import SubstrateGraphRecordV1

    node = ConceptNodeV1(
        node_id=node_id,
        anchor_scope="orion",
        subject_ref="project:orion",
        label="some induced concept label",
        temporal=SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc)),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="concept_induction",
            source_channel="orion:chat:history:log",
            producer="concept_induction",
        ),
        signals=SubstrateSignalBundleV1(confidence=0.8, salience=0.6),
        metadata={},
    )
    return SubstrateGraphRecordV1(graph_id="test-graph", anchor_scope="orion", nodes=[node], edges=[])


def test_node_id_collision_with_unregistered_identity_takes_merge_not_created_branch() -> None:
    store = InMemorySubstrateGraphStore()
    # Simulate a reducer's real direct write, including its real identity_key
    # convention -- this is a genuine, non-None identity_key (matching
    # _write_prediction_error_node's actual call), just one the resolver
    # below never independently reproduces.
    store.upsert_node(
        identity_key="substrate_prediction_error|node:substrate.bus_synaptic",
        node=_reducer_owned_node(node_id="node:substrate.bus_synaptic", prediction_error=0.73),
    )

    materializer = SubstrateGraphMaterializer(store=store)
    record = _colliding_incoming_record_node(node_id="node:substrate.bus_synaptic")
    result = materializer.apply_record(record)

    # Must merge into the existing node, not silently create/overwrite it.
    assert result.nodes_created == 0
    assert result.nodes_merged == 1
    assert result.node_decisions[0].merged is True
    assert result.node_decisions[0].reason == "node_id_match"

    # merge_node() preserves the existing reducer-owned metadata (existing
    # wins for keys incoming doesn't provide) -- confirms this went through
    # the real merge path, not a wholesale replace that would have dropped it.
    merged = store.get_node_by_id("node:substrate.bus_synaptic")
    assert merged is not None
    assert merged.metadata.get("prediction_error") == 0.73


def test_no_collision_still_creates_normally() -> None:
    """Sanity check: the widened fallback must not turn every fresh node
    into a false merge -- a genuinely new node_id still takes the created
    branch."""
    store = InMemorySubstrateGraphStore()
    materializer = SubstrateGraphMaterializer(store=store)
    record = _colliding_incoming_record_node(node_id="sub-concept-brand-new")
    result = materializer.apply_record(record)

    assert result.nodes_created == 1
    assert result.nodes_merged == 0
    assert result.node_decisions[0].reason == "created"
