"""Round-trip test for topic_id through the FalkorDB codec.

Regression for a defect found by code review on the Concept Atlas
readability branch (2026-08-19), the same shape as the pre-existing
perception_staleness/prediction_error_evidence_event_ids traps documented in
falkor_codec.py: topic-foundry's HDBSCAN cluster assignment
(orion/substrate/adapters/topic_foundry.py writes it into
ConceptNodeV1.metadata as "topic_id") was never covered by
encode_node_properties()'s metadata-to-Cypher-property translation, so it
only ever "worked" because FalkorSubstrateStore's in-process cache retained
the original Python object -- the very next cache rehydrate (any snapshot()
call past the write-generation check, or a process restart) durably dropped
it, silently defeating Concept Atlas's community-coloring feature on the
real default backend (SUBSTRATE_STORE_BACKEND=falkor) within ~30 seconds of
any real write.
"""

from __future__ import annotations

from orion.substrate.falkor_codec import (
    TOPIC_FOUNDRY_METADATA_KEYS,
    _topic_foundry_metadata_from_row,
    _topic_foundry_properties_from_metadata,
    decode_concept_node,
    encode_node_properties,
)


def test_key_is_in_the_topic_foundry_allowlist() -> None:
    assert "topic_id" in TOPIC_FOUNDRY_METADATA_KEYS


def test_encoder_promotes_the_value_to_a_native_property() -> None:
    props = _topic_foundry_properties_from_metadata({"topic_id": "cluster-7"})
    assert props["topic_id"] == "cluster-7"


def test_encoder_defaults_to_none_when_absent() -> None:
    props = _topic_foundry_properties_from_metadata({})
    assert props["topic_id"] is None


def test_decoder_omits_the_key_entirely_when_row_value_is_none() -> None:
    # Matches _dynamics_metadata_from_row's "omit rather than store None"
    # convention for optional fields -- a concept node from any other
    # producer should not carry a spurious topic_id: None entry.
    assert _topic_foundry_metadata_from_row({"topic_id": None}) == {}
    assert _topic_foundry_metadata_from_row({}) == {}


def test_decoder_recovers_the_value() -> None:
    assert _topic_foundry_metadata_from_row({"topic_id": "cluster-7"}) == {"topic_id": "cluster-7"}


def _concept_node(**overrides):
    from datetime import datetime, timezone

    from orion.core.schemas.cognitive_substrate import (
        ConceptNodeV1,
        SubstrateProvenanceV1,
        SubstrateTemporalWindowV1,
    )

    defaults = dict(
        node_id="concept-topic-test",
        label="Topic Test",
        anchor_scope="world",
        temporal=SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc)),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="test",
            source_channel="test:falkor_codec_topic_id",
            producer="test_falkor_codec_topic_id",
        ),
    )
    defaults.update(overrides)
    return ConceptNodeV1(**defaults)


def test_full_encode_decode_round_trip_preserves_topic_id() -> None:
    """The end-to-end shape of the real bug: encode a node with topic_id set,
    simulate the Cypher row FalkorDB would hand back (encode_node_properties'
    own output IS that row shape, minus DB-generated fields), decode it, and
    confirm topic_id survives -- this is exactly the path that silently
    dropped it before this fix."""
    node = _concept_node(metadata={"topic_id": "cluster-42"})
    row = encode_node_properties(node, identity_key="concept:topic-test")
    assert row["topic_id"] == "cluster-42"

    row["node_kind"] = "concept"
    decoded = decode_concept_node(row)
    assert decoded is not None
    assert decoded.metadata.get("topic_id") == "cluster-42"


def test_full_encode_decode_round_trip_untagged_node_has_no_topic_id() -> None:
    node = _concept_node(metadata={})
    row = encode_node_properties(node, identity_key="concept:topic-test")
    assert row["topic_id"] is None

    row["node_kind"] = "concept"
    decoded = decode_concept_node(row)
    assert decoded is not None
    assert "topic_id" not in decoded.metadata
