from __future__ import annotations

from datetime import datetime, timezone

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    NodeRefV1,
    SubstrateActivationV1,
    SubstrateEdgeV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
    SubstrateTemporalWindowV1,
)
from orion.substrate.falkor_codec import (
    decode_concept_node,
    decode_edge,
    encode_edge_properties,
    encode_node_properties,
    node_label_for_kind,
)


def _provenance() -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="test",
        source_channel="test:falkor_codec",
        producer="test_falkor_codec",
    )


def _temporal() -> SubstrateTemporalWindowV1:
    return SubstrateTemporalWindowV1(observed_at=datetime(2026, 7, 16, tzinfo=timezone.utc))


def _concept() -> ConceptNodeV1:
    return ConceptNodeV1(
        node_id="concept-alpha",
        label="Alpha",
        definition="A test concept",
        anchor_scope="orion",
        promotion_state="canonical",
        temporal=_temporal(),
        provenance=_provenance(),
        signals=SubstrateSignalBundleV1(
            confidence=0.8,
            salience=0.7,
            activation=SubstrateActivationV1(
                activation=0.6,
                recency_score=0.5,
                decay_floor=0.1,
            ),
        ),
        metadata={"quarantine": "not durable SoR"},
    )


def test_node_label_for_kind_uses_closed_mapping():
    assert node_label_for_kind("concept") == "Concept"
    assert node_label_for_kind("state_snapshot") == "StateSnapshot"
    assert node_label_for_kind("unknown_kind") == "Generic"


def test_encode_concept_node_properties_are_native_scalars_without_payload_json():
    props = encode_node_properties(_concept(), identity_key="concept:alpha")

    assert "payload_json" not in props
    assert "metadata" not in props
    assert props == {
        "node_id": "concept-alpha",
        "node_kind": "concept",
        "identity_key": "concept:alpha",
        "anchor_scope": "orion",
        "subject_ref": None,
        "promotion_state": "canonical",
        "risk_tier": "low",
        "confidence": 0.8,
        "salience": 0.7,
        "activation": 0.6,
        "recency_score": 0.5,
        "decay_half_life_seconds": None,
        "decay_floor": 0.1,
        "observed_at": "2026-07-16T00:00:00+00:00",
        "valid_from": None,
        "valid_to": None,
        "provenance_authority": "local_inferred",
        "provenance_source_kind": "test",
        "provenance_source_channel": "test:falkor_codec",
        "provenance_producer": "test_falkor_codec",
        "provenance_model_name": None,
        "provenance_correlation_id": None,
        "provenance_trace_id": None,
        "provenance_tier_rank": None,
        "label": "Alpha",
        "definition": "A test concept",
    }


def test_decode_concept_node_reconstructs_minimal_typed_model():
    row = encode_node_properties(_concept(), identity_key="concept:alpha")

    node = decode_concept_node(row)

    assert node is not None
    assert node.node_id == "concept-alpha"
    assert node.node_kind == "concept"
    assert node.label == "Alpha"
    assert node.definition == "A test concept"
    assert node.promotion_state == "canonical"
    assert node.signals.confidence == 0.8
    assert node.signals.salience == 0.7
    assert node.signals.activation.activation == 0.6
    assert node.metadata == {}


def test_encode_edge_properties_are_native_scalars_without_payload_json():
    edge = SubstrateEdgeV1(
        edge_id="edge-alpha-beta",
        source=NodeRefV1(node_id="concept-alpha", node_kind="concept"),
        target=NodeRefV1(node_id="concept-beta", node_kind="concept"),
        predicate="contradicts",
        temporal=_temporal(),
        confidence=0.9,
        salience=0.4,
        provenance=_provenance(),
        metadata={"ignored": "not durable SoR"},
    )

    props = encode_edge_properties(edge, identity_key="edge:alpha-beta")

    assert "payload_json" not in props
    assert "metadata" not in props
    assert props["edge_id"] == "edge-alpha-beta"
    assert props["source_id"] == "concept-alpha"
    assert props["target_id"] == "concept-beta"
    assert props["predicate"] == "contradicts"
    assert props["identity_key"] == "edge:alpha-beta"
    assert props["confidence"] == 0.9
    assert props["salience"] == 0.4


def test_decode_edge_reconstructs_typed_edge():
    edge = SubstrateEdgeV1(
        edge_id="edge-alpha-beta",
        source=NodeRefV1(node_id="concept-alpha", node_kind="concept"),
        target=NodeRefV1(node_id="concept-beta", node_kind="concept"),
        predicate="contradicts",
        temporal=_temporal(),
        confidence=0.9,
        salience=0.4,
        provenance=_provenance(),
    )
    row = encode_edge_properties(edge, identity_key="edge:alpha-beta")

    decoded = decode_edge(row)

    assert decoded is not None
    assert decoded.edge_id == "edge-alpha-beta"
    assert decoded.source.node_id == "concept-alpha"
    assert decoded.target.node_id == "concept-beta"
    assert decoded.predicate == "contradicts"
    assert decoded.confidence == 0.9
    assert decoded.salience == 0.4
