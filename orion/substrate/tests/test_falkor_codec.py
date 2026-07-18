from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    DriveNodeV1,
    EvidenceNodeV1,
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
    decode_evidence_node,
    decode_node,
    encode_edge_properties,
    encode_node_properties,
    node_label_for_kind,
)


def _provenance(**kwargs) -> SubstrateProvenanceV1:
    base = dict(
        authority="local_inferred",
        source_kind="test",
        source_channel="test:falkor_codec",
        producer="test_falkor_codec",
    )
    base.update(kwargs)
    return SubstrateProvenanceV1(**base)


def _temporal() -> SubstrateTemporalWindowV1:
    return SubstrateTemporalWindowV1(observed_at=datetime(2026, 7, 16, tzinfo=timezone.utc))


def _concept() -> ConceptNodeV1:
    return ConceptNodeV1(
        node_id="concept-alpha",
        label="Alpha",
        definition="A test concept",
        taxonomy_path=["root", "branch"],
        anchor_scope="orion",
        promotion_state="canonical",
        temporal=_temporal(),
        provenance=_provenance(evidence_refs=["ev:1", "ev:2"]),
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


def _evidence() -> EvidenceNodeV1:
    return EvidenceNodeV1(
        node_id="evidence-alpha",
        evidence_type="chat_turn",
        content_ref="ev-content-ref-1",
        anchor_scope="orion",
        promotion_state="canonical",
        temporal=_temporal(),
        provenance=_provenance(evidence_refs=["ev:1", "ev:2"]),
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
        "evidence_refs_json": '["ev:1", "ev:2"]',
        "label": "Alpha",
        "definition": "A test concept",
        "taxonomy_path_json": '["root", "branch"]',
        "dynamic_pressure": 0.0,
        "dynamic_pressure_reason": None,
        "dormant": False,
        "dormancy_updated_at": None,
        "prediction_error": None,
        "contributing_turn_ids_json": "[]",
    }


def test_encode_preserves_evidence_refs_and_taxonomy_path_round_trip():
    row = encode_node_properties(_concept(), identity_key="concept:alpha")
    node = decode_concept_node(row)

    assert node is not None
    assert node.provenance.evidence_refs == ["ev:1", "ev:2"]
    assert node.taxonomy_path == ["root", "branch"]


def test_encode_node_properties_rejects_non_concept():
    drive = DriveNodeV1(
        node_id="drive-curiosity",
        drive_kind="curiosity",
        anchor_scope="orion",
        temporal=_temporal(),
        provenance=_provenance(),
    )
    with pytest.raises(ValueError, match="concept and evidence nodes only"):
        encode_node_properties(drive, identity_key="drive:curiosity")


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
    # dynamic_pressure/dormant always come back as real scalars (0.0/False
    # defaults); dynamics.py/pressure.py already tolerate these via
    # `.get(key) or default`, so this is not an empty-metadata regression --
    # see test_concept_with_no_dynamics_metadata_encodes_and_decodes_with_sane_defaults
    # for the full defaults contract.
    assert node.metadata == {"dynamic_pressure": 0.0, "dormant": False}


def test_encode_evidence_node_properties_are_native_scalars():
    props = encode_node_properties(_evidence(), identity_key="evidence:alpha")

    assert "payload_json" not in props
    assert "metadata" not in props
    assert "label" not in props
    assert "definition" not in props
    assert "taxonomy_path_json" not in props
    assert props == {
        "node_id": "evidence-alpha",
        "node_kind": "evidence",
        "identity_key": "evidence:alpha",
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
        "evidence_refs_json": '["ev:1", "ev:2"]',
        "evidence_type": "chat_turn",
        "content_ref": "ev-content-ref-1",
    }


def test_decode_evidence_node_reconstructs_minimal_typed_model():
    row = encode_node_properties(_evidence(), identity_key="evidence:alpha")

    node = decode_evidence_node(row)

    assert node is not None
    assert node.node_id == "evidence-alpha"
    assert node.node_kind == "evidence"
    assert node.evidence_type == "chat_turn"
    assert node.content_ref == "ev-content-ref-1"
    assert node.provenance.evidence_refs == ["ev:1", "ev:2"]


def test_decode_node_dispatches_to_concept_or_evidence():
    concept_row = encode_node_properties(_concept(), identity_key="concept:alpha")
    evidence_row = encode_node_properties(_evidence(), identity_key="evidence:alpha")

    concept_result = decode_node(concept_row)
    evidence_result = decode_node(evidence_row)

    assert isinstance(concept_result, ConceptNodeV1)
    assert isinstance(evidence_result, EvidenceNodeV1)

    unrelated_row = dict(concept_row)
    unrelated_row["node_kind"] = "drive"
    assert decode_node(unrelated_row) is None


def test_encode_edge_properties_are_native_scalars_without_payload_json():
    edge = SubstrateEdgeV1(
        edge_id="edge-alpha-beta",
        source=NodeRefV1(node_id="concept-alpha", node_kind="concept"),
        target=NodeRefV1(node_id="concept-beta", node_kind="concept"),
        predicate="contradicts",
        temporal=_temporal(),
        confidence=0.9,
        salience=0.4,
        provenance=_provenance(evidence_refs=["edge-ev:1"]),
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
    assert props["evidence_refs_json"] == '["edge-ev:1"]'


def test_decode_rejects_corrupt_evidence_refs_json():
    row = encode_node_properties(_concept(), identity_key="concept:alpha")
    row["evidence_refs_json"] = "{not-json"

    with pytest.raises(ValueError, match="evidence_refs_json"):
        decode_concept_node(row)


def _concept_with_dynamics_metadata() -> ConceptNodeV1:
    base = _concept()
    return base.model_copy(
        update={
            "metadata": {
                "dynamic_pressure": 0.42,
                "dynamic_pressure_reason": "prediction_error_seed",
                "dormant": True,
                "dormancy_updated_at": "2026-07-17T00:00:00+00:00",
                "prediction_error": 0.8,
                "contributing_turn_ids": ["turn-1", "turn-2"],
            }
        }
    )


def test_encode_promotes_dynamics_metadata_keys_to_native_scalars():
    props = encode_node_properties(_concept_with_dynamics_metadata(), identity_key="concept:alpha")

    assert "metadata" not in props
    assert "payload_json" not in props
    assert props["dynamic_pressure"] == 0.42
    assert props["dynamic_pressure_reason"] == "prediction_error_seed"
    assert props["dormant"] is True
    assert props["dormancy_updated_at"] == "2026-07-17T00:00:00+00:00"
    assert props["prediction_error"] == 0.8
    assert props["contributing_turn_ids_json"] == '["turn-1", "turn-2"]'


def test_decode_reconstructs_dynamics_metadata_from_row():
    row = encode_node_properties(_concept_with_dynamics_metadata(), identity_key="concept:alpha")

    node = decode_concept_node(row)

    assert node is not None
    assert node.metadata.get("dynamic_pressure") == 0.42
    assert node.metadata.get("dynamic_pressure_reason") == "prediction_error_seed"
    assert node.metadata.get("dormant") is True
    assert node.metadata.get("dormancy_updated_at") == "2026-07-17T00:00:00+00:00"
    assert node.metadata.get("prediction_error") == 0.8
    assert node.metadata.get("contributing_turn_ids") == ["turn-1", "turn-2"]


def test_dynamics_metadata_round_trips_through_encode_decode():
    original = _concept_with_dynamics_metadata()

    decoded = decode_concept_node(encode_node_properties(original, identity_key="concept:alpha"))

    assert decoded is not None
    for key in (
        "dynamic_pressure",
        "dynamic_pressure_reason",
        "dormant",
        "dormancy_updated_at",
        "prediction_error",
        "contributing_turn_ids",
    ):
        assert decoded.metadata.get(key) == original.metadata.get(key)


def test_contributing_turn_ids_empty_list_is_omitted_on_decode():
    """metadata['contributing_turn_ids'] absent entirely (the common case --
    most concept nodes are never touched by _write_prediction_error_node)
    must encode to an empty JSON list and decode back to the key being
    absent from metadata, not an empty list sitting in the dict -- mirrors
    the existing omit-when-absent convention for prediction_error/
    dynamic_pressure_reason/dormancy_updated_at.
    """
    row = encode_node_properties(_concept(), identity_key="concept:alpha")
    assert row["contributing_turn_ids_json"] == "[]"

    node = decode_concept_node(row)
    assert node is not None
    assert "contributing_turn_ids" not in node.metadata


def test_contributing_turn_ids_missing_json_key_decodes_without_raising():
    """A row missing contributing_turn_ids_json entirely (e.g. a node written
    before this patch shipped) must decode fail-open, not raise -- same
    tolerance every other decode field in this module gives a missing key.
    """
    row = encode_node_properties(_concept(), identity_key="concept:alpha")
    del row["contributing_turn_ids_json"]

    node = decode_concept_node(row)
    assert node is not None
    assert "contributing_turn_ids" not in node.metadata


def test_contributing_turn_ids_malformed_json_decodes_fail_open_not_raise():
    """Unlike evidence_refs_json (load-bearing identity data, intentionally
    raises on corruption -- see test_decode_rejects_corrupt_evidence_refs_json),
    contributing_turn_ids is best-effort attribution metadata: a corrupted
    value must not abort decoding the whole node.
    """
    row = encode_node_properties(_concept(), identity_key="concept:alpha")
    row["contributing_turn_ids_json"] = "{not-json"

    node = decode_concept_node(row)
    assert node is not None
    assert "contributing_turn_ids" not in node.metadata


def test_contributing_turn_ids_wrong_json_shape_decodes_fail_open_not_raise():
    """A JSON value that parses but isn't a list (e.g. an object) must also
    be tolerated, not raised -- same fail-open contract as the malformed-JSON
    case above.
    """
    row = encode_node_properties(_concept(), identity_key="concept:alpha")
    row["contributing_turn_ids_json"] = '{"not": "a list"}'

    node = decode_concept_node(row)
    assert node is not None
    assert "contributing_turn_ids" not in node.metadata


def test_concept_with_no_dynamics_metadata_encodes_and_decodes_with_sane_defaults():
    node = _concept()  # metadata has no dynamics keys at all -- the common case
    row = encode_node_properties(node, identity_key="concept:alpha")

    assert row["dynamic_pressure"] == 0.0
    assert row["dynamic_pressure_reason"] is None
    assert row["dormant"] is False
    assert row["dormancy_updated_at"] is None
    assert row["prediction_error"] is None

    decoded = decode_concept_node(row)
    assert decoded is not None

    # Mirror the exact accessor patterns dynamics.py/pressure.py/attention_broadcast.py
    # use in production -- no KeyError, no None-vs-0.0 confusion.
    assert float(decoded.metadata.get("dynamic_pressure") or 0.0) == 0.0
    assert bool(decoded.metadata.get("dormant", False)) is False
    assert decoded.metadata.get("dynamic_pressure_reason") is None
    assert float(decoded.metadata.get("prediction_error") or 0.0) == 0.0
    assert decoded.metadata.get("dormancy_updated_at") is None


def test_decode_edge_reconstructs_typed_edge():
    edge = SubstrateEdgeV1(
        edge_id="edge-alpha-beta",
        source=NodeRefV1(node_id="concept-alpha", node_kind="concept"),
        target=NodeRefV1(node_id="concept-beta", node_kind="concept"),
        predicate="contradicts",
        temporal=_temporal(),
        confidence=0.9,
        salience=0.4,
        provenance=_provenance(evidence_refs=["edge-ev:1"]),
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
    assert decoded.provenance.evidence_refs == ["edge-ev:1"]
