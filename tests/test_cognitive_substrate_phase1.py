from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    ContradictionNodeV1,
    DriveNodeV1,
    EntityNodeV1,
    EventNodeV1,
    EvidenceNodeV1,
    GoalNodeV1,
    HypothesisNodeV1,
    NodeRefV1,
    OntologyBranchNodeV1,
    StateSnapshotNodeV1,
    SubstrateEdgeV1,
    SubstrateGraphRecordV1,
    SubstrateNodeV1,
    SubstrateProvenanceV1,
    TensionNodeV1,
    SubstrateTemporalWindowV1,
)
from orion.core.schemas.endogenous_runtime import EndogenousRuntimeExecutionRecordV1
from orion.core.schemas.reasoning_summary import ReasoningSummaryV1
from orion.schemas.registry import _REGISTRY


NOW = datetime.now(timezone.utc)


def _provenance() -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="unit",
        source_channel="orion:test",
        producer="pytest",
        evidence_refs=["ev:1"],
    )


def _temporal() -> SubstrateTemporalWindowV1:
    return SubstrateTemporalWindowV1(observed_at=NOW)


def test_canonical_node_variants_validate() -> None:
    nodes = [
        EntityNodeV1(anchor_scope="orion", temporal=_temporal(), provenance=_provenance(), label="orion", entity_type="system"),
        ConceptNodeV1(anchor_scope="orion", temporal=_temporal(), provenance=_provenance(), label="substrate"),
        EventNodeV1(anchor_scope="session", temporal=_temporal(), provenance=_provenance(), event_type="observation", summary="event"),
        EvidenceNodeV1(anchor_scope="session", temporal=_temporal(), provenance=_provenance(), evidence_type="trace", content_ref="trace:1"),
        ContradictionNodeV1(
            anchor_scope="orion",
            temporal=_temporal(),
            provenance=_provenance(),
            summary="conflict",
            involved_node_ids=["n1", "n2"],
        ),
        TensionNodeV1(anchor_scope="orion", temporal=_temporal(), provenance=_provenance(), tension_kind="goal_conflict"),
        DriveNodeV1(anchor_scope="orion", temporal=_temporal(), provenance=_provenance(), drive_kind="coherence"),
        GoalNodeV1(anchor_scope="orion", temporal=_temporal(), provenance=_provenance(), goal_text="stabilize"),
        StateSnapshotNodeV1(anchor_scope="orion", temporal=_temporal(), provenance=_provenance(), dimensions={"focus": 0.7}),
        HypothesisNodeV1(anchor_scope="orion", temporal=_temporal(), provenance=_provenance(), hypothesis_text="maybe x causes y"),
        OntologyBranchNodeV1(anchor_scope="orion", temporal=_temporal(), provenance=_provenance(), branch_key="autonomy", branch_label="Autonomy"),
    ]
    assert len(nodes) == 11


def test_discriminated_union_rejects_invalid_node_kind() -> None:
    adapter = TypeAdapter(SubstrateNodeV1)
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "node_kind": "not_a_kind",
                "anchor_scope": "orion",
                "temporal": {"observed_at": NOW.isoformat()},
                "provenance": {
                    "authority": "local_inferred",
                    "source_kind": "unit",
                    "source_channel": "orion:test",
                    "producer": "pytest",
                },
            }
        )


def test_edge_schema_validation_and_predicates() -> None:
    edge = SubstrateEdgeV1(
        source=NodeRefV1(node_id="sub-node-1", node_kind="concept"),
        target=NodeRefV1(node_id="sub-node-2", node_kind="evidence"),
        predicate="supports",
        temporal=_temporal(),
        provenance=_provenance(),
        confidence=0.8,
        salience=0.6,
    )
    assert edge.predicate == "supports"

    with pytest.raises(ValidationError):
        SubstrateEdgeV1(
            source=NodeRefV1(node_id="x", node_kind="concept"),
            target=NodeRefV1(node_id="sub-node-2", node_kind="evidence"),
            predicate="invalid_predicate",
            temporal=_temporal(),
            provenance=_provenance(),
        )


def test_temporal_and_activation_support_fields_validate() -> None:
    node = ConceptNodeV1(
        anchor_scope="orion",
        temporal=SubstrateTemporalWindowV1(observed_at=NOW, valid_from=NOW),
        provenance=_provenance(),
        label="activation-schema-support",
        signals={"confidence": 0.9, "salience": 0.8, "activation": {"activation": 0.4, "recency_score": 0.7, "decay_half_life_seconds": 60}},
        promotion_state="provisional",
        risk_tier="medium",
        subject_ref="project:orion_sapienform",
    )
    assert node.signals.activation.decay_half_life_seconds == 60


def test_registry_and_exports_include_substrate_contracts() -> None:
    assert "SubstrateEdgeV1" in _REGISTRY
    assert "EntityNodeV1" in _REGISTRY
    assert "SubstrateGraphRecordV1" in _REGISTRY


def test_substrate_graph_record_and_non_destructive_existing_imports() -> None:
    graph = SubstrateGraphRecordV1(
        anchor_scope="orion",
        subject_ref="project:orion_sapienform",
        nodes=[
            ConceptNodeV1(anchor_scope="orion", temporal=_temporal(), provenance=_provenance(), label="coherence"),
            EvidenceNodeV1(anchor_scope="orion", temporal=_temporal(), provenance=_provenance(), evidence_type="trace", content_ref="trace:2"),
        ],
        edges=[
            SubstrateEdgeV1(
                source=NodeRefV1(node_id="sub-node-1", node_kind="concept"),
                target=NodeRefV1(node_id="sub-node-2", node_kind="evidence"),
                predicate="supports",
                temporal=_temporal(),
                provenance=_provenance(),
            )
        ],
    )
    assert len(graph.nodes) == 2

    # Existing families remain importable/usable (non-destructive introduction).
    assert ReasoningSummaryV1.model_json_schema()["title"] == "ReasoningSummaryV1"
    assert EndogenousRuntimeExecutionRecordV1.model_json_schema()["title"] == "EndogenousRuntimeExecutionRecordV1"
