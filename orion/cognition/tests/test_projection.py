from __future__ import annotations

from datetime import datetime, timezone

from orion.cognition.projection import project_unified_beliefs_for_mind
from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
    SubstrateTemporalWindowV1,
)
from orion.substrate.relational.beliefs import AnchorBeliefSliceV1, UnifiedRelationalBeliefSetV1


def _concept(label: str, salience: float, confidence: float) -> ConceptNodeV1:
    return ConceptNodeV1(
        node_kind="concept",
        anchor_scope="orion",
        label=label,
        definition=f"Definition for {label}",
        temporal=SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc)),
        signals=SubstrateSignalBundleV1(salience=salience, confidence=confidence),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="test",
            source_channel="unit",
            producer="test_projection",
            tier_rank=3,
            evidence_refs=["ev1"],
        ),
    )


def test_project_unified_beliefs_for_mind_compacts_items_and_lineage() -> None:
    beliefs = UnifiedRelationalBeliefSetV1(
        anchors={
            "orion": AnchorBeliefSliceV1(
                anchor="orion",
                concepts=[
                    _concept("low", salience=0.1, confidence=0.9),
                    _concept("high", salience=0.9, confidence=0.8),
                ],
                tier_outcomes=["concept_induced_accepted:2"],
            )
        },
        cold_anchors=["orion"],
        degraded_producers=["producer_x"],
        lineage=["test_projection:concept_induced"],
    )

    projection = project_unified_beliefs_for_mind(beliefs, max_items_per_bucket=2, max_total_items=4)

    assert projection is not None
    assert projection.schema_version == "cognitive.projection.v1"
    assert projection.source == "cognitive_unification_layer"
    assert projection.item_count == 2
    assert projection.cold_anchors == ["orion"]
    assert projection.degraded_producers == ["producer_x"]
    assert "cold_path_materialized" in projection.notes
    assert "degraded_producers_present" in projection.notes
    assert projection.anchors["orion"].tier_outcomes == ["concept_induced_accepted:2"]
    assert projection.anchors["orion"].items[0].label == "high"
    assert projection.anchors["orion"].items[0].producer == "test_projection"
    assert projection.anchors["orion"].items[0].evidence_refs == ["ev1"]


def test_project_unified_beliefs_for_mind_empty_beliefs_is_still_signal() -> None:
    beliefs = UnifiedRelationalBeliefSetV1(
        anchors={},
        lineage=["skipped:introspect_spark_or_unified_beliefs_disabled"],
    )

    projection = project_unified_beliefs_for_mind(beliefs)

    assert projection is not None
    assert projection.item_count == 0
    assert "no_anchors" in projection.notes
    assert "no_active_projection_items" in projection.notes
    assert projection.lineage == ["skipped:introspect_spark_or_unified_beliefs_disabled"]
