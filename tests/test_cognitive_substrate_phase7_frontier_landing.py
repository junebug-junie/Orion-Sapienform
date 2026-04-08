from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    HypothesisNodeV1,
    SubstrateProvenanceV1,
    SubstrateTemporalWindowV1,
)
from orion.core.schemas.frontier_expansion import FrontierGraphDeltaBundleV1, FrontierSourceProvenanceV1
from orion.core.schemas.frontier_landing import FrontierDeltaLandingDecisionV1, FrontierLandingRequestV1
from orion.substrate import FrontierLandingEvaluator, InMemorySubstrateGraphStore


def _now() -> SubstrateTemporalWindowV1:
    return SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc))


def _prov() -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="test",
        source_channel="orion:test",
        producer="pytest",
    )


def _bundle(*, zone: str, confidence: float, node=None, contradiction: bool = False, evidence_gap: bool = False) -> FrontierGraphDeltaBundleV1:
    return FrontierGraphDeltaBundleV1(
        request_id="req-1",
        response_id="res-1",
        target_zone=zone,
        task_type="ontology_expand" if zone == "world_ontology" else "concept_expand",
        suggested_landing_posture="fast_track_proposal" if zone == "world_ontology" else "strict_proposal_only" if zone == "self_relationship_graph" else "moderate_proposal",
        candidate_nodes=[node] if node is not None else [],
        contradiction_candidates=["contradiction?"] if contradiction else [],
        evidence_gap_candidates=["missing evidence?"] if evidence_gap else [],
        source_provenance=FrontierSourceProvenanceV1(provider="stub", model="stub-v1"),
        confidence=confidence,
        notes=["test"],
    )


def test_landing_contract_validation_enforces_consistency() -> None:
    with pytest.raises(ValidationError):
        FrontierDeltaLandingDecisionV1(
            delta_item_id="d1",
            decision="blocked_due_to_risk",
            target_zone="concept_graph",
            confidence=0.5,
            risk_tier="high",
        )

    with pytest.raises(ValidationError):
        FrontierDeltaLandingDecisionV1(
            delta_item_id="d2",
            decision="proposed_only",
            target_zone="concept_graph",
            blocked_reason="risk",
            confidence=0.5,
            risk_tier="low",
        )

    req = FrontierLandingRequestV1(bundle_id="b-1", request_id="r-1", target_zone="world_ontology")
    assert req.bundle_id == "b-1"


def test_zone_policy_world_more_permissive_than_autonomy_and_self() -> None:
    store = InMemorySubstrateGraphStore()
    evaluator = FrontierLandingEvaluator(store=store)

    world_node = ConceptNodeV1(anchor_scope="orion", label="World", temporal=_now(), provenance=_prov(), risk_tier="low")
    world_bundle = _bundle(zone="world_ontology", confidence=0.86, node=world_node)
    world_result = evaluator.evaluate_and_land(
        request=FrontierLandingRequestV1(bundle_id=world_bundle.bundle_id, request_id="req-1", target_zone="world_ontology"),
        bundle=world_bundle,
    )
    assert world_result.landing_result.outcome_counts.get("materialize_now", 0) >= 1

    autonomy_node = ConceptNodeV1(
        anchor_scope="orion",
        label="Autonomy",
        temporal=_now(),
        provenance=_prov(),
        risk_tier="medium",
        metadata={"frontier_identity_protected": True},
    )
    autonomy_bundle = _bundle(zone="autonomy_graph", confidence=0.9, node=autonomy_node)
    autonomy_result = evaluator.evaluate_and_land(
        request=FrontierLandingRequestV1(bundle_id=autonomy_bundle.bundle_id, request_id="req-1", target_zone="autonomy_graph"),
        bundle=autonomy_bundle,
    )
    assert autonomy_result.landing_result.outcome_counts.get("hitl_required", 0) >= 1

    self_node = ConceptNodeV1(anchor_scope="orion", subject_ref="entity:orion", label="Self", temporal=_now(), provenance=_prov(), risk_tier="high")
    self_bundle = _bundle(zone="self_relationship_graph", confidence=0.95, node=self_node)
    self_result = evaluator.evaluate_and_land(
        request=FrontierLandingRequestV1(bundle_id=self_bundle.bundle_id, request_id="req-1", target_zone="self_relationship_graph"),
        bundle=self_bundle,
    )
    assert self_result.landing_result.outcome_counts.get("hitl_required", 0) >= 1


def test_allowed_materialization_uses_existing_materializer_and_preserves_source_metadata() -> None:
    store = InMemorySubstrateGraphStore()
    evaluator = FrontierLandingEvaluator(store=store)
    node = ConceptNodeV1(anchor_scope="orion", label="Materialize", temporal=_now(), provenance=_prov(), risk_tier="low")
    bundle = _bundle(zone="world_ontology", confidence=0.9, node=node)
    result = evaluator.evaluate_and_land(
        request=FrontierLandingRequestV1(bundle_id=bundle.bundle_id, request_id="req-1", target_zone="world_ontology", landing_context={"subject_ref": "entity:orion"}),
        bundle=bundle,
    )

    assert result.materialization_result is not None
    snapshot = store.snapshot()
    landed_nodes = [n for n in snapshot.nodes.values() if n.metadata.get("frontier_provider") == "stub"]
    assert landed_nodes
    assert all(n.metadata.get("frontier_source_authority") == "frontier_model" for n in landed_nodes)


def test_contradiction_and_evidence_gap_land_as_bounded_hypotheses() -> None:
    store = InMemorySubstrateGraphStore()
    evaluator = FrontierLandingEvaluator(store=store)
    bundle = _bundle(zone="concept_graph", confidence=0.84, contradiction=True, evidence_gap=True)
    result = evaluator.evaluate_and_land(
        request=FrontierLandingRequestV1(bundle_id=bundle.bundle_id, request_id="req-1", target_zone="concept_graph"),
        bundle=bundle,
    )

    assert result.landing_result.materialization_summary["materialized_nodes"] >= 2
    hypothesis_nodes = [n for n in store.snapshot().nodes.values() if n.node_kind == "hypothesis"]
    assert hypothesis_nodes
    assert all(n.metadata.get("frontier_hypothesis_marker") is True for n in hypothesis_nodes)


def test_non_destructive_blocked_or_hitl_items_do_not_materialize() -> None:
    store = InMemorySubstrateGraphStore()
    evaluator = FrontierLandingEvaluator(store=store)
    self_node = ConceptNodeV1(anchor_scope="orion", subject_ref="entity:orion", label="Blocked", temporal=_now(), provenance=_prov(), risk_tier="high")
    bundle = _bundle(zone="self_relationship_graph", confidence=0.95, node=self_node)
    result = evaluator.evaluate_and_land(
        request=FrontierLandingRequestV1(bundle_id=bundle.bundle_id, request_id="req-1", target_zone="self_relationship_graph"),
        bundle=bundle,
    )

    assert result.materialization_result is None
    assert not store.snapshot().nodes
