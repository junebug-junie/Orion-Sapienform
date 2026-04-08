from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    GoalNodeV1,
    HypothesisNodeV1,
    NodeRefV1,
    SubstrateEdgeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateTemporalWindowV1,
)
from orion.core.schemas.frontier_expansion import (
    FrontierDeltaItemV1,
    FrontierExpansionRequestV1,
    FrontierExpansionResponseV1,
)
from orion.substrate import (
    FrontierContextPackBuilder,
    FrontierExpansionService,
    InMemorySubstrateGraphStore,
    SubstrateGraphMaterializer,
)


def _prov() -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="test",
        source_channel="orion:test",
        producer="pytest",
    )


def _temporal(minutes_ago: int = 0) -> SubstrateTemporalWindowV1:
    return SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago))


def _edge(source: str, source_kind: str, target: str, target_kind: str, predicate: str) -> SubstrateEdgeV1:
    return SubstrateEdgeV1(
        source=NodeRefV1(node_id=source, node_kind=source_kind),
        target=NodeRefV1(node_id=target, node_kind=target_kind),
        predicate=predicate,
        temporal=_temporal(0),
        confidence=1.0,
        salience=0.3,
        provenance=_prov(),
    )


class _StubProvider:
    def __init__(self, response: FrontierExpansionResponseV1) -> None:
        self._response = response

    def expand(self, *, request: FrontierExpansionRequestV1, context):
        return self._response


def _build_store() -> InMemorySubstrateGraphStore:
    materializer = SubstrateGraphMaterializer(store=InMemorySubstrateGraphStore())
    concept = ConceptNodeV1(
        node_id="concept-frontier",
        anchor_scope="orion",
        subject_ref="entity:orion",
        label="Frontier",
        temporal=_temporal(10),
        provenance=_prov(),
        signals={"salience": 0.8},
    )
    goal = GoalNodeV1(
        node_id="goal-frontier",
        anchor_scope="orion",
        subject_ref="entity:orion",
        goal_text="expand ontology",
        temporal=_temporal(5),
        provenance=_prov(),
        metadata={"dynamic_pressure": 0.7},
    )
    hypothesis = HypothesisNodeV1(
        node_id="hyp-frontier",
        anchor_scope="orion",
        subject_ref="entity:orion",
        hypothesis_text="possible relation",
        temporal=_temporal(1),
        provenance=_prov(),
    )
    record = SubstrateGraphRecordV1(
        anchor_scope="orion",
        subject_ref="entity:orion",
        nodes=[concept, goal, hypothesis],
        edges=[_edge("concept-frontier", "concept", "goal-frontier", "goal", "supports")],
    )
    materializer.apply_record(record)
    return materializer.store


def test_contract_validation_enforces_task_zone_and_item_shapes() -> None:
    with pytest.raises(ValidationError):
        FrontierExpansionRequestV1(task_type="invalid", anchor_scope="orion", target_zone="world_ontology", topic="x", expansion_goal="y")

    with pytest.raises(ValidationError):
        FrontierDeltaItemV1(item_kind="edge_add")

    node = ConceptNodeV1(anchor_scope="orion", label="n", temporal=_temporal(), provenance=_prov())
    item = FrontierDeltaItemV1(item_kind="node_add", candidate_node=node)
    response = FrontierExpansionResponseV1(
        request_id="req-1",
        provider="stub",
        model="stub-1",
        task_type="ontology_expand",
        target_zone="world_ontology",
        delta_items=[item],
    )
    assert response.delta_items


def test_context_packaging_is_bounded_and_zone_safe() -> None:
    store = _build_store()
    state = store.snapshot()
    request = FrontierExpansionRequestV1(
        request_id="req-context",
        task_type="self_or_relationship_hypothesis",
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_zone="self_relationship_graph",
        topic="self interpretation",
        expansion_goal="propose hypotheses only",
        graph_region={"focal_node_ids": ["concept-frontier", "goal-frontier", "hyp-frontier"]},
    )
    context = FrontierContextPackBuilder(max_nodes=2, max_edges=1).build(state=state, request=request)
    assert len(context.focal_node_ids) <= 2
    assert len(context.focal_edge_ids) <= 1
    assert "concept-frontier" not in context.focal_node_ids


def test_mapping_preserves_zone_provenance_and_delta_types() -> None:
    store = _build_store()
    request = FrontierExpansionRequestV1(
        request_id="req-map",
        task_type="relation_discovery",
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_zone="concept_graph",
        topic="discover relation",
        expansion_goal="propose relation and contradiction",
    )
    new_node = ConceptNodeV1(anchor_scope="orion", subject_ref="entity:orion", label="New Concept", temporal=_temporal(), provenance=_prov())
    new_edge = _edge("concept-frontier", "concept", new_node.node_id, "concept", "associated_with")
    response = FrontierExpansionResponseV1(
        request_id="req-map",
        provider="stub",
        model="stub-v1",
        task_type="relation_discovery",
        target_zone="concept_graph",
        delta_items=[
            FrontierDeltaItemV1(item_kind="node_add", candidate_node=new_node, confidence=0.8),
            FrontierDeltaItemV1(item_kind="edge_add", candidate_edge=new_edge, confidence=0.75),
            FrontierDeltaItemV1(item_kind="contradiction_flag", contradiction_summary="possible inconsistency", confidence=0.6),
            FrontierDeltaItemV1(item_kind="evidence_gap", evidence_gap_question="missing source for linkage", confidence=0.7),
        ],
        confidence=0.77,
    )
    service = FrontierExpansionService(store=store, provider=_StubProvider(response))
    result = service.expand(request=request)

    assert result.delta_bundle.target_zone == "concept_graph"
    assert result.delta_bundle.candidate_nodes
    assert result.delta_bundle.candidate_edges
    assert "possible inconsistency" in result.delta_bundle.contradiction_candidates
    assert "missing source for linkage" in result.delta_bundle.evidence_gap_candidates


def test_safety_blocks_canonical_writes_in_strict_zones() -> None:
    store = _build_store()
    request = FrontierExpansionRequestV1(
        request_id="req-safe",
        task_type="self_or_relationship_hypothesis",
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_zone="self_relationship_graph",
        topic="self-model",
        expansion_goal="proposal only",
    )
    bad_node = ConceptNodeV1(
        anchor_scope="orion",
        subject_ref="entity:orion",
        label="Forbidden canonical",
        temporal=_temporal(),
        provenance=_prov(),
        promotion_state="canonical",
    )
    response = FrontierExpansionResponseV1(
        request_id="req-safe",
        provider="stub",
        model="stub-v1",
        task_type="self_or_relationship_hypothesis",
        target_zone="self_relationship_graph",
        delta_items=[FrontierDeltaItemV1(item_kind="node_add", candidate_node=bad_node)],
    )
    service = FrontierExpansionService(store=store, provider=_StubProvider(response))
    with pytest.raises(ValueError):
        service.expand(request=request)


def test_non_destructive_integration_without_live_provider_dependency() -> None:
    store = _build_store()
    request = FrontierExpansionRequestV1(
        request_id="req-int",
        task_type="ontology_expand",
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_zone="world_ontology",
        topic="ontology gap",
        expansion_goal="propose new branch",
    )
    node = ConceptNodeV1(anchor_scope="orion", subject_ref="entity:orion", label="Branch", temporal=_temporal(), provenance=_prov())
    response = FrontierExpansionResponseV1(
        request_id="req-int",
        provider="stub",
        model="stub-v1",
        task_type="ontology_expand",
        target_zone="world_ontology",
        delta_items=[FrontierDeltaItemV1(item_kind="taxonomy_branch", candidate_node=node)],
    )
    service = FrontierExpansionService(store=store, provider=_StubProvider(response))
    result = service.expand(request=request)

    assert result.delta_bundle.suggested_landing_posture == "fast_track_proposal"
    assert result.context_pack.request_id == "req-int"
    assert store.snapshot().nodes
