from __future__ import annotations

from datetime import datetime, timezone

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    ContradictionNodeV1,
    GoalNodeV1,
    HypothesisNodeV1,
    NodeRefV1,
    SubstrateEdgeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateTemporalWindowV1,
)
from orion.core.schemas.frontier_landing import FrontierLandingResultV1
from orion.core.schemas.substrate_consolidation import GraphConsolidationRequestV1, GraphReviewCycleRecordV1
from orion.graph_cognition.evidence import EvidenceSpanV1, SignalEvidenceBundleV1
from orion.graph_cognition.interpreters import (
    CoherenceAssessmentV1,
    ConceptDriftSignalV1,
    ContradictionCandidateSetV1,
    GoalPressureStateV1,
    GraphCognitionReportV1,
    IdentityConflictSignalV1,
    SocialContinuityAssessmentV1,
)
from orion.substrate.consolidation import GraphConsolidationEvaluator
from orion.substrate.frontier_landing import FrontierLandingExecutionResultV1
from orion.substrate.materializer import SubstrateGraphMaterializer
from orion.substrate.store import InMemorySubstrateGraphStore


def _temporal() -> SubstrateTemporalWindowV1:
    return SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc))


def _prov() -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="test",
        source_channel="orion:test",
        producer="pytest",
    )


def _edge(source: str, target: str, predicate: str = "supports") -> SubstrateEdgeV1:
    return SubstrateEdgeV1(
        source=NodeRefV1(node_id=source, node_kind="concept" if source.startswith("c") else "goal"),
        target=NodeRefV1(node_id=target, node_kind="concept" if target.startswith("c") else "goal"),
        predicate=predicate,
        temporal=_temporal(),
        confidence=1.0,
        salience=0.5,
        provenance=_prov(),
    )


def _cognition(pressure: float = 0.8) -> GraphCognitionReportV1:
    evidence = SignalEvidenceBundleV1(spans=(EvidenceSpanV1(node_ids=("n",), edge_ids=(), reason="r", weight=1.0),), truncated=False, degraded=False, notes=())
    return GraphCognitionReportV1(
        coherence=CoherenceAssessmentV1(score=0.5, confidence=0.7, evidence=evidence, notes=()),
        identity_conflict=IdentityConflictSignalV1(conflict_score=0.2, active=False, confidence=0.7, evidence=evidence),
        goal_pressure=GoalPressureStateV1(pressure_score=pressure, stalled_goal_count=1, competing_goal_density=0.2, confidence=0.8, evidence=evidence),
        social_continuity=SocialContinuityAssessmentV1(continuity_score=0.6, confidence=0.7, degraded=False, evidence=evidence),
        concept_drift=ConceptDriftSignalV1(drift_score=0.3, active=False, confidence=0.7, evidence=evidence),
        contradiction_candidates=ContradictionCandidateSetV1(candidates=(), confidence=0.6, evidence=evidence),
    )


def _landing_exec(materialized_items: int = 0) -> FrontierLandingExecutionResultV1:
    landing = FrontierLandingResultV1(
        bundle_id="b",
        request_id="r",
        target_zone="concept_graph",
        decisions=[],
        outcome_counts={},
        hitl_summary={"required": 0},
        materialization_summary={"materialized_items": materialized_items},
        blocked_summary={},
        confidence=0.5,
    )
    return FrontierLandingExecutionResultV1(landing_result=landing, materialization_result=None)


def _build_store(stable: bool = True, with_contradiction: bool = False, with_gap: bool = False, isolated_frontier: bool = False) -> InMemorySubstrateGraphStore:
    store = InMemorySubstrateGraphStore()
    mat = SubstrateGraphMaterializer(store=store)
    concept = ConceptNodeV1(node_id="concept-1", anchor_scope="orion", subject_ref="entity:orion", label="C1", temporal=_temporal(), provenance=_prov(), signals={"activation": {"activation": 0.7}, "salience": 0.7}, metadata={"dynamic_pressure": 0.6})
    goal = GoalNodeV1(node_id="goal-1", anchor_scope="orion", subject_ref="entity:orion", goal_text="G1", temporal=_temporal(), provenance=_prov(), metadata={"dynamic_pressure": 0.7})
    nodes = [concept, goal]
    edges = [] if isolated_frontier else [_edge("concept-1", "goal-1")]
    if with_contradiction:
        nodes.append(
            ContradictionNodeV1(node_id="contra-1", anchor_scope="orion", subject_ref="entity:orion", summary="conflict", involved_node_ids=["concept-1", "goal-1"], temporal=_temporal(), provenance=_prov(), metadata={"severity": 0.8})
        )
    if with_gap:
        nodes.append(
            HypothesisNodeV1(node_id="hyp-gap-1", anchor_scope="orion", subject_ref="entity:orion", hypothesis_text="gap", temporal=_temporal(), provenance=_prov(), metadata={"frontier_hypothesis_marker": True, "frontier_source_authority": "frontier_model"}, signals={"activation": {"activation": 0.05 if not stable else 0.5}})
        )
    record = SubstrateGraphRecordV1(anchor_scope="orion", subject_ref="entity:orion", nodes=nodes, edges=edges)
    mat.apply_record(record)
    return store


def test_comparison_detects_stability_vs_fragmentation() -> None:
    store = _build_store(stable=True)
    evaluator = GraphConsolidationEvaluator(store=store)
    prior = GraphReviewCycleRecordV1(request_id="old", focal_node_refs=["concept-1", "goal-1"], focal_edge_refs=[], mean_activation=0.4, mean_pressure=0.4, contradiction_count=0, evidence_gap_count=0, isolated_frontier_count=0, outcome_counts={})
    req = GraphConsolidationRequestV1(anchor_scope="orion", subject_ref="entity:orion", focal_node_refs=["concept-1", "goal-1"], reason_for_review="stable check", target_zone="concept_graph")
    out = evaluator.consolidate(request=req, prior_cycle=prior)
    assert out.result.state_delta_digest is not None
    assert out.result.state_delta_digest.node_persistence_ratio > 0.5


def test_outcomes_cover_reinforce_requeue_damp_retire_and_priority() -> None:
    # contradiction + pressure -> maintain_priority/requeue
    store1 = _build_store(with_contradiction=True)
    eval1 = GraphConsolidationEvaluator(store=store1)
    req1 = GraphConsolidationRequestV1(anchor_scope="orion", subject_ref="entity:orion", reason_for_review="contradiction", target_zone="concept_graph")
    res1 = eval1.consolidate(request=req1)
    outcomes1 = {d.outcome for d in res1.result.decisions}
    assert "maintain_priority" in outcomes1 or "requeue_review" in outcomes1

    # isolated low-activation frontier gap -> damp/retire
    store2 = _build_store(stable=False, with_gap=True, isolated_frontier=True)
    eval2 = GraphConsolidationEvaluator(store=store2)
    req2 = GraphConsolidationRequestV1(anchor_scope="orion", subject_ref="entity:orion", reason_for_review="gap", target_zone="concept_graph")
    res2 = eval2.consolidate(request=req2)
    outcomes2 = {d.outcome for d in res2.result.decisions}
    assert "damp" in outcomes2 or "retire" in outcomes2 or "requeue_review" in outcomes2


def test_strict_zone_is_operator_only_and_no_hidden_escalation() -> None:
    store = _build_store(with_gap=True)
    evaluator = GraphConsolidationEvaluator(store=store)
    req = GraphConsolidationRequestV1(anchor_scope="orion", subject_ref="entity:orion", reason_for_review="strict", target_zone="self_relationship_graph")
    res = evaluator.consolidate(request=req)
    assert {d.outcome for d in res.result.decisions} == {"operator_only"}


def test_integration_reads_cognition_and_landing_and_remains_bounded() -> None:
    store = _build_store(with_gap=True)
    evaluator = GraphConsolidationEvaluator(store=store, max_region_nodes=2, max_region_edges=1)
    req = GraphConsolidationRequestV1(anchor_scope="orion", subject_ref="entity:orion", reason_for_review="integration", target_zone="autonomy_graph")
    res = evaluator.consolidate(request=req, cognition_report=_cognition(pressure=0.9), landing_result=_landing_exec(materialized_items=0))

    assert len(res.cycle_record.focal_node_refs) <= 2
    assert len(res.cycle_record.focal_edge_refs) <= 1
    assert any(d.outcome in {"keep_provisional", "maintain_priority", "requeue_review"} for d in res.result.decisions)
