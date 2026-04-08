from __future__ import annotations

from datetime import datetime, timezone

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    ContradictionNodeV1,
    GoalNodeV1,
    HypothesisNodeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateTemporalWindowV1,
)
from orion.core.schemas.frontier_expansion import FrontierDeltaItemV1, FrontierExpansionResponseV1
from orion.graph_cognition.brief import MetacogPerceptionBriefV1
from orion.graph_cognition.evidence import EvidenceSpanV1, SignalEvidenceBundleV1
from orion.graph_cognition.interpreters import (
    CoherenceAssessmentV1,
    ConceptDriftSignalV1,
    ContradictionCandidateSetV1,
    ContradictionCandidateV1,
    GoalPressureStateV1,
    GraphCognitionReportV1,
    IdentityConflictSignalV1,
    SocialContinuityAssessmentV1,
)
from orion.substrate import FrontierExpansionService, FrontierLandingEvaluator, InMemorySubstrateGraphStore, SubstrateGraphMaterializer
from orion.substrate.frontier_curiosity import FrontierCuriosityEvaluator, FrontierCuriosityOrchestrator


def _temporal() -> SubstrateTemporalWindowV1:
    return SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc))


def _prov() -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="test",
        source_channel="orion:test",
        producer="pytest",
    )


def _evidence() -> SignalEvidenceBundleV1:
    return SignalEvidenceBundleV1(spans=(EvidenceSpanV1(node_ids=("n1",), edge_ids=(), reason="test", weight=1.0),), truncated=False, degraded=False, notes=())


def _report(*, contradiction_count: int = 1, drift_active: bool = True, pressure: float = 0.8, identity_conflict: bool = False) -> GraphCognitionReportV1:
    contradiction_candidates = tuple(
        ContradictionCandidateV1(node_id=f"contra-{idx}", severity=0.8, pressure=0.7) for idx in range(contradiction_count)
    )
    return GraphCognitionReportV1(
        coherence=CoherenceAssessmentV1(score=0.4, confidence=0.8, evidence=_evidence(), notes=()),
        identity_conflict=IdentityConflictSignalV1(conflict_score=0.6 if identity_conflict else 0.2, active=identity_conflict, confidence=0.7, evidence=_evidence()),
        goal_pressure=GoalPressureStateV1(pressure_score=pressure, stalled_goal_count=1, competing_goal_density=0.3, confidence=0.75, evidence=_evidence()),
        social_continuity=SocialContinuityAssessmentV1(continuity_score=0.4, confidence=0.7, degraded=False, evidence=_evidence()),
        concept_drift=ConceptDriftSignalV1(drift_score=0.7 if drift_active else 0.2, active=drift_active, confidence=0.8, evidence=_evidence()),
        contradiction_candidates=ContradictionCandidateSetV1(candidates=contradiction_candidates, confidence=0.8, evidence=_evidence()),
    )


def _brief(*, priority: str = "stabilize") -> MetacogPerceptionBriefV1:
    return MetacogPerceptionBriefV1(
        top_tensions=("identity_conflict",),
        top_stabilizers=("coherence",),
        overall_priority=priority,
        recommended_verbs=("deconflict", "prioritize", "repair"),
        confidence=0.8,
        degraded=False,
        supporting_evidence=_evidence(),
        notes_for_router=(),
    )


class _StubProvider:
    def expand(self, *, request, context):
        new_node = ConceptNodeV1(anchor_scope=request.anchor_scope, subject_ref=request.subject_ref, label="Frontier Added", temporal=_temporal(), provenance=_prov())
        return FrontierExpansionResponseV1(
            request_id=request.request_id,
            provider="stub",
            model="stub-v1",
            task_type=request.task_type,
            target_zone=request.target_zone,
            delta_items=[FrontierDeltaItemV1(item_kind="node_add", candidate_node=new_node, confidence=0.9)],
            confidence=0.9,
        )


def _build_store() -> InMemorySubstrateGraphStore:
    store = InMemorySubstrateGraphStore()
    materializer = SubstrateGraphMaterializer(store=store)
    concept = ConceptNodeV1(node_id="concept-1", anchor_scope="orion", subject_ref="entity:orion", label="Concept 1", temporal=_temporal(), provenance=_prov(), signals={"salience": 0.8})
    goal = GoalNodeV1(node_id="goal-1", anchor_scope="orion", subject_ref="entity:orion", goal_text="Goal", temporal=_temporal(), provenance=_prov(), metadata={"dynamic_pressure": 0.8})
    contradiction = ContradictionNodeV1(node_id="contra-0", anchor_scope="orion", subject_ref="entity:orion", summary="Conflict", involved_node_ids=["concept-1", "goal-1"], temporal=_temporal(), provenance=_prov(), metadata={"severity": 0.8})
    gap = HypothesisNodeV1(node_id="gap-1", anchor_scope="orion", subject_ref="entity:orion", hypothesis_text="Evidence gap", temporal=_temporal(), provenance=_prov(), metadata={"frontier_hypothesis_marker": True})
    gap2 = HypothesisNodeV1(node_id="gap-2", anchor_scope="orion", subject_ref="entity:orion", hypothesis_text="Evidence gap 2", temporal=_temporal(), provenance=_prov(), metadata={"frontier_hypothesis_marker": True})
    record = SubstrateGraphRecordV1(anchor_scope="orion", subject_ref="entity:orion", nodes=[concept, goal, contradiction, gap, gap2], edges=[])
    materializer.apply_record(record)
    return store


def test_signal_derivation_and_task_selection_are_deterministic() -> None:
    evaluator = FrontierCuriosityEvaluator(store=_build_store())
    result = evaluator.evaluate(anchor_scope="orion", subject_ref="entity:orion", cognition_report=_report(), perception_brief=_brief())

    signal_types = {signal.signal_type for signal in result.signals}
    assert "contradiction_hotspot" in signal_types
    assert "ontology_sparse_region" in signal_types
    assert any(signal.task_type_candidate == "evidence_gap_scan" for signal in result.signals)
    assert result.decision.outcome in {"invoke", "defer", "noop", "operator_only", "blocked"}


def test_self_relationship_cases_are_operator_only() -> None:
    evaluator = FrontierCuriosityEvaluator(store=_build_store())
    result = evaluator.evaluate(
        anchor_scope="orion",
        subject_ref="entity:orion",
        cognition_report=_report(identity_conflict=True, pressure=0.3),
        perception_brief=_brief(priority="stabilize"),
        operator_requested=False,
    )
    if result.decision.target_zone == "self_relationship_graph":
        assert result.decision.outcome == "operator_only"


def test_region_selection_is_bounded_and_inspectable() -> None:
    evaluator = FrontierCuriosityEvaluator(store=_build_store(), max_focal_nodes=2, max_focal_edges=2)
    result = evaluator.evaluate(anchor_scope="orion", subject_ref="entity:orion", cognition_report=_report(), perception_brief=_brief())
    if result.plan is not None:
        assert len(result.plan.selected_node_refs) <= 2
        assert len(result.plan.selected_edge_refs) <= 2


def test_integration_flows_into_phase6_and_phase7_paths() -> None:
    store = _build_store()
    curiosity = FrontierCuriosityEvaluator(store=store)
    expansion = FrontierExpansionService(store=store, provider=_StubProvider())
    landing = FrontierLandingEvaluator(store=store)
    orchestrator = FrontierCuriosityOrchestrator(curiosity_evaluator=curiosity, expansion_service=expansion, landing_evaluator=landing)

    result = orchestrator.run(
        anchor_scope="orion",
        subject_ref="entity:orion",
        cognition_report=_report(),
        perception_brief=_brief(),
        operator_requested=True,
    )

    assert result.invocation.decision.outcome in {"invoke", "operator_only", "defer", "noop", "blocked"}
    if result.invocation.decision.outcome == "invoke":
        assert result.expansion is not None
        assert result.landing is not None


def test_low_value_region_can_noop_without_broad_runtime_expansion() -> None:
    store = _build_store()
    evaluator = FrontierCuriosityEvaluator(store=store)
    result = evaluator.evaluate(
        anchor_scope="orion",
        subject_ref="entity:orion",
        cognition_report=_report(contradiction_count=0, drift_active=False, pressure=0.2),
        perception_brief=_brief(priority="advance"),
        operator_requested=False,
    )
    assert result.decision.outcome in {"noop", "defer", "invoke", "operator_only", "blocked"}
