from __future__ import annotations

from datetime import datetime, timezone

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    ContradictionNodeV1,
    NodeRefV1,
    SubstrateEdgeV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
    SubstrateTemporalWindowV1,
)
from orion.core.schemas.substrate_consolidation import GraphConsolidationRequestV1
from orion.graph_cognition.brief import build_metacog_perception_brief
from orion.graph_cognition.interpreters import (
    CoherenceAssessmentV1,
    ConceptDriftSignalV1,
    ContradictionCandidateSetV1,
    GoalPressureStateV1,
    GraphCognitionReportV1,
    IdentityConflictSignalV1,
    SocialContinuityAssessmentV1,
)
from orion.graph_cognition.views import build_graph_views_from_store
from orion.graph_cognition.evidence import EvidenceSpanV1, SignalEvidenceBundleV1
from orion.substrate.consolidation import GraphConsolidationEvaluator
from orion.substrate.frontier_curiosity import FrontierCuriosityEvaluator
from orion.substrate.query_planning import SubstrateQueryPlanner, SubstrateSemanticReadCoordinator
from orion.substrate.store import InMemorySubstrateGraphStore


class CountingSemanticStore(InMemorySubstrateGraphStore):
    def __init__(self) -> None:
        super().__init__()
        self.calls: dict[str, int] = {}

    def _count(self, kind: str) -> None:
        self.calls[kind] = self.calls.get(kind, 0) + 1

    def query_hotspot_region(self, *, min_salience: float = 0.6, limit_nodes: int = 32, limit_edges: int = 64):
        self._count("hotspot_region")
        return super().query_hotspot_region(min_salience=min_salience, limit_nodes=limit_nodes, limit_edges=limit_edges)

    def query_concept_region(self, *, limit_nodes: int = 32, limit_edges: int = 64):
        self._count("concept_region")
        return super().query_concept_region(limit_nodes=limit_nodes, limit_edges=limit_edges)

    def query_contradiction_region(self, *, limit_nodes: int = 32, limit_edges: int = 64):
        self._count("contradiction_region")
        return super().query_contradiction_region(limit_nodes=limit_nodes, limit_edges=limit_edges)

    def query_focal_slice(self, *, node_ids: list[str], max_edges: int = 64):
        self._count("focal_slice")
        return super().query_focal_slice(node_ids=node_ids, max_edges=max_edges)

    def query_provenance_neighborhood(self, *, evidence_ref: str, limit_nodes: int = 32, limit_edges: int = 64):
        self._count("provenance_neighborhood")
        return super().query_provenance_neighborhood(evidence_ref=evidence_ref, limit_nodes=limit_nodes, limit_edges=limit_edges)


def _seed_store(store: InMemorySubstrateGraphStore) -> None:
    now = datetime.now(timezone.utc)
    temporal = SubstrateTemporalWindowV1(observed_at=now)
    provenance = SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="test",
        source_channel="pytest",
        producer="unit",
        evidence_refs=["e:1"],
    )
    concept = ConceptNodeV1(
        node_id="node-concept",
        anchor_scope="orion",
        subject_ref="orion",
        temporal=temporal,
        provenance=provenance,
        signals=SubstrateSignalBundleV1(confidence=0.8, salience=0.75),
        label="Core Concept",
        definition="concept",
        metadata={"concept_id": "c1", "dynamic_pressure": 0.7, "frontier_hypothesis_marker": True},
    )
    contradiction = ContradictionNodeV1(
        node_id="node-contradiction",
        anchor_scope="orion",
        subject_ref="orion",
        temporal=temporal,
        provenance=provenance,
        signals=SubstrateSignalBundleV1(confidence=0.9, salience=0.9),
        summary="conflict",
        involved_node_ids=["node-concept", "node-concept"],
        metadata={"dynamic_pressure": 0.85},
    )
    edge = SubstrateEdgeV1(
        edge_id="edge-rel",
        source=NodeRefV1(node_id=concept.node_id, node_kind="concept"),
        target=NodeRefV1(node_id=contradiction.node_id, node_kind="contradiction"),
        predicate="contradicts",
        temporal=temporal,
        confidence=0.9,
        salience=0.75,
        provenance=provenance,
    )
    store.upsert_node(identity_key="concept|orion|orion|c1", node=concept)
    store.upsert_node(identity_key="contradiction|orion|orion|x", node=contradiction)
    store.upsert_edge(identity_key=f"{concept.node_id}|contradicts|{contradiction.node_id}", edge=edge)


def _report() -> GraphCognitionReportV1:
    evidence = SignalEvidenceBundleV1(spans=(EvidenceSpanV1(node_ids=("node-concept",), edge_ids=(), reason="test", weight=1.0),), truncated=False, degraded=False, notes=())
    return GraphCognitionReportV1(
        coherence=CoherenceAssessmentV1(score=0.7, confidence=0.8, evidence=evidence, notes=()),
        identity_conflict=IdentityConflictSignalV1(conflict_score=0.2, active=False, confidence=0.7, evidence=evidence),
        goal_pressure=GoalPressureStateV1(pressure_score=0.82, stalled_goal_count=1, competing_goal_density=0.2, confidence=0.8, evidence=evidence),
        social_continuity=SocialContinuityAssessmentV1(continuity_score=0.6, confidence=0.7, degraded=False, evidence=evidence),
        concept_drift=ConceptDriftSignalV1(drift_score=0.6, active=True, confidence=0.7, evidence=evidence),
        contradiction_candidates=ContradictionCandidateSetV1(candidates=(), confidence=0.7, evidence=evidence),
    )


def test_query_planner_is_bounded_and_deterministic() -> None:
    first = SubstrateQueryPlanner.graph_view_basis(subject_ref="orion", max_nodes=32, max_edges=64)
    second = SubstrateQueryPlanner.graph_view_basis(subject_ref="orion", max_nodes=32, max_edges=64)
    assert first.signature == second.signature
    assert len(first.steps) == 4
    assert all(step.params.get("limit_nodes", 1) <= 32 for step in first.steps if "limit_nodes" in step.params)


def test_query_coordinator_reuses_identical_reads_with_explicit_meta() -> None:
    store = CountingSemanticStore()
    _seed_store(store)
    coordinator = SubstrateSemanticReadCoordinator(store=store)

    plan = SubstrateQueryPlanner.curiosity_seed(max_nodes=16, max_edges=16)
    first = coordinator.execute(plan)
    second = coordinator.execute(plan)

    assert first.meta.reused_cache is False
    assert second.meta.reused_cache is True
    assert store.calls["hotspot_region"] == 1
    assert store.calls["concept_region"] == 1
    assert store.calls["contradiction_region"] == 1


def test_graph_cognition_and_consumers_surface_planning_metadata() -> None:
    store = CountingSemanticStore()
    _seed_store(store)

    _, basis, _ = build_graph_views_from_store(
        store=store,
        now=datetime.now(timezone.utc),
        scope="orion",
        subject_ref="orion",
        max_nodes=16,
        max_edges=16,
    )
    assert basis.plan_kind == "graph_view_basis"
    assert basis.duration_ms >= 0.0

    consolidation = GraphConsolidationEvaluator(store=store).consolidate(
        request=GraphConsolidationRequestV1(
            anchor_scope="orion",
            subject_ref="orion",
            target_zone="concept_graph",
            reason_for_review="phase16",
        )
    )
    assert consolidation.semantic_plan_kind == "consolidation_region"
    assert any(note.startswith("semantic_plan:") for note in consolidation.result.notes)

    curiosity = FrontierCuriosityEvaluator(store=store, max_focal_nodes=4, max_focal_edges=4).evaluate(
        anchor_scope="orion",
        subject_ref="orion",
        cognition_report=_report(),
        perception_brief=build_metacog_perception_brief(_report()),
    )
    assert any(note.startswith("semantic_plan:") for note in curiosity.notes)
