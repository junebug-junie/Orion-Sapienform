from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    ContradictionNodeV1,
    DriveNodeV1,
    GoalNodeV1,
    NodeRefV1,
    SubstrateEdgeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateTemporalWindowV1,
    TensionNodeV1,
)
from orion.graph_cognition import (
    build_graph_views,
    build_metacog_perception_brief,
    extract_graph_features,
    interpret_graph_cognition,
)
from orion.substrate import InMemorySubstrateGraphStore, SubstrateDynamicsEngine, SubstrateGraphMaterializer


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
        salience=0.5,
        provenance=_prov(),
    )


def _build_state() -> InMemorySubstrateGraphStore:
    concept = ConceptNodeV1(
        node_id="concept-a",
        anchor_scope="orion",
        subject_ref="entity:orion",
        label="Coherence",
        temporal=_temporal(5),
        provenance=_prov(),
        signals={"salience": 0.8},
    )
    concept_social = ConceptNodeV1(
        node_id="concept-social",
        anchor_scope="orion",
        subject_ref="entity:orion",
        label="Reciprocity",
        temporal=_temporal(20),
        provenance=_prov(),
        metadata={"social": True},
    )
    drive = DriveNodeV1(
        node_id="drive-a",
        anchor_scope="orion",
        subject_ref="entity:orion",
        drive_kind="stability",
        temporal=_temporal(1),
        provenance=_prov(),
        signals={"salience": 0.9},
        metadata={"drive_status": "active", "pressure": 0.85},
    )
    goal_blocked = GoalNodeV1(
        node_id="goal-a",
        anchor_scope="orion",
        subject_ref="entity:orion",
        goal_text="stabilize context",
        priority=0.9,
        temporal=_temporal(2),
        provenance=_prov(),
        metadata={"goal_status": "blocked", "retry_count": 2},
    )
    goal_satisfied = GoalNodeV1(
        node_id="goal-b",
        anchor_scope="orion",
        subject_ref="entity:orion",
        goal_text="close loop",
        priority=0.6,
        temporal=_temporal(3),
        provenance=_prov(),
        metadata={"goal_status": "satisfied"},
    )
    tension = TensionNodeV1(
        node_id="tension-a",
        anchor_scope="orion",
        subject_ref="entity:orion",
        tension_kind="conflict",
        intensity=0.7,
        temporal=_temporal(4),
        provenance=_prov(),
    )
    contradiction = ContradictionNodeV1(
        node_id="contradiction-a",
        anchor_scope="orion",
        subject_ref="entity:orion",
        summary="goal conflict",
        involved_node_ids=["goal-a", "concept-a"],
        temporal=_temporal(120),
        provenance=_prov(),
        metadata={"resolved": False, "severity": 0.8},
    )
    record = SubstrateGraphRecordV1(
        anchor_scope="orion",
        subject_ref="entity:orion",
        nodes=[concept, concept_social, drive, goal_blocked, goal_satisfied, tension, contradiction],
        edges=[
            _edge("drive-a", "drive", "goal-a", "goal", "seeks"),
            _edge("drive-a", "drive", "goal-b", "goal", "seeks"),
            _edge("goal-a", "goal", "tension-a", "tension", "blocks"),
            _edge("concept-a", "concept", "goal-a", "goal", "supports"),
            _edge("concept-a", "concept", "contradiction-a", "contradiction", "contradicts"),
            _edge("goal-a", "goal", "goal-b", "goal", "co_occurs_with"),
        ],
    )
    materializer = SubstrateGraphMaterializer(store=InMemorySubstrateGraphStore())
    materializer.apply_record(record)
    SubstrateDynamicsEngine(store=materializer.store).tick(now=datetime.now(timezone.utc))
    return materializer.store


def test_view_builders_respect_bounds_and_scope() -> None:
    store = _build_state()
    state = store.snapshot()
    views = build_graph_views(
        state=state,
        now=datetime.now(timezone.utc),
        scope="orion",
        subject_ref="entity:orion",
        max_nodes=1,
        max_edges=4,
        time_window_seconds=24 * 3600,
    )
    assert len(views.semantic.node_ids) <= 1
    assert len(views.executive.edge_ids) <= 4
    assert views.semantic.scope == "orion"
    assert views.semantic.truncated is True


def test_feature_extraction_is_deterministic_and_dynamic_aware() -> None:
    store = _build_state()
    state = store.snapshot()
    now = datetime.now(timezone.utc)
    views = build_graph_views(state=state, now=now, scope="orion", subject_ref="entity:orion")
    features = extract_graph_features(state=state, views=views, now=now)
    assert features.structural["node_count"] > 0
    assert features.semantic["contradiction_density"] > 0
    assert features.dynamic["mean_pressure"] > 0
    assert "social_view_sparse" in features.notes or features.social_executive["reciprocity_balance"] >= 0.0


def test_assessments_respond_to_conflict_and_goal_pressure() -> None:
    store = _build_state()
    state = store.snapshot()
    now = datetime.now(timezone.utc)
    views = build_graph_views(state=state, now=now, scope="orion", subject_ref="entity:orion")
    features = extract_graph_features(state=state, views=views, now=now)
    report = interpret_graph_cognition(state=state, views=views, features=features)

    assert report.coherence.score <= 1.0
    assert report.identity_conflict.conflict_score > 0.0
    assert report.goal_pressure.stalled_goal_count >= 1
    assert report.concept_drift.drift_score >= 0.0
    assert report.contradiction_candidates.candidates


def test_brief_assembly_is_compact_and_typed() -> None:
    store = _build_state()
    state = store.snapshot()
    now = datetime.now(timezone.utc)
    views = build_graph_views(state=state, now=now, scope="orion", subject_ref="entity:orion")
    features = extract_graph_features(state=state, views=views, now=now)
    report = interpret_graph_cognition(state=state, views=views, features=features)
    brief = build_metacog_perception_brief(report)

    assert brief.overall_priority in {"stabilize", "reframe", "advance"}
    assert len(brief.top_tensions) <= 3
    assert len(brief.recommended_verbs) == 3
    assert 0.0 <= brief.confidence <= 1.0


def test_non_destructive_integration_with_substrate_phases_1_to_4() -> None:
    store = _build_state()
    state = store.snapshot()
    now = datetime.now(timezone.utc)

    views = build_graph_views(state=state, now=now, scope="orion", subject_ref="entity:orion")
    features = extract_graph_features(state=state, views=views, now=now)
    report = interpret_graph_cognition(state=state, views=views, features=features)
    brief = build_metacog_perception_brief(report)

    assert state.nodes
    assert report.goal_pressure.pressure_score >= 0.0
    assert brief.supporting_evidence.spans
    assert brief.recommended_verbs
