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
from orion.substrate import InMemorySubstrateGraphStore, SubstrateDynamicsEngine, SubstrateGraphMaterializer


def _temporal(minutes_ago: int = 0) -> SubstrateTemporalWindowV1:
    return SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago))


def _prov() -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="test",
        source_channel="orion:test",
        producer="pytest",
    )


def _edge(source_id: str, source_kind: str, target_id: str, target_kind: str, predicate: str) -> SubstrateEdgeV1:
    return SubstrateEdgeV1(
        source=NodeRefV1(node_id=source_id, node_kind=source_kind),
        target=NodeRefV1(node_id=target_id, node_kind=target_kind),
        predicate=predicate,
        temporal=_temporal(),
        confidence=1.0,
        salience=0.5,
        provenance=_prov(),
    )


def test_activation_propagates_only_on_allowed_predicates_and_attenuates() -> None:
    concept_a = ConceptNodeV1(
        node_id="concept-a",
        anchor_scope="orion",
        subject_ref="entity:orion",
        label="A",
        temporal=_temporal(0),
        provenance=_prov(),
        signals={"salience": 0.9},
    )
    concept_b = ConceptNodeV1(
        node_id="concept-b",
        anchor_scope="orion",
        subject_ref="entity:orion",
        label="B",
        temporal=_temporal(15),
        provenance=_prov(),
    )
    concept_c = ConceptNodeV1(
        node_id="concept-c",
        anchor_scope="orion",
        subject_ref="entity:orion",
        label="C",
        temporal=_temporal(15),
        provenance=_prov(),
    )
    concept_d = ConceptNodeV1(
        node_id="concept-d",
        anchor_scope="orion",
        subject_ref="entity:orion",
        label="D",
        temporal=_temporal(15),
        provenance=_prov(),
    )
    record = SubstrateGraphRecordV1(
        anchor_scope="orion",
        subject_ref="entity:orion",
        nodes=[concept_a, concept_b, concept_c, concept_d],
        edges=[
            _edge("concept-a", "concept", "concept-b", "concept", "supports"),
            _edge("concept-b", "concept", "concept-c", "concept", "supports"),
            _edge("concept-a", "concept", "concept-d", "concept", "subtype_of"),
        ],
    )
    materializer = SubstrateGraphMaterializer(store=InMemorySubstrateGraphStore())
    materializer.apply_record(record)

    engine = SubstrateDynamicsEngine(store=materializer.store)
    result = engine.tick(now=datetime.now(timezone.utc))
    assert result.activation_updates

    snap = materializer.store.snapshot()
    a = snap.nodes["concept-a"].signals.activation.activation
    b = snap.nodes["concept-b"].signals.activation.activation
    c = snap.nodes["concept-c"].signals.activation.activation
    d = snap.nodes["concept-d"].signals.activation.activation
    assert a > b > c
    assert d < b


def test_decay_dormancy_and_revival_are_deterministic() -> None:
    stale = ConceptNodeV1(
        node_id="stale-node",
        anchor_scope="orion",
        subject_ref="entity:orion",
        label="Stale",
        temporal=_temporal(240),
        provenance=_prov(),
        signals={"activation": {"activation": 0.05, "recency_score": 0.01, "decay_half_life_seconds": 60}},
    )
    drive = DriveNodeV1(
        node_id="drive-1",
        anchor_scope="orion",
        subject_ref="entity:orion",
        drive_kind="coherence",
        temporal=_temporal(0),
        provenance=_prov(),
        signals={"salience": 0.9},
        metadata={"drive_status": "active", "pressure": 0.8},
    )
    record = SubstrateGraphRecordV1(
        anchor_scope="orion",
        subject_ref="entity:orion",
        nodes=[stale, drive],
        edges=[_edge("drive-1", "drive", "stale-node", "concept", "activates")],
    )
    materializer = SubstrateGraphMaterializer(store=InMemorySubstrateGraphStore())
    materializer.apply_record(record)
    engine = SubstrateDynamicsEngine(store=materializer.store)

    first = engine.tick(now=datetime.now(timezone.utc))
    assert any(t.node_id == "stale-node" and t.to_state == "dormant" for t in first.dormancy_transitions)

    stale_now = materializer.store.snapshot().nodes["stale-node"]
    assert stale_now.metadata.get("dormant") is True

    revived_node = stale_now.model_copy(update={"temporal": _temporal(0), "metadata": {**stale_now.metadata, "dormant": True}})
    materializer.store.upsert_node(identity_key=None, node=revived_node)
    second = engine.tick(now=datetime.now(timezone.utc))
    assert any(t.node_id == "stale-node" and t.to_state == "active" for t in second.dormancy_transitions)


def test_contradiction_amplification_respects_resolution_and_severity() -> None:
    concept = ConceptNodeV1(
        node_id="concept-x",
        anchor_scope="orion",
        subject_ref="entity:orion",
        label="X",
        temporal=_temporal(30),
        provenance=_prov(),
    )
    unresolved = ContradictionNodeV1(
        node_id="contra-open",
        anchor_scope="orion",
        subject_ref="entity:orion",
        summary="conflict",
        involved_node_ids=["concept-x", "concept-y"],
        temporal=_temporal(60 * 24 * 4),
        provenance=_prov(),
        metadata={"resolved": False, "severity": 0.9},
    )
    resolved = ContradictionNodeV1(
        node_id="contra-closed",
        anchor_scope="orion",
        subject_ref="entity:orion",
        summary="old conflict",
        involved_node_ids=["concept-x", "concept-z"],
        temporal=_temporal(60 * 24 * 4),
        provenance=_prov(),
        metadata={"resolved": True, "severity": 1.0},
    )
    record = SubstrateGraphRecordV1(anchor_scope="orion", subject_ref="entity:orion", nodes=[concept, unresolved, resolved], edges=[])
    materializer = SubstrateGraphMaterializer(store=InMemorySubstrateGraphStore())
    materializer.apply_record(record)
    engine = SubstrateDynamicsEngine(store=materializer.store)
    engine.tick(now=datetime.now(timezone.utc))
    snap = materializer.store.snapshot()

    assert snap.nodes["contra-open"].metadata.get("dynamic_pressure", 0.0) > 0.4
    assert snap.nodes["contra-closed"].metadata.get("dynamic_pressure", 0.0) == 0.0
    assert snap.nodes["concept-x"].metadata.get("dynamic_pressure", 0.0) > 0.0


def test_drive_goal_pressure_propagation_accounts_for_blocked_and_satisfied_goals() -> None:
    drive = DriveNodeV1(
        node_id="drive-main",
        anchor_scope="orion",
        subject_ref="entity:orion",
        drive_kind="stability",
        temporal=_temporal(0),
        provenance=_prov(),
        signals={"salience": 0.9},
        metadata={"drive_status": "active", "pressure": 0.8},
    )
    blocked_goal = GoalNodeV1(
        node_id="goal-blocked",
        anchor_scope="orion",
        subject_ref="entity:orion",
        goal_text="remove blocker",
        priority=0.9,
        temporal=_temporal(0),
        provenance=_prov(),
        metadata={"goal_status": "blocked"},
    )
    satisfied_goal = GoalNodeV1(
        node_id="goal-satisfied",
        anchor_scope="orion",
        subject_ref="entity:orion",
        goal_text="completed",
        priority=0.9,
        temporal=_temporal(0),
        provenance=_prov(),
        metadata={"goal_status": "satisfied"},
    )
    tension = TensionNodeV1(
        node_id="tension-1",
        anchor_scope="orion",
        subject_ref="entity:orion",
        tension_kind="resource_conflict",
        intensity=0.8,
        temporal=_temporal(0),
        provenance=_prov(),
    )
    record = SubstrateGraphRecordV1(
        anchor_scope="orion",
        subject_ref="entity:orion",
        nodes=[drive, blocked_goal, satisfied_goal, tension],
        edges=[
            _edge("drive-main", "drive", "goal-blocked", "goal", "seeks"),
            _edge("drive-main", "drive", "goal-satisfied", "goal", "seeks"),
            _edge("goal-blocked", "goal", "tension-1", "tension", "blocks"),
        ],
    )
    materializer = SubstrateGraphMaterializer(store=InMemorySubstrateGraphStore())
    materializer.apply_record(record)

    engine = SubstrateDynamicsEngine(store=materializer.store)
    engine.tick(now=datetime.now(timezone.utc))
    snap = materializer.store.snapshot()

    blocked_p = snap.nodes["goal-blocked"].metadata.get("dynamic_pressure", 0.0)
    satisfied_p = snap.nodes["goal-satisfied"].metadata.get("dynamic_pressure", 0.0)
    tension_p = snap.nodes["tension-1"].metadata.get("dynamic_pressure", 0.0)
    assert blocked_p > satisfied_p
    assert 0.0 < tension_p <= 1.0


def test_non_destructive_integration_materialization_still_works() -> None:
    materializer = SubstrateGraphMaterializer(store=InMemorySubstrateGraphStore())
    concept = ConceptNodeV1(
        node_id="concept-int",
        anchor_scope="orion",
        subject_ref="entity:orion",
        label="Integration",
        temporal=_temporal(0),
        provenance=_prov(),
    )
    first = SubstrateGraphRecordV1(anchor_scope="orion", subject_ref="entity:orion", nodes=[concept], edges=[])
    second = SubstrateGraphRecordV1(anchor_scope="orion", subject_ref="entity:orion", nodes=[concept], edges=[])
    r1 = materializer.apply_record(first)
    r2 = materializer.apply_record(second)
    assert r1.nodes_created == 1
    assert r2.nodes_merged == 1

    engine = SubstrateDynamicsEngine(store=materializer.store)
    result = engine.tick(now=datetime.now(timezone.utc))
    assert result.tick_at
    assert materializer.store.snapshot().nodes["concept-int"].node_kind == "concept"
