from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    ContradictionNodeV1,
    GoalNodeV1,
    NodeRefV1,
    SubstrateEdgeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateTemporalWindowV1,
)
from orion.core.schemas.substrate_review_runtime import GraphReviewRuntimeRequestV1
from orion.substrate.consolidation import GraphConsolidationEvaluator
from orion.substrate.materializer import SubstrateGraphMaterializer
from orion.substrate.review_bootstrap import GraphReviewBootstrapper
from orion.substrate.review_queue import GraphReviewQueue
from orion.substrate.review_runtime import GraphReviewRuntimeExecutor
from orion.substrate.review_schedule import GraphReviewScheduler
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


def _edge(source: str, source_kind: str, target: str, target_kind: str) -> SubstrateEdgeV1:
    return SubstrateEdgeV1(
        source=NodeRefV1(node_id=source, node_kind=source_kind),
        target=NodeRefV1(node_id=target, node_kind=target_kind),
        predicate="supports",
        temporal=_temporal(),
        confidence=1.0,
        salience=0.7,
        provenance=_prov(),
    )


def _healthy_store() -> InMemorySubstrateGraphStore:
    store = InMemorySubstrateGraphStore()
    mat = SubstrateGraphMaterializer(store=store)
    concept = ConceptNodeV1(
        node_id="concept-1",
        anchor_scope="orion",
        subject_ref="entity:orion",
        label="C1",
        temporal=_temporal(),
        provenance=_prov(),
        signals={"activation": {"activation": 0.8}, "salience": 0.81},
        metadata={"dynamic_pressure": 0.75},
    )
    goal = GoalNodeV1(
        node_id="goal-1",
        anchor_scope="orion",
        subject_ref="entity:orion",
        goal_text="G1",
        temporal=_temporal(),
        provenance=_prov(),
        signals={"activation": {"activation": 0.73}, "salience": 0.77},
        metadata={"dynamic_pressure": 0.7},
    )
    contradiction = ContradictionNodeV1(
        node_id="contra-1",
        anchor_scope="orion",
        subject_ref="entity:orion",
        summary="conflict",
        involved_node_ids=["concept-1", "goal-1"],
        temporal=_temporal(),
        provenance=_prov(),
        signals={"activation": {"activation": 0.82}, "salience": 0.84},
        metadata={"severity": 0.8, "dynamic_pressure": 0.86},
    )
    record = SubstrateGraphRecordV1(
        anchor_scope="orion",
        subject_ref="entity:orion",
        nodes=[concept, goal, contradiction],
        edges=[
            _edge("concept-1", "concept", "goal-1", "goal"),
            _edge("concept-1", "concept", "contra-1", "contradiction"),
        ],
    )
    mat.apply_record(record)
    return store


def test_bootstrap_enqueues_on_meaningful_substrate_and_execute_once_runs() -> None:
    store = _healthy_store()
    queue = GraphReviewQueue(max_items=20)
    scheduler = GraphReviewScheduler(queue=queue)
    bootstrapper = GraphReviewBootstrapper(scheduler=scheduler, semantic_store=store)

    seeded = bootstrapper.bootstrap(query_limit=12)
    assert seeded.items_before == 0
    assert seeded.items_enqueued >= 1
    assert seeded.due_after >= 1

    runtime = GraphReviewRuntimeExecutor(
        queue=queue,
        consolidation_evaluator=GraphConsolidationEvaluator(store=store),
        scheduler=scheduler,
    )
    result = runtime.execute_once(
        request=GraphReviewRuntimeRequestV1(invocation_surface="operator_review"),
        now=datetime.now(timezone.utc),
    )
    assert result.outcome == "executed"


def test_bootstrap_is_noop_on_empty_semantic_state() -> None:
    store = InMemorySubstrateGraphStore()
    queue = GraphReviewQueue(max_items=20)
    scheduler = GraphReviewScheduler(queue=queue)
    bootstrapper = GraphReviewBootstrapper(scheduler=scheduler, semantic_store=store)

    seeded = bootstrapper.bootstrap(query_limit=12)
    assert seeded.items_enqueued == 0
    assert seeded.due_after == 0
    assert any(note.startswith("seed_skipped:") for note in seeded.notes)


def test_bootstrap_is_dedup_safe_on_repeat_runs() -> None:
    store = _healthy_store()
    queue = GraphReviewQueue(max_items=20)
    scheduler = GraphReviewScheduler(queue=queue)
    bootstrapper = GraphReviewBootstrapper(scheduler=scheduler, semantic_store=store)

    first = bootstrapper.bootstrap(query_limit=12)
    second = bootstrapper.bootstrap(query_limit=12)

    assert first.items_enqueued >= 1
    assert second.items_enqueued == 0


def test_sqlite_refresh_sees_cross_process_queue_updates(tmp_path) -> None:
    db_path = str(tmp_path / "review-queue.db")
    q1 = GraphReviewQueue(max_items=20, sql_db_path=db_path)
    q2 = GraphReviewQueue(max_items=20, sql_db_path=db_path)
    scheduler = GraphReviewScheduler(queue=q1)
    store = _healthy_store()
    bootstrapper = GraphReviewBootstrapper(scheduler=scheduler, semantic_store=store)

    seeded = bootstrapper.bootstrap(now=datetime.now(timezone.utc) - timedelta(hours=1), query_limit=12)
    assert seeded.items_enqueued >= 1

    q2.refresh_from_storage()
    assert len(q2.snapshot(limit=50).queue_items) >= 1
