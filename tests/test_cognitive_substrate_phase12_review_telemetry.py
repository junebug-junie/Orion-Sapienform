from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    GoalNodeV1,
    NodeRefV1,
    SubstrateEdgeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateTemporalWindowV1,
)
from orion.core.schemas.substrate_consolidation import GraphConsolidationDecisionV1, GraphConsolidationResultV1
from orion.core.schemas.substrate_review_queue import GraphReviewCyclePolicyV1
from orion.core.schemas.substrate_review_runtime import GraphReviewRuntimeRequestV1
from orion.core.schemas.substrate_review_telemetry import GraphReviewTelemetryQueryV1
from orion.substrate.consolidation import GraphConsolidationEvaluator
from orion.substrate.materializer import SubstrateGraphMaterializer
from orion.substrate.review_queue import GraphReviewQueue
from orion.substrate.review_runtime import GraphReviewRuntimeExecutor
from orion.substrate.review_schedule import GraphReviewScheduler
from orion.substrate.review_telemetry import GraphReviewCalibrationAnalyzer, GraphReviewTelemetryRecorder
from orion.substrate.store import InMemorySubstrateGraphStore


def _temporal() -> SubstrateTemporalWindowV1:
    return SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc))


def _prov() -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(authority="local_inferred", source_kind="test", source_channel="orion:test", producer="pytest")


def _edge(source: str, target: str) -> SubstrateEdgeV1:
    return SubstrateEdgeV1(
        source=NodeRefV1(node_id=source, node_kind="concept" if source.startswith("concept") else "goal"),
        target=NodeRefV1(node_id=target, node_kind="concept" if target.startswith("concept") else "goal"),
        predicate="supports",
        temporal=_temporal(),
        confidence=1.0,
        salience=0.5,
        provenance=_prov(),
    )


def _build_store() -> InMemorySubstrateGraphStore:
    store = InMemorySubstrateGraphStore()
    mat = SubstrateGraphMaterializer(store=store)
    concept = ConceptNodeV1(
        node_id="concept-1",
        anchor_scope="orion",
        subject_ref="entity:orion",
        label="C1",
        temporal=_temporal(),
        provenance=_prov(),
        signals={"activation": {"activation": 0.7}, "salience": 0.7},
        metadata={"dynamic_pressure": 0.6},
    )
    goal = GoalNodeV1(
        node_id="goal-1",
        anchor_scope="orion",
        subject_ref="entity:orion",
        goal_text="G1",
        temporal=_temporal(),
        provenance=_prov(),
        metadata={"dynamic_pressure": 0.7},
    )
    mat.apply_record(
        SubstrateGraphRecordV1(
            anchor_scope="orion",
            subject_ref="entity:orion",
            nodes=[concept, goal],
            edges=[_edge("concept-1", "goal-1")],
        )
    )
    return store


def _enqueue(queue: GraphReviewQueue, scheduler: GraphReviewScheduler) -> str:
    result = GraphConsolidationResultV1(
        request_id="r-telemetry",
        decisions=[
            GraphConsolidationDecisionV1(
                target_refs=["concept-1", "goal-1"],
                outcome="requeue_review",
                reason="test",
                confidence=0.8,
                zone="concept_graph",
                priority=70,
                notes=["t"],
                evidence_summary="t",
            )
        ],
        outcome_counts={},
        regions_reviewed=[],
        unresolved_regions=[],
        confidence=0.8,
    )
    scheduled = scheduler.apply_consolidation_result(
        consolidation_result=result,
        anchor_scope="orion",
        subject_ref="entity:orion",
        now=datetime.now(timezone.utc) - timedelta(hours=3),
    )
    return scheduled.enqueued_items[0].queue_item_id


def test_runtime_execution_emits_telemetry_and_filters_work() -> None:
    queue = GraphReviewQueue(max_items=10)
    scheduler = GraphReviewScheduler(queue=queue, policy=GraphReviewCyclePolicyV1(max_cycles_concept=2))
    _enqueue(queue, scheduler)
    recorder = GraphReviewTelemetryRecorder()
    executor = GraphReviewRuntimeExecutor(
        queue=queue,
        consolidation_evaluator=GraphConsolidationEvaluator(store=_build_store()),
        scheduler=scheduler,
        telemetry_recorder=recorder,
    )

    result = executor.execute_once(request=GraphReviewRuntimeRequestV1(invocation_surface="chat_reflective_lane"))
    assert result.outcome == "executed"

    records = recorder.query(GraphReviewTelemetryQueryV1(limit=10))
    assert len(records) == 1
    assert records[0].execution_outcome == "executed"

    filtered = recorder.query(GraphReviewTelemetryQueryV1(invocation_surface="chat_reflective_lane", outcome="executed", limit=10))
    assert len(filtered) == 1


def test_noop_and_failed_outcomes_are_telemetry_visible_and_non_fatal() -> None:
    recorder = GraphReviewTelemetryRecorder()
    queue = GraphReviewQueue(max_items=10)
    scheduler = GraphReviewScheduler(queue=queue)

    class _BoomEvaluator:
        def consolidate(self, **_: object):
            raise RuntimeError("boom")

    # noop (empty queue)
    noop_exec = GraphReviewRuntimeExecutor(
        queue=queue,
        consolidation_evaluator=GraphConsolidationEvaluator(store=_build_store()),
        scheduler=scheduler,
        telemetry_recorder=recorder,
    )
    noop_result = noop_exec.execute_once(request=GraphReviewRuntimeRequestV1(invocation_surface="operator_review"))
    assert noop_result.outcome == "noop"

    # failed (evaluator raises)
    _enqueue(queue, scheduler)
    fail_exec = GraphReviewRuntimeExecutor(
        queue=queue,
        consolidation_evaluator=_BoomEvaluator(),
        scheduler=scheduler,
        telemetry_recorder=recorder,
    )
    failed_result = fail_exec.execute_once(request=GraphReviewRuntimeRequestV1(invocation_surface="operator_review"))
    assert failed_result.outcome == "failed"

    records = recorder.query(GraphReviewTelemetryQueryV1(limit=20))
    outcomes = {r.execution_outcome for r in records}
    assert "noop" in outcomes
    assert "failed" in outcomes


def test_calibration_recommendations_and_insufficient_data_hold() -> None:
    recorder = GraphReviewTelemetryRecorder()
    queue = GraphReviewQueue(max_items=10)
    scheduler = GraphReviewScheduler(queue=queue, policy=GraphReviewCyclePolicyV1(max_cycles_concept=1))
    executor = GraphReviewRuntimeExecutor(
        queue=queue,
        consolidation_evaluator=GraphConsolidationEvaluator(store=_build_store()),
        scheduler=scheduler,
        telemetry_recorder=recorder,
    )

    # produce enough suppressed/noop/executed mix
    for _ in range(10):
        _enqueue(queue, scheduler)
        executor.execute_once(request=GraphReviewRuntimeRequestV1(invocation_surface="chat_reflective_lane"))

    summary = recorder.summary(GraphReviewTelemetryQueryV1(limit=200))
    analyzer = GraphReviewCalibrationAnalyzer()
    recs = analyzer.recommend(summary=summary)
    assert len(recs) >= 1

    small_summary = recorder.summary(GraphReviewTelemetryQueryV1(limit=1))
    hold = analyzer.recommend(summary=small_summary)
    assert hold[0].recommendation_type == "hold"


def test_telemetry_write_failure_does_not_break_runtime_path() -> None:
    class _BrokenRecorder(GraphReviewTelemetryRecorder):
        def record(self, entry):  # type: ignore[override]
            raise RuntimeError("telemetry down")

    queue = GraphReviewQueue(max_items=10)
    scheduler = GraphReviewScheduler(queue=queue)
    _enqueue(queue, scheduler)
    executor = GraphReviewRuntimeExecutor(
        queue=queue,
        consolidation_evaluator=GraphConsolidationEvaluator(store=_build_store()),
        scheduler=scheduler,
        telemetry_recorder=_BrokenRecorder(),
    )
    result = executor.execute_once(request=GraphReviewRuntimeRequestV1(invocation_surface="operator_review"))
    assert result.outcome == "executed"
