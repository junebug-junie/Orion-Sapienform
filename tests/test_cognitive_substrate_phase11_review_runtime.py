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
from orion.core.schemas.substrate_consolidation import GraphConsolidationDecisionV1, GraphConsolidationResultV1
from orion.core.schemas.substrate_review_queue import GraphReviewCycleBudgetV1, GraphReviewCyclePolicyV1, GraphReviewQueueItemV1
from orion.core.schemas.substrate_review_runtime import GraphReviewRuntimeRequestV1
from orion.substrate.consolidation import GraphConsolidationEvaluator
from orion.substrate.materializer import SubstrateGraphMaterializer
from orion.substrate.review_queue import GraphReviewQueue
from orion.substrate.review_runtime import GraphReviewRuntimeExecutor
from orion.substrate.review_schedule import GraphReviewScheduler
from orion.substrate.review_telemetry import GraphReviewTelemetryRecorder
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


def _build_store(*, with_contradiction: bool = False) -> InMemorySubstrateGraphStore:
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
    if with_contradiction:
        contradiction = ContradictionNodeV1(
            node_id="contra-1",
            anchor_scope="orion",
            subject_ref="entity:orion",
            summary="conflict",
            involved_node_ids=["concept-1", "goal-1"],
            temporal=_temporal(),
            provenance=_prov(),
            metadata={"severity": 0.8},
        )
        record = SubstrateGraphRecordV1(anchor_scope="orion", subject_ref="entity:orion", nodes=[concept, goal, contradiction], edges=[_edge("concept-1", "goal-1")])
    else:
        record = SubstrateGraphRecordV1(anchor_scope="orion", subject_ref="entity:orion", nodes=[concept, goal], edges=[_edge("concept-1", "goal-1")])
    mat.apply_record(record)
    return store


def _enqueue_requeue_item(*, queue: GraphReviewQueue, scheduler: GraphReviewScheduler, zone: str = "concept_graph") -> str:
    result = GraphConsolidationResultV1(
        request_id="r-1",
        decisions=[
            GraphConsolidationDecisionV1(
                target_refs=["concept-1", "goal-1"],
                outcome="requeue_review",
                reason="test requeue",
                confidence=0.8,
                zone=zone,
                priority=70,
                notes=["test"],
                evidence_summary="test",
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


class _FollowupStub:
    def __init__(self) -> None:
        self.calls = 0

    def invoke_for_review(self, **_: object) -> bool:
        self.calls += 1
        return True


def test_runtime_selects_eligible_item_and_executes_single_cycle() -> None:
    queue = GraphReviewQueue(max_items=10)
    scheduler = GraphReviewScheduler(queue=queue, policy=GraphReviewCyclePolicyV1(max_cycles_concept=2))
    item_id = _enqueue_requeue_item(queue=queue, scheduler=scheduler)
    evaluator = GraphConsolidationEvaluator(store=_build_store())
    executor = GraphReviewRuntimeExecutor(queue=queue, consolidation_evaluator=evaluator, scheduler=scheduler)

    result = executor.execute_once(request=GraphReviewRuntimeRequestV1(invocation_surface="chat_reflective_lane"))
    assert result.outcome == "executed"
    assert result.selected_queue_item_id == item_id
    assert result.cycle_budget_summary["after"]["cycle_count"] == 1


def test_runtime_skips_suppressed_terminated_and_returns_noop() -> None:
    queue = GraphReviewQueue(max_items=10)
    scheduler = GraphReviewScheduler(queue=queue)
    item_id = _enqueue_requeue_item(queue=queue, scheduler=scheduler)
    queue.apply_cycle_feedback(item_id, no_change=True)
    queue.apply_cycle_feedback(item_id, no_change=True)

    evaluator = GraphConsolidationEvaluator(store=_build_store())
    executor = GraphReviewRuntimeExecutor(queue=queue, consolidation_evaluator=evaluator, scheduler=scheduler)
    result = executor.execute_once(request=GraphReviewRuntimeRequestV1(invocation_surface="chat_reflective_lane", explicit_queue_item_id=item_id))
    assert result.outcome == "suppressed"


def test_strict_zone_items_blocked_on_non_operator_surface() -> None:
    queue = GraphReviewQueue(max_items=10)
    scheduler = GraphReviewScheduler(queue=queue)
    queue.upsert(
        GraphReviewQueueItemV1(
            focal_node_refs=["concept-1", "goal-1"],
            focal_edge_refs=[],
            anchor_scope="orion",
            subject_ref="entity:orion",
            target_zone="self_relationship_graph",
            originating_decision_id="d-1",
            originating_request_id="r-1",
            reason_for_revisit="strict zone",
            priority=80,
            next_review_at=datetime.now(timezone.utc) - timedelta(minutes=10),
            cycle_budget=GraphReviewCycleBudgetV1(cycle_count=0, max_cycles=1, remaining_cycles=1),
        )
    )

    evaluator = GraphConsolidationEvaluator(store=_build_store())
    executor = GraphReviewRuntimeExecutor(queue=queue, consolidation_evaluator=evaluator, scheduler=scheduler)
    result = executor.execute_once(request=GraphReviewRuntimeRequestV1(invocation_surface="chat_reflective_lane"))
    assert result.outcome == "operator_only"


def test_followup_branch_is_optional_and_default_off() -> None:
    queue = GraphReviewQueue(max_items=10)
    scheduler = GraphReviewScheduler(queue=queue)
    _enqueue_requeue_item(queue=queue, scheduler=scheduler)
    followup = _FollowupStub()
    evaluator = GraphConsolidationEvaluator(store=_build_store(with_contradiction=True))
    executor = GraphReviewRuntimeExecutor(queue=queue, consolidation_evaluator=evaluator, scheduler=scheduler, frontier_followup_executor=followup)

    off = executor.execute_once(request=GraphReviewRuntimeRequestV1(invocation_surface="operator_review"))
    assert off.frontier_followup_invoked is False
    assert followup.calls == 0

    # enqueue fresh item and enable follow-up
    _enqueue_requeue_item(queue=queue, scheduler=scheduler)
    on = executor.execute_once(
        request=GraphReviewRuntimeRequestV1(
            invocation_surface="operator_review",
            execute_frontier_followup_allowed=True,
        )
    )
    assert on.outcome == "executed"
    assert on.frontier_followup_invoked is True
    assert followup.calls == 1


def test_second_review_of_a_stable_region_reaches_reinforce_not_keep_provisional() -> None:
    """2026-09-08 regression: consolidate() was never given a real prior_cycle
    at all -- node_persistence_ratio was permanently 0.0, so `reinforce` was
    structurally unreachable no matter how stable a region actually was.
    Confirmed live: 100% of recorded consolidation_outcomes, all of history,
    were `keep_provisional`. See
    docs/plans/substrate/2026-09-08-consolidation-prior-cycle-wiring.md.

    Store here never changes between cycles (no contradiction, activation
    0.35 mean -- above the >=0.2 reinforce floor) -- the *only* thing that
    can move the outcome from the first cycle to the second is a real
    prior_cycle now being looked up and threaded through.
    """
    queue = GraphReviewQueue(max_items=10)
    scheduler = GraphReviewScheduler(queue=queue)
    item_id = _enqueue_requeue_item(queue=queue, scheduler=scheduler)
    evaluator = GraphConsolidationEvaluator(store=_build_store())
    telemetry = GraphReviewTelemetryRecorder()
    executor = GraphReviewRuntimeExecutor(
        queue=queue, consolidation_evaluator=evaluator, scheduler=scheduler, telemetry_recorder=telemetry
    )

    # No prior_cycle exists yet -- lookup must be empty before the first review.
    assert executor._lookup_prior_cycle(queue_item_id=item_id) is None  # noqa: SLF001

    first = executor.execute_once(request=GraphReviewRuntimeRequestV1(invocation_surface="operator_review"))
    assert first.outcome == "executed"
    assert first.audit_summary["consolidation_outcomes"] == ["keep_provisional"]

    # Force the item eligible again regardless of the scheduler's real cadence --
    # this test is about the prior_cycle comparison, not revisit timing.
    second = executor.execute_once(
        request=GraphReviewRuntimeRequestV1(invocation_surface="operator_review", explicit_queue_item_id=item_id),
        now=datetime.now(timezone.utc) + timedelta(days=365),
    )
    assert second.outcome == "executed"
    assert second.audit_summary["consolidation_outcomes"] == ["reinforce"]


def test_organic_requeue_of_an_unchanged_region_preserves_queue_item_id() -> None:
    """The prior-cycle lookup is keyed on queue_item_id. That id is only
    stable across cycles when GraphReviewQueue.upsert()'s exact-match
    region_key (zone + sorted focal_node_refs) matches on re-enqueue -- see
    test_region_key_changes_orphan_the_queue_item below for what happens
    when it doesn't. This test proves the common case works end-to-end
    through the real scheduler re-enqueue path (apply_consolidation_result
    -> queue.upsert()), not just via the explicit_queue_item_id shortcut
    the other tests in this module use to force a second cycle.
    """
    queue = GraphReviewQueue(max_items=10)
    scheduler = GraphReviewScheduler(queue=queue)
    item_id = _enqueue_requeue_item(queue=queue, scheduler=scheduler)
    evaluator = GraphConsolidationEvaluator(store=_build_store())
    telemetry = GraphReviewTelemetryRecorder()
    executor = GraphReviewRuntimeExecutor(
        queue=queue, consolidation_evaluator=evaluator, scheduler=scheduler, telemetry_recorder=telemetry
    )

    first = executor.execute_once(request=GraphReviewRuntimeRequestV1(invocation_surface="operator_review"))
    assert first.selected_queue_item_id == item_id

    # Auto-select, not explicit_queue_item_id -- whatever the scheduler
    # actually re-enqueued after cycle 1 is what gets picked up here.
    second = executor.execute_once(
        request=GraphReviewRuntimeRequestV1(invocation_surface="operator_review"),
        now=datetime.now(timezone.utc) + timedelta(days=365),
    )
    assert second.selected_queue_item_id == item_id
    assert second.audit_summary["consolidation_outcomes"] == ["reinforce"]


def test_region_key_changes_orphan_the_queue_item() -> None:
    """Known limitation, not fixed by this patch: GraphReviewQueue.upsert()
    matches re-enqueued items by an EXACT region_key (zone + sorted
    focal_node_refs, review_queue.py's _region_key) -- not by any more
    durable region identity (anchor_scope/subject_ref/originating decision).
    Any drift in the live-resolved node set (e.g. a node entering or
    leaving scope between reviews -- an ordinary substrate event, not an
    error) mints a brand-new queue_item_id instead of continuing the old
    one. _lookup_prior_cycle() is keyed on queue_item_id, so a review that
    drifts loses its prior_cycle exactly when the comparison matters most --
    it silently falls back to today's pre-patch behavior for that one
    cycle, not a crash, but a real gap named in
    docs/plans/substrate/2026-09-08-consolidation-prior-cycle-wiring.md's
    follow-ups rather than solved here.
    """
    queue = GraphReviewQueue(max_items=10)
    scheduler = GraphReviewScheduler(queue=queue)
    same_region_narrower = GraphConsolidationDecisionV1(
        target_refs=["concept-1", "goal-1"],
        outcome="requeue_review",
        reason="baseline",
        confidence=0.8,
        zone="concept_graph",
        priority=70,
        notes=[],
        evidence_summary="baseline",
    )
    drifted_region = same_region_narrower.model_copy(update={"target_refs": ["concept-1", "goal-1", "concept-2"]})

    scheduler.apply_consolidation_result(
        consolidation_result=GraphConsolidationResultV1(
            request_id="r-1", decisions=[same_region_narrower], outcome_counts={}, regions_reviewed=[], unresolved_regions=[], confidence=0.8
        ),
        anchor_scope="orion",
        subject_ref="entity:orion",
        now=datetime.now(timezone.utc),
    )
    first_id = queue.snapshot(limit=10).queue_items[0].queue_item_id

    scheduler.apply_consolidation_result(
        consolidation_result=GraphConsolidationResultV1(
            request_id="r-2", decisions=[drifted_region], outcome_counts={}, regions_reviewed=[], unresolved_regions=[], confidence=0.8
        ),
        anchor_scope="orion",
        subject_ref="entity:orion",
        now=datetime.now(timezone.utc),
    )
    items = queue.snapshot(limit=10).queue_items
    assert len(items) == 2, "a one-node-wider region should not merge with the original -- it should orphan it"
    assert first_id in {item.queue_item_id for item in items}


def test_prior_cycle_lookup_reconstructs_the_last_real_cycle() -> None:
    queue = GraphReviewQueue(max_items=10)
    scheduler = GraphReviewScheduler(queue=queue)
    item_id = _enqueue_requeue_item(queue=queue, scheduler=scheduler)
    evaluator = GraphConsolidationEvaluator(store=_build_store())
    telemetry = GraphReviewTelemetryRecorder()
    executor = GraphReviewRuntimeExecutor(
        queue=queue, consolidation_evaluator=evaluator, scheduler=scheduler, telemetry_recorder=telemetry
    )

    assert executor._lookup_prior_cycle(queue_item_id=item_id) is None  # noqa: SLF001

    executor.execute_once(request=GraphReviewRuntimeRequestV1(invocation_surface="operator_review"))

    prior = executor._lookup_prior_cycle(queue_item_id=item_id)  # noqa: SLF001
    assert prior is not None
    assert set(prior.focal_node_refs) == {"concept-1", "goal-1"}
    assert prior.contradiction_count == 0
    assert prior.mean_activation is not None and prior.mean_activation > 0.0


def test_lookup_prior_cycle_returns_none_without_a_telemetry_recorder() -> None:
    queue = GraphReviewQueue(max_items=10)
    scheduler = GraphReviewScheduler(queue=queue)
    evaluator = GraphConsolidationEvaluator(store=_build_store())
    executor = GraphReviewRuntimeExecutor(queue=queue, consolidation_evaluator=evaluator, scheduler=scheduler)

    assert executor._lookup_prior_cycle(queue_item_id="anything") is None  # noqa: SLF001


def test_runtime_failure_is_captured_without_raising() -> None:
    class _BoomEvaluator:
        def consolidate(self, **_: object):
            raise RuntimeError("boom")

    queue = GraphReviewQueue(max_items=10)
    scheduler = GraphReviewScheduler(queue=queue)
    _enqueue_requeue_item(queue=queue, scheduler=scheduler)
    executor = GraphReviewRuntimeExecutor(queue=queue, consolidation_evaluator=_BoomEvaluator(), scheduler=scheduler)

    result = executor.execute_once(request=GraphReviewRuntimeRequestV1(invocation_surface="operator_review"))
    assert result.outcome == "failed"
    assert "runtime_review_execution_failed" in result.notes
