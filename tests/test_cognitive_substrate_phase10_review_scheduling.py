from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.core.schemas.substrate_consolidation import (
    GraphConsolidationDecisionV1,
    GraphConsolidationResultV1,
)
from orion.core.schemas.substrate_review_queue import GraphReviewCyclePolicyV1
from orion.substrate.review_queue import GraphReviewQueue
from orion.substrate.review_schedule import GraphReviewScheduler


def _result_with_outcomes(outcomes: tuple[str, ...], zone: str = "concept_graph") -> GraphConsolidationResultV1:
    decisions = [
        GraphConsolidationDecisionV1(
            target_refs=[f"node-{idx}", f"node-{idx+1}"],
            outcome=outcome,
            reason=f"reason-{outcome}",
            confidence=0.8,
            zone=zone,
            priority=70,
            notes=["test"],
            evidence_summary="evidence",
        )
        for idx, outcome in enumerate(outcomes)
    ]
    return GraphConsolidationResultV1(
        request_id="req-1",
        decisions=decisions,
        outcome_counts={},
        regions_reviewed=[],
        unresolved_regions=[],
        confidence=0.8,
    )


def test_requeue_and_priority_create_queue_items_with_cadence() -> None:
    queue = GraphReviewQueue(max_items=10)
    scheduler = GraphReviewScheduler(queue=queue)
    result = _result_with_outcomes(outcomes=("requeue_review", "maintain_priority"))

    scheduled = scheduler.apply_consolidation_result(consolidation_result=result, anchor_scope="orion", subject_ref="entity:orion")
    assert len(scheduled.enqueued_items) == 2
    assert {d.outcome for d in scheduled.schedule_decisions} == {"enqueue_now"}


def test_low_value_and_retire_map_to_slow_or_terminated_states() -> None:
    queue = GraphReviewQueue(max_items=10)
    scheduler = GraphReviewScheduler(queue=queue)
    result = _result_with_outcomes(outcomes=("damp", "retire"))
    scheduled = scheduler.apply_consolidation_result(consolidation_result=result, anchor_scope="orion", subject_ref="entity:orion")

    outcomes = {d.outcome for d in scheduled.schedule_decisions}
    assert "schedule_later" in outcomes
    assert "terminate" in outcomes


def test_strict_zone_is_operator_only_not_autonomously_queued() -> None:
    queue = GraphReviewQueue(max_items=10)
    scheduler = GraphReviewScheduler(queue=queue)
    result = _result_with_outcomes(outcomes=("operator_only",), zone="self_relationship_graph")
    scheduled = scheduler.apply_consolidation_result(consolidation_result=result, anchor_scope="orion", subject_ref="entity:orion")

    assert scheduled.schedule_decisions[0].outcome == "operator_only"
    assert len(scheduled.enqueued_items) == 0


def test_queue_dedup_selection_and_bounds_are_deterministic() -> None:
    queue = GraphReviewQueue(max_items=2)
    scheduler = GraphReviewScheduler(queue=queue)

    r1 = _result_with_outcomes(outcomes=("requeue_review",))
    r2 = _result_with_outcomes(outcomes=("requeue_review",))
    scheduler.apply_consolidation_result(consolidation_result=r1, anchor_scope="orion", subject_ref="entity:orion")
    scheduler.apply_consolidation_result(consolidation_result=r2, anchor_scope="orion", subject_ref="entity:orion")

    snapshot = queue.snapshot()
    assert len(snapshot.queue_items) <= 2
    eligible = queue.list_eligible(now=datetime.now(timezone.utc) + timedelta(days=1))
    assert len(eligible) >= 1


def test_cycle_budget_and_feedback_suppression_controls() -> None:
    queue = GraphReviewQueue(max_items=10)
    policy = GraphReviewCyclePolicyV1(max_cycles_concept=1, suppress_after_low_value_cycles=2)
    scheduler = GraphReviewScheduler(queue=queue, policy=policy)
    result = _result_with_outcomes(outcomes=("requeue_review",), zone="concept_graph")
    scheduled = scheduler.apply_consolidation_result(consolidation_result=result, anchor_scope="orion", subject_ref="entity:orion")
    item = scheduled.enqueued_items[0]

    reviewed = queue.mark_reviewed(item.queue_item_id)
    assert reviewed is not None
    assert reviewed.cycle_budget.remaining_cycles == 0

    updated = queue.apply_cycle_feedback(item.queue_item_id, no_change=True)
    updated = queue.apply_cycle_feedback(item.queue_item_id, no_change=True)
    assert updated is not None
    assert updated.suppression_state is True

    eligible = queue.list_eligible(now=datetime.now(timezone.utc) + timedelta(days=1))
    assert all(it.queue_item_id != item.queue_item_id for it in eligible)


def test_calibration_knobs_change_cadence_deterministically() -> None:
    queue = GraphReviewQueue(max_items=10)
    policy_fast = GraphReviewCyclePolicyV1(urgent_revisit_seconds=120)
    scheduler_fast = GraphReviewScheduler(queue=queue, policy=policy_fast)
    result = _result_with_outcomes(outcomes=("maintain_priority",), zone="autonomy_graph")
    now = datetime.now(timezone.utc)
    scheduled = scheduler_fast.apply_consolidation_result(consolidation_result=result, anchor_scope="orion", subject_ref="entity:orion", now=now)
    next_review = scheduled.schedule_decisions[0].next_review_at
    assert next_review is not None
    assert int((next_review - now).total_seconds()) == 120


def test_queue_feedback_uses_policy_threshold_not_hardcoded() -> None:
    queue = GraphReviewQueue(max_items=10)
    policy = GraphReviewCyclePolicyV1(suppress_after_low_value_cycles=3)
    scheduler = GraphReviewScheduler(queue=queue, policy=policy)
    result = _result_with_outcomes(outcomes=("requeue_review",), zone="concept_graph")
    scheduled = scheduler.apply_consolidation_result(consolidation_result=result, anchor_scope="orion", subject_ref="entity:orion")
    item = scheduled.enqueued_items[0]

    for _ in range(2):
        updated = queue.apply_cycle_feedback(item.queue_item_id, no_change=True)
        assert updated is not None
        assert updated.suppression_state is False

    final = queue.apply_cycle_feedback(item.queue_item_id, no_change=True)
    assert final is not None
    assert final.suppression_state is True
