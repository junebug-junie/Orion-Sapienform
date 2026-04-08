from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from orion.core.schemas.cognitive_substrate import SubstrateAnchorScopeV1
from orion.core.schemas.substrate_consolidation import GraphConsolidationDecisionV1, GraphConsolidationResultV1
from orion.core.schemas.substrate_review_queue import (
    GraphReviewCycleBudgetV1,
    GraphReviewCyclePolicyV1,
    GraphReviewQueueItemV1,
    GraphReviewScheduleDecisionV1,
)
from orion.substrate.review_queue import GraphReviewQueue


@dataclass(frozen=True)
class GraphReviewSchedulingResultV1:
    schedule_decisions: list[GraphReviewScheduleDecisionV1]
    enqueued_items: list[GraphReviewQueueItemV1]


class GraphReviewScheduler:
    def __init__(self, *, queue: GraphReviewQueue | None = None, policy: GraphReviewCyclePolicyV1 | None = None) -> None:
        self._policy = policy or GraphReviewCyclePolicyV1()
        self._queue = queue or GraphReviewQueue(max_items=self._policy.queue_max_items)

    @property
    def queue(self) -> GraphReviewQueue:
        return self._queue

    def apply_consolidation_result(
        self,
        *,
        consolidation_result: GraphConsolidationResultV1,
        anchor_scope: SubstrateAnchorScopeV1,
        subject_ref: str | None,
        now: datetime | None = None,
    ) -> GraphReviewSchedulingResultV1:
        t = now or datetime.now(timezone.utc)
        decisions: list[GraphReviewScheduleDecisionV1] = []
        enqueued: list[GraphReviewQueueItemV1] = []

        for consolidation_decision in consolidation_result.decisions:
            sched_decision, queue_item = self._schedule_for_decision(
                consolidation_decision=consolidation_decision,
                request_id=consolidation_result.request_id,
                anchor_scope=anchor_scope,
                subject_ref=subject_ref,
                now=t,
            )
            decisions.append(sched_decision)
            if queue_item is not None:
                self._queue.upsert(queue_item)
                enqueued.append(queue_item)

        return GraphReviewSchedulingResultV1(schedule_decisions=decisions, enqueued_items=enqueued)

    def _schedule_for_decision(
        self,
        *,
        consolidation_decision: GraphConsolidationDecisionV1,
        request_id: str,
        anchor_scope: SubstrateAnchorScopeV1,
        subject_ref: str | None,
        now: datetime,
    ) -> tuple[GraphReviewScheduleDecisionV1, GraphReviewQueueItemV1 | None]:
        outcome = consolidation_decision.outcome
        zone = consolidation_decision.zone

        if outcome == "operator_only" or zone == "self_relationship_graph":
            return (
                GraphReviewScheduleDecisionV1(
                    target_refs=consolidation_decision.target_refs,
                    outcome="operator_only",
                    cadence_reason="strict-zone remains operator mediated",
                    cycle_budget_reason="strict-zone no autonomous scheduling",
                    priority=consolidation_decision.priority,
                    notes=["strict_zone_guardrail"],
                ),
                None,
            )

        if outcome in {"retire", "noop"}:
            return (
                GraphReviewScheduleDecisionV1(
                    target_refs=consolidation_decision.target_refs,
                    outcome="terminate",
                    cadence_reason="retired/noop region should stop revisiting",
                    cycle_budget_reason="termination by consolidation outcome",
                    priority=consolidation_decision.priority,
                ),
                None,
            )

        if outcome == "damp":
            next_time = now + timedelta(seconds=self._policy.slow_revisit_seconds)
            decision = GraphReviewScheduleDecisionV1(
                target_refs=consolidation_decision.target_refs,
                outcome="schedule_later",
                next_review_at=next_time,
                cadence_reason="damped region gets slow cadence",
                cycle_budget_reason="low-value cadence",
                priority=max(0, consolidation_decision.priority - 20),
            )
            return decision, self._build_item(
                consolidation_decision=consolidation_decision,
                request_id=request_id,
                anchor_scope=anchor_scope,
                subject_ref=subject_ref,
                next_review_at=next_time,
                priority=decision.priority,
            )

        if outcome in {"maintain_priority", "requeue_review"}:
            seconds = (
                self._policy.urgent_revisit_seconds
                if outcome == "maintain_priority"
                else self._policy.normal_revisit_seconds
            )
            next_time = now + timedelta(seconds=seconds)
            decision = GraphReviewScheduleDecisionV1(
                target_refs=consolidation_decision.target_refs,
                outcome="enqueue_now",
                next_review_at=next_time,
                cadence_reason="high unresolved priority" if outcome == "maintain_priority" else "requeue follow-up",
                cycle_budget_reason="active review budget",
                priority=min(100, consolidation_decision.priority + (10 if outcome == "maintain_priority" else 0)),
            )
            return decision, self._build_item(
                consolidation_decision=consolidation_decision,
                request_id=request_id,
                anchor_scope=anchor_scope,
                subject_ref=subject_ref,
                next_review_at=next_time,
                priority=decision.priority,
            )

        if outcome in {"keep_provisional", "reinforce"}:
            next_time = now + timedelta(seconds=self._policy.normal_revisit_seconds)
            decision = GraphReviewScheduleDecisionV1(
                target_refs=consolidation_decision.target_refs,
                outcome="schedule_later",
                next_review_at=next_time,
                cadence_reason="provisional/reinforce follow-up cadence",
                cycle_budget_reason="bounded monitoring",
                priority=max(0, consolidation_decision.priority - 5),
            )
            return decision, self._build_item(
                consolidation_decision=consolidation_decision,
                request_id=request_id,
                anchor_scope=anchor_scope,
                subject_ref=subject_ref,
                next_review_at=next_time,
                priority=decision.priority,
            )

        return (
            GraphReviewScheduleDecisionV1(
                target_refs=consolidation_decision.target_refs,
                outcome="suppress",
                cadence_reason="unrecognized/low-value outcome suppressed",
                cycle_budget_reason="default suppression",
                priority=0,
            ),
            None,
        )

    def _build_item(
        self,
        *,
        consolidation_decision: GraphConsolidationDecisionV1,
        request_id: str,
        anchor_scope: SubstrateAnchorScopeV1,
        subject_ref: str | None,
        next_review_at: datetime,
        priority: int,
    ) -> GraphReviewQueueItemV1:
        max_cycles_by_zone = {
            "world_ontology": self._policy.max_cycles_world,
            "concept_graph": self._policy.max_cycles_concept,
            "autonomy_graph": self._policy.max_cycles_autonomy,
            "self_relationship_graph": self._policy.max_cycles_self_relationship,
        }
        max_cycles = max_cycles_by_zone[consolidation_decision.zone]
        return GraphReviewQueueItemV1(
            focal_node_refs=consolidation_decision.target_refs,
            focal_edge_refs=[],
            anchor_scope=anchor_scope,
            subject_ref=subject_ref,
            target_zone=consolidation_decision.zone,
            originating_decision_id=consolidation_decision.decision_id,
            originating_request_id=request_id,
            reason_for_revisit=consolidation_decision.reason,
            priority=priority,
            next_review_at=next_review_at,
            cycle_budget=GraphReviewCycleBudgetV1(
                cycle_count=0,
                max_cycles=max_cycles,
                remaining_cycles=max_cycles,
                no_change_cycles=0,
                suppress_after_low_value_cycles=self._policy.suppress_after_low_value_cycles,
            ),
            suppression_state=False,
            termination_state=False,
            notes=consolidation_decision.notes[:8],
        )
