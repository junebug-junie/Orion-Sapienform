from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from orion.core.schemas.substrate_consolidation import GraphConsolidationRequestV1
from orion.core.schemas.substrate_review_queue import GraphReviewQueueItemV1
from orion.core.schemas.substrate_review_runtime import GraphReviewRuntimeRequestV1, GraphReviewRuntimeResultV1
from orion.core.schemas.substrate_review_telemetry import GraphReviewTelemetryRecordV1
from orion.substrate.consolidation import GraphConsolidationEvaluator, GraphConsolidationExecutionV1
from orion.substrate.policy_profiles import SubstratePolicyProfileStore
from orion.substrate.review_queue import GraphReviewQueue
from orion.substrate.review_schedule import GraphReviewScheduler
from orion.substrate.review_telemetry import GraphReviewTelemetryRecorder


class FrontierFollowupExecutor(Protocol):
    def invoke_for_review(self, *, queue_item: GraphReviewQueueItemV1, consolidation: GraphConsolidationExecutionV1) -> bool: ...


@dataclass(frozen=True)
class GraphReviewRuntimeExecutor:
    queue: GraphReviewQueue
    consolidation_evaluator: GraphConsolidationEvaluator
    scheduler: GraphReviewScheduler
    frontier_followup_executor: FrontierFollowupExecutor | None = None
    telemetry_recorder: GraphReviewTelemetryRecorder | None = None
    policy_profiles: SubstratePolicyProfileStore | None = None

    def execute_once(self, *, request: GraphReviewRuntimeRequestV1, now: datetime | None = None) -> GraphReviewRuntimeResultV1:
        started = time.perf_counter()
        runtime_now = now or datetime.now(timezone.utc)
        try:
            item, blocked_outcome, blocked_reason = self._select_item(request=request, now=runtime_now)
            if item is None:
                result = GraphReviewRuntimeResultV1(
                    request_id=request.request_id,
                    correlation_id=request.correlation_id,
                    outcome=blocked_outcome,
                    audit_summary={
                        "invocation_surface": request.invocation_surface,
                        "selection_reason": blocked_reason,
                    },
                    notes=[blocked_reason],
                )
                self._record_telemetry(
                    request=request,
                    result=result,
                    selected_item=None,
                    selection_reason=blocked_reason,
                    started=started,
                )
                return result

            pre_budget = item.cycle_budget.model_dump()
            pre_suppressed = item.suppression_state
            pre_terminated = item.termination_state
            reviewed_item = self.queue.mark_reviewed(item.queue_item_id, reviewed_at=runtime_now)
            if reviewed_item is None:
                result = GraphReviewRuntimeResultV1(
                    request_id=request.request_id,
                    correlation_id=request.correlation_id,
                    selected_queue_item_id=item.queue_item_id,
                    outcome="failed",
                    audit_summary={"invocation_surface": request.invocation_surface, "failure_reason": "queue item missing during review"},
                    notes=["queue_inconsistency"],
                )
                self._record_telemetry(
                    request=request,
                    result=result,
                    selected_item=item,
                    selection_reason="selected_queue_item_missing",
                    started=started,
                )
                return result

            consolidation_request = GraphConsolidationRequestV1(
                correlation_id=request.correlation_id,
                anchor_scope=reviewed_item.anchor_scope,
                subject_ref=reviewed_item.subject_ref,
                focal_node_refs=reviewed_item.focal_node_refs,
                focal_edge_refs=reviewed_item.focal_edge_refs,
                reason_for_review=reviewed_item.reason_for_revisit,
                prior_cycle_refs=[reviewed_item.originating_decision_id],
                target_zone=reviewed_item.target_zone,
                bounded_context_refs=[f"queue_item:{reviewed_item.queue_item_id}"],
            )
            policy_resolution = self._resolve_policy(request=request, queue_item=reviewed_item)
            overrides = policy_resolution.overrides if policy_resolution.mode == "adopted" else {}
            consolidation = self.consolidation_evaluator.consolidate(
                request=consolidation_request,
                max_region_nodes=int(overrides.get("query_limit_nodes")) if overrides.get("query_limit_nodes") is not None else None,
                max_region_edges=int(overrides.get("query_limit_edges")) if overrides.get("query_limit_edges") is not None else None,
                query_cache_enabled=bool(overrides.get("query_cache_enabled", True)),
            )

            no_change = self._no_change_cycle(consolidation)
            updated_item = self.queue.apply_cycle_feedback(reviewed_item.queue_item_id, no_change=no_change) or reviewed_item

            # Re-schedule this reviewed region from fresh consolidation outputs.
            self.scheduler.apply_consolidation_result(
                consolidation_result=consolidation.result,
                anchor_scope=reviewed_item.anchor_scope,
                subject_ref=reviewed_item.subject_ref,
                now=runtime_now,
                invocation_surface=request.invocation_surface,
            )

            frontier_followup_invoked = False
            followup_allowed_by_policy = bool(overrides.get("frontier_followup_allowed", True))
            if (
                request.execute_frontier_followup_allowed
                and followup_allowed_by_policy
                and request.invocation_surface == "operator_review"
                and self.frontier_followup_executor is not None
                and any(d.outcome in {"requeue_review", "maintain_priority", "keep_provisional"} for d in consolidation.result.decisions)
            ):
                frontier_followup_invoked = self.frontier_followup_executor.invoke_for_review(
                    queue_item=updated_item,
                    consolidation=consolidation,
                )

            result = GraphReviewRuntimeResultV1(
                request_id=request.request_id,
                correlation_id=request.correlation_id,
                selected_queue_item_id=updated_item.queue_item_id,
                outcome="executed",
                consolidation_result_ref=consolidation.result.request_id,
                queue_update_summary={
                    "suppression_state_before": pre_suppressed,
                    "suppression_state": updated_item.suppression_state,
                    "termination_state_before": pre_terminated,
                    "termination_state": updated_item.termination_state,
                    "next_review_at": updated_item.next_review_at.isoformat(),
                },
                cycle_budget_summary={
                    "before": pre_budget,
                    "after": updated_item.cycle_budget.model_dump(),
                },
                frontier_followup_invoked=frontier_followup_invoked,
                audit_summary={
                    "invocation_surface": request.invocation_surface,
                    "selection_reason": "eligible_item_selected",
                    "consolidation_outcomes": [d.outcome for d in consolidation.result.decisions],
                    "semantic_source": consolidation.semantic_source,
                    "semantic_degraded": consolidation.semantic_degraded,
                    "semantic_plan": consolidation.semantic_plan_kind,
                    "semantic_reused_cache": consolidation.semantic_reused_cache,
                    "policy_mode": policy_resolution.mode,
                    "policy_profile_id": policy_resolution.profile_id,
                    "policy_query_cache_enabled": bool(overrides.get("query_cache_enabled", True)),
                },
                notes=["single_cycle_execution", f"policy_mode:{policy_resolution.mode}"],
            )
            self._record_telemetry(
                request=request,
                result=result,
                selected_item=updated_item,
                selection_reason="eligible_item_selected",
                started=started,
                consolidation=consolidation,
                pre_suppressed=pre_suppressed,
                pre_terminated=pre_terminated,
            )
            return result
        except Exception as exc:  # fail-safe boundary for narrow runtime seam
            result = GraphReviewRuntimeResultV1(
                request_id=request.request_id,
                correlation_id=request.correlation_id,
                outcome="failed",
                audit_summary={
                    "invocation_surface": request.invocation_surface,
                    "failure_reason": str(exc),
                },
                notes=["runtime_review_execution_failed"],
            )
            self._record_telemetry(
                request=request,
                result=result,
                selected_item=None,
                selection_reason="runtime_exception",
                started=started,
            )
            return result

    def _select_item(
        self,
        *,
        request: GraphReviewRuntimeRequestV1,
        now: datetime,
    ) -> tuple[GraphReviewQueueItemV1 | None, str, str]:
        if request.explicit_queue_item_id is not None:
            snapshot = self.queue.snapshot(limit=request.max_items_to_consider)
            item = next((it for it in snapshot.queue_items if it.queue_item_id == request.explicit_queue_item_id), None)
            if item is None:
                return None, "noop", "explicit queue item not found"
            if item.termination_state:
                return None, "terminated", "queue item terminated"
            if item.suppression_state:
                return None, "suppressed", "queue item suppressed"
            if item.cycle_budget.remaining_cycles <= 0:
                return None, "terminated", "queue item exhausted cycle budget"
            if item.next_review_at > now:
                return None, "noop", "explicit queue item not due yet"
            if (
                item.target_zone == "self_relationship_graph"
                and request.invocation_surface != "operator_review"
                and not request.operator_override_strict_zone
            ):
                return None, "operator_only", "strict-zone item blocked on non-operator surface"
            return item, "noop", "selected explicit queue item"

        eligible = self.queue.list_eligible(now=now, limit=request.max_items_to_consider)
        filtered = [
            it
            for it in eligible
            if (request.anchor_scope is None or it.anchor_scope == request.anchor_scope)
            and (request.subject_ref is None or it.subject_ref == request.subject_ref)
        ]
        if not filtered:
            return None, "noop", "no eligible queue items"

        for item in filtered:
            if item.target_zone == "self_relationship_graph" and request.invocation_surface != "operator_review":
                continue
            return item, "noop", "selected highest-priority eligible item"

        return None, "operator_only", "eligible items require operator surface"

    def _record_telemetry(
        self,
        *,
        request: GraphReviewRuntimeRequestV1,
        result: GraphReviewRuntimeResultV1,
        selected_item: GraphReviewQueueItemV1 | None,
        selection_reason: str,
        started: float,
        consolidation: GraphConsolidationExecutionV1 | None = None,
        pre_suppressed: bool | None = None,
        pre_terminated: bool | None = None,
    ) -> None:
        if self.telemetry_recorder is None:
            return
        try:
            duration_ms = int((time.perf_counter() - started) * 1000)
            self.telemetry_recorder.record(
                GraphReviewTelemetryRecordV1(
                    correlation_id=request.correlation_id,
                    invocation_surface=request.invocation_surface,
                    queue_item_id=selected_item.queue_item_id if selected_item else None,
                    anchor_scope=selected_item.anchor_scope if selected_item else None,
                    subject_ref=selected_item.subject_ref if selected_item else None,
                    target_zone=selected_item.target_zone if selected_item else None,
                    selection_reason=selection_reason,
                    selected_priority=selected_item.priority if selected_item else None,
                    cycle_count_before=(selected_item.cycle_budget.cycle_count - 1) if selected_item else None,
                    cycle_count_after=selected_item.cycle_budget.cycle_count if selected_item else None,
                    remaining_cycles_before=(selected_item.cycle_budget.remaining_cycles + 1) if selected_item else None,
                    remaining_cycles_after=selected_item.cycle_budget.remaining_cycles if selected_item else None,
                    consolidation_outcomes=[d.outcome for d in consolidation.result.decisions] if consolidation else [],
                    suppression_state_before=pre_suppressed,
                    suppression_state_after=selected_item.suppression_state if selected_item else None,
                    termination_state_before=pre_terminated,
                    termination_state_after=selected_item.termination_state if selected_item else None,
                    frontier_followup_invoked=result.frontier_followup_invoked,
                    execution_outcome=result.outcome,
                    runtime_duration_ms=max(0, duration_ms),
                    notes=list(result.notes),
                    degraded=result.outcome == "failed",
                )
            )
        except Exception:
            # Telemetry is advisory; never break runtime path.
            return

    @staticmethod
    def _no_change_cycle(consolidation: GraphConsolidationExecutionV1) -> bool:
        meaningful = {"reinforce", "damp", "retire", "requeue_review", "maintain_priority"}
        return not any(decision.outcome in meaningful for decision in consolidation.result.decisions)

    def _resolve_policy(
        self,
        *,
        request: GraphReviewRuntimeRequestV1,
        queue_item: GraphReviewQueueItemV1,
    ):
        if self.policy_profiles is None:
            from orion.core.schemas.substrate_policy_adoption import SubstratePolicyResolutionV1

            return SubstratePolicyResolutionV1(mode="baseline", reason="policy_store_unconfigured")
        return self.policy_profiles.resolve(
            invocation_surface=request.invocation_surface,
            target_zone=queue_item.target_zone,
            operator_mode=request.invocation_surface == "operator_review",
        )
