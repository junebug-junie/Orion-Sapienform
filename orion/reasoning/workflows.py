from __future__ import annotations

from datetime import datetime, timezone

from orion.core.schemas.endogenous import (
    EndogenousHistoryEntryV1,
    EndogenousTriggerDecisionV1,
    EndogenousTriggerRequestV1,
    EndogenousWorkflowActionV1,
    EndogenousWorkflowExecutionResultV1,
    EndogenousWorkflowPlanV1,
)
from orion.core.schemas.mentor import MentorRequestV1
from orion.core.schemas.reasoning_summary import ReasoningSummaryRequestV1
from orion.reasoning.mentor_gateway import MentorGateway
from orion.reasoning.summary import ReasoningSummaryCompiler
from orion.reasoning.trigger_history import InMemoryTriggerHistoryStore
from orion.reasoning.triggers import EndogenousTriggerEvaluator


class EndogenousWorkflowPlanner:
    """Bounded deterministic planner for endogenous self-revision actions."""

    def __init__(self, *, max_actions: int = 6) -> None:
        self._max_actions = max_actions

    def plan(self, decision: EndogenousTriggerDecisionV1, request: EndogenousTriggerRequestV1) -> EndogenousWorkflowPlanV1:
        actions: list[EndogenousWorkflowActionV1] = []
        if decision.outcome in {"trigger", "coalesce"}:
            actions.extend(self._actions_for_workflow(decision.workflow_type, request))
        if actions and actions[-1].action_type != "stop":
            actions.append(EndogenousWorkflowActionV1(action_type="stop"))
        if len(actions) > self._max_actions:
            actions = actions[: self._max_actions]
            if actions[-1].action_type != "stop":
                actions[-1] = EndogenousWorkflowActionV1(action_type="stop")

        return EndogenousWorkflowPlanV1(
            request_id=request.request_id,
            workflow_type=decision.workflow_type,
            trigger_outcome=decision.outcome,
            reasons=list(decision.reasons),
            actions=actions,
            max_actions=self._max_actions,
            coalesced_with=decision.debug.cause_signature if decision.coalesced else None,
            audit={
                "cooldown_applied": decision.cooldown_applied,
                "debounce_applied": decision.debounce_applied,
                "selected_workflow": decision.workflow_type,
                "signal_total_pressure": decision.signal.total_pressure,
            },
        )

    def _actions_for_workflow(self, workflow_type: str, request: EndogenousTriggerRequestV1) -> list[EndogenousWorkflowActionV1]:
        base = [EndogenousWorkflowActionV1(action_type="compile_context_slice", params={"artifact_ids": request.selected_artifact_ids[:8]})]
        if workflow_type == "contradiction_review":
            return base + [
                EndogenousWorkflowActionV1(action_type="run_contradiction_check", params={"contradiction_refs": request.contradiction_refs[:8]}),
                EndogenousWorkflowActionV1(action_type="emit_reflective_journal", params={"journal_kind": "contradiction_review"}),
                EndogenousWorkflowActionV1(action_type="emit_audit_trace", params={"workflow": workflow_type}),
            ]
        if workflow_type == "concept_refinement":
            return base + [
                EndogenousWorkflowActionV1(action_type="run_concept_refinement", params={"fragmentation": request.concept_fragmentation_score}),
                EndogenousWorkflowActionV1(action_type="emit_reflective_journal", params={"journal_kind": "concept_refinement"}),
                EndogenousWorkflowActionV1(action_type="emit_audit_trace", params={"workflow": workflow_type}),
            ]
        if workflow_type == "autonomy_review":
            return base + [
                EndogenousWorkflowActionV1(action_type="review_autonomy_state", params={"autonomy_pressure": request.autonomy_pressure}),
                EndogenousWorkflowActionV1(action_type="emit_reflective_journal", params={"journal_kind": "autonomy_review"}),
                EndogenousWorkflowActionV1(action_type="emit_audit_trace", params={"workflow": workflow_type}),
            ]
        if workflow_type == "mentor_critique":
            return base + [
                EndogenousWorkflowActionV1(
                    action_type="invoke_mentor_gateway",
                    params={"task_type": "contradiction_review" if request.unresolved_contradiction_count else "concept_refinement"},
                ),
                EndogenousWorkflowActionV1(action_type="materialize_advisory_proposals"),
                EndogenousWorkflowActionV1(action_type="promotion_gate_check", params={"mode": "advisory_only"}),
                EndogenousWorkflowActionV1(action_type="emit_audit_trace", params={"workflow": workflow_type}),
            ]
        if workflow_type == "reflective_journal":
            return base + [
                EndogenousWorkflowActionV1(action_type="emit_reflective_journal", params={"journal_kind": "pressure_capture"}),
                EndogenousWorkflowActionV1(action_type="emit_audit_trace", params={"workflow": workflow_type}),
            ]
        return []


class EndogenousWorkflowOrchestrator:
    """Bounded orchestration seam for trigger-evaluate -> plan -> optional mentor execution."""

    def __init__(
        self,
        *,
        evaluator: EndogenousTriggerEvaluator,
        planner: EndogenousWorkflowPlanner | None = None,
        history: InMemoryTriggerHistoryStore | None = None,
        mentor_gateway: MentorGateway | None = None,
        summary_compiler: ReasoningSummaryCompiler | None = None,
    ) -> None:
        self._evaluator = evaluator
        self._planner = planner or EndogenousWorkflowPlanner()
        self._history = history or evaluator.history
        self._mentor_gateway = mentor_gateway
        self._summary_compiler = summary_compiler

    def orchestrate(self, request: EndogenousTriggerRequestV1, *, execute_actions: bool = False) -> EndogenousWorkflowExecutionResultV1:
        return self._orchestrate(
            request,
            execute_actions=execute_actions,
            allowed_workflow_types=None,
            allow_mentor_execution=True,
            invocation_surface=None,
        )

    def _orchestrate(
        self,
        request: EndogenousTriggerRequestV1,
        *,
        execute_actions: bool,
        allowed_workflow_types: set[str] | None,
        allow_mentor_execution: bool,
        invocation_surface: str | None,
    ) -> EndogenousWorkflowExecutionResultV1:
        if request.reasoning_summary is None and self._summary_compiler is not None:
            compiled = self._summary_compiler.compile(
                ReasoningSummaryRequestV1(anchor_scope=request.anchor_scope, subject_refs=[request.subject_ref] if request.subject_ref else [])
            )
            request = request.model_copy(update={"reasoning_summary": compiled})

        decision = self._evaluator.evaluate(request)
        if (
            allowed_workflow_types is not None
            and decision.workflow_type not in allowed_workflow_types
            and decision.outcome in {"trigger", "coalesce"}
        ):
            decision = decision.model_copy(
                update={
                    "outcome": "suppress",
                    "workflow_type": "no_action",
                    "reasons": list(decision.reasons) + [f"runtime_workflow_not_allowed:{decision.workflow_type}"],
                    "coalesced": False,
                }
            )

        if decision.workflow_type == "mentor_critique" and not allow_mentor_execution and decision.outcome == "trigger":
            decision = decision.model_copy(
                update={
                    "outcome": "suppress",
                    "workflow_type": "no_action",
                    "reasons": list(decision.reasons) + ["mentor_runtime_disabled"],
                }
            )
        plan = self._planner.plan(decision, request)

        now = request.evaluated_at if request.evaluated_at.tzinfo else request.evaluated_at.replace(tzinfo=timezone.utc)
        self._history.record(
            EndogenousHistoryEntryV1(
                workflow_type=decision.workflow_type,
                subject_ref=request.subject_ref,
                cause_signature=decision.debug.cause_signature,
                outcome=decision.outcome,
                recorded_at=now,
            )
        )

        mentor_invoked = False
        materialized_ids: list[str] = []
        audit_events = [f"decision:{decision.outcome}", f"workflow:{decision.workflow_type}"]
        if invocation_surface:
            audit_events.append(f"surface:{invocation_surface}")

        if (
            execute_actions
            and allow_mentor_execution
            and decision.outcome == "trigger"
            and decision.workflow_type == "mentor_critique"
            and self._mentor_gateway is not None
        ):
            mentor_invoked = True
            task_type = "contradiction_review" if request.unresolved_contradiction_count else "concept_refinement"
            mentor_req = MentorRequestV1(
                mentor_provider="openai",
                mentor_model="gpt-4.1-mini",
                task_type=task_type,
                anchor_scope=request.anchor_scope,
                subject_ref=request.subject_ref,
                context={
                    "artifact_ids": request.selected_artifact_ids[:8],
                    "summary_refs": [request.reasoning_summary.request_id] if request.reasoning_summary else [],
                    "evidence_refs": request.contradiction_refs[:8],
                },
                correlation_id=request.request_id,
            )
            result = self._mentor_gateway.execute(mentor_req)
            materialized_ids = list(result.materialized_artifact_ids)
            audit_events.append(f"mentor_success:{result.success}")

        return EndogenousWorkflowExecutionResultV1(
            request_id=request.request_id,
            decision=decision,
            plan=plan,
            executed=execute_actions,
            mentor_invoked=mentor_invoked,
            materialized_artifact_ids=materialized_ids,
            audit_events=audit_events,
        )

    def orchestrate_runtime(
        self,
        request: EndogenousTriggerRequestV1,
        *,
        invocation_surface: str,
        allowed_workflow_types: set[str],
        allow_mentor_execution: bool,
        execute_actions: bool = True,
    ) -> EndogenousWorkflowExecutionResultV1:
        return self._orchestrate(
            request,
            execute_actions=execute_actions,
            allowed_workflow_types=allowed_workflow_types,
            allow_mentor_execution=allow_mentor_execution,
            invocation_surface=invocation_surface,
        )
