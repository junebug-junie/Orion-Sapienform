from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.core.schemas.endogenous import EndogenousHistoryEntryV1, EndogenousTriggerRequestV1
from orion.core.schemas.mentor import MentorProposalItemV1, MentorResponseV1
from orion.reasoning.materializer import ReasoningMaterializer
from orion.reasoning.mentor_gateway import MentorGateway
from orion.reasoning.repository import InMemoryReasoningRepository
from orion.reasoning.trigger_history import InMemoryTriggerHistoryStore
from orion.reasoning.triggers import EndogenousTriggerEvaluator
from orion.reasoning.workflows import EndogenousWorkflowOrchestrator, EndogenousWorkflowPlanner


class _MentorProvider:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, request, *, context_packet: list[dict]) -> MentorResponseV1:
        self.calls += 1
        return MentorResponseV1(
            proposal_batch_id=f"batch-{self.calls}",
            mentor_provider=request.mentor_provider,
            mentor_model=request.mentor_model,
            task_type=request.task_type,
            proposals=[
                MentorProposalItemV1(
                    proposal_id=f"proposal-{self.calls}",
                    proposal_type="missing_evidence",
                    confidence=0.7,
                    rationale="narrow advisory critique",
                    evidence_refs=request.context.evidence_refs,
                )
            ],
        )


def _request(**overrides) -> EndogenousTriggerRequestV1:
    base = dict(
        anchor_scope="orion",
        subject_ref="project:orion_sapienform",
        selected_artifact_ids=["artifact-a", "artifact-b"],
        contradiction_refs=["contradiction-1"],
        unresolved_contradiction_count=0,
        contradiction_severity_score=0.0,
        spark_pressure=0.1,
        spark_instability=0.1,
        autonomy_pressure=0.1,
        concept_fragmentation_score=0.1,
        low_confidence_artifact_count=0,
        mentor_gap_count=0,
        lifecycle_state="active",
        evaluated_at=datetime.now(timezone.utc),
    )
    base.update(overrides)
    return EndogenousTriggerRequestV1(**base)


def test_trigger_evaluation_selects_contradiction_review_for_heavy_contradictions() -> None:
    evaluator = EndogenousTriggerEvaluator()
    decision = evaluator.evaluate(
        _request(
            unresolved_contradiction_count=4,
            contradiction_severity_score=0.9,
            contradiction_refs=["c1", "c2", "c3"],
        )
    )
    assert decision.workflow_type == "contradiction_review"
    assert decision.outcome == "trigger"
    assert "unresolved_contradictions_high" in decision.reasons


def test_trigger_evaluation_selects_concept_refinement_for_fragmentation() -> None:
    evaluator = EndogenousTriggerEvaluator()
    decision = evaluator.evaluate(
        _request(
            concept_fragmentation_score=0.95,
            low_confidence_artifact_count=6,
            unresolved_contradiction_count=0,
        )
    )
    assert decision.workflow_type == "concept_refinement"
    assert decision.outcome == "trigger"


def test_trigger_evaluation_selects_autonomy_review_for_drive_strain() -> None:
    evaluator = EndogenousTriggerEvaluator()
    decision = evaluator.evaluate(
        _request(
            autonomy_pressure=0.95,
            spark_pressure=0.8,
            spark_instability=0.4,
        )
    )
    assert decision.workflow_type == "autonomy_review"
    assert decision.outcome == "trigger"


def test_insufficient_pressure_returns_noop() -> None:
    evaluator = EndogenousTriggerEvaluator()
    decision = evaluator.evaluate(_request())
    assert decision.workflow_type == "no_action"
    assert decision.outcome == "noop"


def test_repeated_trigger_under_cooldown_is_suppressed() -> None:
    history = InMemoryTriggerHistoryStore()
    evaluator = EndogenousTriggerEvaluator(history=history)
    now = datetime.now(timezone.utc)
    request = _request(concept_fragmentation_score=0.9, low_confidence_artifact_count=8, evaluated_at=now)
    first = evaluator.evaluate(request)
    history.record(
        EndogenousHistoryEntryV1(
            workflow_type=first.workflow_type,
            subject_ref=request.subject_ref,
            cause_signature=first.debug.cause_signature,
            outcome=first.outcome,
            recorded_at=now,
        )
    )
    second = evaluator.evaluate(request.model_copy(update={"evaluated_at": now + timedelta(seconds=10)}))
    assert second.outcome == "suppress"
    assert second.cooldown_applied is True


def test_workflow_planning_bounds_actions_and_no_action_is_empty() -> None:
    evaluator = EndogenousTriggerEvaluator()
    planner = EndogenousWorkflowPlanner(max_actions=4)

    no_action_decision = evaluator.evaluate(_request())
    no_action_plan = planner.plan(no_action_decision, _request())
    assert no_action_plan.actions == []

    decision = evaluator.evaluate(_request(unresolved_contradiction_count=4, contradiction_severity_score=1.0))
    plan = planner.plan(decision, _request(unresolved_contradiction_count=4, contradiction_severity_score=1.0))
    assert plan.workflow_type == "contradiction_review"
    assert len(plan.actions) <= 4
    assert plan.actions[-1].action_type == "stop"


def test_each_supported_workflow_produces_ordered_actions() -> None:
    evaluator = EndogenousTriggerEvaluator()
    planner = EndogenousWorkflowPlanner(max_actions=8)

    cases = [
        _request(unresolved_contradiction_count=4, contradiction_severity_score=0.9),
        _request(concept_fragmentation_score=0.9, low_confidence_artifact_count=7),
        _request(autonomy_pressure=0.9, spark_pressure=0.8),
        _request(mentor_gap_count=5, spark_instability=0.9),
        _request(concept_fragmentation_score=0.6, low_confidence_artifact_count=2),
    ]
    for req in cases:
        decision = evaluator.evaluate(req)
        plan = planner.plan(decision, req)
        if decision.workflow_type == "no_action":
            continue
        assert plan.actions[0].action_type == "compile_context_slice"
        assert plan.actions[-1].action_type == "stop"


def test_mentor_branch_routes_through_gateway_and_is_bounded() -> None:
    repo = InMemoryReasoningRepository()
    provider = _MentorProvider()
    gateway = MentorGateway(repository=repo, materializer=ReasoningMaterializer(repo), provider=provider)

    history = InMemoryTriggerHistoryStore()
    evaluator = EndogenousTriggerEvaluator(history=history)
    orchestrator = EndogenousWorkflowOrchestrator(evaluator=evaluator, history=history, mentor_gateway=gateway)

    req = _request(mentor_gap_count=5, spark_instability=0.95, unresolved_contradiction_count=0)
    result = orchestrator.orchestrate(req, execute_actions=True)

    assert result.decision.workflow_type == "mentor_critique"
    assert result.mentor_invoked is True
    assert provider.calls == 1
    assert result.materialized_artifact_ids


def test_mentor_cooldown_prevents_repeated_hammering() -> None:
    history = InMemoryTriggerHistoryStore()
    evaluator = EndogenousTriggerEvaluator(history=history)
    now = datetime.now(timezone.utc)

    req = _request(mentor_gap_count=6, spark_instability=0.9, evaluated_at=now)
    first = evaluator.evaluate(req)
    history.record(
        EndogenousHistoryEntryV1(
            workflow_type=first.workflow_type,
            subject_ref=req.subject_ref,
            cause_signature=first.debug.cause_signature,
            outcome=first.outcome,
            recorded_at=now,
        )
    )
    second = evaluator.evaluate(req.model_copy(update={"evaluated_at": now + timedelta(seconds=60)}))
    assert second.outcome == "suppress"
    assert second.workflow_type == "mentor_critique"


def test_contradiction_debounce_coalesces_unchanged_signature() -> None:
    history = InMemoryTriggerHistoryStore()
    evaluator = EndogenousTriggerEvaluator(history=history)
    now = datetime.now(timezone.utc)

    req = _request(unresolved_contradiction_count=5, contradiction_severity_score=0.95, contradiction_refs=["c-cluster"])
    first = evaluator.evaluate(req.model_copy(update={"evaluated_at": now - timedelta(seconds=250)}))
    history.record(
        EndogenousHistoryEntryV1(
            workflow_type=first.workflow_type,
            subject_ref=req.subject_ref,
            cause_signature=first.debug.cause_signature,
            outcome=first.outcome,
            recorded_at=now - timedelta(seconds=100),
        )
    )
    second = evaluator.evaluate(req.model_copy(update={"evaluated_at": now}))
    assert second.outcome == "coalesce"
    assert second.debounce_applied is True


def test_dynamic_subject_active_triggers_and_dormant_suppresses() -> None:
    evaluator = EndogenousTriggerEvaluator()
    active = evaluator.evaluate(
        _request(subject_ref="person:alex", lifecycle_state="active", autonomy_pressure=0.92, spark_pressure=0.85)
    )
    dormant = evaluator.evaluate(
        _request(subject_ref="person:alex", lifecycle_state="dormant", autonomy_pressure=0.92, spark_pressure=0.85)
    )
    assert active.outcome == "trigger"
    assert dormant.outcome == "suppress"


def test_audit_debug_payload_exposes_reasons_counters_and_noop_visibility() -> None:
    evaluator = EndogenousTriggerEvaluator()
    orchestrator = EndogenousWorkflowOrchestrator(evaluator=evaluator)
    result = orchestrator.orchestrate(_request(), execute_actions=False)

    assert result.decision.outcome == "noop"
    assert result.decision.debug.cause_signature
    assert "decision:noop" in result.audit_events
    assert result.plan.workflow_type == "no_action"


def test_runtime_orchestrate_enforces_allowed_workflows() -> None:
    evaluator = EndogenousTriggerEvaluator()
    orchestrator = EndogenousWorkflowOrchestrator(evaluator=evaluator)
    req = _request(unresolved_contradiction_count=4, contradiction_severity_score=0.9)
    result = orchestrator.orchestrate_runtime(
        req,
        invocation_surface="chat_reflective_lane",
        allowed_workflow_types={"reflective_journal"},
        allow_mentor_execution=False,
    )
    assert result.decision.outcome == "suppress"
    assert result.decision.workflow_type == "no_action"
    assert any("runtime_workflow_not_allowed" in reason for reason in result.decision.reasons)
    assert "surface:chat_reflective_lane" in result.audit_events
