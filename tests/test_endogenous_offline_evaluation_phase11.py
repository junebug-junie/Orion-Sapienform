from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.core.schemas.endogenous import EndogenousTriggerDecisionV1, EndogenousTriggerRequestV1, EndogenousWorkflowPlanV1
from orion.core.schemas.endogenous_eval import EndogenousEvaluationRequestV1
from orion.core.schemas.endogenous_runtime import EndogenousRuntimeExecutionRecordV1, EndogenousRuntimeSignalDigestV1
from orion.reasoning.calibration import EndogenousCalibrationEngine, render_evaluation_report
from orion.reasoning.evaluation import EndogenousOfflineEvaluator


def _record(
    *,
    surface: str = "chat_reflective_lane",
    workflow: str = "contradiction_review",
    outcome: str = "trigger",
    mentor_invoked: bool = False,
    cooldown: bool = False,
    debounce: bool = False,
    reasons: list[str] | None = None,
    success: bool = True,
    mentor_gap: int = 0,
    materialized: int = 0,
    subject_ref: str = "project:orion_sapienform",
) -> EndogenousRuntimeExecutionRecordV1:
    request = EndogenousTriggerRequestV1(subject_ref=subject_ref)
    decision = EndogenousTriggerDecisionV1(
        request_id=request.request_id,
        outcome=outcome,
        workflow_type=workflow,
        reasons=reasons or [],
        cooldown_applied=cooldown,
        debounce_applied=debounce,
    )
    plan = EndogenousWorkflowPlanV1(
        request_id=request.request_id,
        workflow_type=workflow,
        trigger_outcome=outcome,
        reasons=list(decision.reasons),
    )
    return EndogenousRuntimeExecutionRecordV1(
        invocation_surface=surface,
        subject_ref=subject_ref,
        trigger_request=request,
        signal_digest=EndogenousRuntimeSignalDigestV1(mentor_gap_count=mentor_gap),
        decision=decision,
        plan=plan,
        mentor_invoked=mentor_invoked,
        materialized_artifact_ids=[f"artifact-{idx}" for idx in range(materialized)],
        execution_success=success,
        created_at=datetime.now(timezone.utc),
    )


def test_metrics_are_deterministic_and_include_breakdowns() -> None:
    records = [
        _record(workflow="contradiction_review", outcome="trigger", materialized=1),
        _record(workflow="reflective_journal", outcome="noop"),
        _record(workflow="concept_refinement", outcome="suppress", cooldown=True),
        _record(workflow="contradiction_review", outcome="coalesce", debounce=True),
        _record(workflow="mentor_critique", outcome="suppress", reasons=["mentor_runtime_disabled"]),
    ]
    evaluator = EndogenousOfflineEvaluator()
    request = EndogenousEvaluationRequestV1(limit=50, min_sample_size=3)
    result = evaluator.evaluate(request, records)
    assert result.metrics.sample_size == 5
    assert result.metrics.by_workflow["contradiction_review"] == 2
    assert result.metrics.by_outcome["suppress"] == 2
    assert result.metrics.cooldown_hit_rate > 0.0
    assert result.metrics.mentor_disabled_suppression_rate > 0.0


def test_insufficient_data_returns_hold_recommendation() -> None:
    evaluator = EndogenousOfflineEvaluator()
    calibration = EndogenousCalibrationEngine()
    result = evaluator.evaluate(EndogenousEvaluationRequestV1(limit=50, min_sample_size=20), [_record()])
    calibrated = calibration.recommend(result)
    assert calibrated.recommendations[0].direction == "hold"
    assert "insufficient_sample_size" in calibrated.recommendations[0].rationale


def test_over_triggering_recommends_stricter_thresholds() -> None:
    records = [_record(outcome="trigger") for _ in range(40)]
    evaluator = EndogenousOfflineEvaluator()
    calibration = EndogenousCalibrationEngine()
    result = evaluator.evaluate(EndogenousEvaluationRequestV1(limit=100, min_sample_size=20), records)
    calibrated = calibration.recommend(result)
    assert any(rec.target == "trigger_threshold" and rec.direction == "increase" for rec in calibrated.recommendations)


def test_under_triggering_recommends_looser_thresholds() -> None:
    records = [_record(outcome="noop", workflow="no_action") for _ in range(36)] + [
        _record(outcome="suppress", workflow="no_action") for _ in range(10)
    ]
    evaluator = EndogenousOfflineEvaluator()
    calibration = EndogenousCalibrationEngine()
    result = evaluator.evaluate(EndogenousEvaluationRequestV1(limit=100, min_sample_size=20), records)
    calibrated = calibration.recommend(result)
    assert any(rec.target == "trigger_threshold" and rec.direction == "decrease" for rec in calibrated.recommendations)


def test_summary_and_promotion_guidance_and_report_generation() -> None:
    records = []
    for _ in range(35):
        records.append(_record(outcome="trigger", mentor_gap=1, reasons=["runtime_workflow_not_allowed:contradiction_review"]))
    evaluator = EndogenousOfflineEvaluator()
    calibration = EndogenousCalibrationEngine()
    result = evaluator.evaluate(EndogenousEvaluationRequestV1(limit=200, min_sample_size=20), records)
    calibrated = calibration.recommend(result)

    assert calibrated.reasoning_summary.recommendation in {"loosen_summary_inclusion", "hold", "tighten_summary_inclusion"}
    assert calibrated.promotion.recommendation in {"review_promotion_evidence_threshold", "hold"}
    markdown = render_evaluation_report(calibrated)
    assert "# Endogenous Offline Evaluation" in markdown
    assert "## Recommendations" in markdown


def test_safety_evaluator_is_advisory_only_and_non_mutating() -> None:
    record = _record()
    original_status = record.decision.outcome

    evaluator = EndogenousOfflineEvaluator()
    calibration = EndogenousCalibrationEngine()
    result = evaluator.evaluate(EndogenousEvaluationRequestV1(limit=10, min_sample_size=2), [record])
    calibrated = calibration.recommend(result)

    assert record.decision.outcome == original_status
    assert all(rec.advisory_only for rec in calibrated.recommendations)
    assert calibrated.generated_profile is not None
    assert calibrated.generated_profile.advisory_only is True


def test_time_and_filtering_inputs_are_honored() -> None:
    old = _record()
    old.created_at = datetime.now(timezone.utc) - timedelta(days=2)
    recent = _record(surface="operator_review", workflow="reflective_journal")
    evaluator = EndogenousOfflineEvaluator()
    request = EndogenousEvaluationRequestV1(
        limit=100,
        invocation_surfaces=["operator_review"],
        workflow_types=["reflective_journal"],
        created_after=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    result = evaluator.evaluate(request, [old, recent])
    assert result.metrics.sample_size == 1
    assert result.metrics.by_surface == {"operator_review": 1}
