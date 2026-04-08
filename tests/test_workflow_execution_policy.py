from __future__ import annotations

from datetime import datetime, timezone

from orion.cognition.workflows.execution_policy import derive_workflow_execution_policy


def test_policy_parses_immediate_notify_prompt() -> None:
    policy = derive_workflow_execution_policy(
        workflow_id="dream_cycle",
        prompt="Run your dream cycle now and message me when it's done",
        session_id="sid",
        user_id="u1",
        now_utc=datetime(2026, 3, 24, 3, 0, tzinfo=timezone.utc),
    )
    assert policy.invocation_mode == "immediate"
    assert policy.notify_on == "completion"
    assert policy.schedule is None


def test_policy_parses_one_shot_schedule_prompt() -> None:
    policy = derive_workflow_execution_policy(
        workflow_id="self_review",
        prompt="Schedule a self review for tomorrow morning",
        session_id="sid",
        user_id="u1",
        now_utc=datetime(2026, 3, 24, 3, 0, tzinfo=timezone.utc),
    )
    assert policy.invocation_mode == "scheduled"
    assert policy.schedule is not None
    assert policy.schedule.kind == "one_shot"
    assert policy.schedule.run_at_utc is not None


def test_policy_parses_recurring_failure_notify_prompt() -> None:
    policy = derive_workflow_execution_policy(
        workflow_id="concept_induction_pass",
        prompt="Run concept induction every Sunday and notify me only on failure",
        session_id="sid",
        user_id="u1",
        now_utc=datetime(2026, 3, 24, 3, 0, tzinfo=timezone.utc),
    )
    assert policy.invocation_mode == "scheduled"
    assert policy.notify_on == "failure"
    assert policy.schedule is not None
    assert policy.schedule.kind == "recurring"
    assert policy.schedule.cadence == "weekly"
