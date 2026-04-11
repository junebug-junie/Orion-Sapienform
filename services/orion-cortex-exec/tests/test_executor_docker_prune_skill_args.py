"""Exec rebuilds PlanExecutionRequest from ctx; docker prune must see outer user text for NL run_mode."""

from uuid import uuid4

from app.executor import _plan_request_from_step_ctx
from orion.schemas.cortex.schemas import ExecutionStep, PlanExecutionRequest


def _skill_args_from_plan(req: PlanExecutionRequest) -> dict:
    ctx = req.context if isinstance(req.context, dict) else {}
    meta = ctx.get("metadata") if isinstance(ctx.get("metadata"), dict) else {}
    sa = meta.get("skill_args")
    return dict(sa) if isinstance(sa, dict) else {}


def test_plan_request_injects_user_request_from_raw_user_text():
    step = ExecutionStep(
        verb_name="skills.runtime.docker_prune_stopped_containers.v1",
        step_name="skills.runtime.docker_prune_stopped_containers.v1",
        order=0,
        services=[],
    )
    ctx = {"raw_user_text": "Prune stopped containers.", "plan_metadata": {}}
    req = _plan_request_from_step_ctx(step, ctx, str(uuid4()))
    sa = _skill_args_from_plan(req)
    assert sa.get("user_request") == "Prune stopped containers."


def test_plan_request_injects_from_messages_when_raw_absent():
    step = ExecutionStep(
        verb_name="skills.runtime.docker_prune_stopped_containers.v1",
        step_name="skills.runtime.docker_prune_stopped_containers.v1",
        order=0,
        services=[],
    )
    ctx = {
        "plan_metadata": {},
        "messages": [{"role": "user", "content": "Dry-run cleanup of stopped containers."}],
    }
    req = _plan_request_from_step_ctx(step, ctx, str(uuid4()))
    sa = _skill_args_from_plan(req)
    assert sa.get("user_request") == "Dry-run cleanup of stopped containers."


def test_plan_request_does_not_override_explicit_skill_args():
    step = ExecutionStep(
        verb_name="skills.runtime.docker_prune_stopped_containers.v1",
        step_name="skills.runtime.docker_prune_stopped_containers.v1",
        order=0,
        services=[],
    )
    ctx = {
        "raw_user_text": "Prune stopped containers.",
        "plan_metadata": {"skill_args": {"execute": True, "text": "custom"}},
    }
    req = _plan_request_from_step_ctx(step, ctx, str(uuid4()))
    sa = _skill_args_from_plan(req)
    assert sa.get("text") == "custom"
    assert sa.get("execute") is True
