from uuid import uuid4

from app.executor import _plan_request_from_step_ctx
from orion.schemas.cortex.schemas import ExecutionStep, PlanExecutionRequest


def _skill_args(req: PlanExecutionRequest) -> dict:
    meta = req.context.get("metadata") if isinstance(req.context, dict) else {}
    sa = meta.get("skill_args") if isinstance(meta.get("skill_args"), dict) else {}
    return dict(sa)


def test_plan_request_merges_capability_bridge_skill_args_from_metadata():
    step = ExecutionStep(
        verb_name="skills.biometrics.raw_recent.v1",
        step_name="skills.biometrics.raw_recent.v1",
        order=0,
        services=[],
    )
    ctx = {
        "plan_metadata": {},
        "metadata": {"capability_bridge_skill_args": {"limit": 25}},
    }
    req = _plan_request_from_step_ctx(step, ctx, str(uuid4()))
    assert _skill_args(req).get("limit") == 25
