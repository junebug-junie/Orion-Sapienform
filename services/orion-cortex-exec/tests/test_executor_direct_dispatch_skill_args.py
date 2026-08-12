"""Regression test for a real live incident (2026-08-12): a direct chat-dispatch caller
(services/orion-hub/scripts/api_routes.py's POST /api/debug/container-bringup) sets
CortexChatRequest(metadata={"skill_args": {"service": "..."}}). That metadata survives
Hub -> Gateway -> Orch verbatim into ctx["metadata"] (orion-cortex-orch/app/orchestrator.py's
_build_context sets ctx["metadata"] = req.context.metadata), but _plan_request_from_step_ctx
only read ctx["metadata"]["capability_bridge_skill_args"] (a narrower key meant for nested
LLM-driven capability-bridge calls) -- not the plain "skill_args" key the verb's own
_skill_args() helper (and this same file's PlanExecutionRequest.context.metadata contract)
expects. Confirmed live: skills.docker.compose_service_bringup.v1 always saw service=None
despite the caller setting it correctly.
"""

from uuid import uuid4

from app.executor import _plan_request_from_step_ctx
from orion.schemas.cortex.schemas import ExecutionStep, PlanExecutionRequest


def _skill_args(req: PlanExecutionRequest) -> dict:
    meta = req.context.get("metadata") if isinstance(req.context, dict) else {}
    sa = meta.get("skill_args") if isinstance(meta.get("skill_args"), dict) else {}
    return dict(sa)


def test_plan_request_merges_plain_skill_args_from_metadata():
    step = ExecutionStep(
        verb_name="skills.docker.compose_service_bringup.v1",
        step_name="skills.docker.compose_service_bringup.v1",
        order=0,
        services=[],
    )
    ctx = {
        "plan_metadata": {},
        "metadata": {"skill_args": {"service": "orion-collapse-mirror"}},
    }
    req = _plan_request_from_step_ctx(step, ctx, str(uuid4()))
    assert _skill_args(req).get("service") == "orion-collapse-mirror"


def test_capability_bridge_skill_args_still_overrides_plain_skill_args():
    # Ordering guarantee: capability_bridge_skill_args is applied last and should still win over
    # a plain metadata.skill_args value for the same key, since it's the more specific channel.
    step = ExecutionStep(
        verb_name="skills.biometrics.raw_recent.v1",
        step_name="skills.biometrics.raw_recent.v1",
        order=0,
        services=[],
    )
    ctx = {
        "plan_metadata": {},
        "metadata": {
            "skill_args": {"limit": 10},
            "capability_bridge_skill_args": {"limit": 25},
        },
    }
    req = _plan_request_from_step_ctx(step, ctx, str(uuid4()))
    assert _skill_args(req).get("limit") == 25
