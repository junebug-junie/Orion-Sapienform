"""Exec rebuilds PlanExecutionRequest from ctx; ask_camera must see the
user's real question via ctx (raw_user_text/messages), the same gap
docker_prune_stopped_containers and notify_chat_message already had --
capability_bridge's nested call carries user text on ctx, not
plan.metadata.skill_args, unless _plan_request_from_step_ctx injects it.
Review finding, 2026-08-21: caught before shipping, not after a live miss
like the two prior occurrences of this exact gap."""

import asyncio
import json
from uuid import uuid4

from app.executor import _plan_request_from_step_ctx
from app.verb_adapters import AskCameraVerb
from orion.core.verbs.base import VerbContext
from orion.schemas.cortex.schemas import ExecutionStep, PlanExecutionRequest


def _skill_args_from_plan(req: PlanExecutionRequest) -> dict:
    ctx = req.context if isinstance(req.context, dict) else {}
    meta = ctx.get("metadata") if isinstance(ctx.get("metadata"), dict) else {}
    sa = meta.get("skill_args")
    return dict(sa) if isinstance(sa, dict) else {}


def test_plan_request_injects_question_from_raw_user_text():
    step = ExecutionStep(
        verb_name="skills.perception.ask_camera.v1",
        step_name="skills.perception.ask_camera.v1",
        order=0,
        services=[],
    )
    ctx = {"raw_user_text": "Is the door open?", "plan_metadata": {}}
    req = _plan_request_from_step_ctx(step, ctx, str(uuid4()))
    sa = _skill_args_from_plan(req)
    assert sa.get("question") == "Is the door open?"


def test_plan_request_injects_question_from_messages_when_raw_absent():
    step = ExecutionStep(
        verb_name="skills.perception.ask_camera.v1",
        step_name="skills.perception.ask_camera.v1",
        order=0,
        services=[],
    )
    ctx = {
        "plan_metadata": {},
        "messages": [{"role": "user", "content": "How many monitors are on the desk?"}],
    }
    req = _plan_request_from_step_ctx(step, ctx, str(uuid4()))
    sa = _skill_args_from_plan(req)
    assert sa.get("question") == "How many monitors are on the desk?"


def test_plan_request_does_not_override_explicit_question():
    step = ExecutionStep(
        verb_name="skills.perception.ask_camera.v1",
        step_name="skills.perception.ask_camera.v1",
        order=0,
        services=[],
    )
    ctx = {
        "raw_user_text": "Is the door open?",
        "plan_metadata": {"skill_args": {"question": "custom explicit question"}},
    }
    req = _plan_request_from_step_ctx(step, ctx, str(uuid4()))
    sa = _skill_args_from_plan(req)
    assert sa.get("question") == "custom explicit question"


def test_plan_request_leaves_unrelated_verbs_untouched():
    """The injection is scoped to ask_camera specifically -- an unrelated
    verb with the same ctx must not get a spurious "question" key."""
    step = ExecutionStep(
        verb_name="skills.gpu.nvidia_smi_snapshot.v1",
        step_name="skills.gpu.nvidia_smi_snapshot.v1",
        order=0,
        services=[],
    )
    ctx = {"raw_user_text": "Is the door open?", "plan_metadata": {}}
    req = _plan_request_from_step_ctx(step, ctx, str(uuid4()))
    sa = _skill_args_from_plan(req)
    assert "question" not in sa


def test_full_chain_from_capability_bridge_ctx_to_ask_camera_verb(monkeypatch):
    """The actual integration this whole fix is for: a request shaped
    exactly the way capability_bridge's nested call produces it (user text
    on ctx, no plan.metadata.skill_args) rebuilt by
    _plan_request_from_step_ctx, then fed into AskCameraVerb.execute()
    unmodified -- confirms the two real functions compose correctly, not
    just that each one is independently correct against a hand-built
    fixture. This is the exact round trip that was silently broken before
    this patch (AskCameraVerb would have seen skill_args={} and returned
    missing_question for every real chat-driven "ask the camera" turn)."""
    step = ExecutionStep(
        verb_name="skills.perception.ask_camera.v1",
        step_name="skills.perception.ask_camera.v1",
        order=0,
        services=[],
    )
    ctx = {"raw_user_text": "Is the door open?", "plan_metadata": {}}
    req = _plan_request_from_step_ctx(step, ctx, str(uuid4()))

    seen = {}

    def _fake_post(url, *, body, timeout_sec):
        seen["body"] = body
        return {
            "ok": True,
            "artifacts": {
                "model_id": "Salesforce/blip-image-captioning-base",
                "vqa": {"question": body["request"]["question"], "answer": "yes", "confidence": 1.0},
            },
            "warnings": [],
        }

    monkeypatch.setattr("app.verb_adapters._http_json_post", _fake_post)
    vctx = VerbContext(meta={"correlation_id": str(uuid4())})
    out, effects = asyncio.run(AskCameraVerb().execute(vctx, req))

    assert effects == []
    assert out.ok is True
    assert seen["body"]["request"]["question"] == "Is the door open?"
    data = json.loads(out.final_text)
    assert data["answer"] == "yes"
