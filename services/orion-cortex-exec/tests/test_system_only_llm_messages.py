"""Cortex Exec must give chat-template workers a task turn for system-only verbs."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.executor import _build_hop_messages, call_step_services
from orion.core.bus.bus_schemas import ChatResponsePayload, ServiceRef
from orion.schemas.cortex.schemas import ExecutionStep


@pytest.mark.parametrize("verb", ["journal.compose", "reverie_question", "visual_context_interpret"])
def test_system_only_metacog_verb_sends_user_task_turn(verb):
    step = ExecutionStep(
        step_name="llm_step", verb_name=verb, services=["LLMGatewayService"],
        order=0, prompt_template="Write the requested JSON artifact.",
    )
    with patch("app.executor.LLMGatewayClient.chat", new=AsyncMock(return_value=ChatResponsePayload(content="{}"))) as chat:
        result = asyncio.run(call_step_services(
            bus=MagicMock(), source=ServiceRef(name="test"), step=step,
            ctx={"mode": "metacog", "llm_route": "metacog"}, correlation_id=str(uuid4()),
        ))

    assert result.status == "success"
    req = chat.await_args.kwargs["req"]
    assert req.route == "metacog"
    assert [m.role for m in req.messages] == ["system", "user"]
    assert "Write the requested JSON artifact." in req.messages[0].content
    assert req.messages[1].content.strip()


def test_existing_user_turn_is_not_duplicated():
    messages = _build_hop_messages(
        prompt="Answer the question.",
        ctx_messages=[{"role": "user", "content": "What changed?"}],
    )
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"] == "What changed?"


def test_no_task_prompt_or_user_turn_fails_locally():
    with pytest.raises(ValueError, match="no task prompt or user message"):
        _build_hop_messages(prompt="", ctx_messages=None)


def _hop_request(ctx):
    step = ExecutionStep(
        # A neutral verb: stance_react's own recall steps are not what this checks.
        step_name="llm_step", verb_name="json_artifact", services=["LLMGatewayService"],
        order=0, prompt_template="Output the requested JSON object.",
    )
    with patch("app.executor.LLMGatewayClient.chat", new=AsyncMock(return_value=ChatResponsePayload(content="{}"))) as chat:
        asyncio.run(call_step_services(
            bus=MagicMock(), source=ServiceRef(name="test"), step=step, ctx=ctx, correlation_id=str(uuid4()),
        ))
    return chat.await_args.kwargs["req"]


def test_json_hop_without_user_turn_gets_the_json_task_turn():
    # Regression, 2026-11-17: stance_react on the Strata agent lane replied "{" and ended (Strata
    # monitor: output_tokens=2, finish=stop). The generic task turn raised the model's chance of
    # ending right after "{" to 0.07-0.15 on two real failing prompts; JSON_TASK_TURN kept it near 0.01-0.03.
    req = _hop_request({"mode": "metacog", "llm_route": "metacog", "response_format": {"type": "json_object"},
                        "chat_template_kwargs": {"enable_thinking": False}})
    assert [m.role for m in req.messages] == ["system", "user"]
    assert req.messages[1].content == "Respond now with the JSON object the system message asks for."


# The sentences are written out, not imported: the wording is what was measured, so a change to it
# should show up here as a failing test.


@pytest.mark.parametrize("json_keys", [
    {"return_json": True},
    {"structured_output_schema": {"type": "object", "properties": {"a": {"type": "string"}}}},
    {"options": {"structured_output_method": "json_schema"}},
])
def test_other_json_requests_get_the_json_task_turn(json_keys):
    req = _hop_request({"mode": "metacog", "llm_route": "metacog", **json_keys})
    assert req.messages[1].content == "Respond now with the JSON object the system message asks for."


def test_wants_json_output_reads_the_gateway_keys():
    from app.executor import _wants_json_output
    assert _wants_json_output({"response_format": {"type": "json_object"}})
    assert _wants_json_output({"return_json": True})
    assert _wants_json_output({"structured_output_schema": {"type": "object"}})
    assert _wants_json_output({"options": {"structured_output_method": "json_schema"}})
    assert not _wants_json_output({"return_json": False, "response_format": None})
    assert not _wants_json_output({"response_format": "json", "options": {}})
    assert not _wants_json_output({})


def test_prose_hop_keeps_the_generic_task_turn():
    messages = _build_hop_messages(prompt="Write a short narration.", ctx_messages=None)
    assert messages[-1] == {"role": "user", "content": "Carry out the task in the system message."}


def test_json_hop_with_a_user_turn_is_left_alone():
    messages = _build_hop_messages(
        prompt="Return JSON.", ctx_messages=[{"role": "user", "content": "hi"}], json_output=True,
    )
    assert [m["content"] for m in messages if m["role"] == "user"] == ["hi"]
