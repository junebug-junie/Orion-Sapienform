"""Cortex Exec must give chat-template workers a task turn for system-only verbs."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.executor import _build_hop_messages, call_step_services
from orion.core.bus.bus_schemas import ChatResponsePayload, ServiceRef
from orion.schemas.cortex.schemas import ExecutionStep


@pytest.mark.parametrize("verb", ["journal.compose", "reverie_narrate", "visual_context_interpret"])
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
