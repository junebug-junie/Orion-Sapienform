"""stance_react's thinking-off request survives the Cortex Exec -> Gateway seam.

orion-thought puts ``chat_template_kwargs={"enable_thinking": False}`` at the top level of
the stance_react plan context (services/orion-thought/app/bus_listener.py). Live
2026-10-14..31, without it, the thinking-on agent lanes spent ~3-6k completion tokens per
stance (chat lane: ~340) and 108 reading turns failed at this verb's 235 s gateway budget.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.executor import call_step_services
from orion.core.bus.bus_schemas import ChatResponsePayload, ServiceRef
from orion.schemas.cortex.schemas import ExecutionStep


def _run_stance_step(ctx):
    step = ExecutionStep(step_name="llm_stance_react", verb_name="stance_react",
                         services=["LLMGatewayService"], order=0, prompt_template="{{ raw_user_text }}")
    base = {"mode": "brain", "session_id": "orion_world_pulse_read", "raw_user_text": "read this",
            "messages": [{"role": "user", "content": "read this"}]}
    with patch("app.executor.LLMGatewayClient.chat", new=AsyncMock(
        return_value=ChatResponsePayload(content='{"disposition":"proceed"}')
    )) as chat:
        asyncio.run(call_step_services(bus=MagicMock(), source=ServiceRef(name="test"),
                    step=step, ctx={**base, **ctx}, correlation_id=str(uuid4())))
    return chat.await_args.kwargs["req"]


def test_stance_react_thinking_off_reaches_the_gateway_on_the_agent_route():
    req = _run_stance_step({
        "llm_route": "agent",
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    })
    assert req.route == "agent"
    assert req.options["chat_template_kwargs"] == {"enable_thinking": False}
    assert req.options["response_format"] == {"type": "json_object"}
