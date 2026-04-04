import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.executor import call_step_services
from orion.core.bus.bus_schemas import ChatResponsePayload, ServiceRef
from orion.schemas.cortex.schemas import ExecutionStep


def _base_ctx(mode: str = "brain") -> dict:
    return {
        "mode": mode,
        "messages": [{"role": "user", "content": "hello"}],
        "raw_user_text": "hello",
        "session_id": "s-test",
    }


def test_chat_general_stance_step_uses_helper_route() -> None:
    step = ExecutionStep(
        step_name="synthesize_chat_stance_brief",
        verb_name="chat_general",
        services=["LLMGatewayService"],
        order=0,
        prompt_template="{{ raw_user_text }}",
    )
    source = ServiceRef(name="test", node="test", version="1.0")

    with patch("app.executor.LLMGatewayClient.chat", new=AsyncMock(return_value=ChatResponsePayload(content='{"conversation_frame":"supportive"}'))) as llm_chat:
        result = asyncio.run(
            call_step_services(
                bus=MagicMock(),
                source=source,
                step=step,
                ctx=_base_ctx(),
                correlation_id=str(uuid4()),
            )
        )

    assert result.status == "success"
    sent_req = llm_chat.await_args.kwargs["req"]
    assert sent_req.route == "helper"


def test_chat_general_final_step_uses_chat_route() -> None:
    step = ExecutionStep(
        step_name="llm_chat_general",
        verb_name="chat_general",
        services=["LLMGatewayService"],
        order=1,
        prompt_template="{{ raw_user_text }}",
    )
    source = ServiceRef(name="test", node="test", version="1.0")

    with patch("app.executor.LLMGatewayClient.chat", new=AsyncMock(return_value=ChatResponsePayload(content="final"))) as llm_chat:
        result = asyncio.run(
            call_step_services(
                bus=MagicMock(),
                source=source,
                step=step,
                ctx=_base_ctx(),
                correlation_id=str(uuid4()),
            )
        )

    assert result.status == "success"
    sent_req = llm_chat.await_args.kwargs["req"]
    assert sent_req.route == "chat"
