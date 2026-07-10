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


def test_harness_finalize_reflect_uses_chat_route() -> None:
    """Fat finalize prompts exceed metacog's 4k/slot; chat lane has 131k and
    runs after the FCC motor on the same turn (sequential, one Hub FCC at a time)."""
    step = ExecutionStep(
        step_name="llm_harness_finalize_reflect",
        verb_name="harness_finalize_reflect",
        services=["LLMGatewayService"],
        order=0,
        prompt_template="{{ raw_user_text }}",
    )
    source = ServiceRef(name="test", node="test", version="1.0")

    with patch(
        "app.executor.LLMGatewayClient.chat",
        new=AsyncMock(return_value=ChatResponsePayload(content='{"verdict":"aligned"}')),
    ) as llm_chat:
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


def test_orion_voice_finalize_uses_quick_route() -> None:
    step = ExecutionStep(
        step_name="llm_orion_voice_finalize",
        verb_name="orion_voice_finalize",
        services=["LLMGatewayService"],
        order=0,
        prompt_template="{{ raw_user_text }}",
    )
    source = ServiceRef(name="test", node="test", version="1.0")

    with patch(
        "app.executor.LLMGatewayClient.chat",
        new=AsyncMock(return_value=ChatResponsePayload(content="final voice")),
    ) as llm_chat:
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
    assert sent_req.route == "quick"
