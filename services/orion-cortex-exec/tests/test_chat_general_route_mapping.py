import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.executor import _resolve_llm_chat_max_tokens, _resolve_llm_max_tokens, call_step_services
from orion.core.bus.bus_schemas import ChatResponsePayload, ServiceRef
from orion.schemas.cortex.schemas import ExecutionStep


def _base_ctx(mode: str = "brain") -> dict:
    return {
        "mode": mode,
        "messages": [{"role": "user", "content": "hello"}],
        "raw_user_text": "hello",
        "session_id": "s-test",
    }


def test_chat_general_stance_step_uses_quick_route() -> None:
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
    assert sent_req.route == "quick"
    assert sent_req.options["max_tokens"] == 384


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
    assert sent_req.options["max_tokens"] == 768


def test_chat_quick_step_uses_quick_route() -> None:
    step = ExecutionStep(
        step_name="llm_chat_quick",
        verb_name="chat_quick",
        services=["LLMGatewayService"],
        order=0,
        prompt_template="{{ raw_user_text }}",
    )
    source = ServiceRef(name="test", node="test", version="1.0")

    with patch("app.executor.LLMGatewayClient.chat", new=AsyncMock(return_value=ChatResponsePayload(content="quick"))) as llm_chat:
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
    assert sent_req.options["max_tokens"] == 384


def test_introspect_spark_uses_quick_route() -> None:
    step = ExecutionStep(
        step_name="llm_introspect_spark",
        verb_name="introspect_spark",
        services=["LLMGatewayService"],
        order=0,
        prompt_template="{{ raw_user_text }}",
    )
    source = ServiceRef(name="test", node="test", version="1.0")

    with patch(
        "app.executor.LLMGatewayClient.chat",
        new=AsyncMock(return_value=ChatResponsePayload(content="spark introspection")),
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


def test_ctx_override_max_tokens_takes_precedence() -> None:
    step = ExecutionStep(
        step_name="llm_chat_general",
        verb_name="chat_general",
        services=["LLMGatewayService"],
        order=1,
        prompt_template="{{ raw_user_text }}",
    )
    source = ServiceRef(name="test", node="test", version="1.0")
    ctx = _base_ctx()
    ctx["max_tokens"] = 222
    with patch("app.executor.LLMGatewayClient.chat", new=AsyncMock(return_value=ChatResponsePayload(content="final"))) as llm_chat:
        result = asyncio.run(
            call_step_services(
                bus=MagicMock(),
                source=source,
                step=step,
                ctx=ctx,
                correlation_id=str(uuid4()),
            )
        )
    assert result.status == "success"
    sent_req = llm_chat.await_args.kwargs["req"]
    assert sent_req.options["max_tokens"] == 222


def test_dream_synthesis_uses_dream_max_tokens_budget(monkeypatch) -> None:
    """dream_cycle + dream_synthesis must not fall through to default chat completion cap."""
    import app.executor as executor_mod

    class _DreamSettings:
        llm_dream_max_tokens = 22222

    monkeypatch.setattr(executor_mod, "settings", _DreamSettings())
    step = ExecutionStep(
        step_name="dream_synthesis",
        verb_name="dream_cycle",
        services=["LLMGatewayService"],
        order=1,
        prompt_template="x",
    )
    eff_chat, _req, src_chat = _resolve_llm_chat_max_tokens(step, {})
    assert eff_chat == 22222
    assert src_chat == "settings.llm_dream_max_tokens"

    eff_max, src_max, _ = _resolve_llm_max_tokens(ctx={}, step=step)
    assert eff_max == 22222
    assert src_max == "dream_default"


def test_dream_synthesis_ctx_max_tokens_override(monkeypatch) -> None:
    import app.executor as executor_mod

    monkeypatch.setattr(executor_mod, "settings", type("S", (), {"llm_dream_max_tokens": 99999})())
    step = ExecutionStep(
        step_name="dream_synthesis",
        verb_name="dream_cycle",
        services=["LLMGatewayService"],
        order=1,
        prompt_template="x",
    )
    eff, req, src = _resolve_llm_chat_max_tokens(step, {"max_tokens": 50})
    assert eff == 50
    assert src == "ctx.max_tokens"
    assert req == 50


def test_journal_compose_draft_uses_general_lane_max_tokens(monkeypatch) -> None:
    import app.executor as executor_mod

    monkeypatch.setattr(
        executor_mod,
        "settings",
        type("S", (), {"llm_chat_general_max_tokens": 2048, "llm_chat_max_tokens_default": 512})(),
    )
    step = ExecutionStep(
        step_name="draft_journal_entry",
        verb_name="journal.compose",
        services=["LLMGatewayService"],
        order=0,
        prompt_template="x",
    )
    eff, req, src = _resolve_llm_chat_max_tokens(step, {})
    assert eff == 2048
    assert src == "settings.llm_chat_general_max_tokens_journal_compose"
    assert req is None
