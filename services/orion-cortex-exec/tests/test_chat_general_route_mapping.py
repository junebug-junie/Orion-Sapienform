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


def test_chat_general_final_step_requests_logprobs_when_enabled(monkeypatch) -> None:
    """CORTEX_CHAT_RETURN_LOGPROBS on: the real user-facing chat=route reply asks the
    gateway for return_logprobs, riding the existing OpenAI-compat call (no
    logprob_probe_mode set, so no native-completion endpoint switch)."""
    import app.executor as executor_mod

    monkeypatch.setattr(executor_mod.settings, "cortex_chat_return_logprobs", True)
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
    assert sent_req.options["return_logprobs"] is True
    # No explicit top_k: the gateway already defaults logprobs_top_k to
    # settings.llm_logprob_top_k_default when the key is absent (llm_backend.py:914,
    # :1138) -- setting it here would just duplicate that default and drift silently
    # if the gateway's own default is ever retuned.
    assert "logprobs_top_k" not in sent_req.options
    assert "logprob_probe_mode" not in sent_req.options


def test_chat_general_final_step_no_logprobs_when_disabled(monkeypatch) -> None:
    """Default (CORTEX_CHAT_RETURN_LOGPROBS off): unchanged, no return_logprobs key."""
    import app.executor as executor_mod

    monkeypatch.setattr(executor_mod.settings, "cortex_chat_return_logprobs", False)
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
    assert not sent_req.options.get("return_logprobs")


def test_chat_general_stance_step_never_requests_logprobs_even_when_enabled(monkeypatch) -> None:
    """Narrow scope: route=quick steps (e.g. the stance-brief pass) never get
    return_logprobs, even with the flag on -- only the real route=chat reply does."""
    import app.executor as executor_mod

    monkeypatch.setattr(executor_mod.settings, "cortex_chat_return_logprobs", True)
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
    assert not sent_req.options.get("return_logprobs")


def test_chat_general_final_step_skips_logprobs_with_response_format(monkeypatch) -> None:
    """A JSON-constrained chat=route reply skips return_logprobs -- constrained decoding
    collapses logprob entropy on structured output, same reason mind's synthesis calls
    use native completion instead of this path rather than piggybacking on it here."""
    import app.executor as executor_mod

    monkeypatch.setattr(executor_mod.settings, "cortex_chat_return_logprobs", True)
    step = ExecutionStep(
        step_name="llm_chat_general",
        verb_name="chat_general",
        services=["LLMGatewayService"],
        order=1,
        prompt_template="{{ raw_user_text }}",
    )
    source = ServiceRef(name="test", node="test", version="1.0")
    ctx = _base_ctx()
    ctx["response_format"] = {"type": "json_object"}

    with patch("app.executor.LLMGatewayClient.chat", new=AsyncMock(return_value=ChatResponsePayload(content="{}"))) as llm_chat:
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
    assert not sent_req.options.get("return_logprobs")


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


def test_chat_kids_story_step_uses_quick_route() -> None:
    step = ExecutionStep(
        step_name="llm_chat_kids_story",
        verb_name="chat_kids_story",
        services=["LLMGatewayService"],
        order=1,
        prompt_template="{{ raw_user_text }}",
    )
    source = ServiceRef(name="test", node="test", version="1.0")

    with patch("app.executor.LLMGatewayClient.chat", new=AsyncMock(return_value=ChatResponsePayload(content="once"))) as llm_chat:
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
    assert sent_req.options.get("llm_lane") == "spark"


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
