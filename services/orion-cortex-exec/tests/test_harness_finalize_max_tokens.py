from __future__ import annotations

from types import SimpleNamespace

from app.executor import _resolve_llm_chat_max_tokens, _resolve_llm_max_tokens
from orion.schemas.cortex.types import ExecutionStep


def _step(*, verb_name: str, step_name: str) -> ExecutionStep:
    return ExecutionStep(
        verb_name=verb_name,
        step_name=step_name,
        order=0,
        services=["LLMGatewayService"],
        prompt_template="x.j2",
    )


def test_harness_finalize_reflect_uses_general_max_tokens_budget(monkeypatch) -> None:
    import app.executor as executor_mod

    monkeypatch.setattr(
        executor_mod,
        "settings",
        SimpleNamespace(
            llm_chat_general_max_tokens=8000,
            llm_chat_max_tokens_default=512,
            llm_dream_max_tokens=32768,
            llm_chat_quick_max_tokens=384,
            llm_memory_graph_suggest_max_tokens=4096,
        ),
    )
    step = _step(verb_name="harness_finalize_reflect", step_name="llm_harness_finalize_reflect")
    eff, _req, src = _resolve_llm_chat_max_tokens(step, {})
    assert eff == 8000
    assert src == "settings.llm_chat_general_max_tokens_harness_finalize"


def test_orion_voice_finalize_uses_general_max_tokens_budget(monkeypatch) -> None:
    import app.executor as executor_mod

    monkeypatch.setattr(
        executor_mod,
        "settings",
        SimpleNamespace(
            llm_chat_general_max_tokens=8000,
            llm_chat_max_tokens_default=512,
            llm_dream_max_tokens=32768,
            llm_chat_quick_max_tokens=384,
            llm_memory_graph_suggest_max_tokens=4096,
            llm_chat_fallback_max_tokens=512,
        ),
    )
    step = _step(verb_name="orion_voice_finalize", step_name="llm_orion_voice_finalize")
    eff, src, _ = _resolve_llm_max_tokens(ctx={}, step=step)
    assert eff == 8000
    assert src == "harness_finalize_default"
