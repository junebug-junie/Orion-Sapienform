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


def test_stance_react_uses_general_max_tokens_budget(monkeypatch) -> None:
    """Regression, 2026-08-20: stance_react was missing from this mapping too
    (the same incident that also found it missing from the llm_route mapping
    -- see _default_llm_route_for_step's docstring). Live-confirmed even
    after the route fix alone: corr=3cafe6db-36b4-476d-9e62-11f88c6c34b8
    still hit finish_reason=length/emitted_chars=0 at the 512-token default,
    truncating the strict-JSON ThoughtEventV1 payload mid-object, so
    orion-thought's extract_stance_react_payload() found nothing usable and
    the turn deferred. Same fix, same settings field as its four siblings."""
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
    step = _step(verb_name="stance_react", step_name="llm_stance_react")
    eff, _req, src = _resolve_llm_chat_max_tokens(step, {})
    assert eff == 8000
    assert src == "settings.llm_chat_general_max_tokens_stance_react"


def test_self_study_reflect_uses_general_max_tokens_budget(monkeypatch) -> None:
    """Regression, 2026-09-04: self_study.reflect (Layer 3 real reflection,
    PR #2080) was missing from this mapping too -- same disease as its six
    siblings above, discovered the same way: live-confirmed
    corr=871934fd-4095-55b6-82b0-a8784eda8998 hit payload_max_tokens=512
    (this function's own un-special-cased default), completion_tokens=512,
    finish_reason=length -- the agent-route reasoning model (Qwen3-27B)
    spent its entire budget on reasoning_content (2417 chars) and never
    wrote a single character of the strict-JSON findings answer, so
    self_study.py's own json.loads() found nothing to parse. Same fix, same
    settings field as its six siblings."""
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
    step = _step(verb_name="self_study.reflect", step_name="draft_self_study_reflection")
    eff, _req, src = _resolve_llm_chat_max_tokens(step, {})
    assert eff == 8000
    assert src == "settings.llm_chat_general_max_tokens_self_study_reflect"


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
