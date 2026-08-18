from __future__ import annotations

from types import SimpleNamespace

from app.llm_lane import resolve_llm_lane_for_step


def _settings(**kwargs: object) -> object:
    d = {"exec_lane": "legacy"}
    d.update(kwargs)
    return SimpleNamespace(**d)


def test_chat_general_chat_lane() -> None:
    step = SimpleNamespace(verb_name="chat_general", step_name="llm_chat_general")
    out = resolve_llm_lane_for_step(step=step, ctx={"execution_lane": "chat"}, settings=_settings(exec_lane="chat"))
    assert out["llm_lane"] == "chat"
    assert out["allow_chat_fallback"] is True
    assert out["priority"] == "high"


def test_introspect_spark_lane() -> None:
    step = SimpleNamespace(verb_name="introspect_spark", step_name="llm_introspect_spark")
    out = resolve_llm_lane_for_step(step=step, ctx={}, settings=_settings())
    assert out["llm_lane"] == "spark"
    assert out["allow_chat_fallback"] is False


def test_explicit_options_llm_lane_wins() -> None:
    step = SimpleNamespace(verb_name="chat_quick", step_name="llm_chat_quick")
    out = resolve_llm_lane_for_step(
        step=step,
        ctx={"options": {"llm_lane": "background"}},
        settings=_settings(),
    )
    assert out["llm_lane"] == "background"


def test_dream_verb_background_lane() -> None:
    step = SimpleNamespace(verb_name="dream_cycle", step_name="x")
    out = resolve_llm_lane_for_step(step=step, ctx={}, settings=_settings())
    assert out["llm_lane"] == "background"


def test_reverie_narrate_verb_background_lane() -> None:
    step = SimpleNamespace(verb_name="reverie_narrate", step_name="llm_reverie_narrate")
    out = resolve_llm_lane_for_step(step=step, ctx={}, settings=_settings())
    assert out["llm_lane"] == "background"
    assert out["allow_chat_fallback"] is False


def test_spark_lane_allow_chat_fallback_from_options() -> None:
    step = SimpleNamespace(verb_name="introspect_spark", step_name="llm_introspect_spark")
    out = resolve_llm_lane_for_step(
        step=step,
        ctx={"options": {"allow_chat_fallback": True}},
        settings=_settings(),
    )
    assert out["llm_lane"] == "spark"
    assert out["allow_chat_fallback"] is True


def test_spark_lane_allow_chat_fallback_false_explicit() -> None:
    step = SimpleNamespace(verb_name="introspect_spark", step_name="llm_introspect_spark")
    out = resolve_llm_lane_for_step(
        step=step,
        ctx={"options": {"allow_chat_fallback": False}},
        settings=_settings(),
    )
    assert out["allow_chat_fallback"] is False


def test_chat_lane_allow_chat_fallback_can_be_false() -> None:
    step = SimpleNamespace(verb_name="chat_general", step_name="llm_chat_general")
    out = resolve_llm_lane_for_step(
        step=step,
        ctx={"execution_lane": "chat", "options": {"allow_chat_fallback": False}},
        settings=_settings(exec_lane="chat"),
    )
    assert out["llm_lane"] == "chat"
    assert out["allow_chat_fallback"] is False


def test_finalize_reflect_ctx_llm_lane_resolves_agent() -> None:
    """Mirror the finalize_reflect context (top-level llm_lane) as cortex-exec merges it
    into ctx; the 5b reflection must resolve to the agent lane (circe-worker-agent-1),
    not chat or background.

    Was `background` until confirmed wrong live 2026-08-16
    (corr=d9c3a9fc-0bc3-4e42-86cc-622613dfedbd): 5c's own orion_voice_finalize call also
    resolves to `background`/atlas-worker-2 and can occupy it for 90s+, starving this
    call's LLMGatewayService RPC entirely. `chat` was considered and rejected: it maps
    to circe-worker-1, the same worker chat_general's own live draft generation uses,
    with no admission/concurrency throttling on that route -- would trade the 5b-vs-5c
    collision for 5b-vs-live-user-chat contention. `agent` (circe-worker-agent-1,
    verified live) is currently unused by any other verb, isolating this call from both.
    """
    step = SimpleNamespace(verb_name="harness_finalize_reflect", step_name="llm_harness_finalize_reflect")
    out = resolve_llm_lane_for_step(
        step=step,
        ctx={"llm_lane": "agent", "allow_chat_fallback": True, "metadata": {"mode": "brain"}},
        settings=_settings(),
    )
    assert out["llm_lane"] == "agent"
    assert out["allow_chat_fallback"] is True
