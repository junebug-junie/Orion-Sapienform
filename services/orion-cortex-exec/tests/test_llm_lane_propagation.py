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
