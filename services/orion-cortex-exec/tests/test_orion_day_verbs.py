"""Orion's Day verbs in cortex-exec: their own completion budgets, the agent lane by default,
and plain-text output (never the structured-output path)."""

from __future__ import annotations

from types import SimpleNamespace

from app.executor import _default_llm_route_for_step, _resolve_llm_chat_max_tokens
from app.router import _structured_output_expected
from app.settings import Settings
from orion.schemas.cortex.types import ExecutionStep


def _step(verb_name: str, step_name: str) -> ExecutionStep:
    return ExecutionStep(verb_name=verb_name, step_name=step_name, order=0,
                         services=["LLMGatewayService"], prompt_template="x.j2")


def _settings(**over):
    base = dict(llm_chat_general_max_tokens=8000, llm_chat_max_tokens_default=512, llm_dream_max_tokens=32768,
                llm_chat_quick_max_tokens=384, llm_memory_graph_suggest_max_tokens=4096,
                llm_orion_day_note_max_tokens=12000, llm_orion_day_carry_forward_max_tokens=4000)
    base.update(over)
    return SimpleNamespace(**base)


def test_note_and_carry_forward_have_their_own_budgets(monkeypatch):
    import app.executor as executor_mod

    monkeypatch.setattr(executor_mod, "settings", _settings(llm_orion_day_note_max_tokens=12345,
                                                            llm_orion_day_carry_forward_max_tokens=2345))
    eff, _, src = _resolve_llm_chat_max_tokens(_step("orion_day_note_v1", "draft_orion_day_note"), {})
    assert (eff, src) == (12345, "settings.llm_orion_day_note_max_tokens")
    eff, _, src = _resolve_llm_chat_max_tokens(_step("orion_day_carry_forward_v1", "draft_orion_day_carry_forward"), {})
    assert (eff, src) == (2345, "settings.llm_orion_day_carry_forward_max_tokens")


def test_settings_defaults_and_env_aliases(monkeypatch):
    monkeypatch.setenv("LLM_ORION_DAY_NOTE_MAX_TOKENS", "9000")
    s = Settings()
    assert s.llm_orion_day_note_max_tokens == 9000
    assert Settings.model_fields["llm_orion_day_carry_forward_max_tokens"].default == 4000
    assert Settings.model_fields["llm_orion_day_note_max_tokens"].default == 12000


def test_unstamped_default_route_is_agent_never_chat_or_quick():
    for verb in ("orion_day_note_v1", "orion_day_carry_forward_v1"):
        assert _default_llm_route_for_step(verb_name=verb, step_name="x", mode="brain") == "agent"


def test_both_verbs_stay_out_of_structured_output():
    assert not _structured_output_expected("orion_day_note_v1")
    assert not _structured_output_expected("orion_day_carry_forward_v1")
