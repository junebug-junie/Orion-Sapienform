from __future__ import annotations

from pathlib import Path

from jinja2 import Environment


def _render(path: str, **ctx):
    env = Environment(autoescape=False)
    tpl = env.from_string(Path(path).read_text(encoding="utf-8"))
    return tpl.render(**ctx)


def _base_ctx():
    return {
        "user_message": "hello",
        "message_history": [],
        "memory_digest": "",
        "orion_identity_summary": "id",
        "juniper_relationship_summary": "rel",
        "response_policy_summary": "policy",
        "chat_stance_brief": {"task_mode": "direct_response"},
        "metadata": {},
    }


def test_prompts_render_without_situation_fragment():
    base = _base_ctx()
    quick = _render("orion/cognition/prompts/chat_quick.j2", **base)
    general = _render("orion/cognition/prompts/chat_general.j2", **base)
    # Conditional situation blocks should stay dark when fragment is unset.
    assert "Situation context is grounding context only" not in quick
    assert "do not force time/weather/location commentary" not in general


def test_prompts_render_with_situation_fragment_scenarios():
    base = _base_ctx()
    scenarios = [
        "solo same-breath",
        "child present",
        "long gap",
        "heading out with rain",
        "late-night risky hardware task",
    ]
    for scenario in scenarios:
        fragment = {"compact_text": f"Situation: {scenario}", "summary_lines": [scenario]}
        quick = _render("orion/cognition/prompts/chat_quick.j2", **{**base, "situation_prompt_fragment": fragment})
        general = _render("orion/cognition/prompts/chat_general.j2", **{**base, "situation_prompt_fragment": fragment})
        assert "situation_prompt_fragment" in quick
        assert "situation_prompt_fragment" in general
        assert "grounding context only" in quick
        assert "do not force" in general.lower()
