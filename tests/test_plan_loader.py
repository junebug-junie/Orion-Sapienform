from __future__ import annotations

from orion.cognition.plan_loader import build_plan_for_verb


def test_build_plan_for_verb_loads_prompt_text():
    plan = build_plan_for_verb("chat_general", mode="brain")
    assert plan.verb_name == "chat_general"
    assert plan.steps
    assert isinstance(plan.steps[0].prompt_template, str)
    assert "chat_mode" in (plan.steps[0].prompt_template or "")
