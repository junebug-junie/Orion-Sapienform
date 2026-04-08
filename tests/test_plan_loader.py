from __future__ import annotations

from orion.cognition.plan_loader import build_plan_for_verb, load_verb_yaml


def test_build_plan_for_verb_loads_prompt_text():
    plan = build_plan_for_verb("chat_general", mode="brain")
    assert plan.verb_name == "chat_general"
    assert plan.steps
    assert all(isinstance(step.prompt_template, str) for step in plan.steps if step.prompt_template is not None)
    assert all((step.prompt_template or "").strip() for step in plan.steps)



def test_build_plan_for_verb_exposes_personality_file_metadata():
    plan = build_plan_for_verb("chat_general", mode="brain")
    assert plan.metadata.get("personality_file") == "orion/cognition/personality/orion_identity.yaml"


def test_chat_general_yaml_declares_personality_file():
    data = load_verb_yaml("chat_general")
    assert data.get("personality_file") == "orion/cognition/personality/orion_identity.yaml"
