from __future__ import annotations

from pathlib import Path

import yaml


def test_chat_quick_plan_is_single_pass_llm_gateway() -> None:
    doc = yaml.safe_load(Path("orion/cognition/verbs/chat_quick.yaml").read_text(encoding="utf-8"))
    assert doc["name"] == "chat_quick"
    steps = doc.get("plan") or []
    assert len(steps) == 1
    assert steps[0]["name"] == "llm_chat_quick"
    assert steps[0]["services"] == ["LLMGatewayService"]
    assert steps[0]["prompt_template"] == "chat_quick.j2"


def test_chat_quick_prompt_has_identity_context_and_no_stance_dependency() -> None:
    prompt = Path("orion/cognition/prompts/chat_quick.j2").read_text(encoding="utf-8")
    assert "orion_identity_summary" in prompt
    assert "juniper_relationship_summary" in prompt
    assert "response_policy_summary" in prompt
    assert "Do not synthesize a formal stance brief." in prompt
    assert "Do not imitate the full chat_general stance-brief style." in prompt
    assert "chat_stance_brief" not in prompt
