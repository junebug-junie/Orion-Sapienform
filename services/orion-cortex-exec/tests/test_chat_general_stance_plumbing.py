from __future__ import annotations

from pathlib import Path

import yaml


def test_chat_general_plan_runs_stance_before_final() -> None:
    doc = yaml.safe_load(Path("orion/cognition/verbs/chat_general.yaml").read_text(encoding="utf-8"))
    steps = doc.get("plan") or []
    assert len(steps) >= 2
    assert steps[0]["name"] == "synthesize_chat_stance_brief"
    assert steps[0]["prompt_template"] == "chat_stance_brief.j2"
    assert steps[1]["name"] == "llm_chat_general"


def test_final_prompt_consumes_stance_brief_and_keeps_rails() -> None:
    prompt = Path("orion/cognition/prompts/chat_general.j2").read_text(encoding="utf-8")
    assert "chat_stance_brief" in prompt
    assert "Default to answering directly" in prompt
    assert "Do not use customer-support language" in prompt
    assert "Do not collapse into generic \"better chatbot\" goals." in prompt


def test_synthesis_prompt_is_structured_and_toolless() -> None:
    prompt = Path("orion/cognition/prompts/chat_stance_brief.j2").read_text(encoding="utf-8")
    assert "output only strict JSON" in prompt
    assert "not writing the user-facing answer" in prompt
    assert "Return JSON with exactly these keys" in prompt
