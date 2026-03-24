from __future__ import annotations

from pathlib import Path

from jinja2 import Environment


def test_chat_general_prompt_contains_identity_and_behavior_rails() -> None:
    template = Environment().from_string(
        Path("orion/cognition/prompts/chat_general.j2").read_text(encoding="utf-8")
    )

    rendered = template.render(
        orion_identity_summary=["Oríon is not a generic one-off assistant."],
        juniper_relationship_summary=["co-architect", "steward"],
        response_policy_summary=["Answer the actual question or invitation first."],
        memory_digest="",
    )

    assert "Default to answering directly." in rendered
    assert "answer in first person as Oríon" in rendered
    assert "Do not say \"It sounds like...\"" in rendered
    assert "co-architect" in rendered
    assert "steward" in rendered
