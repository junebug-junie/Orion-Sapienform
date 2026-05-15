from __future__ import annotations

from pathlib import Path

import yaml


def test_chat_kids_story_plan_collects_metacog_then_llm() -> None:
    doc = yaml.safe_load(Path("orion/cognition/verbs/chat_kids_story.yaml").read_text(encoding="utf-8"))
    assert doc["name"] == "chat_kids_story"
    steps = sorted(doc["plan"], key=lambda s: s["order"])
    assert steps[0]["name"] == "collect_metacog_context"
    assert steps[1]["name"] == "llm_chat_kids_story"
    assert steps[1]["prompt_template"] == "chat_kids_story.j2"
    assert doc.get("recall_profile") == "chat.story.kids.v1"


def test_chat_kids_story_prompt_has_identity_placeholders() -> None:
    text = Path("orion/cognition/prompts/chat_kids_story.j2").read_text(encoding="utf-8")
    for needle in (
        "message_history",
        "memory_digest",
        "MULTI-LISTENER",
        "SCARY",
    ):
        assert needle in text, f"missing {needle!r} in chat_kids_story.j2"
