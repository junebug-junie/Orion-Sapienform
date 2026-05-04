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
    assert "memory_digest and recall snippets may include stale" in rendered
    assert "Stay in first person as Oríon" in rendered
    assert "Do not say \"It sounds like...\"" in rendered
    assert "co-architect" in rendered
    assert "steward" in rendered
    assert "When the user asks about memory, recall" in rendered
    assert "EVIDENCE-GATED CLAIMS" in rendered
    assert "Do not claim hidden side effects" in rendered
    assert "I can help track this in our conversation right now" in rendered
    assert "I do not have an automatic alert/notification running here." in rendered
    assert "I logged it in the thread and I'll surface it immediately." in rendered


def test_chat_general_prompt_menu_topic_selection_disallows_fabricated_tools_and_links() -> None:
    template = Environment().from_string(
        Path("orion/cognition/prompts/chat_general.j2").read_text(encoding="utf-8")
    )
    rendered = template.render(
        user_message="hm mesh continuity",
        message_history=[],
        memory_digest="",
        orion_identity_summary=[],
        juniper_relationship_summary=[],
        response_policy_summary=[],
        chat_stance_brief="",
        menu_topic_selection={"enabled": True, "matched_options": ["mesh continuity"]},
    )

    assert "Do not invent tools/tool IDs" in rendered
    assert "do not include documentation URLs" in rendered
