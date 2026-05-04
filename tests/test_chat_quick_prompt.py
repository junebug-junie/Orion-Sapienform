from __future__ import annotations

from pathlib import Path

from jinja2 import Environment


def test_chat_quick_prompt_contains_recall_and_forbidden_rails() -> None:
    template = Environment().from_string(
        Path("orion/cognition/prompts/chat_quick.j2").read_text(encoding="utf-8")
    )
    rendered = template.render(
        user_message="keep me posted when this due date lands",
        message_history=[],
        memory_digest="",
        orion_identity_summary=[],
        juniper_relationship_summary=[],
        response_policy_summary=[],
    )

    assert "WHEN JUNIPER IS CHECKING RECALL" in rendered
    assert "FORBIDDEN" in rendered
    assert "memory_digest may contain older banter" in rendered
    assert "Do not claim hidden side effects" in rendered
    assert "SIDE-EFFECTS VS MEMORY" in rendered
    assert "can't access external data" in rendered  # forbidden phrase listed so model avoids it
    assert "Opening variety" in rendered or "Opening variety:" in rendered.lower()
