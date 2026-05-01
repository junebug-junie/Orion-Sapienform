from __future__ import annotations

from pathlib import Path

from jinja2 import Environment


def test_chat_quick_prompt_contains_evidence_gated_claim_rails() -> None:
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

    assert "EVIDENCE-GATED CLAIMS" in rendered
    assert "Do not claim hidden side effects" in rendered
    assert "I can keep this in-thread for us here" in rendered
    assert "I do not have an automatic reminder/notification firing in this lane." in rendered
    assert "I've logged that and will ping you when it lands." in rendered
