from __future__ import annotations

from orion.cognition.answer_grounding import build_answer_grounding_context


def test_orion_question_without_discord_skips_adapter_paragraph() -> None:
    ctx = build_answer_grounding_context(
        user_text="How does Orion route from hub to exec?",
        output_mode="implementation_guide",
    )
    assert ctx["delivery_grounding_mode"] == "orion_repo_architecture"
    assert "Discord bot" not in (ctx.get("grounding_context") or "")
