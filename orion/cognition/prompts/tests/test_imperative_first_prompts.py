from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_response_repair_uses_grammar_not_contract_flags() -> None:
    text = (REPO_ROOT / "orion/cognition/prompts/orion_response_repair.j2").read_text(encoding="utf-8")
    assert "grammar_receipts" in text
    assert "requires_repo_grounding" not in text
    assert "smallest necessary correction" in text


def test_reflect_prompt_includes_grammar_receipts() -> None:
    text = (REPO_ROOT / "orion/cognition/prompts/harness_finalize_reflect.j2").read_text(encoding="utf-8")
    assert "grammar_receipts" in text


def _render_reflect() -> str:
    from jinja2 import Environment

    text = (REPO_ROOT / "orion/cognition/prompts/harness_finalize_reflect.j2").read_text(encoding="utf-8")
    return Environment().from_string(text).render(
        user_message="What have you read about graphics cards lately?",
        draft_text="I read three GPU pieces...",
        thought_event={"imperative": "Ground it in circe_gpu rendering."},
        substrate_appraisal={},
        grammar_receipts=[],
        tool_execution="",
        repair_overlay={},
    )


def test_reflect_prompt_judges_the_task_before_the_imperative() -> None:
    rendered = _render_reflect()
    assert rendered.index("The task is user_message") < rendered.index("thought_event.imperative")
    assert "not misaligned for that reason alone" in rendered


def test_reflect_prompt_does_not_excuse_skipped_verification() -> None:
    rendered = _render_reflect()
    excuse = rendered.index("not misaligned for that reason alone")
    verification = rendered.index(
        "Verification or world-contact the imperative commanded in order to answer "
        "user_message is part of how to answer, not an extra: skipping it can still be misaligned."
    )
    assert excuse < verification


def test_reflect_prompt_world_contact_rule_keys_on_the_task() -> None:
    rendered = _render_reflect()
    assert "when imperative required world-contact" not in rendered.lower()
    assert "when the task required world-contact" in rendered.lower()
