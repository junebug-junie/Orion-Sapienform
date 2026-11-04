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


def test_reflect_prompt_grades_answer_and_grounding_not_obedience() -> None:
    # Unified-turn spec A2: the judge stops grading obedience to stance's imperative.
    rendered = _render_reflect()
    assert rendered.index("Does draft_text respond to what user_message asked?") < rendered.index("thought_event (imperative")
    assert "Is what draft_text says grounded in what this turn actually saw" in rendered
    assert "Never call a draft misaligned or uncertain for departing from thought_event.imperative." in rendered
    assert "corrects the imperative from evidence" in rendered
    # The old obedience framing is gone.
    assert "compare draft_text against thought_event.imperative" not in rendered
    assert "felt stance" not in rendered


def test_reflect_prompt_does_not_excuse_skipped_verification() -> None:
    # Dropping obedience must not drop grounding: a draft that needed world contact and had none
    # still leans misaligned, keyed on the answer, not on what the imperative commanded.
    rendered = _render_reflect()
    assert ("When answering user_message needed world contact (current facts, a page, a file, a\n"
            "  camera) and grammar_receipts is empty, lean misaligned or uncertain.") in rendered


def test_reflect_prompt_world_contact_rule_keys_on_the_answer() -> None:
    rendered = _render_reflect().lower()
    assert "when imperative required world-contact" not in rendered
    assert "imperative commanded" not in rendered
    assert "when answering user_message needed world contact" in rendered
