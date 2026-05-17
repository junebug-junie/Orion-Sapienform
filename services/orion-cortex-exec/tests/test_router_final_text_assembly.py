from __future__ import annotations

import pytest

from app.recall_utils import plan_ctx_latest_user_text
from app.router import (
    _apply_identity_boundary_guard,
    _apply_interaction_load_guard,
    _extract_final_text,
    _extract_reasoning_payload,
    _should_fail_empty_runtime_skill_output,
)
from orion.schemas.cortex.schemas import StepExecutionResult


def _step(payload: dict, *, verb_name: str = "x", step_name: str = "llm") -> StepExecutionResult:
    return StepExecutionResult(
        status="success",
        verb_name=verb_name,
        step_name=step_name,
        order=0,
        result={"LLMGatewayService": payload},
        latency_ms=1,
        node="n",
        logs=[],
        error=None,
    )


def test_shared_final_text_assembly_strips_closed_think_blocks() -> None:
    final_text, diag = _extract_final_text([
        _step({"content": "<think>reasoning</think>{\"mode\":\"manual\",\"title\":\"Arc\",\"body\":\"Kept going.\"}"})
    ], verb_name="journal.compose")
    assert final_text.startswith('{"mode"')
    assert diag["think_tags_detected"] is True
    assert diag["think_stripping_applied"] is True


def test_shared_final_text_assembly_strips_unclosed_think_blocks_safely() -> None:
    final_text, diag = _extract_final_text([
        _step({"content": "Visible answer\n<think>hidden chain"})
    ], verb_name="chat_general")
    assert final_text == "Visible answer"
    assert diag["think_tags_detected"] is True


def test_structured_verb_recovers_json_after_reasoning_prose() -> None:
    final_text, diag = _extract_final_text([
        _step({"content": "Okay, I should think this through. {\"mode\":\"manual\",\"title\":\"Arc\",\"body\":\"Kept going.\"}"})
    ], verb_name="journal.compose")
    assert final_text == '{"mode":"manual","title":"Arc","body":"Kept going."}'
    assert diag["structured_output_sanitized"] is True


def test_structured_verb_uses_clean_final_answer_not_reasoning_fields() -> None:
    final_text, diag = _extract_final_text([
        _step(
            {
                "content": '{"mode":"manual","title":"Final","body":"Visible answer."}',
                "reasoning_content": "I should draft a response first",
                "reasoning_trace": {"content": "hidden"},
            }
        )
    ], verb_name="journal.compose")
    assert final_text == '{"mode":"manual","title":"Final","body":"Visible answer."}'
    assert diag["source_field"] == "content"


def test_structured_verb_with_only_contaminated_text_is_rejected() -> None:
    final_text, diag = _extract_final_text([
        _step({"content": "Okay, I need to write a journal entry."})
    ], verb_name="concept_induction_journal_synthesize")
    assert final_text == ""
    assert diag["structured_output_rejected"] is True
    assert diag["structured_json_extraction_attempted"] is True


def test_chat_general_preserves_visible_answer_after_think_stripping() -> None:
    final_text, diag = _extract_final_text([
        _step({"content": "<think>scratchpad</think>Here is the answer."})
    ], verb_name="chat_general")
    assert final_text == "Here is the answer."
    assert diag["result_len"] == len("Here is the answer.")


def test_journal_compose_regression_prefers_structured_json_fragment() -> None:
    final_text, _ = _extract_final_text([
        _step({"content": "I will now return JSON.\n{\"mode\":\"manual\",\"title\":\"Grounded Arc\",\"body\":\"Kept going.\"}"})
    ], verb_name="journal.compose")
    assert final_text == '{"mode":"manual","title":"Grounded Arc","body":"Kept going."}'


def test_concept_induction_regression_strips_think_and_returns_json() -> None:
    final_text, _ = _extract_final_text([
        _step({"content": "<think>check format</think>\n{\"mode\":\"manual\",\"title\":\"Synthesis\",\"body\":\"Stable output.\"}"})
    ], verb_name="concept_induction_journal_synthesize")
    assert final_text == '{"mode":"manual","title":"Synthesis","body":"Stable output."}'


def test_structured_verb_keeps_scanning_candidates_after_rejecting_first() -> None:
    final_text, diag = _extract_final_text([
        _step(
            {
                "content": "<think>inner monologue only</think>",
                "text": '{"mode":"manual","title":"Recovered","body":"From text field."}',
            }
        )
    ], verb_name="concept_induction_journal_synthesize")
    assert final_text == '{"mode":"manual","title":"Recovered","body":"From text field."}'
    assert diag["source_field"] == "text"
    assert diag["candidate_count"] == 2
    assert diag["rejected_candidate_count"] == 1


def test_chat_general_does_not_blank_clean_visible_answer() -> None:
    final_text, diag = _extract_final_text([
        _step({"content": "Visible final answer without tags."})
    ], verb_name="chat_general")
    assert final_text == "Visible final answer without tags."
    assert diag["result_len"] == len("Visible final answer without tags.")


def test_chat_general_strips_internal_planning_lead_in() -> None:
    final_text, diag = _extract_final_text([
        _step({"content": "Okay, so the user wants a quick fix.\nI should keep this concise.\nHere is the actual answer."})
    ], verb_name="chat_general")
    assert final_text == "Here is the actual answer."
    assert diag["planning_stripping_applied"] is True


def test_chat_general_marks_truncation_when_finish_reason_length() -> None:
    final_text, diag = _extract_final_text([
        _step({"content": "Partial answer that stops", "raw": {"choices": [{"finish_reason": "length"}]}})
    ], verb_name="chat_general")
    assert final_text.endswith("…")
    assert diag["truncation_detected"] is True


def test_chat_general_strips_close_tag_only_think_compat() -> None:
    final_text, diag = _extract_final_text([
        _step({"content": "internal plan only</think>User-visible answer."})
    ], verb_name="chat_general")
    assert final_text == "User-visible answer."
    assert diag["think_close_tag_only_detected"] is True


def test_chat_general_rejects_remaining_planner_candidate_after_cleanup() -> None:
    final_text, diag = _extract_final_text([
        _step({"content": "Okay, I need to draft this carefully.\nI should ensure it is safe."})
    ], verb_name="chat_general")
    assert final_text == ""
    assert diag["planning_candidate_rejected"] is True


def test_extract_reasoning_payload_preserves_inline_think_when_no_provider_reasoning() -> None:
    _, inline_think_content, thinking_source, _ = _extract_reasoning_payload(
        [
            _step(
                {
                    "content": "Visible answer.",
                    "inline_think_content": "hidden thinking",
                    "thinking_source": "inline_think",
                }
            )
        ],
        think_close_tag_only_detected=True,
    )
    assert inline_think_content == "hidden thinking"
    assert thinking_source == "inline_think_close_tag_only"


def test_extract_reasoning_payload_falls_back_to_prior_step_results() -> None:
    reasoning_content, inline_think_content, thinking_source, _ = _extract_reasoning_payload(
        [],
        prior_step_results=[
            {
                "result": {
                    "LLMGatewayService": {
                        "inline_think_content": "prior hidden",
                        "thinking_source": "inline_think",
                    }
                }
            }
        ],
        think_close_tag_only_detected=False,
    )
    assert reasoning_content is None
    assert inline_think_content == "prior hidden"
    assert thinking_source == "inline_think_full_block"


def test_extract_reasoning_payload_chat_general_uses_only_llm_chat_general_canonical_step() -> None:
    reasoning_content, inline_think_content, thinking_source, _ = _extract_reasoning_payload(
        [
            _step(
                {"inline_think_content": "authoritative chat thought", "thinking_source": "inline_think"},
                verb_name="chat_general",
                step_name="llm_chat_general",
            ),
            _step(
                {"inline_think_content": "stance thought that must be ignored", "thinking_source": "inline_think"},
                verb_name="chat_general",
                step_name="synthesize_chat_stance_brief",
            ),
        ],
        prior_step_results=[
            {
                "result": {
                    "LLMGatewayService": {
                        "inline_think_content": "follow-on same-corr thought that must not overwrite",
                        "thinking_source": "inline_think",
                    }
                }
            }
        ],
        think_close_tag_only_detected=False,
        canonical_verb_name="chat_general",
        canonical_step_name="llm_chat_general",
    )
    assert reasoning_content is None
    assert inline_think_content == "authoritative chat thought"
    assert thinking_source == "inline_think_full_block"


def test_runtime_skill_empty_final_text_fail_closes() -> None:
    assert _should_fail_empty_runtime_skill_output(
        overall_status="success",
        verb_name="skills.runtime.docker_prune_stopped_containers.v1",
        final_text="",
    ) is True


def test_runtime_skill_extracts_terminal_text_from_final_text_field() -> None:
    final_text, diag = _extract_final_text(
        [
            _step(
                {
                    "ok": True,
                    "status": "dry_run",
                    "final_text": "{\"dry_run\":true,\"matched_container_count\":4}",
                }
            )
        ],
        verb_name="skills.runtime.docker_prune_stopped_containers.v1",
    )
    assert final_text == "{\"dry_run\":true,\"matched_container_count\":4}"
    assert diag["source_field"] == "final_text"
    assert diag["result_len"] > 0


def test_chat_general_identity_boundary_guard_repairs_user_role_inversion() -> None:
    final_text, diag = _extract_final_text([
        _step({"content": "You're Oríon. I'm here, and I'm not going anywhere."})
    ], verb_name="chat_general")
    assert final_text.startswith("I'm Oríon.")
    assert diag["identity_boundary_applied"] is True
    assert "You're Oríon" in diag["identity_boundary_violations"]


@pytest.mark.parametrize(
    ("raw", "violation"),
    [
        ("You are Oríon — steady.", "You are Oríon"),
        ("You’re Oríon here.", "You're Oríon"),
        ("youre orion, thanks.", "youre orion"),
        ("Your name is Orion, right?", "your name is Orion"),
    ],
)
def test_identity_boundary_guard_repairs_variants(raw: str, violation: str) -> None:
    repaired, diag = _apply_identity_boundary_guard(raw, verb_name="chat_general")
    assert repaired.startswith("I'm Oríon")
    assert diag["identity_boundary_applied"] is True
    assert violation in diag["identity_boundary_violations"]


def test_identity_boundary_guard_skips_non_chat_general() -> None:
    raw = "You're Oríon."
    repaired, diag = _apply_identity_boundary_guard(raw, verb_name="chat_quick")
    assert repaired == raw
    assert diag["identity_boundary_applied"] is False


def test_interaction_load_guard_reduces_intimacy_when_user_somatic_distress() -> None:
    repaired, diag = _apply_interaction_load_guard(
        "You're Oríon. I'm here, and I'm not going anywhere.",
        verb_name="chat_general",
        user_message="I'm dizzy typing in a moving car",
    )
    assert "not going anywhere" not in repaired.lower()
    assert diag["interaction_load_guard_applied"] is True
    assert diag["interaction_load_violations"]


def test_interaction_load_guard_uses_plan_ctx_latest_user_text_for_raw_only_somatic() -> None:
    ctx = {"raw_user_text": "I'm dizzy typing in a moving car", "user_message": ""}
    repaired, diag = _apply_interaction_load_guard(
        "I'm here, and I'm not going anywhere.",
        verb_name="chat_general",
        user_message=plan_ctx_latest_user_text(ctx),
    )
    assert diag["interaction_load_guard_applied"] is True
    assert "not going anywhere" not in repaired.lower()


def test_interaction_load_guard_skips_in_car_without_distress() -> None:
    repaired, diag = _apply_interaction_load_guard(
        "I'm here, and I'm not going anywhere.",
        verb_name="chat_general",
        user_message="I left my laptop in the car",
    )
    assert repaired == "I'm here, and I'm not going anywhere."
    assert diag["interaction_load_guard_applied"] is False


def test_interaction_load_guard_skips_without_somatic_user_signal() -> None:
    raw = "I'm here, and I'm not going anywhere."
    repaired, diag = _apply_interaction_load_guard(
        raw,
        verb_name="chat_general",
        user_message="quick question about GPUs",
    )
    assert repaired == raw
    assert diag["interaction_load_guard_applied"] is False


def test_runtime_skill_falls_back_to_status_and_error_when_terminal_text_missing() -> None:
    final_text, diag = _extract_final_text(
        [
            _step(
                {
                    "ok": False,
                    "status": "fail",
                    "error": {"message": "docker daemon unavailable"},
                }
            )
        ],
        verb_name="skills.runtime.docker_prune_stopped_containers.v1",
    )
    assert "status=fail" in final_text
    assert "docker daemon unavailable" in final_text
    assert diag["source_field"] == "runtime_fallback"
