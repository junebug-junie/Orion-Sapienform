from __future__ import annotations

from app.router import _extract_final_text
from orion.schemas.cortex.schemas import StepExecutionResult


def _step(payload: dict) -> StepExecutionResult:
    return StepExecutionResult(
        status="success",
        verb_name="x",
        step_name="llm",
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
