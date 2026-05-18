from __future__ import annotations

from app.executor import _resolve_llm_chat_max_tokens, _resolve_llm_max_tokens
from app.router import (
    _extract_final_text,
    _should_fail_empty_structured_verb_output,
)
from orion.schemas.cortex.schemas import StepExecutionResult


def _step(payload: dict, *, step_name: str = "llm_memory_graph_suggest") -> StepExecutionResult:
    return StepExecutionResult(
        status="success",
        verb_name="memory_graph_suggest",
        step_name=step_name,
        order=0,
        result={"LLMGatewayService": payload},
        latency_ms=1,
        node="n",
        logs=[],
        error=None,
    )


def test_memory_graph_suggest_max_tokens_budget() -> None:
    from types import SimpleNamespace

    step = SimpleNamespace(verb_name="memory_graph_suggest", step_name="llm_memory_graph_suggest")
    settings = SimpleNamespace(llm_memory_graph_suggest_max_tokens=1536, llm_chat_max_tokens_default=512)
    eff, _req, src = _resolve_llm_chat_max_tokens(step, {})
    assert eff == 1536
    assert src == "settings.llm_memory_graph_suggest_max_tokens"
    eff2, src2, _ = _resolve_llm_max_tokens(ctx={}, step=step)
    assert eff2 == 1536
    assert src2 == "memory_graph_suggest_default"


def test_structured_verb_rejects_non_json_and_marks_empty() -> None:
    final_text, diag = _extract_final_text(
        [_step({"content": "Okay, I will extract a memory graph now."})],
        verb_name="memory_graph_suggest",
    )
    assert final_text == ""
    assert diag["structured_output_rejected"] is True
    assert diag["structured_rejection_preview"]


def test_structured_verb_accepts_json_in_content() -> None:
    draft = (
        '{"ontology_version":"orionmem-2026-05","utterance_ids":["u1"],'
        '"entities":[],"situations":[],"edges":[],"dispositions":[]}'
    )
    final_text, diag = _extract_final_text(
        [_step({"content": draft})],
        verb_name="memory_graph_suggest",
    )
    assert final_text == draft
    assert diag["structured_output_rejected"] is False


def test_should_fail_empty_structured_verb_output() -> None:
    diag = {"structured_output_rejected": True, "candidate_count": 1, "result_len": 0}
    assert _should_fail_empty_structured_verb_output(
        overall_status="success",
        verb_name="memory_graph_suggest",
        final_text="",
        final_text_diag=diag,
    )
