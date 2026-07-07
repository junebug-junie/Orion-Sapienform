"""Option B guard: finalize contexts surface a deterministic tool-execution digest.

This is the anti-confabulation seam. When grammar receipts show a tool call was
executed, both the reflect (5b) and voice (5c) prompts receive an explicit
tool_execution digest so the LLM cannot claim "no tool call succeeded".
"""

from __future__ import annotations

from orion.harness.finalize import (
    build_finalize_reflect_context,
    build_voice_finalize_context,
    format_tool_execution_digest,
    tools_called_this_turn,
)
from orion.harness.tests.fixtures import (
    make_appraisal,
    make_reflection,
    make_repair_overlay,
    make_thought,
)
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.harness_finalize import GrammarReceiptV1


def _receipts() -> list[GrammarReceiptV1]:
    return [
        GrammarReceiptV1(step_index=0, tool_name=None, summary="[0] system init", grammar_event_id="g0"),
        GrammarReceiptV1(
            step_index=1,
            tool_name="mcp__github__list_pull_requests",
            summary="[1] assistant: tool_use mcp__github__list_pull_requests(owner=junebug-junie)",
            grammar_event_id="g1",
        ),
        GrammarReceiptV1(
            step_index=2,
            tool_name=None,
            summary='[2] user: tool_result (72 chars): [{"number":854,"title":"fix(action-outcome)"}]',
            grammar_event_id="g2",
        ),
    ]


def test_tools_called_this_turn_lists_only_tool_steps() -> None:
    calls = tools_called_this_turn(_receipts())
    assert calls == [{"step": "1", "tool": "mcp__github__list_pull_requests"}]


def test_format_tool_execution_digest_lists_calls() -> None:
    digest = format_tool_execution_digest(_receipts())
    assert "mcp__github__list_pull_requests" in digest
    assert "step 1" in digest


def test_format_tool_execution_digest_none_when_no_tools() -> None:
    digest = format_tool_execution_digest(
        [GrammarReceiptV1(step_index=0, tool_name=None, summary="s", grammar_event_id="g")]
    )
    assert "none" in digest.lower()


def test_reflect_context_includes_tool_execution() -> None:
    ctx = build_finalize_reflect_context(
        correlation_id="c-1",
        draft_text="fix(action-outcome): reconcile index ownership",
        thought=make_thought(),
        substrate_appraisal=make_appraisal(),
        repair_overlay=make_repair_overlay(),
        user_message="grab the latest pr title; skip bash",
        grammar_receipts=_receipts(),
    )
    assert "tool_execution" in ctx
    assert "mcp__github__list_pull_requests" in ctx["tool_execution"]


def test_voice_context_includes_tool_execution() -> None:
    thought = make_thought()
    ctx = build_voice_finalize_context(
        correlation_id="c-1",
        draft_text="fix(action-outcome): reconcile index ownership",
        thought=thought,
        substrate_appraisal=make_appraisal(),
        reflection=make_reflection(alignment_verdict="misaligned"),
        stance_harness_slice=thought.stance_harness_slice,
        voice_contract=AnswerContract(),
        repair_overlay=make_repair_overlay(),
        user_message="grab the latest pr title; skip bash",
        grammar_receipts=_receipts(),
    )
    assert "tool_execution" in ctx
    assert "mcp__github__list_pull_requests" in ctx["tool_execution"]
