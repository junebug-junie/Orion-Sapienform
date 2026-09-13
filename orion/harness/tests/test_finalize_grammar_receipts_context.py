from __future__ import annotations

from orion.harness.finalize import (
    build_finalize_reflect_context,
    build_response_repair_context,
)
from orion.harness.tests.fixtures import (
    make_appraisal,
    make_reflection,
    make_repair_overlay,
    make_thought,
)
from orion.schemas.harness_finalize import GrammarReceiptV1


def test_build_finalize_reflect_context_includes_grammar_receipts() -> None:
    thought = make_thought()
    receipts = [
        GrammarReceiptV1(step_index=0, tool_name="Read", summary="read coalition.py"),
    ]
    ctx = build_finalize_reflect_context(
        correlation_id="c-1",
        draft_text="draft",
        thought=thought,
        substrate_appraisal=make_appraisal(),
        repair_overlay=make_repair_overlay(),
        user_message="hello",
        grammar_receipts=receipts,
    )
    assert ctx["grammar_receipts"] == [
        {"step": "0", "tool": "Read", "summary": "read coalition.py"},
    ]


def test_build_response_repair_context_includes_grammar_receipts() -> None:
    receipts = [GrammarReceiptV1(step_index=1, tool_name="Grep", summary="grep coalition")]
    ctx = build_response_repair_context(
        correlation_id="c-1",
        draft_text="draft",
        reflection=make_reflection(),
        user_message="hello",
        grammar_receipts=receipts,
    )
    assert ctx["grammar_receipts"][0]["tool"] == "Grep"
