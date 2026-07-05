"""Agent-claude harness steps must land in chat history thought_process seam."""

from scripts.fcc_claude_bridge import (
    build_harness_reasoning_trace,
    summarize_harness_steps_for_history,
)


def test_summarize_harness_steps_for_history_includes_assistant_lines() -> None:
    steps = [
        {"type": "system", "raw": {"type": "system"}},
        {
            "type": "assistant",
            "raw": {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Checking repo layout."}]},
            },
        },
        {"type": "result", "raw": {"type": "result", "result": "done"}},
    ]
    text = summarize_harness_steps_for_history(steps)
    assert "[0] system" in text
    assert "assistant: Checking repo layout." in text
    assert "[2] result: done" in text


def test_build_harness_reasoning_trace_shape_for_sql_writer() -> None:
    trace = build_harness_reasoning_trace(
        steps=[{"type": "system", "raw": {"type": "system"}}],
        correlation_id="corr-1",
        session_id="sess-1",
        model_label="MODEL_HAIKU",
    )
    assert trace is not None
    assert trace["trace_role"] == "reasoning"
    assert trace["metadata"]["source"] == "agent_claude_harness"
    assert trace["content"] == "[0] system"


def test_build_harness_reasoning_trace_empty_when_no_steps() -> None:
    assert build_harness_reasoning_trace(steps=[], correlation_id="x") is None
