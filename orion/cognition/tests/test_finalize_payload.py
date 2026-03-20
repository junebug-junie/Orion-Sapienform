"""Pass 2: finalize_response receives trace + modes."""

from __future__ import annotations

from orion.cognition.finalize_payload import build_finalize_tool_input


def test_finalize_payload_contains_trace_and_modes():
    snap = [{"step_index": 0, "action": {"tool_id": "plan_action"}, "observation": {"llm_output": "Steps: 1. x"}}]
    d = build_finalize_tool_input(
        user_text="deploy to Discord",
        trace_snapshot=snap,
        output_mode="implementation_guide",
        response_profile="technical_delivery",
    )
    assert d["original_request"] == "deploy to Discord"
    assert "plan_action" in d["trace"]
    assert d["output_mode"] == "implementation_guide"
    assert d["response_profile"] == "technical_delivery"
