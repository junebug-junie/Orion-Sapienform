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
    assert d["trace_preferred_output"] == "Steps: 1. x"
    assert d["finalization_source_trace_used"] is True


def test_finalize_payload_adds_orion_grounding_for_discord_delivery():
    snap = [{"step_index": 0, "observation": {"llm_output": "Bridge Orion via Discord bot adapter"}}]
    d = build_finalize_tool_input(
        user_text="would you write up detailed developer instructions on how to build Orion (AI) into a Discord server?",
        trace_snapshot=snap,
        output_mode="implementation_guide",
        response_profile="technical_delivery",
    )

    assert d["delivery_grounding_mode"] == "orion_repo_architecture"
    assert "Hub/Client -> Cortex-Orch -> Cortex-Exec" in d["grounding_context"]
    assert "Do not silently substitute a random stack" in d["anti_generic_drift"]
