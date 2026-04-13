"""Pass 2: finalize_response receives trace + modes."""

from __future__ import annotations

from orion.cognition.finalize_payload import (
    answer_contract_expects_findings_rendering,
    build_finalize_tool_input,
)


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


def test_personal_contract_does_not_attach_findings_bundle_to_prompt_payload():
    """Empty synthetic bundles must not reach delivery verbs for coaching-shaped contracts."""
    fb = {
        "findings": [],
        "missing_evidence": ["repo"],
        "unsupported_requests": [],
        "next_checks": [],
        "grounded_status": "insufficient_grounding",
    }
    d = build_finalize_tool_input(
        user_text="how do I become a better version of myself?",
        trace_snapshot=[],
        output_mode="direct_answer",
        response_profile="direct_answer",
        findings_bundle=fb,
        answer_contract={
            "request_kind": "personal",
            "requires_repo_grounding": False,
            "requires_runtime_grounding": False,
        },
    )
    assert "findings_bundle" not in d
    assert "findings_bundle_json" not in d


def test_repo_contract_attaches_findings_bundle():
    fb = {"findings": [], "missing_evidence": [], "unsupported_requests": [], "next_checks": [], "grounded_status": "grounded_partial"}
    d = build_finalize_tool_input(
        user_text="fix import in services/foo",
        trace_snapshot=[],
        output_mode="implementation_guide",
        response_profile="technical_delivery",
        findings_bundle=fb,
        answer_contract={"request_kind": "repo_technical", "requires_repo_grounding": True},
    )
    assert "findings_bundle" in d


def test_answer_contract_expects_findings_rendering_heuristic():
    assert answer_contract_expects_findings_rendering({"request_kind": "repo_technical"}) is True
    assert answer_contract_expects_findings_rendering({"request_kind": "personal"}) is False
    assert (
        answer_contract_expects_findings_rendering(
            {"request_kind": "personal", "requires_repo_grounding": True}
        )
        is False
    )


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
