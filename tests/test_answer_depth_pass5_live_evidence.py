"""Pass 5: Live evidence schema and runner output tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Evidence schema keys required by Pass 5
EVIDENCE_RUNTIME_KEYS = [
    "output_mode",
    "response_profile",
    "packs",
    "resolved_tool_ids",
    "triage_blocked_post_step0",
    "repeated_plan_action_escalation",
    "finalize_response_invoked",
    "quality_evaluator_rewrite",
]

PASS_CHECK_KEYS = [
    "real_orch_path_observed",
    "plannerreact_bus_observed",
    "llm_bus_observed",
    "output_mode_expected",
    "response_profile_expected",
    "delivery_pack_active",
    "delivery_verbs_visible",
    "triage_not_after_step0",
    "repeated_plan_action_not_shallow",
    "finalization_when_needed",
    "quality_evaluator_rewrite",
    "answer_quality_concrete",
]

DELIVERY_VERBS = {"write_guide", "finalize_response", "write_tutorial", "write_runbook"}


def test_evidence_schema_includes_runtime_keys():
    """Evidence schema must include all Pass 5 runtime fields."""
    from scripts.run_answer_depth_live_proof import _summarize_live_result

    raw = {
        "correlation_id": "test-corr",
        "request_channel": "orion:cortex:gateway:request",
        "entrypoint": "gateway",
        "reply_channel": "orion:cortex:gateway:result:test",
        "result_kind": "cortex.gateway.chat.result",
        "error": None,
        "request_payload": {"prompt": "Deploy to Discord"},
        "response_payload": {
            "cortex_result": {
                "status": "success",
                "mode": "agent",
                "verb": "agent_runtime",
                "final_text": "Discord deployment steps: token, Developer Portal, deploy.",
                "metadata": {
                    "answer_depth": {
                        "output_mode": "implementation_guide",
                        "response_profile": "technical_delivery",
                        "packs": ["executive_pack", "memory_pack", "delivery_pack"],
                        "resolved_tool_ids": ["plan_action", "write_guide", "finalize_response"],
                        "triage_blocked_post_step0": True,
                        "repeated_plan_action_escalation": True,
                        "finalize_response_invoked": True,
                        "quality_evaluator_rewrite": True,
                    }
                },
                "steps": [{"step_name": "agent_chain", "order": 100}],
            },
            "final_text": "Discord deployment steps: token, Developer Portal, deploy.",
        },
        "probe_events": [
            {"kind": "cortex.gateway.chat.request"},
            {"kind": "cortex.orch.request"},
            {"kind": "verb.request"},
            {"kind": "agent.planner.request"},
            {"kind": "agent.chain.request"},
            {"kind": "llm.chat.request"},
        ],
    }

    evidence = _summarize_live_result(raw, scenario_name="discord_deploy_live")

    for key in EVIDENCE_RUNTIME_KEYS:
        assert key in evidence, f"Evidence must include '{key}'"

    assert evidence["output_mode"] == "implementation_guide"
    assert evidence["response_profile"] == "technical_delivery"
    assert evidence["packs"] == ["executive_pack", "memory_pack", "delivery_pack"]
    assert "write_guide" in (evidence["resolved_tool_ids"] or [])
    assert evidence["triage_blocked_post_step0"] is True
    assert evidence["repeated_plan_action_escalation"] is True
    assert evidence["finalize_response_invoked"] is True
    assert evidence["quality_evaluator_rewrite"] is True


def test_evidence_schema_pass_checks_keys():
    """Pass checks must include quality_evaluator_rewrite and all expected keys."""
    from scripts.run_answer_depth_live_proof import _summarize_live_result

    raw = {
        "correlation_id": "test-corr",
        "request_channel": "orion:cortex:gateway:request",
        "entrypoint": "gateway",
        "reply_channel": "orion:cortex:gateway:result:test",
        "result_kind": "cortex.gateway.chat.result",
        "error": None,
        "request_payload": {"prompt": "Deploy to Discord"},
        "response_payload": {
            "cortex_result": {
                "status": "success",
                "mode": "agent",
                "verb": "agent_runtime",
                "final_text": "Discord: Developer Portal, token, OAuth, deploy, test.",
                "metadata": {"answer_depth": {"output_mode": "implementation_guide"}},
                "steps": [{"step_name": "agent_chain", "order": 100}],
            },
            "final_text": "Discord: Developer Portal, token, OAuth, deploy, test.",
        },
        "probe_events": [
            {"kind": "cortex.gateway.chat.request"},
            {"kind": "cortex.orch.request"},
            {"kind": "verb.request"},
            {"kind": "agent.planner.request"},
            {"kind": "agent.chain.request"},
            {"kind": "llm.chat.request"},
        ],
    }

    evidence = _summarize_live_result(raw, scenario_name="discord_deploy_live")
    pass_checks = evidence.get("pass_checks") or {}

    for key in PASS_CHECK_KEYS:
        assert key in pass_checks, f"pass_checks must include '{key}'"


def test_delivery_verbs_visible_when_in_resolved_tool_ids():
    """When resolved_tool_ids includes delivery verbs, delivery_verbs_visible should be True."""
    from scripts.run_answer_depth_live_proof import _summarize_live_result

    raw = {
        "correlation_id": "test-corr",
        "request_channel": "orion:cortex:gateway:request",
        "entrypoint": "gateway",
        "reply_channel": "orion:cortex:gateway:result:test",
        "result_kind": "cortex.gateway.chat.result",
        "error": None,
        "request_payload": {"prompt": "Deploy to Discord"},
        "response_payload": {
            "cortex_result": {
                "status": "success",
                "mode": "agent",
                "verb": "agent_runtime",
                "final_text": "Discord deployment: token, Developer Portal, deploy.",
                "metadata": {
                    "answer_depth": {
                        "output_mode": "implementation_guide",
                        "response_profile": "technical_delivery",
                        "packs": ["executive_pack", "delivery_pack"],
                        "resolved_tool_ids": [
                            "plan_action",
                            "write_guide",
                            "finalize_response",
                            "write_tutorial",
                        ],
                    }
                },
                "steps": [{"step_name": "agent_chain", "order": 100}],
            },
            "final_text": "Discord deployment: token, Developer Portal, deploy.",
        },
        "probe_events": [
            {"kind": "cortex.gateway.chat.request"},
            {"kind": "cortex.orch.request"},
            {"kind": "verb.request"},
            {"kind": "agent.planner.request"},
            {"kind": "agent.chain.request"},
            {"kind": "llm.chat.request"},
        ],
    }

    evidence = _summarize_live_result(raw, scenario_name="discord_deploy_live")

    assert evidence["pass_checks"]["delivery_verbs_visible"] is True
    resolved = evidence.get("resolved_tool_ids") or []
    assert any(v in resolved for v in DELIVERY_VERBS)


def test_runner_writes_evidence_files_schema(tmp_path):
    """When evidence is written, files must contain the required keys."""
    from scripts.run_answer_depth_live_proof import LIVE_PROOF_DIR, _summarize_live_result, _write_evidence

    raw = {
        "correlation_id": "test-corr",
        "request_channel": "orion:cortex:gateway:request",
        "entrypoint": "gateway",
        "reply_channel": "orion:cortex:gateway:result:test",
        "result_kind": "cortex.gateway.chat.result",
        "error": None,
        "request_payload": {"prompt": "Deploy to Discord"},
        "response_payload": {
            "cortex_result": {
                "status": "success",
                "mode": "agent",
                "verb": "agent_runtime",
                "final_text": "Discord: Developer Portal, token, deploy, test.",
                "metadata": {"answer_depth": {"output_mode": "implementation_guide"}},
                "steps": [{"step_name": "agent_chain", "order": 100}],
            },
            "final_text": "Discord: Developer Portal, token, deploy, test.",
        },
        "probe_events": [
            {"kind": "cortex.gateway.chat.request"},
            {"kind": "cortex.orch.request"},
            {"kind": "verb.request"},
            {"kind": "agent.planner.request"},
            {"kind": "agent.chain.request"},
            {"kind": "llm.chat.request"},
        ],
    }

    evidence = _summarize_live_result(raw, scenario_name="pass5_schema_test")

    # Write to a temp dir to avoid polluting real evidence
    import scripts.run_answer_depth_live_proof as live_proof_module
    original_dir = live_proof_module.LIVE_PROOF_DIR
    live_proof_module.LIVE_PROOF_DIR = tmp_path
    try:
        _write_evidence("pass5_schema_test", evidence)
        json_path = tmp_path / "pass5_schema_test.json"
        assert json_path.exists()
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        for key in EVIDENCE_RUNTIME_KEYS:
            assert key in loaded, f"Written evidence must include '{key}'"
        assert "quality_evaluator_rewrite" in loaded
    finally:
        live_proof_module.LIVE_PROOF_DIR = original_dir


def test_supervisor_quality_gate_with_test_verify_content():
    """Supervisor scenario with test/verify in answer should pass quality checks."""
    from scripts.run_answer_depth_live_proof import _quality_checks

    # Answer that includes test/verify (aligned with updated SUPERVISOR_PROMPT)
    text = (
        "Concise Discord deployment: 1) Discord Developer Portal, create app, add bot. "
        "2) Copy token. 3) OAuth2 invite, select permissions. 4) Install discord.py, configure. "
        "5) Deploy to server. 6) Verify the bot comes online and responds to commands."
    )
    quality = _quality_checks(text)
    assert quality["positive_pass"] is True
    assert quality["negative_pass"] is True
    assert quality["overall_pass"] is True
