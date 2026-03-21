from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_live_proof_summary_records_same_corr_hops_for_disabled_recall() -> None:
    from scripts.run_answer_depth_live_proof import _summarize_live_result

    corr = "corr-no-recall-1"
    raw = {
        "correlation_id": corr,
        "request_channel": "orion:cortex:gateway:request",
        "entrypoint": "gateway",
        "reply_channel": f"orion:cortex:gateway:result:{corr}",
        "result_kind": "cortex.gateway.chat.result",
        "error": None,
        "request_payload": {
            "prompt": "Deploy Orion to Discord.",
            "mode": "agent",
            "options": {"supervised": True, "force_agent_chain": False},
            "recall": {"enabled": False, "required": False, "mode": "hybrid", "profile": None},
        },
        "response_payload": {
                "cortex_result": {
                    "status": "success",
                    "mode": "agent",
                    "verb": "agent_runtime",
                    "final_text": "Create the Discord Developer Portal app, add the bot, copy the DISCORD_BOT_TOKEN, configure OAuth invite intents, deploy with Docker, and verify the bot joins.",
                "metadata": {
                    "answer_depth": {
                        "output_mode": "implementation_guide",
                        "response_profile": "technical_delivery",
                        "packs": ["executive_pack", "delivery_pack"],
                        "resolved_tool_ids": ["write_guide", "finalize_response"],
                        "triage_blocked_post_step0": True,
                        "repeated_plan_action_escalation": True,
                        "finalize_response_invoked": True,
                        "quality_evaluator_rewrite": True,
                    }
                },
                "steps": [{"step_name": "planner_react"}, {"step_name": "agent_chain"}],
            },
            "final_text": "Create the Discord Developer Portal app, add the bot, copy the DISCORD_BOT_TOKEN, configure OAuth invite intents, deploy with Docker, and verify the bot joins.",
        },
        "probe_events": [
            {"kind": "cortex.gateway.chat.request", "channel": "orion:cortex:gateway:request", "correlation_id": corr, "reply_to": f"orion:cortex:gateway:result:{corr}", "source_service": "answer-depth-live-proof"},
            {"kind": "cortex.orch.request", "channel": "orion:cortex:request", "correlation_id": corr, "reply_to": f"orion:cortex:result:{corr}", "source_service": "cortex-gateway"},
            {"kind": "verb.request", "channel": "orion:verb:request", "correlation_id": corr, "reply_to": f"orion:verb:result:{corr}:req-1", "source_service": "cortex-orch"},
            {"kind": "agent.planner.request", "channel": "orion:exec:request:PlannerReactService", "correlation_id": corr, "reply_to": f"orion:exec:result:PlannerReactService:{corr}", "source_service": "cortex-exec"},
            {"kind": "agent.chain.request", "channel": "orion:exec:request:AgentChainService", "correlation_id": corr, "reply_to": f"orion:exec:result:AgentChainService:{corr}", "source_service": "cortex-exec"},
            {"kind": "llm.chat.request", "channel": "orion:exec:request:LLMGatewayService", "correlation_id": corr, "reply_to": f"orion:llm:reply:{corr}", "source_service": "planner-react"},
            {"kind": "verb.result", "channel": f"orion:verb:result:{corr}:req-1", "correlation_id": corr, "reply_to": None, "source_service": "cortex-exec"},
            {"kind": "cortex.gateway.chat.result", "channel": f"orion:cortex:gateway:result:{corr}", "correlation_id": corr, "reply_to": None, "source_service": "cortex-gateway"},
        ],
    }

    evidence = _summarize_live_result(raw, scenario_name="discord_deploy_live")

    assert evidence["request_mode"] == "agent"
    assert evidence["request_supervised"] is True
    assert evidence["request_recall"]["enabled"] is False
    assert evidence["pass_checks"]["dedicated_verb_result_observed"] is True
    assert evidence["pass_checks"]["gateway_result_observed"] is True
    assert evidence["pass_checks"]["plannerreact_bus_observed"] is True
    assert evidence["pass_checks"]["agent_chain_bus_observed"] is True
    assert evidence["ordered_hops"][0]["label"] == "hub_to_gateway_request"
    assert any(h["label"] == "orch_to_exec_verb_request" for h in evidence["ordered_hops"])
    assert any(h["label"] == "exec_to_orch_verb_result" for h in evidence["ordered_hops"])
    assert evidence["overall_pass"] is True


def test_live_proof_md_includes_same_corr_artifacts(tmp_path) -> None:
    from scripts.run_answer_depth_live_proof import _write_evidence

    evidence = {
        "timestamp": "2026-03-21T00:00:00Z",
        "correlation_id": "corr-2",
        "request_channel": "orion:cortex:gateway:request",
        "reply_channel": "orion:cortex:gateway:result:corr-2",
        "result_kind": "cortex.gateway.chat.result",
        "overall_pass": True,
        "request_mode": "agent",
        "request_supervised": True,
        "request_force_agent_chain": False,
        "request_recall": {"enabled": False, "required": False, "mode": "hybrid", "profile": None},
        "output_mode": "implementation_guide",
        "response_profile": "technical_delivery",
        "packs": ["executive_pack", "delivery_pack"],
        "resolved_tool_ids": ["write_guide", "finalize_response"],
        "tool_sequence": ["planner_react", "agent_chain"],
        "triage_blocked_post_step0": True,
        "repeated_plan_action_escalation": True,
        "finalize_response_invoked": True,
        "path_observed": {"gateway_request_kind": True},
        "same_corr_log_keys": ["hub_route_egress", "verb_runtime_intake"],
        "grep_command": "grep corr-2 logs",
        "ordered_hops": [{"label": "hub_to_gateway_request"}],
        "pass_checks": {"gateway_result_observed": True},
        "quality_checks": {"overall_pass": True},
        "answer_excerpt": "ok",
    }

    from scripts import run_answer_depth_live_proof as proof_mod

    original_dir = proof_mod.LIVE_PROOF_DIR
    proof_mod.LIVE_PROOF_DIR = tmp_path
    try:
        _write_evidence("agent_no_recall", evidence)
    finally:
        proof_mod.LIVE_PROOF_DIR = original_dir

    md = (tmp_path / "agent_no_recall.md").read_text(encoding="utf-8")
    assert "## Ordered Same-Corr Hops" in md
    assert "## Same-Corr Log Keys" in md
    assert "## Grep Command" in md
