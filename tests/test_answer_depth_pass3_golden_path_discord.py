from __future__ import annotations

import importlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from orion.schemas.agents.schemas import AgentChainRequest
from orion.schemas.cortex.schemas import ExecutionPlan, PlanExecutionArgs, PlanExecutionRequest
from orion.schemas.cortex.types import StepExecutionResult


PROMPT_DISCORD_DEPLOY = "Please provide instructions on how to deploy you onto Discord."
META_PLAN_TEXT = "Gather requirements. Create a guide. Review and refine."


DISCORD_GUIDE = """\
Discord Deployment Guide (Orion Discord Bot Bridge)

1) Create the Discord application
- Go to the Discord Developer Portal.
- Create a new Application and add a Bot.
- Enable the Bot and copy the Bot token.

2) Configure environment variables
- Set `DISCORD_BOT_TOKEN` in your runtime environment (never hard-code it).
- Example:
  - export DISCORD_BOT_TOKEN="your-token-here"

3) Choose Gateway intents + permissions
- Enable only the intents you need.
- Common starting point for bridges:
  - `Guilds`
  - `GuildMessages`
  - `MessageContent` (if applicable in your server settings)

4) Host the bot process
- Start the bridge process with a persistent runner (systemd, Docker, or a supervised container).
- Ensure the process restarts on failure.

5) Invite the bot to your server
- Generate the OAuth2 invite URL for the application.
- Use the required scopes (typically bot) and permissions.
- Copy the invite URL into your browser and authorize to the target server.

6) Test and troubleshoot
- Verify the bot comes online (watch logs for successful login).
- If messages are not received:
  - confirm bot has the right OAuth2 permissions
  - confirm intents are enabled in code and server settings
- Common token issues:
  - ensure `DISCORD_BOT_TOKEN` is present in the environment
"""


def _last_user_message(ctx: Dict[str, Any]) -> str:
    msgs = ctx.get("messages") or []
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content") or "")
    return str(ctx.get("user_message") or "")


def _import_service_app(
    service_root: Path,
    module_qualified: str,
    *,
    env_vars: Optional[Dict[str, str]] = None,
):
    """
    Import a service's local `app.*` module without colliding with other services' `app` packages.

    Each service under `services/<svc>/app` uses the top-level package name `app`.
    This test needs both cortex-exec and orion-agent-chain; we therefore re-bind `app` at import time.
    """

    old_sys_path = list(sys.path)
    # Reset module resolution for `app` so relative imports bind to the chosen service root.
    for k in list(sys.modules.keys()):
        if k == "app" or k.startswith("app."):
            sys.modules.pop(k, None)

    sys.path.insert(0, str(service_root))

    # Set env vars for import-time wiring (agent-chain creates ToolRegistry at import time).
    old_env: Dict[str, Optional[str]] = {}
    if env_vars:
        for k, v in env_vars.items():
            old_env[k] = os.environ.get(k)
            os.environ[k] = v

    try:
        return importlib.import_module(module_qualified)
    finally:
        sys.path = old_sys_path
        # Restore only if unset previously. (ToolRegistry base_dir is already captured at import time.)
        if env_vars:
            for k, _old in old_env.items():
                if _old is None:
                    os.environ.pop(k, None)


@dataclass
class _Captured:
    agent_runtime_debug: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Tuple[str, Dict[str, Any]]]] = None
    final_text: Optional[str] = None
    log_excerpt: Optional[str] = None


@pytest.mark.timeout(30)
def test_orion_cortex_golden_path_discord_delivery_hard_wiring(monkeypatch, caplog):
    repo_root = Path(__file__).resolve().parents[1]
    cortex_exec_root = repo_root / "services" / "orion-cortex-exec"
    agent_chain_root = repo_root / "services" / "orion-agent-chain"

    # --- Import cortex-exec PlanRunner + Supervisor (real control flow) ---
    router_mod = _import_service_app(cortex_exec_root, "app.router")
    PlanRunner = router_mod.PlanRunner
    # PlanRunner captured Supervisor at import time inside app.router, so use that symbol.
    Supervisor = router_mod.Supervisor

    captured = _Captured()

    # --- Import agent-chain API + patch planner + tool execution (real tool resolution stays intact) ---
    agent_base_dir = str((repo_root / "orion" / "cognition").resolve())
    agent_api = _import_service_app(
        agent_chain_root,
        "app.api",
        env_vars={"COGNITION_BASE_DIR": agent_base_dir},
    )

    tool_calls: List[Tuple[str, Dict[str, Any]]] = []

    class _FakeToolExecutor:
        def __init__(self, *args, **kwargs):
            self.calls: List[Tuple[str, Dict[str, Any]]] = []

        async def execute_llm_verb(self, tool_id, tool_input, *, parent_correlation_id=None):
            # Store only a compact view of tool_input for evidence.
            if isinstance(tool_input, dict):
                compact = {
                    k: v
                    for k, v in tool_input.items()
                    if k in {"text", "request", "original_request", "output_mode", "response_profile"}
                }
            else:
                compact = {}
            self.calls.append((str(tool_id), compact))
            tool_calls.append((str(tool_id), compact))
            if str(tool_id) == "finalize_response":
                return {"llm_output": DISCORD_GUIDE}
            if str(tool_id) == "write_guide":
                return {"llm_output": "WRITE_GUIDE_OBSERVATION"}
            return {"llm_output": f"OBS_{tool_id}"}

    planner_queue: List[Dict[str, Any]] = [
        {
            "status": "ok",
            "trace": [
                {
                    "step_index": 0,
                    "thought": "plan first",
                    "action": {"tool_id": "plan_action", "input": {"goal": PROMPT_DISCORD_DEPLOY}},
                    "observation": None,
                }
            ],
        },
        {
            "status": "ok",
            "trace": [
                {
                    "step_index": 1,
                    "thought": "plan again",
                    "action": {"tool_id": "plan_action", "input": {"goal": PROMPT_DISCORD_DEPLOY}},
                    "observation": None,
                }
            ],
        },
        {
            "status": "ok",
            "trace": [
                {
                    "step_index": 2,
                    "thought": "triage after plan",
                    "action": {"tool_id": "triage", "input": {}},
                    "observation": None,
                }
            ],
        },
        {
            "status": "ok",
            "final_answer": {"content": META_PLAN_TEXT, "structured": {}},
        },
    ]

    async def _fake_call_planner_react(payload, *, parent_correlation_id=None, rpc_bus=None):
        if not planner_queue:
            raise AssertionError("planner_queue exhausted unexpectedly")
        return planner_queue.pop(0)

    monkeypatch.setattr(agent_api, "call_planner_react", _fake_call_planner_react)
    monkeypatch.setattr(agent_api, "ToolExecutor", _FakeToolExecutor)

    # --- Patch cortex-exec Supervisor to (1) force an agent_chain action, and (2) execute real agent-chain directly ---
    async def _fake_planner_step(
        self: Supervisor,
        *,
        source,
        goal_text: str,
        toolset: list,
        trace: list,
        ctx: Dict[str, Any],
        correlation_id: str,
        diagnostic: bool = False,
    ):
        planner_step = StepExecutionResult(
            status="success",
            verb_name="planner",
            step_name="planner_react",
            order=-1,
            result={
                "PlannerReactService": {
                    "trace": [
                        {
                            "thought": "delegate to agent_chain",
                            "action": {"tool_id": "agent_chain", "input": {}},
                        }
                    ],
                    "stop_reason": "delegate",
                }
            },
            latency_ms=1,
            node="test",
            logs=["planner ok (stubbed)"],
            error=None,
        )
        action = {"tool_id": "agent_chain", "input": {}}
        return None, planner_step, None, action

    async def _fake_agent_chain_escalation(
        self: Supervisor,
        *,
        source,
        correlation_id: str,
        ctx: Dict[str, Any],
        packs: List[str],
    ):
        text = _last_user_message(ctx)
        agent_req = AgentChainRequest(
            text=text,
            mode=ctx.get("mode") or "agent",
            session_id=ctx.get("session_id"),
            user_id=ctx.get("user_id"),
            messages=ctx.get("messages") or [],
            packs=packs,
            output_mode=ctx.get("output_mode"),
            response_profile=ctx.get("response_profile"),
        )
        agent_out = await agent_api.execute_agent_chain(
            agent_req,
            correlation_id=correlation_id,
            rpc_bus=MagicMock(),
        )
        captured.agent_runtime_debug = dict(agent_out.runtime_debug or {})
        captured.tool_calls = list(tool_calls)
        captured.final_text = agent_out.text

        return StepExecutionResult(
            status="success",
            verb_name="agent_chain",
            step_name="agent_chain",
            order=100,
            result={"AgentChainService": agent_out.model_dump(mode="json")},
            latency_ms=1,
            node="test",
            logs=["agent_chain ok (real tool resolution)"],
            error=None,
        )

    monkeypatch.setattr(Supervisor, "_planner_step", _fake_planner_step)
    monkeypatch.setattr(Supervisor, "_agent_chain_escalation", _fake_agent_chain_escalation)

    # --- Execute real cortex-exec router -> supervisor control flow ---
    req = PlanExecutionRequest(
        plan=ExecutionPlan(
            verb_name="agent_runtime",
            label="agent-runtime",
            description="golden-path test plan",
            category="agentic",
            priority="normal",
            interruptible=True,
            can_interrupt_others=False,
            timeout_ms=10_000,
            max_recursion_depth=1,
            steps=[],
            metadata={"mode": "agent", "execution_depth": "2"},
        ),
        args=PlanExecutionArgs(
            request_id="req-test",
            user_id="u1",
            trigger_source="test",
            extra={"mode": "agent", "recall": {"enabled": False}},
        ),
        context={},
    )

    ctx = {
        "mode": "agent",
        "max_steps": 1,
        "messages": [{"role": "user", "content": PROMPT_DISCORD_DEPLOY}],
        # Provide only base packs here. agent-chain must activate delivery_pack itself.
        "packs": ["executive_pack", "memory_pack"],
    }

    caplog.set_level(logging.INFO)
    runner = PlanRunner()
    bus = MagicMock()
    source = MagicMock()
    correlation_id = "corr-discord-deploy"

    # PlanRunner.run_plan is async.
    import asyncio

    res = asyncio.run(
        runner.run_plan(
            bus,
            source=source,
            req=req,
            correlation_id=correlation_id,
            ctx=ctx,
        )
    )

    assert res.final_text is not None

    # --- Assertions: runtime wiring signals from agent-chain ---
    dbg = captured.agent_runtime_debug or {}
    assert dbg.get("output_mode") == "implementation_guide"
    assert dbg.get("response_profile") == "technical_delivery"

    packs = dbg.get("packs") or []
    assert "delivery_pack" in packs

    resolved = set(dbg.get("resolved_tool_ids") or [])
    assert "write_guide" in resolved
    assert "finalize_response" in resolved

    assert dbg.get("triage_blocked_post_step0") is True
    assert dbg.get("repeated_plan_action_escalation") is True
    assert dbg.get("finalize_response_invoked") is True
    assert dbg.get("quality_evaluator_rewrite") is True

    # triage should be overridden to finalize_response (tool executor sees finalize_response, not triage)
    called = [t for (t, _inp) in (captured.tool_calls or [])]
    assert "triage" not in called
    assert "write_guide" in called
    assert "finalize_response" in called

    # --- Assertions: final answer shape quality gate (Discord-specific) ---
    final_text = res.final_text or ""
    assert "Discord Developer Portal" in final_text or "Discord application" in final_text
    assert "DISCORD_BOT_TOKEN" in final_text or "BOT_TOKEN" in final_text
    assert "intents" in final_text.lower()
    assert "permissions" in final_text.lower()
    assert "invite" in final_text.lower()
    assert "troubleshoot" in final_text.lower()

    # Must not leak meta-plan scaffolding language.
    forbidden = ["gather requirements", "create a guide", "review and refine"]
    assert not any(p in final_text.lower() for p in forbidden)

    # --- Evidence bundle (optional) ---
    if os.environ.get("ORION_PROOF_WRITE") == "1":
        proof_dir = repo_root / "docs" / "postflight" / "proof"
        proof_dir.mkdir(parents=True, exist_ok=True)
        log_excerpt = "\n".join(
            [
                rec.message
                for rec in caplog.records
                if "triage_blocked_post_step0" in rec.message
                or "repeated_plan_action_escalation" in rec.message
                or "quality_evaluator_rewrite" in rec.message
                or "finalize_response" in rec.message
                or "orch_plan_wiring" in rec.message
            ][:200]
        )
        dbg_resolved = dbg.get("resolved_tool_ids") or []
        evidence = {
            "prompt": PROMPT_DISCORD_DEPLOY,
            "output_mode": dbg.get("output_mode"),
            "response_profile": dbg.get("response_profile"),
            "packs": dbg.get("packs"),
            "resolved_tool_ids_excerpt": dbg_resolved[:30] if isinstance(dbg_resolved, list) else dbg_resolved,
            "runtime_debug_flags": {
                "triage_blocked_post_step0": dbg.get("triage_blocked_post_step0"),
                "repeated_plan_action_escalation": dbg.get("repeated_plan_action_escalation"),
                "finalize_response_invoked": dbg.get("finalize_response_invoked"),
                "quality_evaluator_rewrite": dbg.get("quality_evaluator_rewrite"),
            },
            "tool_call_sequence": [t for (t, _inp) in (captured.tool_calls or [])],
            "final_answer_excerpt": final_text[:900],
            "log_excerpt": log_excerpt,
        }
        (proof_dir / "discord_deploy_golden_path_evidence.json").write_text(
            json.dumps(evidence, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (proof_dir / "discord_deploy_golden_path_evidence.md").write_text(
            "\n".join(
                [
                    "# Discord Golden-Path Evidence (Pass 3)",
                    "",
                    "This evidence is produced by a deterministic integration-style harness (LLM/planner are stubbed; tool resolution and guards are real).",
                    "",
                    "## Runtime-debug excerpt",
                    "```json",
                    json.dumps(
                        {
                            "output_mode": evidence["output_mode"],
                            "response_profile": evidence["response_profile"],
                            "packs": evidence["packs"],
                            "resolved_tool_ids_excerpt": evidence["resolved_tool_ids_excerpt"],
                            "flags": evidence["runtime_debug_flags"],
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    "```",
                    "",
                    "## Log excerpt",
                    "```",
                    evidence["log_excerpt"],
                    "```",
                    "",
                    "## Tool call sequence",
                    "```",
                    " -> ".join(evidence["tool_call_sequence"][:40]),
                    "```",
                    "",
                    "## Final answer excerpt",
                    "```text",
                    evidence["final_answer_excerpt"],
                    "```",
                ]
            ),
            encoding="utf-8",
        )

