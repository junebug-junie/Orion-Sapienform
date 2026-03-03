# services/orion-agent-chain/app/api.py
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, List

from fastapi import APIRouter, HTTPException
from orion.core.bus.async_service import OrionBusAsync

from .planner_rpc import call_planner_react
from .settings import settings
from .tool_executor import ToolExecutor
from .tool_registry import ToolRegistry
from orion.schemas.agents.schemas import AgentChainRequest, AgentChainResult, ToolDef

logger = logging.getLogger("agent-chain.api")
TOOL_REGISTRY = ToolRegistry(base_dir=Path("/app/orion/cognition"))
router = APIRouter()


def _resolve_tools(body: AgentChainRequest) -> List[ToolDef]:
    if body.tools:
        return [ToolDef(**t) for t in body.tools]
    pack_names = body.packs or ["executive_pack", "memory_pack"]
    local_tools = TOOL_REGISTRY.tools_for_packs(pack_names)
    return [ToolDef(**(t.dict() if hasattr(t, "dict") else t)) for t in local_tools]


async def execute_agent_chain(
    body: AgentChainRequest,
    *,
    correlation_id: str | None = None,
    rpc_bus: OrionBusAsync | None = None,
) -> AgentChainResult:
    parent_corr_id = str(correlation_id or uuid.uuid4())
    tools = _resolve_tools(body)
    logger.info("[agent-chain] resolved_tools count=%s tool_ids=%s", len(tools), [t.tool_id for t in tools])

    planner_payload: dict[str, Any] = {
        "request_id": str(uuid.uuid4()),
        "parent_correlation_id": parent_corr_id,
        "caller": "agent-chain",
        "goal": {
            "type": "chat",
            "description": body.goal_description or f"Agentic mode={body.mode}: {body.text}",
        },
        "context": {
            "conversation_history": [m.model_dump() for m in (body.messages or [])]
            or [{"role": "user", "content": body.text}],
            "orion_state_snapshot": {},
            "external_facts": {},
        },
        "toolset": [t.model_dump() for t in tools],
        "trace": [],
        "limits": {
            "max_steps": settings.default_max_steps,
            "timeout_seconds": settings.default_timeout_seconds,
        },
        "preferences": {
            "style": "neutral",
            "delegate_tool_execution": True,
            "return_trace": True,
        },
    }

    owns_bus = rpc_bus is None
    working_bus = rpc_bus or OrionBusAsync(url=settings.orion_bus_url)
    if owns_bus:
        await working_bus.connect()
    tool_executor = ToolExecutor(working_bus)

    try:
        for step_idx in range(settings.default_max_steps):
            logger.info("[agent-chain] planner step=%s parent_corr=%s", step_idx, parent_corr_id)
            raw_resp = await call_planner_react(
                planner_payload,
                parent_correlation_id=parent_corr_id,
                rpc_bus=working_bus,
            )

            if not isinstance(raw_resp, dict):
                raise RuntimeError(f"Invalid Planner Response: {raw_resp}")
            if raw_resp.get("status") != "ok":
                raise RuntimeError(f"Planner Failed: {raw_resp.get('error')}")

            final = raw_resp.get("final_answer") or {}
            text = final.get("content") or ""
            structured = final.get("structured") or {}
            if text or structured:
                if not text and structured:
                    text = json.dumps(structured, indent=2)
                return AgentChainResult(mode=body.mode, text=text, structured=structured, planner_raw=raw_resp)

            trace = list(raw_resp.get("trace") or [])
            if not trace:
                raise RuntimeError("Planner returned no trace/action in delegate mode")

            last = trace[-1] if isinstance(trace[-1], dict) else {}
            action = last.get("action") or {}
            tool_id = action.get("tool_id")
            tool_input = action.get("input") or {}
            logger.info("[agent-chain] planner_action tool_id=%s input_keys=%s", tool_id, sorted(tool_input.keys()) if isinstance(tool_input, dict) else [])
            if not tool_id:
                stop_reason = str(raw_resp.get("stop_reason") or "")
                continue_reason = str(raw_resp.get("continue_reason") or "")
                # Planner may conclude without emitting a delegate action in this step.
                if stop_reason in {"final_answer", "continue"} or continue_reason != "action_present":
                    fallback = ""
                    if isinstance(last.get("thought"), str):
                        fallback = last.get("thought") or ""
                    if not fallback and isinstance(last.get("observation"), dict):
                        fallback = str(last.get("observation", {}).get("llm_output") or "")
                    if not fallback:
                        fallback = "Planner completed without explicit action; no final answer content provided."
                    return AgentChainResult(
                        mode=body.mode,
                        text=fallback,
                        structured={},
                        planner_raw=raw_resp,
                    )
                raise RuntimeError("Planner delegate response missing action.tool_id")

            observation = await tool_executor.execute_llm_verb(
                tool_id,
                tool_input if isinstance(tool_input, dict) else {},
                parent_correlation_id=parent_corr_id,
            )
            last["observation"] = observation
            trace[-1] = last
            planner_payload["trace"] = trace

        raise RuntimeError("Agent-chain reached max delegated steps without final_answer")
    finally:
        if owns_bus:
            await working_bus.close()


@router.post("/run", response_model=AgentChainResult)
async def run_chain(body: AgentChainRequest) -> AgentChainResult:
    if not body.text.strip():
        raise HTTPException(400, "text required")
    try:
        return await execute_agent_chain(body)
    except Exception as e:
        logger.exception("Agent Chain Error")
        raise HTTPException(500, str(e))
