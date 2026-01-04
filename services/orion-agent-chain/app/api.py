# services/orion-agent-chain/app/api.py
from __future__ import annotations
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from .settings import settings
from .planner_rpc import call_planner_react
from .tool_registry import ToolRegistry

from orion.schemas.agents.schemas import (
    AgentChainRequest, AgentChainResult, ToolDef, PlannerRequest
)

logger = logging.getLogger("agent-chain.api")
TOOL_REGISTRY = ToolRegistry(base_dir=Path("/app/orion/cognition"))
router = APIRouter()

def _resolve_tools(body: AgentChainRequest) -> List[ToolDef]:
    if body.tools:
        return [ToolDef(**t) for t in body.tools]
    pack_names = body.packs or ["executive_pack", "memory_pack"]
    local_tools = TOOL_REGISTRY.tools_for_packs(pack_names)
    # Map local dicts to shared ToolDef
    return [ToolDef(**(t.dict() if hasattr(t, 'dict') else t)) for t in local_tools]

async def execute_agent_chain(body: AgentChainRequest) -> AgentChainResult:
    request_id = str(uuid.uuid4())

    # Map shared AgentChainRequest -> shared PlannerRequest (dict)
    tools = _resolve_tools(body)

    # Construct Planner Payload manually to match schema expectations
    # (Planner expects dict input over the wire)
    planner_payload = {
        "request_id": request_id,
        "caller": "agent-chain",
        "goal": {
            "type": "chat", 
            "description": body.goal_description or f"Agentic mode={body.mode}: {body.text}"
        },
        "context": {
            "conversation_history": [m.model_dump() for m in (body.messages or [])] or [{"role":"user","content":body.text}],
            "orion_state_snapshot": {},
            "external_facts": {}
        },
        "toolset": [t.model_dump() for t in tools],
        "limits": {
            "max_steps": settings.default_max_steps,
            "timeout_seconds": settings.default_timeout_seconds
        },
        "preferences": {"style": "neutral"}
    }

    logger.info("[agent-chain] Calling Planner %s", request_id)
    raw_resp = await call_planner_react(planner_payload)

    if not isinstance(raw_resp, dict):
        raise RuntimeError(f"Invalid Planner Response: {raw_resp}")

    if raw_resp.get("status") != "ok":
        raise RuntimeError(f"Planner Failed: {raw_resp.get('error')}")

    final = raw_resp.get("final_answer") or {}
    text = final.get("content") or ""
    structured = final.get("structured") or {}

    if not text and structured:
        text = json.dumps(structured, indent=2)

    return AgentChainResult(
        mode=body.mode,
        text=text,
        structured=structured,
        planner_raw=raw_resp
    )

@router.post("/run", response_model=AgentChainResult)
async def run_chain(body: AgentChainRequest) -> AgentChainResult:
    if not body.text.strip(): raise HTTPException(400, "text required")
    try:
        return await execute_agent_chain(body)
    except Exception as e:
        logger.exception("Agent Chain Error")
        raise HTTPException(500, str(e))
