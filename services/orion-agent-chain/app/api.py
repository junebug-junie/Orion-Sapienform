from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .settings import settings
from .planner_rpc import call_planner_react
from .tool_registry import ToolRegistry
from .models import ToolDef

logger = logging.getLogger("agent-chain.api")

TOOL_REGISTRY = ToolRegistry(base_dir=Path("/app/orion/cognition"))
router = APIRouter()


# ─────────────────────────────────────────────
# 1. AGENT CHAIN MODELS (API-Facing)
# ─────────────────────────────────────────────

Role = Literal["user", "assistant", "system"]


class Message(BaseModel):
    role: Role
    content: str


class AgentChainRequest(BaseModel):
    text: str
    mode: str = "chat"
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    goal_description: Optional[str] = None
    messages: Optional[List[Message]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    packs: Optional[List[str]] = None


class AgentChainResult(BaseModel):
    mode: str
    text: str
    structured: Dict[str, Any] = Field(default_factory=dict)
    planner_raw: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# 2. HELPERS: TOOLSET & PLANNER PAYLOAD
# ─────────────────────────────────────────────

def _resolve_tools_for_request(body: AgentChainRequest) -> List[ToolDef]:
    """
    Decide which tools to offer the Planner.
    """
    # 1) Explicit tool definitions win
    if body.tools:
        return [ToolDef(**t) for t in body.tools]

    # 2) Pack names from the request or default combo
    if body.packs:
        pack_names = body.packs
    else:
        # Default executive + memory when caller doesn't specify
        pack_names = ["executive_pack", "memory_pack"]

    return TOOL_REGISTRY.tools_for_packs(pack_names)


def _build_planner_request(body: AgentChainRequest) -> Dict[str, Any]:
    """
    Build a *plain dict* shaped like planner-react's PlannerRequest.
    """
    request_id = str(uuid.uuid4())

    # 1) Goal
    goal_description = (
        body.goal_description
        or f"Agentic request in mode={body.mode}: {body.text}"
    )
    goal = {
        "type": "chat",
        "description": goal_description,
        "metadata": {},
    }

    # 2) Context
    conversation_history: List[Dict[str, Any]] = []
    if body.messages:
        for m in body.messages:
            conversation_history.append(
                {"role": m.role, "content": m.content}
            )
    else:
        # minimal history with just this message
        conversation_history.append(
            {"role": "user", "content": body.text}
        )

    context = {
        "conversation_history": conversation_history,
        "orion_state_snapshot": {},
        "external_facts": {},
    }

    # 3) Toolset (from packs → verbs → ToolDef)
    tools: List[ToolDef] = _resolve_tools_for_request(body)
    toolset = [t.model_dump() for t in tools]

    # 4) Limits / Preferences
    limits = {
        "max_steps": settings.default_max_steps,
        "max_tokens_reason": 2048,
        "max_tokens_answer": 1024,
        "timeout_seconds": settings.default_timeout_seconds,
    }

    preferences = {
        "style": "neutral",
        "allow_internal_thought_logging": True,
        "return_trace": True,
    }

    # 5) Final payload dict (planner-react's PlannerRequest shape)
    return {
        "request_id": request_id,
        "caller": "agent-chain",
        "goal": goal,
        "context": context,
        "toolset": toolset,
        "limits": limits,
        "preferences": preferences,
    }


# ─────────────────────────────────────────────
# 3. CORE EXECUTOR (Logic Layer)
# ─────────────────────────────────────────────

async def execute_agent_chain(body: AgentChainRequest) -> AgentChainResult:
    """
    Core business logic. Validates request, calls Planner via Bus,
    and normalizes the response. Raises RuntimeError / TimeoutError on failure.
    """
    # 1. Build Planner payload dict
    planner_payload = _build_planner_request(body)
    trace_id = planner_payload.get("request_id")

    # 2. Call Planner via Bus RPC
    logger.info(
        "[agent-chain] Publishing to %s (trace_id=%s)",
        settings.planner_request_channel, trace_id
    )
    
    raw_resp = await call_planner_react(planner_payload)

    # DEBUG: Log raw planner response
    logger.info(f"[DEBUG-TRACE] Planner Response for {trace_id}: keys={list(raw_resp.keys())}")

    if not isinstance(raw_resp, dict):
        raise RuntimeError(f"Planner returned non-dict response: {raw_resp!r}")

    status = raw_resp.get("status", "error")
    if status != "ok":
        err = raw_resp.get("error") or {}
        if isinstance(err, dict):
            err_msg = err.get("message") or str(err)
        else:
            err_msg = str(err) or "Unknown planner error"

        if status == "timeout":
            raise TimeoutError(f"Planner timed out: {err_msg}")

        raise RuntimeError(f"Planner error: {err_msg}")

    final = raw_resp.get("final_answer") or {}
    if not isinstance(final, dict):
        final = {}

    text = final.get("content") or ""
    structured = final.get("structured") or {}

    # DEBUG: Print initial extraction
    logger.info(f"[DEBUG-TRACE] Extracted text length: {len(text)}")
    logger.info(f"[DEBUG-TRACE] Extracted structured keys: {list(structured.keys())}")

    # ─────────────────────────────────────────────────────────────
    # BUG FIX: Fallback to structured data if text is empty
    # ─────────────────────────────────────────────────────────────
    if not text.strip() and structured:
        logger.info("[DEBUG-TRACE] Text is empty but structured exists. Transforming...")
        try:
            # Pretty print JSON so it is readable in the chat UI
            text = json.dumps(structured, indent=2, ensure_ascii=False)
        except Exception:
            text = str(structured)
        logger.info(f"[DEBUG-TRACE] Transformed text: {text[:100]}...")

    if not text:
        logger.error("[DEBUG-TRACE] FATAL: No text and no structured data found.")
        raise RuntimeError("Planner finished but returned no final answer (text or structured)")

    result_obj = AgentChainResult(
        mode=body.mode,
        text=text,
        structured=structured,
        planner_raw=raw_resp,
    )

    # DEBUG: Log what we are returning to the API/Bus
    logger.info(f"[DEBUG-TRACE] Returning AgentChainResult. Text length={len(result_obj.text)}")
    
    return result_obj


# ─────────────────────────────────────────────
# 4. HTTP ENDPOINT (Transport Layer)
# ─────────────────────────────────────────────

@router.post("/run", response_model=AgentChainResult)
async def run_chain(body: AgentChainRequest) -> AgentChainResult:
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text is required for agent-chain")

    try:
        return await execute_agent_chain(body)

    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except RuntimeError as e:
        # Generic business logic errors -> 502 Bad Gateway
        raise HTTPException(status_code=502, detail=str(e))
    except Exception:
        logger.exception("Unexpected error in agent chain")
        raise HTTPException(status_code=500, detail="Internal agent-chain error")
