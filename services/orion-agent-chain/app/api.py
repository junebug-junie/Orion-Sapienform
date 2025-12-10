# services/orion-agent-chain/app/api.py

from __future__ import annotations

from pathlib import Path
import logging
import uuid
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
  Decide which tools to offer the Planner:

  - If body.tools is provided, treat it as explicit tool defs from the caller.
  - Else, if packs are provided, derive tools from those packs.
  - Else, fall back to default packs.
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
  We do NOT import planner-react's Pydantic models here.
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

  # 2. Call Planner via Bus RPC
  logger.info(
    "[agent-chain] Publishing to %s",
    settings.planner_request_channel,
  )
  raw_resp = await call_planner_react(planner_payload)

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

  if not text:
    raise RuntimeError("Planner finished but returned no final answer")

  return AgentChainResult(
    mode=body.mode,
    text=text,
    structured=structured,
    planner_raw=raw_resp,
  )


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
