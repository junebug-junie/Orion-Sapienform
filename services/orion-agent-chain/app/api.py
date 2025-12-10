# services/orion-agent-chain/app/api.py

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .settings import settings
from orion.core.bus.service import OrionBus  # same as other services

from planner_react.api import (
    Goal,
    ContextBlock,
    ToolDef,
    Limits,
    Preferences,
    PlannerRequest,
    TraceStep,
    Usage,
    FinalAnswer,
    PlannerResponse,
)

logger = logging.getLogger("agent-chain.api")

router = APIRouter()


# ─────────────────────────────────────────────
# Shared message type (like planner-react)
# ─────────────────────────────────────────────

Role = Literal["user", "assistant", "system"]


class Message(BaseModel):
    role: Role
    content: str


# ─────────────────────────────────────────────
# AgentChain request/response models
# ─────────────────────────────────────────────

class AgentChainRequest(BaseModel):
    """
    Same outer shape as planner-react, so you can reuse the test payloads.
    """
    request_id: Optional[str] = None
    caller: str = "hub"
    goal: Goal
    context: ContextBlock = Field(default_factory=ContextBlock)
    toolset: List[ToolDef] = Field(default_factory=list)
    limits: Limits = Field(default_factory=Limits)
    preferences: Preferences = Field(default_factory=Preferences)


class AgentChainResult(BaseModel):
    """
    Normalized output back to Hub.
    """
    mode: str
    text: str
    structured: Dict[str, Any] = Field(default_factory=dict)
    planner_raw: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# Helper: call planner-react over HTTP
# ─────────────────────────────────────────────

async def _call_planner_react(payload: Dict[str, Any]) -> Dict[str, Any]:
    url = settings.planner_base_url.rstrip("/") + "/plan/react"
    logger.info("[agent-chain] POST -> %s", url)

    # NOTE: planner-react is stateless HTTP; we don't need bus here.
    async with httpx.AsyncClient(timeout=settings.default_timeout_seconds) as client:
        try:
            r = await client.post(url, json=payload)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(
                "[agent-chain] planner-react HTTP error %s: %s | body=%s",
                e.response.status_code,
                e,
                e.response.text,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Planner-react HTTP error: {e}",
            ) from e
        except Exception as e:
            logger.error("[agent-chain] planner-react call failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=502,
                detail=f"Planner-react unreachable: {e}",
            ) from e

    data = r.json()
    return data


# ─────────────────────────────────────────────
# Helper: build PlannerRequest from AgentChainRequest
# ─────────────────────────────────────────────

def _build_planner_request(body: AgentChainRequest) -> Dict[str, Any]:
    """
    Shape a PlannerRequest payload from a simpler AgentChainRequest.

    For now we hard-code a single tool 'extract_facts' for the chain demo.
    Later we can expand this into a profile-based tool palette.
    """
    goal_desc = (
        body.goal_description
        or "Extract structured facts from the provided text and summarize them in 2–3 bullets."
    )

    # Use the last user message, or synthesize one from text.
    messages = body.messages or []
    if not messages:
        messages = [Message(role="user", content=body.text)]

    context = {
        "conversation_history": [m.model_dump() for m in messages],
        "orion_state_snapshot": {
            "session_id": body.session_id,
            "user_id": body.user_id,
            "mode": body.mode,
        },
        "external_facts": {
            "text": body.text,
        },
    }

    planner_payload: Dict[str, Any] = {
        "caller": settings.service_name,
        "goal": {
            "type": body.mode,
            "description": goal_desc,
            "metadata": {},
        },
        "context": context,
        "toolset": [
            {
                "tool_id": "extract_facts",
                "description": "Extract structured subject/predicate/object facts from a span of text.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                    },
                    "required": ["text"],
                },
                "output_schema": {},
            }
        ],
        "limits": {
            "max_steps": settings.default_max_steps,
            "timeout_seconds": settings.default_timeout_seconds,
        },
        "preferences": {
            "style": "neutral",
            "allow_internal_thought_logging": True,
            "return_trace": True,
        },
    }

    return planner_payload


# ─────────────────────────────────────────────
# POST /chain/run — main entrypoint
# ─────────────────────────────────────────────

@router.post("/run", response_model=AgentChainResult)
async def run_chain(body: AgentChainRequest) -> AgentChainResult:
    """
    Main AgentChain endpoint.

    Hub calls this with a simple AgentChainRequest. We:
      - Build a PlannerRequest (using vendored planner-react models)
      - Call planner-react over HTTP
      - Normalize the result for Hub
    """
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text is required for agent-chain")

    # Optional: bring bus online so we can add telemetry later
    bus = OrionBus(
        url=settings.orion_bus_url,
        enabled=settings.orion_bus_enabled,
    )
    if not bus.enabled:
        logger.warning("[agent-chain] OrionBus disabled; running without telemetry.")

    # Build a PlannerRequest from the lightweight AgentChainRequest
    planner_req: PlannerRequest = _build_planner_request(body)

    # Call planner-react (this should return a PlannerResponse instance)
    planner_resp = await _call_planner_react(planner_req)

    # Defensive: if someone changed _call_planner_react to return a dict/raw JSON
    if not isinstance(planner_resp, PlannerResponse):
        logger.error(
            "[agent-chain] planner-react returned non-PlannerResponse: %r",
            planner_resp,
        )
        raise HTTPException(
            status_code=502,
            detail="planner-react returned unexpected payload (non-PlannerResponse)",
        )

    # Status / error handling
    if planner_resp.status != "ok":
        detail = (planner_resp.error or {}).get("message") if isinstance(planner_resp.error, dict) else None
        detail = detail or "planner-react error"
        raise HTTPException(status_code=502, detail=detail)

    if planner_resp.final_answer is None:
        logger.error("[agent-chain] planner-react responded ok but final_answer is None")
        raise HTTPException(
            status_code=502,
            detail="planner-react produced no final_answer",
        )

    final: FinalAnswer = planner_resp.final_answer
    text = (final.content or "").strip()
    structured = final.structured or {}

    return AgentChainResult(
        mode=body.mode,
        text=text,
        structured=structured,
        planner_raw=planner_resp.model_dump(),
    )
