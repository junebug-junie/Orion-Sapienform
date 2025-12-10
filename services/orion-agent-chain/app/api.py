# services/orion-agent-chain/app/api.py

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .settings import settings
from .planner_rpc import call_planner_react

logger = logging.getLogger("agent-chain.api")

router = APIRouter()

# ─────────────────────────────────────────────
# 1. VENDORED MODELS (Planner-React Contract)
# ─────────────────────────────────────────────

Role = Literal["user", "assistant", "system"]

class Message(BaseModel):
    role: Role
    content: str

class Goal(BaseModel):
    type: str = "chat"
    description: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ContextBlock(BaseModel):
    conversation_history: List[Message] = Field(default_factory=list)
    orion_state_snapshot: Dict[str, Any] = Field(default_factory=dict)
    external_facts: Dict[str, Any] = Field(default_factory=dict)

class ToolDef(BaseModel):
    tool_id: str
    description: Optional[str] = None
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)

class Limits(BaseModel):
    max_steps: int = 4
    max_tokens_reason: int = 2048
    max_tokens_answer: int = 1024
    timeout_seconds: int = 60

class Preferences(BaseModel):
    style: str = "neutral"
    allow_internal_thought_logging: bool = True
    return_trace: bool = True

class PlannerRequest(BaseModel):
    request_id: Optional[str] = None
    caller: str = "hub"
    goal: Goal
    context: ContextBlock = Field(default_factory=ContextBlock)
    toolset: List[ToolDef] = Field(default_factory=list)
    limits: Limits = Field(default_factory=Limits)
    preferences: Preferences = Field(default_factory=Preferences)

class FinalAnswer(BaseModel):
    content: str
    structured: Dict[str, Any] = Field(default_factory=dict)

class TraceStep(BaseModel):
    step_index: int
    thought: Optional[str] = None
    action: Optional[Dict[str, Any]] = None
    observation: Optional[Dict[str, Any]] = None

class Usage(BaseModel):
    steps: int
    tokens_reason: int
    tokens_answer: int
    tools_called: List[str]
    duration_ms: int

class PlannerResponse(BaseModel):
    request_id: Optional[str] = None
    status: Literal["ok", "error", "timeout"] = "ok"
    error: Optional[Dict[str, Any]] = None
    final_answer: Optional[FinalAnswer] = None
    trace: List[TraceStep] = Field(default_factory=list)
    usage: Optional[Usage] = None


# ─────────────────────────────────────────────
# 2. AGENT CHAIN MODELS (The API Layer)
# ─────────────────────────────────────────────

class AgentChainRequest(BaseModel):
    text: str
    mode: str = "chat"
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    goal_description: Optional[str] = None
    messages: Optional[List[Message]] = None
    tools: Optional[List[Dict[str, Any]]] = None

class AgentChainResult(BaseModel):
    mode: str
    text: str
    structured: Dict[str, Any] = Field(default_factory=dict)
    planner_raw: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# 3. HELPER: BUILD PLANNER OBJECT
# ─────────────────────────────────────────────

def _build_planner_request(body: AgentChainRequest) -> PlannerRequest:
    """
    Constructs a valid PlannerRequest Pydantic object from the incoming body.
    """
    goal_desc = (
        body.goal_description
        or "Extract structured facts from the provided text and summarize them in 2–3 bullets."
    )

    messages = body.messages or []
    if not messages:
        messages = [Message(role="user", content=body.text)]

    context = ContextBlock(
        conversation_history=messages,
        orion_state_snapshot={
            "session_id": body.session_id,
            "user_id": body.user_id,
            "mode": body.mode,
        },
        external_facts={"text": body.text}
    )

    # Dynamic Tool Injection
    tools: List[ToolDef] = []
    if body.tools:
        for t_dict in body.tools:
            try:
                tools.append(ToolDef(**t_dict))
            except Exception as e:
                logger.warning("[agent-chain] Invalid tool definition in request: %s", e)
    else:
        # Fallback to the demo tool if none provided
        tools = [
            ToolDef(
                tool_id="extract_facts",
                description="Extract structured subject/predicate/object facts from a span of text.",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                }
            )
        ]

    return PlannerRequest(
        caller=settings.service_name,
        goal=Goal(type=body.mode, description=goal_desc),
        context=context,
        toolset=tools,
        limits=Limits(
            max_steps=settings.default_max_steps,
            timeout_seconds=settings.default_timeout_seconds
        ),
        preferences=Preferences(
            style="neutral",
            allow_internal_thought_logging=True,
            return_trace=True
        )
    )


# ─────────────────────────────────────────────
# 4. CORE EXECUTOR (The Logic Layer)
# ─────────────────────────────────────────────

async def execute_agent_chain(body: AgentChainRequest) -> AgentChainResult:
    """
    Core business logic. Validates request, calls Planner via Bus, 
    and normalizes the response. Raises RuntimeError on failure.
    """
    # 1. Build Payload Object
    planner_req = _build_planner_request(body)

    # 2. Call Planner via Bus RPC
    logger.info("[agent-chain] Publishing to %s", settings.planner_request_channel)

    # .model_dump() converts Pydantic object to dict for the bus
    raw_resp = await call_planner_react(planner_req.model_dump())

    # 3. Rehydrate Response into Pydantic Object
    try:
        planner_resp = PlannerResponse(**raw_resp)
    except Exception as e:
        logger.error("[agent-chain] Invalid response from planner bus: %s", e)
        raise RuntimeError(f"Invalid bus response: {e}")

    # 4. Handle Errors (Logic Level)
    if planner_resp.status != "ok":
        err_msg = "Unknown planner error"
        if planner_resp.error:
            err_msg = planner_resp.error.get("message", str(planner_resp.error))

        if planner_resp.status == "timeout":
            raise TimeoutError(f"Planner timed out: {err_msg}")

        raise RuntimeError(f"Planner error: {err_msg}")

    if not planner_resp.final_answer:
        raise RuntimeError("Planner finished but returned no final answer")

    # 5. Return Result
    return AgentChainResult(
        mode=body.mode,
        text=planner_resp.final_answer.content,
        structured=planner_resp.final_answer.structured,
        planner_raw=planner_resp.model_dump(),
    )


# ─────────────────────────────────────────────
# 5. HTTP ENDPOINT (The Transport Layer)
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
    except Exception as e:
        logger.exception("Unexpected error in agent chain")
        raise HTTPException(status_code=500, detail="Internal agent-chain error")
