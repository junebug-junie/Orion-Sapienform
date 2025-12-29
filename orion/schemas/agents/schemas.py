# orion/schemas/agents/schemas.py
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

# Reuse core message type so history is compatible system-wide
from orion.core.bus.bus_schemas import LLMMessage

# ─────────────────────────────────────────────
# Shared: Tools & Capabilities
# ─────────────────────────────────────────────

class ToolDef(BaseModel):
    """Definition of a tool available to the planner."""
    model_config = ConfigDict(extra="ignore")
    
    tool_id: str
    description: Optional[str] = None
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# Service: Agent Chain
# ─────────────────────────────────────────────

class AgentChainRequest(BaseModel):
    """Primary entry point for the Agent Chain service."""
    model_config = ConfigDict(extra="ignore")

    text: str
    mode: str = "chat"
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    goal_description: Optional[str] = None
    # Flexible input: accept dicts (which Pydantic converts) or LLMMessage objects
    messages: Optional[List[LLMMessage]] = None 
    tools: Optional[List[Dict[str, Any]]] = None
    packs: Optional[List[str]] = None


class AgentChainResult(BaseModel):
    """Response from Agent Chain."""
    model_config = ConfigDict(extra="ignore")

    mode: str
    text: str
    structured: Dict[str, Any] = Field(default_factory=dict)
    planner_raw: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# Service: Planner (ReAct)
# ─────────────────────────────────────────────

class Goal(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str = "chat"
    description: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContextBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")
    conversation_history: List[LLMMessage] = Field(default_factory=list)
    orion_state_snapshot: Dict[str, Any] = Field(default_factory=dict)
    external_facts: Dict[str, Any] = Field(default_factory=dict)


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
    """Low-level request to the ReAct Planner."""
    model_config = ConfigDict(extra="ignore")

    request_id: Optional[str] = None
    caller: str = "hub"
    goal: Goal
    context: ContextBlock = Field(default_factory=ContextBlock)
    toolset: List[ToolDef] = Field(default_factory=list)
    limits: Limits = Field(default_factory=Limits)
    preferences: Preferences = Field(default_factory=Preferences)


# ─────────────────────────────────────────────
# Planner Outputs
# ─────────────────────────────────────────────

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


class FinalAnswer(BaseModel):
    content: str
    structured: Dict[str, Any] = Field(default_factory=dict)


class PlannerResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request_id: Optional[str] = None
    status: Literal["ok", "error", "timeout"] = "ok"
    error: Optional[Dict[str, Any]] = None
    final_answer: Optional[FinalAnswer] = None
    trace: List[TraceStep] = Field(default_factory=list)
    usage: Optional[Usage] = None
