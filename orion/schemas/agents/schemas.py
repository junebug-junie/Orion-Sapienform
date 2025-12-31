# orion/schemas/agents/schemas.py
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

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
    plan_only: bool = False
    delegate_tool_execution: bool = False


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


class PlannerRequest(BaseModel):
    """Low-level request to the ReAct Planner."""
    model_config = ConfigDict(extra="ignore")

    request_id: Optional[str] = None
    caller: str = "hub"
    goal: Goal
    context: ContextBlock = Field(default_factory=ContextBlock)
    toolset: List[ToolDef] = Field(default_factory=list)
    trace: List[TraceStep] = Field(default_factory=list)
    limits: Limits = Field(default_factory=Limits)
    preferences: Preferences = Field(default_factory=Preferences)


# ─────────────────────────────────────────────
# Planner Outputs
# ─────────────────────────────────────────────

class PlannerResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request_id: Optional[str] = None
    status: Literal["ok", "error", "timeout"] = "ok"
    error: Optional[Dict[str, Any]] = None
    final_answer: Optional[FinalAnswer] = None
    trace: List[TraceStep] = Field(default_factory=list)
    usage: Optional[Usage] = None


# ─────────────────────────────────────────────
# Service: Agent Council (NEW)
# ─────────────────────────────────────────────

# --- φ + SelfField snapshots ---

class PhiSnapshot(BaseModel):
    valence: float = 0.0
    energy: float = 0.0
    coherence: float = 1.0
    novelty: float = 0.0

class SelfField(BaseModel):
    calm: Optional[float] = None
    stress_load: Optional[float] = None
    uncertainty: Optional[float] = None
    focus: Optional[float] = None
    attunement_to_juniper: Optional[float] = None
    curiosity: Optional[float] = None

# --- Opinion Structures ---

class AgentOpinion(BaseModel):
    agent_name: str
    text: str

class BlinkScores(BaseModel):
    coherence: float = 0.7
    faithfulness: float = 0.7
    usefulness: float = 0.7
    risk: float = 0.3
    effort_cost: float = 0.5
    novelty: float = 0.5
    overall: float = 0.7

class BlinkJudgement(BaseModel):
    proposed_answer: str
    scores: BlinkScores
    disagreement: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

class AuditVerdict(BaseModel):
    action: str  # "accept" | "revise_same_round" | "new_round"
    reason: str
    constraints: Dict[str, Any] = Field(default_factory=dict)
    override_answer: Optional[str] = None

class RoundResult(BaseModel):
    round_index: int
    opinions: List[AgentOpinion]

# --- The Contract ---

class DeliberationRequest(BaseModel):
    """Request to the Agent Council for a multi-agent decision."""
    model_config = ConfigDict(extra="ignore")

    event: str = "council_deliberation"
    trace_id: Optional[str] = None
    source: Optional[str] = None

    prompt: str
    history: Optional[List[Dict[str, Any]]] = None
    tags: Optional[List[str]] = None
    universe: Optional[str] = None

    response_channel: Optional[str] = None

    # Context injections
    phi: Optional[PhiSnapshot] = None
    self_field: Optional[SelfField] = None
    persona_state: Optional[Dict[str, Any]] = None


class CouncilResult(BaseModel):
    """Final output from the Council."""
    model_config = ConfigDict(extra="ignore")

    trace_id: str
    prompt: str
    final_text: str
    opinions: List[AgentOpinion]
    blink: BlinkJudgement
    verdict: AuditVerdict
    meta: Dict[str, Any] = Field(default_factory=dict)
