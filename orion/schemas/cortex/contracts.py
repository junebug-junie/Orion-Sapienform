from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from orion.core.bus.bus_schemas import LLMMessage
from orion.schemas.cortex.types import StepExecutionResult
from orion.schemas.metacognitive_trace import MetacognitiveTraceV1


class RecallDirective(BaseModel):
    """Client-facing recall options."""
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    required: bool = False
    mode: Literal["hybrid", "deep", "graph"] = "hybrid"
    profile: Optional[str] = None
    time_window_days: int = 90
    max_items: int = 8


class CortexClientContext(BaseModel):
    """Conversation + caller context supplied by the client."""
    model_config = ConfigDict(extra="forbid")

    messages: List[LLMMessage]
    raw_user_text: Optional[str] = Field(
        default=None,
        description="Canonical user utterance (pre-scaffold) for telemetry + Spark.",
    )
    user_message: Optional[str] = Field(
        default=None,
        description="Latest user-visible message, if convenient to pass explicitly.",
    )
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CortexClientRequest(BaseModel):
    """Public request contract for Cortex-Orch."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    mode: Literal["brain", "agent", "council", "auto"] = "brain"
    route_intent: Literal["none", "auto"] = "none"
    verb: Optional[str] = Field(default=None, alias="verb_name")
    packs: List[str] = Field(default_factory=list)
    options: Dict[str, Any] = Field(default_factory=dict)
    recall: RecallDirective = Field(default_factory=RecallDirective)
    context: CortexClientContext


ToolFamilyLiteral = Literal[
    "reasoning",
    "planning",
    "recall",
    "communication",
    "runtime",
    "external",
    "device",
    "memory",
    "orchestration",
    "unknown",
]

ActionKindLiteral = Literal[
    "inspect",
    "analyze",
    "retrieve",
    "decide",
    "delegate",
    "notify",
    "write",
    "execute",
    "summarize",
    "unknown",
]

EffectKindLiteral = Literal[
    "read_only",
    "state_change",
    "side_effect",
    "external_io",
    "unknown",
]


class AgentTraceToolStatV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_id: str
    tool_family: ToolFamilyLiteral = "unknown"
    action_kind: ActionKindLiteral = "unknown"
    effect_kind: EffectKindLiteral = "unknown"
    count: int = 0
    duration_ms: Optional[int] = None


class AgentTraceStepV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    event_type: str
    tool_id: Optional[str] = None
    tool_family: Optional[ToolFamilyLiteral] = None
    action_kind: ActionKindLiteral = "unknown"
    effect_kind: EffectKindLiteral = "unknown"
    status: str = "unknown"
    duration_ms: Optional[int] = None
    summary: str = ""
    detail: Dict[str, Any] = Field(default_factory=dict)


class AgentTraceSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corr_id: Optional[str] = None
    message_id: Optional[str] = None
    mode: Literal["agent"] = "agent"
    status: str = "unknown"
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    duration_ms: int = 0
    step_count: int = 0
    tool_call_count: int = 0
    unique_tool_count: int = 0
    unique_tool_families: List[ToolFamilyLiteral] = Field(default_factory=list)
    action_counts: Dict[str, int] = Field(default_factory=dict)
    effect_counts: Dict[str, int] = Field(default_factory=dict)
    summary_text: str = ""
    tools: List[AgentTraceToolStatV1] = Field(default_factory=list)
    steps: List[AgentTraceStepV1] = Field(default_factory=list)
    raw: Dict[str, Any] = Field(default_factory=dict)


class CortexClientResult(BaseModel):
    """Reply contract returned to clients via Orch."""
    model_config = ConfigDict(extra="forbid")

    ok: bool
    mode: str
    verb: str
    status: str
    final_text: Optional[str] = None
    reasoning_content: Optional[str] = None
    reasoning_trace: Optional[Dict[str, Any]] = None
    spark_vector: Optional[List[float]] = None
    memory_used: bool = False
    recall_debug: Dict[str, Any] = Field(default_factory=dict)
    steps: List[StepExecutionResult] = Field(default_factory=list)
    error: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None
    agent_trace: Optional[AgentTraceSummaryV1] = None
    metacog_traces: List[MetacognitiveTraceV1] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CortexChatRequest(BaseModel):
    """Simple chat request for Cortex Gateway."""
    prompt: str = Field(..., description="The user's prompt text")
    messages: Optional[List[LLMMessage]] = Field(
        default=None,
        description="Optional bounded conversation turns (user/assistant/system) for continuity.",
    )
    mode: Literal["brain", "agent", "council", "auto"] = Field(default="brain", description="Execution mode: brain, agent, council, auto")
    route_intent: Literal["none", "auto"] = Field(default="none", description="Explicit routing intent. Use 'auto' to enable Orch auto-routing.")

    # Optional overrides
    verb: Optional[str] = Field(default=None, description="Cognition verb override")
    packs: Optional[List[str]] = Field(default=None, description="Cognition packs override")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Execution options")
    recall: Optional[Dict[str, Any]] = Field(default=None, description="Recall configuration overrides")

    # Context overrides
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    trace_id: Optional[str] = Field(default=None, description="Trace identifier")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional context metadata")


class CortexChatResult(BaseModel):
    """Simple chat result from Cortex Gateway."""
    cortex_result: CortexClientResult = Field(..., description="The raw result from Cortex Orchestrator")
    final_text: Optional[str] = Field(default=None, description="Convenience field for the final text response")


class AutoRouteRecallDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    required: bool = False
    profile: Optional[str] = None


class AutoRouteDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route_mode: Literal["chat", "agent", "council"]
    verb: str
    packs: List[str] = Field(default_factory=list)
    recall: AutoRouteRecallDecisionV1 = Field(default_factory=AutoRouteRecallDecisionV1)
    confidence: float = 0.0
    reason: str = ""
    source: Literal["heuristic", "llm", "fallback"] = "heuristic"


OutputMode = Literal[
    "direct_answer",
    "tutorial",
    "implementation_guide",
    "code_delivery",
    "decision_support",
    "comparative_analysis",
    "debug_diagnosis",
    "project_planning",
    "reflective_depth",
]

ResponseProfile = Literal[
    "direct_answer",
    "tutorial_stepwise",
    "technical_delivery",
    "architect",
    "reflective_depth",
]


class OutputModeDecisionV1(BaseModel):
    """Output mode and response profile for answer depth routing."""

    model_config = ConfigDict(extra="forbid")

    output_mode: OutputMode = "direct_answer"
    response_profile: ResponseProfile = "direct_answer"
    direct_answer_bypass_used: bool = False


class AutoDepthDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_depth: Literal[0, 1, 2, 3]
    primary_verb: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""
    source: Literal["heuristic", "llm", "fallback"] = "heuristic"
