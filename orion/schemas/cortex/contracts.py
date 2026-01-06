from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from orion.core.bus.bus_schemas import LLMMessage
from orion.schemas.cortex.schemas import StepExecutionResult


class RecallDirective(BaseModel):
    """Client-facing recall options."""
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    required: bool = False
    mode: Literal["hybrid", "deep"] = "hybrid"
    time_window_days: int = 90
    max_items: int = 8


class CortexClientContext(BaseModel):
    """Conversation + caller context supplied by the client."""
    model_config = ConfigDict(extra="forbid")

    messages: List[LLMMessage]
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CortexClientRequest(BaseModel):
    """Public request contract for Cortex-Orch."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    mode: Literal["brain", "agent", "council"] = "brain"
    verb: str = Field(..., alias="verb_name")
    packs: List[str] = Field(default_factory=list)
    options: Dict[str, Any] = Field(default_factory=dict)
    recall: RecallDirective = Field(default_factory=RecallDirective)
    context: CortexClientContext


class CortexClientResult(BaseModel):
    """Reply contract returned to clients via Orch."""
    model_config = ConfigDict(extra="forbid")

    ok: bool
    mode: str
    verb: str
    status: str
    final_text: Optional[str] = None
    spark_vector: Optional[List[float]] = None
    memory_used: bool = False
    recall_debug: Dict[str, Any] = Field(default_factory=dict)
    steps: List[StepExecutionResult] = Field(default_factory=list)
    error: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CortexChatRequest(BaseModel):
    """Simple chat request for Cortex Gateway."""
    prompt: str = Field(..., description="The user's prompt text")
    mode: Literal["brain", "agent", "council"] = Field(default="brain", description="Execution mode: brain, agent, council")

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
