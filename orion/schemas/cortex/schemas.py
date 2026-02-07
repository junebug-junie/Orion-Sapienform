# orion/schemas/cortex/schemas.py
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.cortex.types import ExecutionStep, StepExecutionResult, SystemState


class ExecutionPlan(BaseModel):
    """The full list of steps to execute."""
    model_config = ConfigDict(extra="forbid")

    verb_name: str
    label: str = ""
    description: str = ""
    category: str = "general"
    priority: str = "normal"
    interruptible: bool = True
    can_interrupt_others: bool = False
    timeout_ms: int = 120000
    max_recursion_depth: int = 2
    steps: List[ExecutionStep]
    blocked: bool = False
    blocked_reason: Optional[str] = None
    system_state: Optional[SystemState] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# 3. The Envelope (Orch -> Exec)
# ─────────────────────────────────────────────────────────────

class PlanExecutionArgs(BaseModel):
    """Context arguments passed from the trigger."""
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    trigger_source: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class PlanExecutionRequest(BaseModel):
    """The Payload Contract."""
    plan: ExecutionPlan
    kind: Literal["cortex.exec.request"] = "cortex.exec.request"
    args: PlanExecutionArgs = Field(default_factory=PlanExecutionArgs)
    context: Dict[str, Any] = Field(default_factory=dict)


class PlanExecutionResult(BaseModel):
    verb_name: str
    request_id: Optional[str] = None
    status: str  # "success" | "partial" | "fail"
    blocked: bool = False
    blocked_reason: Optional[str] = None
    steps: List[StepExecutionResult] = Field(default_factory=list)
    mode: Optional[str] = None
    final_text: Optional[str] = None
    memory_used: bool = False
    recall_debug: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
