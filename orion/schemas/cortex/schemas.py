# orion/schemas/cortex/schemas.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

# ─────────────────────────────────────────────────────────────
# 1. System Awareness
# ─────────────────────────────────────────────────────────────

class SystemState(BaseModel):
    """Snapshot of the node's capacity when the plan was made."""
    model_config = ConfigDict(extra="ignore")

    name: str
    cpu_load: Optional[float] = None
    gpu_load: Optional[float] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# 2. The Plan Structure
# ─────────────────────────────────────────────────────────────

class ExecutionStep(BaseModel):
    """A single atomic action in the plan."""
    model_config = ConfigDict(extra="forbid")  # Strict validation

    verb_name: str
    step_name: str
    description: str = ""
    order: int
    services: List[str]
    prompt_template: Optional[str] = None
    requires_gpu: bool = False
    requires_memory: bool = False
    timeout_ms: int = 120000
    recall_profile: Optional[str] = None


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
    args: PlanExecutionArgs = Field(default_factory=PlanExecutionArgs)
    context: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# 4. The Results (Exec -> Orch)
# ─────────────────────────────────────────────────────────────

class StepExecutionResult(BaseModel):
    status: str  # "success" | "fail" | "retry"
    verb_name: str
    step_name: str
    order: int
    result: Dict[str, Any] = Field(default_factory=dict)
    spark_vector: Optional[List[float]] = None
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: Optional[int] = None
    node: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    error: Optional[str] = None


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
