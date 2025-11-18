# orion-cortex-exec/app/models.py

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# --- Copied / aligned with semantic planner ---

class SystemState(BaseModel):
    name: str
    cpu_load: Optional[float] = None
    gpu_load: Optional[float] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class ExecutionStep(BaseModel):
    verb_name: str
    step_name: str
    description: str
    order: int
    services: List[str]
    prompt_template: Optional[str]
    requires_gpu: bool = False
    requires_memory: bool = False


class ExecutionPlan(BaseModel):
    verb_name: str
    label: str
    description: str
    category: str
    priority: str
    interruptible: bool
    can_interrupt_others: bool
    timeout_ms: int
    max_recursion_depth: int
    steps: List[ExecutionStep]
    blocked: bool = False
    blocked_reason: Optional[str] = None
    system_state: Optional[SystemState] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


# --- Cortex envelopes ---

class PlanExecutionArgs(BaseModel):
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    trigger_source: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class PlanExecutionRequest(BaseModel):
    plan: ExecutionPlan
    args: PlanExecutionArgs = Field(default_factory=PlanExecutionArgs)
    context: Dict[str, Any] = Field(default_factory=dict)


class StepExecutionResult(BaseModel):
    status: str  # "success" | "fail" | "retry"
    verb_name: str
    step_name: str
    order: int
    result: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: Optional[int] = None
    node: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class PlanExecutionResult(BaseModel):
    verb_name: str
    request_id: Optional[str]
    status: str  # "success" | "partial" | "fail"
    blocked: bool = False
    blocked_reason: Optional[str] = None
    steps: List[StepExecutionResult] = Field(default_factory=list)
