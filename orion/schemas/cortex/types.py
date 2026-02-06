from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class SystemState(BaseModel):
    """Snapshot of the node's capacity when the plan was made."""

    model_config = ConfigDict(extra="ignore")

    name: str
    cpu_load: Optional[float] = None
    gpu_load: Optional[float] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


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
