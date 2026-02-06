from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.cortex.schemas import PlanExecutionRequest, PlanExecutionResult


class CortexExecRequestPayload(PlanExecutionRequest):
    """Typed payload for cortex.exec.request."""


class CortexExecResultPayload(BaseModel):
    """Typed payload for cortex.exec.result."""

    model_config = ConfigDict(extra="ignore")

    ok: bool = True
    result: Optional[PlanExecutionResult] = None
    error: Optional[str] = None
    details: Optional[Any] = None
