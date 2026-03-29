from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from orion.core.bus.bus_schemas import Envelope

METACOGNITIVE_TRACE_KIND = "metacognitive.trace.v1"

TraceRole = Literal[
    "reasoning",
    "planning",
    "self_check",
    "critique",
    "reflection",
    "stance",
]

TraceStage = Literal["pre_answer", "mid_answer", "post_answer"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MetacognitiveTraceV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    correlation_id: str
    session_id: Optional[str] = None
    message_id: Optional[str] = None
    trace_role: TraceRole = "reasoning"
    trace_stage: TraceStage = "post_answer"
    content: str
    model: str
    token_count: Optional[int] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now_iso)


class MetacognitiveTraceEnvelope(Envelope[MetacognitiveTraceV1]):
    kind: Literal[METACOGNITIVE_TRACE_KIND] = METACOGNITIVE_TRACE_KIND
    correlation_id: UUID
