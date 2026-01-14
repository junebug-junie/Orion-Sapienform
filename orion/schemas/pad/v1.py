from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

KIND_PAD_EVENT_V1 = "orion.pad.event.v1"
KIND_PAD_FRAME_V1 = "orion.pad.frame.v1"
KIND_PAD_SIGNAL_V1 = "orion.pad.signal.v1"
KIND_PAD_STATS_V1 = "orion.pad.stats.v1"
KIND_PAD_RPC_REQUEST_V1 = "PadRpcRequestV1"
KIND_PAD_RPC_RESPONSE_V1 = "PadRpcResponseV1"

PadEventType = Literal[
    "observation",
    "percept",
    "decision",
    "intent",
    "reflection",
    "memory",
    "metric",
    "anomaly",
    "snapshot",
    "task_state_change",
    "unknown",
]


class PadLinks(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    correlation_id: Optional[UUID] = None
    trace_id: Optional[str] = None


class PadEventV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    event_id: str
    ts_ms: int
    source_service: str
    source_channel: str
    subject: Optional[str] = None
    type: PadEventType = "unknown"
    salience: float = Field(0.0, ge=0.0, le=1.0)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    novelty: float = Field(0.0, ge=0.0, le=1.0)
    payload: Dict[str, Any] = Field(default_factory=dict)
    links: Optional[PadLinks] = None


class TensorBlobV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # Use schema_ with an alias to avoid shadowing Pydantic's internal 'schema' attribute
    schema_: Literal["orion.pad.tensor.v1"] = Field("orion.pad.tensor.v1", alias="schema")
    dim: int
    vector_b64: str
    features: Dict[str, Any] = Field(default_factory=dict)


class StateSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    top_signals: List[str] = Field(default_factory=list)
    active_tasks: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)


class StateBuckets(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    presence: Dict[str, Any] = Field(default_factory=dict)
    affect: Dict[str, Any] = Field(default_factory=dict)
    system: Dict[str, Any] = Field(default_factory=dict)
    cognition: Dict[str, Any] = Field(default_factory=dict)


class StateFrameV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    frame_id: str
    ts_ms: int
    window_ms: int
    summary: StateSummary
    state: StateBuckets
    salient_event_ids: List[str] = Field(default_factory=list)
    tensor: TensorBlobV1


class PadRpcRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    request_id: str
    reply_channel: str
    method: Literal["get_latest_frame", "get_frames", "get_salient_events", "get_latest_tensor"]
    args: Dict[str, Any] = Field(default_factory=dict)


class PadRpcResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    request_id: str
    ok: bool = True
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
