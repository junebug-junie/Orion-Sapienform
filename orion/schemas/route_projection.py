from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RouteArbitrationRunStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    correlation_id: str
    session_id: str | None = None
    turn_id: str | None = None
    node_id: str
    lane: str = "unknown"
    lane_reason: str = "unknown"
    mind_requested: bool = False
    mind_skip_reason: str | None = None
    output_mode: str = "unknown"
    evidence_event_ids: list[str] = Field(default_factory=list)
    last_updated_at: datetime


class RouteArbitrationProjectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["projection.route_arbitration.v1"] = (
        "projection.route_arbitration.v1"
    )
    projection_id: str
    generated_at: datetime
    runs: dict[str, RouteArbitrationRunStateV1] = Field(default_factory=dict)
