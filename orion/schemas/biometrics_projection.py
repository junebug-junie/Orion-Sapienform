from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


AvailabilityStatus = Literal[
    "unknown",
    "online",
    "stale",
    "missing_expected",
    "offline_expected",
]


class NodeBiometricsStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    aliases: list[str] = Field(default_factory=list)
    role: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    expected_online: bool | None = None
    last_seen_at: datetime | None = None
    latest_trace_id: str | None = None
    latest_payload_ref: str | None = None
    latest_sample_event_id: str | None = None
    latest_summary_event_id: str | None = None
    latest_induction_event_id: str | None = None
    availability_status: AvailabilityStatus = "unknown"
    pressure_hints: dict[str, Any] = Field(default_factory=dict)


class NodeBiometricsProjectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["projection.node_biometrics.v1"] = "projection.node_biometrics.v1"
    projection_id: str
    generated_at: datetime
    nodes: dict[str, NodeBiometricsStateV1] = Field(default_factory=dict)


class ActiveNodePressureStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    availability_status: str
    active_pressures: list[str] = Field(default_factory=list)
    suppressed_pressures: list[str] = Field(default_factory=list)
    capability_impacts: list[str] = Field(default_factory=list)
    pressure_score: float = 0.0
    evidence_event_ids: list[str] = Field(default_factory=list)
    last_updated_at: datetime


class ActiveNodePressureProjectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["projection.active_node_pressure.v1"] = (
        "projection.active_node_pressure.v1"
    )
    projection_id: str
    generated_at: datetime
    nodes: dict[str, ActiveNodePressureStateV1] = Field(default_factory=dict)
