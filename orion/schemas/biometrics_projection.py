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
    # Per-pressure-kind merge-window bookkeeping (2026-07-22): keyed by
    # pressure_kind ("strain"/"availability"/etc.), not a single timestamp,
    # since different pressure kinds on the same node can legitimately be
    # accepted at different cadences -- a single shared timestamp would
    # falsely suppress one kind's fresh candidate just because a different
    # kind was reinforced moments earlier. Lives on the durable projection
    # (not a reducer-local dict) specifically so DEFAULT_MERGE_WINDOW_SEC
    # actually holds across separate reduce_node_pressure_candidates() calls
    # -- confirmed live this was previously a no-op: this reducer runs once
    # per trigger event (orion/substrate/biometrics_loop/pipeline.py), so a
    # call-scoped dict could never accumulate dedup history across events.
    last_accepted_at: dict[str, datetime] = Field(default_factory=dict)


class ActiveNodePressureProjectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["projection.active_node_pressure.v1"] = (
        "projection.active_node_pressure.v1"
    )
    projection_id: str
    generated_at: datetime
    nodes: dict[str, ActiveNodePressureStateV1] = Field(default_factory=dict)
