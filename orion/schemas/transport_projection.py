from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TransportBusStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["transport_bus.state.v1"] = "transport_bus.state.v1"

    target_id: str
    node_id: str

    sample_window_id: str
    source_trace_id: str

    redis_ping_ok: bool | None = None

    streams_observed: int = 0
    total_stream_depth: int = 0
    max_stream_depth: int = 0

    uncataloged_stream_count: int = 0
    backpressure_count: int = 0
    observer_failure_count: int = 0
    # Distinct cataloged streams where a bounded XREVRANGE sample failed
    # schema validation against the stream's declared schema_id
    # (orion/bus/channels.yaml). Backs contract_pressure -- genuinely
    # independent of uncataloged_stream_count/catalog_drift_pressure (which
    # measures streams missing from the catalog entirely, a different
    # failure mode). See services/orion-bus/app/bus_observer.py:
    # count_schema_mismatches().
    schema_mismatch_stream_count: int = 0

    bus_health: float = Field(ge=0.0, le=1.0, default=0.5)
    delivery_confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    stream_depth_pressure: float = Field(ge=0.0, le=1.0, default=0.0)
    backpressure: float = Field(ge=0.0, le=1.0, default=0.0)
    catalog_drift_pressure: float = Field(ge=0.0, le=1.0, default=0.0)
    observer_failure_pressure: float = Field(ge=0.0, le=1.0, default=0.0)

    transport_pressure: float = Field(ge=0.0, le=1.0, default=0.0)
    contract_pressure: float = Field(ge=0.0, le=1.0, default=0.0)
    reliability_pressure: float = Field(ge=0.0, le=1.0, default=0.0)

    evidence_event_ids: list[str] = Field(default_factory=list)
    observed_at: datetime | None = None


class TransportBusProjectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["transport_bus.projection.v1"] = "transport_bus.projection.v1"

    updated_at: datetime
    projection_id: str = "active_transport_bus_projection"
    buses: dict[str, TransportBusStateV1] = Field(default_factory=dict)
