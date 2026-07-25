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
    # Mesh-wide census diff (orion.bus.census.compute_census(), Phase 2 of
    # docs/superpowers/specs/2026-07-23-bus-channel-velocity-census-design.md),
    # NOT the same thing as uncataloged_stream_count above -- that's scoped to
    # this observer's own small configured Redis Streams list; this is a real
    # scan of the full ~264-channel catalog against live pub/sub activity.
    # Backs catalog_drift_pressure below (fixed 2026-07-25, see
    # docs/superpowers/specs/2026-07-25-catalog-drift-pressure-mesh-wide-fix.md
    # -- was previously uncataloged_stream_count/streams_observed, capped at
    # whatever this observer's own small stream list covers, structurally
    # incapable of representing general bus health).
    # None (not 0) means "not measured this tick" (census gated off, or the
    # scan failed) -- must stay distinguishable from a real, honest zero.
    undeclared_active_count: int | None = None
    catalog_size: int = 0
    # Distinct cataloged streams where a bounded XREVRANGE sample failed
    # schema validation against the stream's declared schema_id
    # (orion/bus/channels.yaml). Backs contract_pressure -- genuinely
    # independent of uncataloged_stream_count/catalog_drift_pressure (which
    # measures streams missing from the catalog entirely, a different
    # failure mode). See services/orion-bus/app/bus_observer.py:
    # count_schema_mismatches().
    schema_mismatch_stream_count: int = 0

    stream_backlog_health: float = Field(ge=0.0, le=1.0, default=0.5)
    delivery_confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    stream_depth_pressure: float = Field(ge=0.0, le=1.0, default=0.0)
    backpressure: float = Field(ge=0.0, le=1.0, default=0.0)
    catalog_drift_pressure: float = Field(ge=0.0, le=1.0, default=0.0)
    observer_failure_pressure: float = Field(ge=0.0, le=1.0, default=0.0)

    stream_backlog_pressure: float = Field(ge=0.0, le=1.0, default=0.0)
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
