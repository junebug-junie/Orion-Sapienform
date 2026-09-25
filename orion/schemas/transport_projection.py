from __future__ import annotations

from datetime import datetime
from typing import Literal

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Fields retired 2026-09-25 (fix/bus-observer-scope). They were all derived
# from XLEN on two world_pulse Redis Streams (depth/backpressure) or from the
# observer's own PING (stream_backlog_health/delivery_confidence, which could
# never read unhealthy through this path: a failed PING means the same Redis
# the observer publishes to is down). Persisted projections written before the
# deploy still carry them, and this model is extra="forbid", so they are
# dropped on read -- otherwise the first load after deploy would raise and the
# reducer, hub and relational adapter would all lose the projection. Only these
# exact names are dropped; any other unknown key still fails loudly.
RETIRED_TRANSPORT_BUS_STATE_FIELDS: frozenset[str] = frozenset(
    {
        "total_stream_depth",
        "max_stream_depth",
        "backpressure_count",
        "stream_backlog_health",
        "delivery_confidence",
        "stream_depth_pressure",
        "backpressure",
        "stream_backlog_pressure",
    }
)


_logger = logging.getLogger("orion.schemas.transport_projection")


class TransportBusStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _drop_retired_fields(cls, data: Any) -> Any:
        if isinstance(data, dict) and not RETIRED_TRANSPORT_BUS_STATE_FIELDS.isdisjoint(data):
            # Expected once per persisted pre-retirement row (rewritten on the
            # next reducer tick). Seen repeatedly = a live writer still sends them.
            _logger.info(
                "transport_bus_state_retired_fields_dropped fields=%s",
                sorted(RETIRED_TRANSPORT_BUS_STATE_FIELDS.intersection(data)),
            )
            return {k: v for k, v in data.items() if k not in RETIRED_TRANSPORT_BUS_STATE_FIELDS}
        return data

    schema_version: Literal["transport_bus.state.v1"] = "transport_bus.state.v1"

    target_id: str
    node_id: str

    sample_window_id: str
    source_trace_id: str

    redis_ping_ok: bool | None = None

    # Count of BUS_OBSERVER_STREAMS keys checked for catalog membership and
    # schema samples this tick (denominator for contract_pressure and the
    # census-off catalog_drift_pressure fallback). Not a depth reading.
    streams_observed: int = 0

    uncataloged_stream_count: int = 0
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

    catalog_drift_pressure: float = Field(ge=0.0, le=1.0, default=0.0)
    observer_failure_pressure: float = Field(ge=0.0, le=1.0, default=0.0)

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
