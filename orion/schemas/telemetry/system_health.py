from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SystemHealthV1(BaseModel):
    """
    Versioned heartbeat contract for Titanium bus.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    service: str
    node: Optional[str] = None
    version: Optional[str] = None
    instance: Optional[str] = None
    boot_id: str
    status: Literal["ok", "degraded", "down"] = "ok"
    last_seen_ts: datetime
    heartbeat_interval_sec: float = 10.0
    details: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("last_seen_ts")
    @classmethod
    def _ensure_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class EquilibriumServiceState(BaseModel):
    """
    Current service state used by the Equilibrium snapshot publisher.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    service: str
    node: Optional[str] = None
    status: Literal["ok", "degraded", "down"]
    last_seen_ts: datetime
    heartbeat_interval_sec: float
    down_for_ms: int
    uptime_pct: Dict[str, float] = Field(default_factory=dict)
    boot_id: Optional[str] = None
    version: Optional[str] = None
    instance: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("last_seen_ts")
    @classmethod
    def _ensure_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class EquilibriumSnapshotV1(BaseModel):
    """
    Aggregate view of system equilibrium and distress.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source_service: str
    source_node: Optional[str] = None
    producer_boot_id: str
    generated_at: datetime
    grace_multiplier: float
    windows_sec: List[int]
    expected_services: List[str] = Field(default_factory=list)
    services: List[EquilibriumServiceState] = Field(default_factory=list)
    distress_score: float = 0.0
    zen_score: float = 1.0
    correlation_id: Optional[str] = None

    @field_validator("generated_at")
    @classmethod
    def _ensure_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class EquilibriumServiceTransitionV1(BaseModel):
    """
    A single service status transition (ok/degraded/down), emitted by
    orion-equilibrium-service the instant it detects `status` changed for a
    tracked service -- NOT a periodic snapshot. `EquilibriumSnapshotV1`
    (`orion:equilibrium:snapshot`) already broadcasts the full rolling-window
    state every `EQUILIBRIUM_PUBLISH_INTERVAL_SEC`; nothing before this schema
    persisted the *moments* status changed, which is what "how long was
    service X down last week" needs -- see
    docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.

    Recovery transitions (`from_status == "down"`) carry `down_since_ts` and
    `down_duration_ms` so a single row answers a downtime-duration query
    without joining a paired down/up event.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # Row-identity field, named to match the sql-writer persistence model's
    # primary-key column exactly (services/orion-sql-writer/app/models/
    # equilibrium_service_transition.py) -- same convention as
    # CausalGeometrySnapshotV1.snapshot_id.
    transition_id: str = Field(default_factory=lambda: str(uuid4()))

    service: str
    node: Optional[str] = None
    boot_id: Optional[str] = None
    version: Optional[str] = None
    instance: Optional[str] = None

    from_status: Literal["ok", "degraded", "down", "unknown"] = "unknown"
    to_status: Literal["ok", "degraded", "down"]
    transitioned_at: datetime

    # Populated only when from_status == "down" -- when the downtime episode
    # began, and how long it lasted, as observed by this consumer's own
    # in-memory tracking (not a stored-and-replayed value).
    down_since_ts: Optional[datetime] = None
    down_duration_ms: Optional[int] = None

    source_service: str
    source_node: Optional[str] = None
    producer_boot_id: str

    @field_validator("transitioned_at", "down_since_ts")
    @classmethod
    def _ensure_tz(cls, v: Optional[datetime]) -> Optional[datetime]:
        if v is not None and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class BusConsumerReadinessV1(BaseModel):
    """
    Structured readiness snapshot for bus-backed service consumers.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    ok: bool
    http_alive: Optional[bool] = None
    bus_consumer_ready: bool
    intake_channel: str
    subscriber_count: int
    heartbeat_fresh: Optional[bool] = None
    rpc_smoke_ok: Optional[bool] = None
    dependency_status: Literal["available", "unavailable"]
    error: Optional[str] = None
