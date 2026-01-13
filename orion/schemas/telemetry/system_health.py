from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

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
