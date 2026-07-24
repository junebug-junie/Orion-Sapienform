from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RpcHealthSnapshotV1(BaseModel):
    """
    One drained window of a service's real OrionBusAsync.rpc_request() outcomes.

    Mirrors orion.core.bus.rpc_health.RpcHealthSnapshot field-for-field, plus service/node
    identity (same identity fields as SystemHealthV1) so the signal-gateway adapter and any
    other consumer can attribute the window to a real process without re-deriving it from the
    envelope's source metadata.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    service: str
    node: Optional[str] = None
    instance: Optional[str] = None

    window_start: datetime
    window_end: datetime
    success_count: int
    timeout_count: int
    success_latency_ms_p50: Optional[float] = None
    success_latency_ms_p95: Optional[float] = None
    success_latency_ms_max: Optional[float] = None
    timeout_elapsed_ms_max: Optional[float] = None
    channel_counts: Dict[str, int] = Field(default_factory=dict)
    truncated: bool = False

    @field_validator("window_start", "window_end")
    @classmethod
    def _ensure_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v
