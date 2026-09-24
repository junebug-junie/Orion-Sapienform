from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RpcChannelLatencyV1(BaseModel):
    """One hop's outcomes inside one RpcHealthSnapshotV1 window.

    Keyed in ``RpcHealthSnapshotV1.channel_latency`` by a HOP KEY (see
    ``orion.core.bus.rpc_health.hop_key`` and the conventions documented there):
    a bus request channel for ``rpc_request()``, ``"<channel>#<health_label>"`` when the
    caller labels the call, or ``verb:<name>`` / ``governor:<mode>`` /
    ``http:<host><path>`` / ``fcc:<served_model>`` for hand-rolled hops.

    ``log_ms_sum``/``log_ms_sumsq`` are sufficient statistics over SUCCESSES only:
    sum of ``ln(elapsed_ms)`` and sum of ``ln(elapsed_ms)**2``. A consumer recovers the
    window's exact log-latency mean and variance from them
    (``mean = sum / n``, ``var = sumsq / n - mean**2``) without trusting a small-sample
    percentile, and can fold consecutive windows by plain addition. Timeouts are counted
    but never enter the latency statistics -- a timeout's elapsed time is the caller's
    own ceiling, not a round trip (same separation as the pooled fields below).
    ``max_ms`` is the largest success latency in the window, ``None`` if no successes.
    """

    model_config = ConfigDict(extra="forbid")

    success_count: int = 0
    timeout_count: int = 0
    log_ms_sum: float = 0.0
    log_ms_sumsq: float = 0.0
    max_ms: Optional[float] = None


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
    # Per-hop breakdown (docs/superpowers/specs/2026-09-24-metacog-capture-and-transport-
    # ewma-baseline-design.md, A0). Additive; absent/empty from producers that predate it
    # or run with RPC_HEALTH_CHANNEL_LATENCY_ENABLED=false. The pooled fields above keep
    # their original meaning (rpc_request() outcomes only); hops recorded through
    # record_hop_success()/record_hop_timeout() appear ONLY here.
    channel_latency: Dict[str, RpcChannelLatencyV1] = Field(default_factory=dict)

    @field_validator("window_start", "window_end")
    @classmethod
    def _ensure_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v
