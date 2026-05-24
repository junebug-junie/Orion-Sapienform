from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.system_health import SystemHealthV1

from .state import RouterState

BOOT_ID = str(uuid.uuid4())


@dataclass
class RouterMetrics:
    frames_seen_total: int = 0
    frames_dispatched_total: int = 0
    frames_skipped_total: int = 0
    skip_reason_counts: dict[str, int] = field(default_factory=dict)
    host_replies_total: int = 0
    host_errors_total: int = 0
    host_timeouts_total: int = 0
    decode_errors_total: int = 0
    last_error: str | None = None

    def record_skip(self, reason: str) -> None:
        self.frames_skipped_total += 1
        self.skip_reason_counts[reason] = self.skip_reason_counts.get(reason, 0) + 1

    def record_dispatch(self) -> None:
        self.frames_dispatched_total += 1

    def record_seen(self) -> None:
        self.frames_seen_total += 1


def make_health_envelope(
    *,
    service_name: str,
    service_version: str,
    router_enabled: bool,
    dry_run: bool,
    policy_path: str,
    metrics: RouterMetrics,
    state: RouterState,
    status: str = "ok",
) -> BaseEnvelope:
    payload = SystemHealthV1(
        service=service_name,
        version=service_version,
        boot_id=BOOT_ID,
        last_seen_ts=datetime.now(timezone.utc),
        status=status,  # type: ignore[arg-type]
        details={
            "frames_seen_total": metrics.frames_seen_total,
            "frames_dispatched_total": metrics.frames_dispatched_total,
            "frames_skipped_total": metrics.frames_skipped_total,
            "inflight_total": state.inflight_total(),
            "pending_count": len(state.pending),
            "policy_path": policy_path,
            "router_enabled": router_enabled,
            "dry_run": dry_run,
            "skip_reason_counts": dict(metrics.skip_reason_counts),
            "host_replies_total": metrics.host_replies_total,
            "host_errors_total": metrics.host_errors_total,
            "host_timeouts_total": metrics.host_timeouts_total,
            "decode_errors_total": metrics.decode_errors_total,
            "last_error": metrics.last_error,
        },
    )
    return BaseEnvelope(
        kind="system.health.v1",
        source=ServiceRef(name=service_name, version=service_version),
        payload=payload.model_dump(mode="json"),
    )
