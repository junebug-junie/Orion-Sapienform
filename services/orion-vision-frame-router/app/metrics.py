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
    host_trigger_updates_total: int = 0
    host_errors_total: int = 0
    host_timeouts_total: int = 0
    decode_errors_total: int = 0
    # Separate from frames_dispatched_total (2026-08-26) -- conflating the
    # two would make frames_dispatched_total ambiguous (was this a
    # retina_fast dispatch, or the identity_face secondary one?) for a
    # metric other consumers already read.
    identity_dispatched_total: int = 0
    # Face checks that got a full-resolution still, and ones that fell back to the stream frame.
    identity_still_total: int = 0
    identity_still_fallback_total: int = 0
    # Walkway-style captions from the VLM lane (caption_dispatch). Every
    # outcome is counted with its reason: skipped before asking, failed on
    # the lane, answered with no caption (null_answer / sanitizer reason), or
    # answered with a caption.
    caption_dispatched_total: int = 0
    caption_captioned_total: int = 0
    caption_skip_counts: dict[str, int] = field(default_factory=dict)
    caption_no_caption_counts: dict[str, int] = field(default_factory=dict)
    caption_failure_counts: dict[str, int] = field(default_factory=dict)
    # Answered-but-no-caption replies since the last real caption, across all
    # captioned streams (only walkway today). 2026-10-16:
    # 24 of 24 walkway replies were "null_answer" and every counter looked
    # healthy; a long streak now logs a warning (dispatcher).
    caption_no_caption_streak: int = 0
    last_error: str | None = None

    def record_skip(self, reason: str) -> None:
        self.frames_skipped_total += 1
        self.skip_reason_counts[reason] = self.skip_reason_counts.get(reason, 0) + 1

    def record_dispatch(self) -> None:
        self.frames_dispatched_total += 1

    def record_identity_dispatch(self) -> None:
        self.identity_dispatched_total += 1

    def record_seen(self) -> None:
        self.frames_seen_total += 1

    @staticmethod
    def _bump(counts: dict[str, int], key: str) -> None:
        counts[key] = counts.get(key, 0) + 1

    def record_caption_skip(self, reason: str) -> None:
        self._bump(self.caption_skip_counts, reason)

    def record_caption_no_caption(self, reason: str) -> None:
        self._bump(self.caption_no_caption_counts, reason or "unknown")
        self.caption_no_caption_streak += 1

    def record_captioned(self) -> None:
        self.caption_captioned_total += 1
        self.caption_no_caption_streak = 0

    def record_caption_failure(self, reason: str) -> None:
        self._bump(self.caption_failure_counts, reason or "host_error")


def make_health_envelope(
    *,
    service_name: str,
    service_version: str,
    router_enabled: bool,
    dry_run: bool,
    policy_path: str,
    metrics: RouterMetrics,
    state: RouterState,
    heartbeat_interval_sec: float,
    status: str = "ok",
) -> BaseEnvelope:
    # Threaded from the caller's real loop period rather than left at the schema
    # default: orion-equilibrium-service computes grace = interval * 3.0 and marks
    # the service "down" past it, so a hardcoded 10.0 silently mislabels any
    # deployment that overrides HEALTH_INTERVAL_SECONDS.
    payload = SystemHealthV1(
        heartbeat_interval_sec=heartbeat_interval_sec,
        service=service_name,
        version=service_version,
        boot_id=BOOT_ID,
        last_seen_ts=datetime.now(timezone.utc),
        status=status,  # type: ignore[arg-type]
        details={
            "frames_seen_total": metrics.frames_seen_total,
            "frames_dispatched_total": metrics.frames_dispatched_total,
            "identity_dispatched_total": metrics.identity_dispatched_total,
            "identity_still_total": metrics.identity_still_total,
            "identity_still_fallback_total": metrics.identity_still_fallback_total,
            "frames_skipped_total": metrics.frames_skipped_total,
            "inflight_total": state.inflight_total(),
            "pending_count": len(state.pending),
            "policy_path": policy_path,
            "router_enabled": router_enabled,
            "dry_run": dry_run,
            "skip_reason_counts": dict(metrics.skip_reason_counts),
            "host_replies_total": metrics.host_replies_total,
            "host_trigger_updates_total": metrics.host_trigger_updates_total,
            "host_errors_total": metrics.host_errors_total,
            "host_timeouts_total": metrics.host_timeouts_total,
            "decode_errors_total": metrics.decode_errors_total,
            "caption_dispatched_total": metrics.caption_dispatched_total,
            "caption_captioned_total": metrics.caption_captioned_total,
            "caption_inflight": state.inflight_lane("caption"),
            "caption_skip_counts": dict(metrics.caption_skip_counts),
            "caption_no_caption_counts": dict(metrics.caption_no_caption_counts),
            "caption_no_caption_streak": metrics.caption_no_caption_streak,
            "caption_failure_counts": dict(metrics.caption_failure_counts),
            "last_error": metrics.last_error,
        },
    )
    return BaseEnvelope(
        kind="system.health.v1",
        source=ServiceRef(name=service_name, version=service_version),
        payload=payload.model_dump(mode="json"),
    )
