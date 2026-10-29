from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .state import RouterState


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
    # Main-lane dispatches per stream per "<tier>:<triggered_by>" (e.g.
    # "triggered:experiment", "baseline:-"). Monotonic since boot. The
    # switchback runner (orion-prediction) reads deltas at each block edge as
    # its manipulation check: did an on-block actually raise the tier?
    dispatch_tier_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    last_error: str | None = None

    def record_skip(self, reason: str) -> None:
        self.frames_skipped_total += 1
        self.skip_reason_counts[reason] = self.skip_reason_counts.get(reason, 0) + 1

    def record_dispatch(
        self, stream_id: str | None = None, tier: str | None = None, triggered_by: str | None = None
    ) -> None:
        self.frames_dispatched_total += 1
        if stream_id and tier:
            # Copy-on-write: health readers copy this dict from another thread.
            per = dict(self.dispatch_tier_counts.get(stream_id) or {})
            key = f"{tier}:{triggered_by or '-'}"
            per[key] = per.get(key, 0) + 1
            # Replace the outer dict too: inserting a new stream key while asdict() iterates
            # it from the heartbeat thread would raise "dictionary changed size".
            self.dispatch_tier_counts = {**self.dispatch_tier_counts, stream_id: per}

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


def make_health_details(
    *,
    router_enabled: bool,
    dry_run: bool,
    policy_path: str,
    metrics: RouterMetrics,
    state: RouterState,
    experiment_override_streams: list[str] | None = None,
) -> dict[str, Any]:
    """The `details` dict of this service's SystemHealthV1 heartbeat.

    Called by HeartbeatOnly's heartbeat loop in a worker thread
    (asyncio.to_thread), while the event loop keeps mutating `metrics` and
    `state.pending`. So every container is copied once with a single C-level
    call (dict(...) / list(...)), which holds the GIL for the whole copy, and
    the counts are computed from those copies. A Python-level generator over
    the live dict could raise "dictionary changed size during iteration" and
    drop that tick's details.
    """
    pending = list(state.pending.values())
    return {
        "frames_seen_total": metrics.frames_seen_total,
        "frames_dispatched_total": metrics.frames_dispatched_total,
        "identity_dispatched_total": metrics.identity_dispatched_total,
        "identity_still_total": metrics.identity_still_total,
        "identity_still_fallback_total": metrics.identity_still_fallback_total,
        "frames_skipped_total": metrics.frames_skipped_total,
        "inflight_total": sum(1 for t in pending if t.lane == "host"),
        "pending_count": len(pending),
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
        "caption_inflight": sum(1 for t in pending if t.lane == "caption"),
        "caption_skip_counts": dict(metrics.caption_skip_counts),
        "experiment_override_streams": list(experiment_override_streams or []),
        "caption_no_caption_counts": dict(metrics.caption_no_caption_counts),
        "caption_no_caption_streak": metrics.caption_no_caption_streak,
        "caption_failure_counts": dict(metrics.caption_failure_counts),
        "last_error": metrics.last_error,
    }
