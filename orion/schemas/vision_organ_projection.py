"""Vision organ substrate lane (orion-vision-frame-router reporting on the eye).

The frame router is the one process that sees every stage of the eye: frames
arriving per stream (``orion:vision:frames``), the tasks it sends the vision host,
and every host reply or timeout that comes back. Once per window it publishes one
grammar trace (``vision.organ:<router>:<window_id>``,
services/orion-vision-frame-router/app/grammar_emit.py): one atom per stream plus a
closing atom sent even when no frame arrived, so "nothing seen" and "router gone"
stay distinguishable downstream.

The vision_organ reducer (orion/substrate/vision_organ_loop/) folds each window
into per-stream states and one organ reading on ``node:substrate.vision_organ``.
Counts and ages only -- no image, caption text, label, or identity reaches here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---- Wire contract shared by the producer (orion-vision-frame-router) and the
# reducer. Lives here, not in orion/substrate/, so the router can import it
# without executing orion/substrate/__init__.py (graph store, materializer -- the
# import that crash-looped two thin services on 2026-08-19).
VISION_ORGAN_SOURCE_SERVICE = "orion-vision-frame-router"
VISION_ORGAN_TRACE_PREFIX = "vision.organ:"
# One atom per stream per window (self-contained: a reducer batch boundary can
# split a window between streams but never inside one stream's reading).
ROLE_STREAM_WINDOW = "vision_stream_window_observed"
# Sent last, every window, even with zero frames. The organ reading is computed
# only when this arrives, so a window split across two reducer batches never
# yields a reading over a partial set of streams.
ROLE_WINDOW_COMPLETED = "vision_organ_window_completed"

# Failure classes the router can observe for a task it dispatched. Every class
# means "a frame was handed to the eye and no usable answer came back".
FAILURE_TIMEOUT = "timeout"  # no reply within TASK_TIMEOUT_SECONDS
FAILURE_INVALID_REPLY = "invalid_reply"  # reply did not validate as VisionTaskResultPayload
FAILURE_HOST_ERROR = "host_error"  # ok=false with no error_code

STREAM_LIVE = "live"
STREAM_STALE = "stale"
STREAM_NEVER_SEEN = "never_seen"
StreamStatus = Literal["live", "stale", "never_seen"]

ORGAN_REPORTING = "reporting"
# The router stopped publishing windows: the substrate's clock path wrote this,
# not a window. Staleness is 1.0 -- silence converges toward alarm, never calm.
ORGAN_SILENT = "silent"
OrganStatus = Literal["reporting", "silent"]


class VisionOrganStreamStateV1(BaseModel):
    """Everything one router window saw about one camera stream."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["vision_organ.stream_state.v1"] = "vision_organ.stream_state.v1"

    stream_id: str
    window_id: str
    source_trace_id: str
    # listed under `streams:` in config/vision_frame_router.yaml (expected to deliver)
    configured: bool = False

    frames: int = 0
    # seconds from this stream's newest frame to window end; None = no frame since
    # the router started (see router_uptime_sec)
    last_frame_age_sec: float | None = None
    router_uptime_sec: float = 0.0

    # primary tasks / identity_face secondary tasks sent to the host this window
    dispatched: int = 0
    identity_dispatched: int = 0
    # replies_ok / failed count BOTH primary and identity tasks (and a reply can
    # land in a later window than its dispatch), so compare them with
    # dispatched + identity_dispatched over several windows, not one.
    replies_ok: int = 0
    failed: int = 0
    # failure class -> count, e.g. {"timeout": 1}; host error_code values pass through
    failure_classes: dict[str, int] = Field(default_factory=dict)
    # policy skip reason -> count (frame_sampled_out, camera_rate_limited, ...)
    skip_reasons: dict[str, int] = Field(default_factory=dict)

    # yield, record-only (see reducer docstring for why none of this is pressure)
    detect_replies: int = 0
    objects: int = 0
    caption_requested: int = 0
    captions: int = 0

    status: StreamStatus = STREAM_NEVER_SEEN
    # vision_channel_staleness_pressure(age); never_seen is aged from router start
    frame_staleness: float = Field(default=1.0, ge=0.0, le=1.0)
    observed_at: datetime


class VisionOrganWindowCountV1(BaseModel):
    """One window's task outcomes for one stream, kept on the projection so the
    failure reading can span several windows."""

    model_config = ConfigDict(extra="forbid")

    window_id: str
    window_end: datetime
    stream_id: str
    replies_ok: int = 0
    failed: int = 0


class VisionOrganProjectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["vision_organ.projection.v1"] = "vision_organ.projection.v1"
    projection_id: str
    generated_at: datetime
    # One frame router is assumed: a second router (or a worktree deploy with the
    # same SERVICE_NAME) would replace this projection's stream set every window.
    router: str | None = None
    status: OrganStatus = ORGAN_REPORTING
    # stream_id -> latest window state for that stream
    streams: dict[str, VisionOrganStreamStateV1] = Field(default_factory=dict)
    last_window_id: str | None = None
    # router's window end (event time) of the last completed window; the clock
    # path measures silence from here
    last_window_end: datetime | None = None
    # organ readings as of the last completed window (or silence write)
    vision_frame_staleness: float | None = Field(default=None, ge=0.0, le=1.0)
    vision_processing_failure_pressure: float | None = Field(default=None, ge=0.0, le=1.0)
    # rolling task outcomes, oldest first, pruned to the failure window
    recent_windows: list[VisionOrganWindowCountV1] = Field(default_factory=list)


__all__ = [
    "FAILURE_HOST_ERROR",
    "FAILURE_INVALID_REPLY",
    "FAILURE_TIMEOUT",
    "ORGAN_REPORTING",
    "ORGAN_SILENT",
    "ROLE_STREAM_WINDOW",
    "ROLE_WINDOW_COMPLETED",
    "STREAM_LIVE",
    "STREAM_NEVER_SEEN",
    "STREAM_STALE",
    "VISION_ORGAN_SOURCE_SERVICE",
    "VISION_ORGAN_TRACE_PREFIX",
    "VisionOrganProjectionV1",
    "VisionOrganStreamStateV1",
    "VisionOrganWindowCountV1",
]
