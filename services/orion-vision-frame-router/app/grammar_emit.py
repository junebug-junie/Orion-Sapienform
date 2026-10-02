"""The eye reporting on itself (Layer 1 of the vision_organ lane).

The frame router is the only process that sees every stage of the eye at once:
frames arriving per stream, the tasks it hands the vision host, and every reply
or timeout that comes back. It counts those into a fixed window and, once per
window, publishes one grammar trace (``vision.organ:<router>:<window_id>``) on
``orion:grammar:event``: one atom per stream plus a closing atom that is sent
even when no frame arrived, so "the camera went quiet" and "the router is gone"
stay distinguishable downstream.

Which streams are reported: every stream listed under ``streams:`` in
config/vision_frame_router.yaml (expected to deliver), plus any other stream this
process has seen. A configured stream that has sent nothing since the router
started is reported with ``last_frame_age_sec=none`` -- absent, never a calm 0.

Why the router and not the substrate's old artifact listener: that listener saw
only detect-bearing artifacts pooled across all cameras, so one live camera hid
every dead one. Measured 2026-10-02: node:substrate.vision read prediction_error
0.0 on 124,612 of 124,612 field ticks (2026-09-29..10-02) while the carbon
webcam delivered zero frames for that entire span.

Bounded by construction: counts, ages and failure classes only. No image paths,
labels, captions, identities or correlation ids.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from loguru import logger

from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.schemas.vision_organ_projection import (
    FAILURE_HOST_ERROR,
    ROLE_STREAM_WINDOW,
    ROLE_WINDOW_COMPLETED,
    VISION_ORGAN_TRACE_PREFIX,
)

# A module-level literal (same value as the contract constant, pinned by a test) so
# static producer-catalog scans can resolve this file's GrammarProvenanceV1 identity.
SOURCE_SERVICE = "orion-vision-frame-router"

_MAX_STREAMS = 16
_MAX_CLASSES = 12
_SAFE_RE = re.compile(r"[^a-z0-9_.-]")
_OTHER = "other"


def safe_token(value: Any, *, default: str = "unknown") -> str:
    """Lowercase ``[a-z0-9_.-]`` only, so a value can never break the kv summary."""
    raw = _SAFE_RE.sub("", str(value or "").strip().lower())
    return raw[:64] or default


def _bump(counts: dict[str, int], key: str, n: int = 1) -> None:
    key = safe_token(key)
    if key not in counts and len(counts) >= _MAX_CLASSES:
        key = _OTHER
    counts[key] = counts.get(key, 0) + n


def _counts(counts: dict[str, int]) -> str:
    return "|".join(f"{k}:{v}" for k, v in sorted(counts.items()) if v > 0) or "none"


@dataclass
class _StreamBucket:
    frames: int = 0
    dispatched: int = 0
    identity_dispatched: int = 0
    replies_ok: int = 0
    failed: int = 0
    failure_classes: dict[str, int] = field(default_factory=dict)
    skip_reasons: dict[str, int] = field(default_factory=dict)
    detect_replies: int = 0
    objects: int = 0
    caption_requested: int = 0
    captions: int = 0

    def summary(
        self,
        stream: str,
        *,
        configured: bool,
        last_frame_age_sec: float | None,
        uptime_sec: float,
    ) -> str:
        age = "none" if last_frame_age_sec is None else f"{max(0.0, last_frame_age_sec):.1f}"
        return (
            f"stream={stream} configured={int(configured)} frames={self.frames} "
            f"last_frame_age_sec={age} uptime_sec={max(0.0, uptime_sec):.1f} "
            f"dispatched={self.dispatched} identity_dispatched={self.identity_dispatched} "
            f"replies_ok={self.replies_ok} failed={self.failed} "
            f"failure_classes={_counts(self.failure_classes)} "
            f"skips={_counts(self.skip_reasons)} "
            f"detect_replies={self.detect_replies} objects={self.objects} "
            f"caption_requested={self.caption_requested} captions={self.captions}"
        )


@dataclass
class WindowSnapshot:
    start: float
    end: float
    buckets: dict[str, _StreamBucket]
    last_frame_at: dict[str, float]
    configured: frozenset[str]
    started_at: float


class OrganWindowRecorder:
    """Counts what the eye did into the current window. Every ``record_*`` is cheap
    and never raises into the dispatch path; ``drain`` swaps the window out."""

    def __init__(self, *, configured_streams: Iterable[str] = (), clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._configured = frozenset(safe_token(s) for s in configured_streams if str(s or "").strip())
        self._buckets: dict[str, _StreamBucket] = {}
        # newest frame per stream; survives window boundaries (staleness spans windows)
        self._last_frame_at: dict[str, float] = {}
        self._started_at = self._clock()
        self._window_start = self._started_at

    def set_configured_streams(self, streams: Iterable[str]) -> None:
        with self._lock:
            self._configured = frozenset(safe_token(s) for s in streams if str(s or "").strip())

    def _bucket(self, stream_id: Any) -> tuple[str, _StreamBucket]:
        key = safe_token(stream_id)
        known = key in self._buckets or key in self._configured or key in self._last_frame_at
        if not known and len(set(self._buckets) | self._configured | set(self._last_frame_at)) >= _MAX_STREAMS:
            key = _OTHER
        return key, self._buckets.setdefault(key, _StreamBucket())

    def record_frame(self, stream_id: Any) -> None:
        now = self._clock()
        with self._lock:
            key, bucket = self._bucket(stream_id)
            bucket.frames += 1
            self._last_frame_at[key] = now

    def record_skip(self, stream_id: Any, reason: str) -> None:
        with self._lock:
            _bump(self._bucket(stream_id)[1].skip_reasons, reason)

    def record_dispatch(self, stream_id: Any, *, identity: bool = False) -> None:
        with self._lock:
            bucket = self._bucket(stream_id)[1]
            if identity:
                bucket.identity_dispatched += 1
            else:
                bucket.dispatched += 1

    def record_failure(self, stream_id: Any, failure_class: str) -> None:
        with self._lock:
            bucket = self._bucket(stream_id)[1]
            bucket.failed += 1
            _bump(bucket.failure_classes, failure_class or FAILURE_HOST_ERROR)

    def record_reply_ok(
        self,
        stream_id: Any,
        *,
        primary: bool,
        objects: int | None,
        caption_requested: bool,
        caption_present: bool,
    ) -> None:
        """``objects`` is None when the reply carried no ``objects`` key (an embed-only
        or identity task): that is not an empty detection and is not counted as one."""
        with self._lock:
            bucket = self._bucket(stream_id)[1]
            bucket.replies_ok += 1
            if not primary:
                return
            if objects is not None:
                bucket.detect_replies += 1
                bucket.objects += max(0, int(objects))
            if caption_requested:
                bucket.caption_requested += 1
                if caption_present:
                    bucket.captions += 1

    def drain(self) -> WindowSnapshot:
        with self._lock:
            start, end = self._window_start, self._clock()
            buckets, self._buckets = self._buckets, {}
            self._window_start = end
            return WindowSnapshot(
                start=start,
                end=end,
                buckets=buckets,
                last_frame_at=dict(self._last_frame_at),
                configured=self._configured,
                started_at=self._started_at,
            )


def _window_id(start_ts: float) -> str:
    return datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_window_events(*, router: str, snapshot: WindowSnapshot) -> list[GrammarEventV1]:
    name = safe_token(router, default="router")
    trace_id = f"{VISION_ORGAN_TRACE_PREFIX}{name}:{_window_id(snapshot.start)}"
    emitted_at = datetime.fromtimestamp(snapshot.end, tz=timezone.utc)
    uptime = max(0.0, snapshot.end - snapshot.started_at)
    dims = ["perception", "vision"]
    provenance = GrammarProvenanceV1(
        source_service=SOURCE_SERVICE,
        source_component="organ_window",
        source_trace_id=trace_id,
    )

    def _event(idx: int, role: str, summary: str, text_value: str) -> GrammarEventV1:
        event_id = f"{trace_id}:{idx:02d}:{role}"
        return GrammarEventV1(
            event_id=event_id,
            event_kind="atom_emitted",
            trace_id=trace_id,
            emitted_at=emitted_at,
            observed_at=emitted_at,
            layer="perception",
            dimensions=dims,
            atom=GrammarAtomV1(
                atom_id=event_id,
                trace_id=trace_id,
                atom_type="observation",
                semantic_role=role,
                layer="perception",
                dimensions=dims,
                summary=summary,
                text_value=text_value,
                confidence=1.0,
                salience=0.3,
            ),
            provenance=provenance,
        )

    streams = sorted(set(snapshot.configured) | set(snapshot.buckets) | set(snapshot.last_frame_at))
    events: list[GrammarEventV1] = []
    total_frames = 0
    for stream in streams:
        bucket = snapshot.buckets.get(stream) or _StreamBucket()
        total_frames += bucket.frames
        last = snapshot.last_frame_at.get(stream)
        age = None if last is None else max(0.0, snapshot.end - last)
        events.append(
            _event(
                len(events),
                ROLE_STREAM_WINDOW,
                bucket.summary(
                    stream,
                    configured=stream in snapshot.configured,
                    last_frame_age_sec=age,
                    uptime_sec=uptime,
                ),
                stream,
            )
        )
    events.append(
        _event(
            len(events),
            ROLE_WINDOW_COMPLETED,
            (
                f"router={name} streams={len(streams)} frames={total_frames} "
                f"window_sec={max(0.0, snapshot.end - snapshot.start):.1f} uptime_sec={uptime:.1f}"
            ),
            name,
        )
    )
    return events


_recorder: OrganWindowRecorder | None = None


def get_recorder() -> OrganWindowRecorder | None:
    """None until ``install_recorder``: with the grammar flag off nothing is counted."""
    return _recorder


def install_recorder(configured_streams: Iterable[str] = ()) -> OrganWindowRecorder:
    global _recorder
    _recorder = OrganWindowRecorder(configured_streams=configured_streams)
    return _recorder


def reset_recorder_for_tests() -> None:
    global _recorder
    _recorder = None


async def run_window_publisher(
    bus: Any,
    *,
    router: str,
    window_sec: float,
    stop: asyncio.Event,
) -> None:
    """Flush one window every ``window_sec``. A publish failure is logged and the
    window dropped -- telemetry must never back up into the dispatch path."""
    from orion.grammar.publish import publish_grammar_event

    recorder = get_recorder()
    if recorder is None:
        return
    interval = max(5.0, float(window_sec))
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
        snapshot = recorder.drain()
        events = build_window_events(router=router, snapshot=snapshot)
        sent = 0
        try:
            for event in events:
                await publish_grammar_event(bus, event, source_name=SOURCE_SERVICE)
                sent += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"[ROUTER] vision organ grammar publish failed window_start={snapshot.start} "
                f"dropped={len(events) - sent} of {len(events)}: {exc}"
            )
