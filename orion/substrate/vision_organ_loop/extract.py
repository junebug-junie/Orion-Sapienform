"""Parse one router window trace into per-stream states."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

from orion.schemas.grammar import GrammarEventV1
from orion.schemas.vision_organ_projection import (
    ROLE_STREAM_WINDOW,
    ROLE_WINDOW_COMPLETED,
    STREAM_LIVE,
    STREAM_NEVER_SEEN,
    STREAM_STALE,
    VisionOrganStreamStateV1,
    VisionOrganWindowCountV1,
)
from orion.substrate.prediction_error import vision_channel_staleness_pressure

from .constants import VISION_ORGAN_TRACE_PREFIX

_KV_RE = re.compile(r"(\w+)=([^,;\s]+)")
_MAX_STREAMS = 32
_MAX_CLASSES = 16


def _utc(ts: datetime | None, fallback: datetime) -> datetime:
    if ts is None:
        return fallback
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def parse_vision_organ_trace_id(trace_id: str) -> tuple[str, str] | None:
    """``vision.organ:<router>:<window_id>`` -> (router, window_id)."""
    if not trace_id or not trace_id.startswith(VISION_ORGAN_TRACE_PREFIX):
        return None
    parts = trace_id.split(":", 2)
    if len(parts) != 3 or not parts[1].strip() or not parts[2].strip():
        return None
    return parts[1].strip().lower(), parts[2].strip()


def _parse_kv(summary: str) -> dict[str, str]:
    return {k.lower(): v.strip() for k, v in _KV_RE.findall(summary or "")}


def _int(kv: dict[str, str], key: str) -> int:
    try:
        return max(0, int(kv.get(key, "0") or 0))
    except ValueError:
        return 0


def _float(kv: dict[str, str], key: str) -> float | None:
    raw = kv.get(key)
    if raw in (None, "", "none", "None"):
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if value != value or value < 0.0 or value == float("inf"):
        return None
    return value


def _parse_counts(raw: str | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in (raw or "").split("|"):
        name, _, count = part.partition(":")
        name = name.strip()
        if not name or name == "none" or (name not in out and len(out) >= _MAX_CLASSES):
            continue
        try:
            out[name] = out.get(name, 0) + max(0, int(count or 0))
        except ValueError:
            continue
    return out


def stream_status_and_staleness(
    last_frame_age_sec: float | None, router_uptime_sec: float
) -> tuple[str, float]:
    """A stream that has sent nothing since the router started is aged from the
    router's start: inside the deadband that is a normal restart, past it the
    stream is as unseen as a dead one. Absent is never a calm 0."""
    if last_frame_age_sec is None:
        return STREAM_NEVER_SEEN, vision_channel_staleness_pressure(max(0.0, router_uptime_sec))
    staleness = vision_channel_staleness_pressure(last_frame_age_sec)
    return (STREAM_LIVE if staleness <= 0.0 else STREAM_STALE), staleness


@dataclass(frozen=True)
class WindowParse:
    router: str
    window_id: str
    streams: dict[str, VisionOrganStreamStateV1]
    counts: dict[str, VisionOrganWindowCountV1]
    completed: bool
    window_end: datetime
    stream_event_ids: list[str]
    closing_event_id: str | None
    # streams=N from the closing atom: how many stream atoms the window carried
    expected_streams: int | None = None


def extract_vision_organ_window(events: list[GrammarEventV1], *, now: datetime) -> WindowParse:
    if not events:
        raise ValueError("events must not be empty")
    trace_id = events[0].trace_id or ""
    parsed = parse_vision_organ_trace_id(trace_id)
    if not parsed:
        raise ValueError(f"invalid vision_organ trace_id: {trace_id}")
    router, window_id = parsed

    streams: dict[str, VisionOrganStreamStateV1] = {}
    counts: dict[str, VisionOrganWindowCountV1] = {}
    stream_ids: list[str] = []
    closing_id: str | None = None
    expected: int | None = None
    window_end = now
    for event in events:
        if event.atom is None:
            continue
        role = (event.atom.semantic_role or "").strip()
        observed = _utc(event.emitted_at, now)
        if role == ROLE_WINDOW_COMPLETED:
            closing_id = event.event_id
            window_end = observed
            raw = _parse_kv(event.atom.summary).get("streams")
            expected = int(raw) if raw is not None and raw.isdigit() else None
            continue
        if role != ROLE_STREAM_WINDOW:
            continue
        kv = _parse_kv(event.atom.summary)
        stream = kv.get("stream", "").strip().lower()
        if not stream or (stream not in streams and len(streams) >= _MAX_STREAMS):
            continue
        age = _float(kv, "last_frame_age_sec")
        uptime = _float(kv, "uptime_sec") or 0.0
        status, staleness = stream_status_and_staleness(age, uptime)
        state = VisionOrganStreamStateV1(
            stream_id=stream,
            window_id=window_id,
            source_trace_id=trace_id,
            configured=kv.get("configured") == "1",
            frames=_int(kv, "frames"),
            last_frame_age_sec=age,
            router_uptime_sec=uptime,
            dispatched=_int(kv, "dispatched"),
            identity_dispatched=_int(kv, "identity_dispatched"),
            replies_ok=_int(kv, "replies_ok"),
            failed=_int(kv, "failed"),
            failure_classes=_parse_counts(kv.get("failure_classes")),
            skip_reasons=_parse_counts(kv.get("skips")),
            detect_replies=_int(kv, "detect_replies"),
            objects=_int(kv, "objects"),
            caption_requested=_int(kv, "caption_requested"),
            captions=_int(kv, "captions"),
            status=status,  # type: ignore[arg-type]
            frame_staleness=staleness,
            observed_at=observed,
        )
        streams[stream] = state
        counts[stream] = VisionOrganWindowCountV1(
            window_id=window_id,
            window_end=observed,
            stream_id=stream,
            replies_ok=state.replies_ok,
            failed=state.failed,
        )
        stream_ids.append(event.event_id)
    return WindowParse(
        router=router,
        window_id=window_id,
        streams=streams,
        counts=counts,
        completed=closing_id is not None,
        window_end=window_end,
        stream_event_ids=stream_ids,
        closing_event_id=closing_id,
        expected_streams=expected,
    )
