"""``project_recent_arcs``: the last few arcs Orion closed today, for reverie. Pure.

A fixed template over the current ``TemporalSelfFrameV1`` (orion-durable-runs
``GET /temporal-self/frame``). No model writes it. One line per closed arc, most recently ended
first, in local time, each line carrying the ``[arc:<id>]`` it came from. The arc's label is its
``subject_label`` (falling back to ``subject_ref``); the duration is ``ended_at - began_at``.

Same render gate as the stance cue (``stance_cue.project_stance_cue``): nothing (``text == ""``,
with ``skip_reason``) when the frame is missing or invalid, is for another local day, is older
than ``max_age_sec``, or is from the future. A fresh frame with no closed arc renders nothing
too (``skip_reason="no_closed_arcs"``): there is no history to show, and saying so adds nothing
reverie can use. ``arcs_today`` is capped upstream, so "today" means the frame's own arcs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from orion.schemas.temporal_self import ArcSummaryV1, TemporalSelfFrameV1
from orion.temporal_self.day import DEFAULT_TZ, as_utc, day_id_for
from orion.temporal_self.stance_cue import DEFAULT_MAX_AGE_SEC, MAX_FUTURE_SKEW_SEC, span_words

MAX_LIMIT = 5
RECENT_ARCS_MAX_CHARS = 400
HEAD = "Recently closed today (most recent first):"

_KIND_WORDS = {
    "conversation": "a conversation with Juniper",
    "curiosity": "a curiosity run",
    "sleep": "a sleep cycle",
    "imagery": "an image attempt",
}


@dataclass(frozen=True)
class RecentArcsBlock:
    text: str = ""
    arc_ids: list[str] = field(default_factory=list)
    skip_reason: str | None = None
    day_id: str | None = None


def _what(arc: ArcSummaryV1) -> str:
    label = arc.subject_label.strip() or arc.subject_ref
    if arc.kind in _KIND_WORDS:
        return _KIND_WORDS[arc.kind]
    if arc.kind == "attention":
        return f"attention on {label}"
    return f"{arc.kind}: {label}"


def _line(arc: ArcSummaryV1, tz: ZoneInfo) -> str:
    began = as_utc(arc.began_at)
    ended = as_utc(arc.ended_at)
    assert began is not None and ended is not None
    span = span_words((ended - began).total_seconds())
    when = f"{began.astimezone(tz).strftime('%H:%M')}-{ended.astimezone(tz).strftime('%H:%M')}"
    return f"- {when} ({span}) {_what(arc)} [arc:{arc.arc_id}]"


def project_recent_arcs(
    frame: TemporalSelfFrameV1 | dict[str, Any] | None,
    now: datetime,
    *,
    limit: int = 3,
    max_age_sec: float = DEFAULT_MAX_AGE_SEC,
    tz_name: str = DEFAULT_TZ,
    max_chars: int = RECENT_ARCS_MAX_CHARS,
) -> RecentArcsBlock:
    now = now.astimezone(timezone.utc)
    if frame is None:
        return RecentArcsBlock(skip_reason="fetch_failed")
    if not isinstance(frame, TemporalSelfFrameV1):
        try:
            frame = TemporalSelfFrameV1.model_validate(frame)
        except ValidationError:
            return RecentArcsBlock(skip_reason="invalid_frame")
    if frame.day_id != day_id_for(now, tz_name):
        return RecentArcsBlock(skip_reason="wrong_day", day_id=frame.day_id)
    as_of = as_utc(frame.as_of)
    assert as_of is not None
    if (now - as_of).total_seconds() > max_age_sec:
        return RecentArcsBlock(skip_reason="stale", day_id=frame.day_id)
    if (as_of - now).total_seconds() > MAX_FUTURE_SKEW_SEC:
        return RecentArcsBlock(skip_reason="clock_skew", day_id=frame.day_id)

    closed = [a for a in frame.arcs_today if a.status == "closed" and a.ended_at is not None]
    if not closed:
        return RecentArcsBlock(skip_reason="no_closed_arcs", day_id=frame.day_id)
    closed.sort(key=lambda a: (as_utc(a.ended_at), a.arc_id), reverse=True)
    n = max(1, min(int(limit), MAX_LIMIT))
    tz = ZoneInfo(tz_name)
    picked = closed[:n]
    lines = [_line(a, tz) for a in picked]
    # Over the length cap: drop the oldest lines first; one line alone is clipped, never dropped
    # (its [arc:<id>] is at the end, so a clipped line would lose its citation -- shorten the label).
    while len(lines) > 1 and len("\n".join([HEAD] + lines)) > max_chars:
        lines.pop()
        picked.pop()
    text = "\n".join([HEAD] + lines)
    if len(text) > max_chars:
        arc = picked[0]
        ref = f" [arc:{arc.arc_id}]"
        text = text[: max_chars - len(ref) - 3].rstrip() + "..." + ref
    return RecentArcsBlock(text=text, arc_ids=[a.arc_id for a in picked], day_id=frame.day_id)
