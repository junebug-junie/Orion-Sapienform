"""``project_personality_cue``: the personality graph read back to Orion, as a few lines. Pure.

A fixed template over the current ``PersonalityFrameV1`` (orion-durable-runs
``GET /personality/frame``). No model writes it. It names what Orion keeps returning to, in
Orion's own words (each node's label is an excerpt of Orion's evidence), with where it showed up
and how often. It is framed as a record, not an instruction: the design is bottom-up (spec rev 2),
so nothing here tells Orion how to sound.

Renders nothing (``text == ""`` plus ``skip_reason``) when the frame is missing, invalid, older
than ``max_age_sec``, or has no active node and no choice record. Shared by every reader (chat
stance, reverie, curiosity kickoff, outreach) so all lanes see the same self.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from orion.schemas.personality import (
    PERSONALITY_CUE_MAX_CHARS,
    PersonalityCueV1,
    PersonalityFrameV1,
)

DEFAULT_MAX_AGE_SEC = 3600.0  # reducer ticks every 600 s; an hour without a frame is stale
DEFAULT_TOP_K = 5
# The choice line is a pattern claim; below this many choices it is an anecdote, not shown
# (live 2026-10-17: "went your own way in 1 of 1 choices" read to Orion as a tendency).
MIN_CHOICES_SHOWN = 5
LABEL_SHOWN_CHARS = 120

HEADER = (
    "Things you keep coming back to, gathered from your own recent dreams, journal, reverie, "
    "reading and conversations (strongest first; a record, not an instruction):"
)

LANE_WORDS = {
    "dream": "dreams",
    "journal": "journal",
    "reverie": "reverie",
    "reading": "reading",
    "day_letter": "day letter",
    "chat": "conversation",
    "outreach": "outreach",
    "curiosity_choice": "curiosity picks",
    "self_inquiry_seed": "self-inquiry",
}


def _skip(now: datetime, reason: str, frame: PersonalityFrameV1 | None = None) -> PersonalityCueV1:
    return PersonalityCueV1(
        rendered_at=now,
        skip_reason=reason,  # type: ignore[arg-type]
        frame_generated_at=frame.generated_at if frame else None,
    )


def _label(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= LABEL_SHOWN_CHARS:
        return text
    return text[: LABEL_SHOWN_CHARS - 1].rstrip() + "…"


def _line(d: Any) -> str:
    lanes = sorted(d.lane_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    where = ", ".join(LANE_WORDS.get(lane, lane) for lane, n in lanes if n > 0)
    since = d.first_seen_at.strftime("%b %d").replace(" 0", " ")
    return f'- "{_label(d.label)}" ({where}; {d.evidence_count} times since {since})'


def _choice_line(frame: PersonalityFrameV1) -> str | None:
    style = frame.choice_style
    if style is None or style.choices < MIN_CHOICES_SHOWN:
        return None
    days = "day" if style.window_days == 1 else f"{style.window_days} days"
    return (
        f"In curiosity you went your own way in {style.went_own_way} of {style.choices} choices "
        f"over the last {days}, and took the top suggestion in {style.took_top}."
    )


def project_personality_cue(
    frame: PersonalityFrameV1 | dict[str, Any] | None,
    *,
    now: datetime | None = None,
    max_age_sec: float = DEFAULT_MAX_AGE_SEC,
    top_k: int = DEFAULT_TOP_K,
) -> PersonalityCueV1:
    """Never raises."""
    now = now or datetime.now(timezone.utc)
    if frame is None:
        return _skip(now, "fetch_failed")
    try:
        parsed = frame if isinstance(frame, PersonalityFrameV1) else PersonalityFrameV1.model_validate(frame)
    except (ValidationError, TypeError, ValueError):
        return _skip(now, "invalid_frame")
    if (now - parsed.generated_at).total_seconds() > max_age_sec:
        return _skip(now, "stale", parsed)

    shown = [d for d in parsed.dispositions if d.status == "active"][: max(0, top_k)]
    choice = _choice_line(parsed)
    if not shown and choice is None:
        return _skip(now, "empty", parsed)

    # Drop the weakest node until the cue fits; the choice line goes last and is kept if it fits.
    while True:
        lines = [HEADER] + [_line(d) for d in shown] if shown else []
        if choice:
            lines.append(choice)
        text = "\n".join(lines)
        if len(text) <= PERSONALITY_CUE_MAX_CHARS:
            break
        if shown:
            shown = shown[:-1]
        elif choice:
            text = choice[:PERSONALITY_CUE_MAX_CHARS]
            break
    return PersonalityCueV1(
        text=text,
        rendered_at=now,
        frame_generated_at=parsed.generated_at,
        disposition_ids=[d.disposition_id for d in shown],
    )
