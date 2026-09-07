from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

__all__ = ["build_recent_attention_cue"]


def _age_label(age_sec: float) -> str:
    """Bucket an age in seconds into a coarse, human phrase.

    Coarse on purpose: this cue is meant to read like background awareness,
    not a timestamp readout. "moments ago" and "about N minutes ago" are the
    kind of thing a person would actually say about their own last few
    moments of attention; a raw ISO timestamp is not.
    """
    if age_sec < 60:
        return "moments ago"
    if age_sec < 3600:
        minutes = int(age_sec // 60)
        return f"about {minutes} minute{'s' if minutes != 1 else ''} ago"
    if age_sec < 86400:
        hours = int(age_sec // 3600)
        return f"about {hours} hour{'s' if hours != 1 else ''} ago"
    return "more than a day ago"


def _coerce_row(row: Mapping[str, Any], *, now: datetime) -> dict[str, Any] | None:
    """Pull the three fields this cue needs out of one `rows` entry, or
    return None if the row is malformed. Never raises -- a bad row is
    dropped, not a crash, because this function's caller (a prompt-context
    builder) must never be the reason a chat turn fails."""
    process = row.get("process")
    narrative = row.get("reason_narrative")
    generated_at = row.get("generated_at")
    if not process or not narrative or generated_at is None:
        return None
    if not isinstance(generated_at, datetime):
        return None
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=timezone.utc)
    age_sec = max(0.0, (now - generated_at).total_seconds())
    return {
        "process": str(process),
        "narrative": str(narrative),
        "age_label": _age_label(age_sec),
        "generated_at": generated_at.isoformat(),
        "_generated_at_dt": generated_at,
    }


def build_recent_attention_cue(
    rows: list[Mapping[str, Any]],
    *,
    now: datetime | None = None,
    limit: int = 3,
    stale_after_sec: float = 900.0,
) -> dict[str, Any]:
    """Assemble Oríon's own ambient sense of its last few moments of
    attention, for `chat_stance_brief.j2`'s `recent_attention` SOURCES entry.

    `rows` comes from `substrate_attention_schema` -- one row per attention
    event across five producers (cortex_turn, curiosity, reverie,
    substrate_attention, durable_run), each carrying a human-readable
    `reason_narrative`. This is deliberately a compact cue, not a raw
    telemetry dump: the caller should already `ORDER BY generated_at DESC`,
    but this function defensively re-sorts and caps to `limit` itself rather
    than trusting the caller blindly -- the same posture
    `build_recent_trend_signals_cue` takes toward its own inputs.

    The point of this cue is background awareness, not a status report: a
    person doesn't narrate their own heartbeat, they just have a faint,
    ambient sense of it. `stale` means "nothing new to notice", not "an
    error occurred" -- the fastest producer here (`substrate_attention`)
    ticks roughly every 30 seconds, so `stale_after_sec` defaults to 900
    (15 minutes): a window that long with zero rows across all five
    producers means the system has actually gone quiet, not that it's
    merely between ticks.

    Never raises. A malformed row (missing `process`/`reason_narrative`/
    `generated_at`, or a `generated_at` that isn't a real datetime) is
    silently dropped rather than crashing the cue -- this is a pure
    function feeding an internal prompt, not a place that should ever be
    the reason a chat turn fails.
    """
    now = now if now is not None else datetime.now(timezone.utc)

    coerced: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        item = _coerce_row(row, now=now)
        if item is not None:
            coerced.append(item)

    coerced.sort(key=lambda item: item["_generated_at_dt"], reverse=True)
    coerced = coerced[:limit]

    items = [
        {
            "process": item["process"],
            "narrative": item["narrative"],
            "age_label": item["age_label"],
            "generated_at": item["generated_at"],
        }
        for item in coerced
    ]

    if not items:
        stale = True
    else:
        newest_age_sec = (now - coerced[0]["_generated_at_dt"]).total_seconds()
        stale = newest_age_sec > stale_after_sec

    return {
        "items": items,
        "stale": stale,
        "as_of": now.isoformat(),
    }
