"""Format FieldState queue_contention_score for hire role-teach progress.

Pure string helper: Hub (or curiosity) supplies the already-computed score +
driver from the latest ``substrate_field_state`` row. This module never
reads Redis, never recomputes EWMA, and never embeds raw queue counts.
"""

from __future__ import annotations

from orion.field.queue_contention import (
    OLDEST_WAIT_SUFFIX,
    SOURCE_DURABLE,
    SOURCE_GPU_POOL,
    SOURCE_SEED,
    driver_source,
)

_HIRE_NUDGE = (
    "Elevated queue is a reason to hire_cursor (offload), not a reason to "
    "stay on local_crawl — the local agent GPU seat is the expensive one. "
    "Write hire_cursor and HelpRequest now — do not take a short local look first."
)

# A reading-seed queue that has stopped moving is most likely a stalled reader
# pipeline, not shared agent capacity under contention: hiring Cursor does not
# unstick it, so it must not push a hire. Report it as information only.
_STUCK_SEED_NOTE = (
    "This is a stalled queue, not busy capacity — it is not by itself a "
    "reason to hire_cursor."
)

_DRIVER_BLURBS: dict[str, str] = {
    SOURCE_DURABLE: (
        "durable GPU demand is running well above its normal level"
    ),
    SOURCE_SEED: (
        "the reading-seed pipeline backlog is running well above its normal level"
    ),
    SOURCE_GPU_POOL: (
        "work waiting in line for a GPU is running well above its normal level"
    ),
    # Oldest-wait drivers (2026-09-25): the queue may not be longer than usual,
    # but the item at the back of it has waited far past its normal wait --
    # i.e. the queue looks stuck, not just busy. Relative wording only; the
    # raw age never reaches Orion.
    SOURCE_DURABLE + OLDEST_WAIT_SUFFIX: (
        "the oldest durable GPU demand has waited far longer than they normally do"
    ),
    SOURCE_SEED + OLDEST_WAIT_SUFFIX: (
        "the oldest reading seed has waited far longer than seeds normally do — "
        "that queue looks stuck, not just busy"
    ),
    SOURCE_GPU_POOL + OLDEST_WAIT_SUFFIX: (
        "the oldest request waiting for a GPU has waited far longer than normal"
    ),
}


def _band_label(score: float) -> str:
    if score >= 7.0:
        return "high"
    if score >= 4.0:
        return "moderate"
    return "elevated"


# Scores below this round to 0/10 while still landing in the "elevated" band.
_DISCLOSE_FLOOR = 0.5


def _display_score(score: float) -> str:
    # Callers only pass scores at/above _DISCLOSE_FLOOR; clamp so the shown
    # integer is never 0 while a band label claims elevation (banker's round
    # of exactly 0.5 is 0 in Python 3).
    rounded = int(round(score))
    return str(max(1, min(10, rounded)))


def format_queue_contention_progress(
    score: float | None, driver: str | None
) -> list[str]:
    """One advisory line naming score/10 + driving source; omit if unusable.

    Omits when ``score is None`` or ``score < 0.5``. Never includes raw counts.
    """
    if score is None:
        return []
    try:
        value = float(score)
    except (TypeError, ValueError):
        return []
    if value < _DISCLOSE_FLOOR:
        return []

    band = _band_label(value)
    shown = _display_score(value)
    blurb = _DRIVER_BLURBS.get(str(driver or "").strip())
    if blurb is None:
        blurb = "shared agent capacity is under more contention than usual"

    source, is_oldest_wait = driver_source(str(driver or "").strip() or None)
    closing = _STUCK_SEED_NOTE if (is_oldest_wait and source == SOURCE_SEED) else _HIRE_NUDGE
    return [f"Queue pressure: {shown}/10 ({band}) — {blurb}. {closing}"]
