"""Format FieldState queue_contention_score for hire role-teach progress.

Pure string helper: Hub (or curiosity) supplies the already-computed score +
driver from the latest ``substrate_field_state`` row. This module never
reads Redis, never recomputes EWMA, and never embeds raw queue counts.
"""

from __future__ import annotations

from orion.field.queue_contention import (
    SOURCE_DURABLE,
    SOURCE_GATEWAY,
    SOURCE_SEED,
)

_DRIVER_BLURBS: dict[str, str] = {
    SOURCE_DURABLE: (
        "durable GPU demand is running well above its normal level"
    ),
    SOURCE_SEED: (
        "the reading-seed pipeline backlog is running well above its normal level"
    ),
    SOURCE_GATEWAY: (
        "LLM gateway waiting is running well above its normal level"
    ),
}


def _band_label(score: float) -> str:
    if score >= 7.0:
        return "high"
    if score >= 4.0:
        return "moderate"
    return "elevated"


def _display_score(score: float) -> str:
    rounded = int(round(score))
    return str(max(0, min(10, rounded)))


def format_queue_contention_progress(
    score: float | None, driver: str | None
) -> list[str]:
    """One advisory line naming score/10 + driving source; omit if unusable.

    Omits when ``score is None`` or ``score <= 0``. Never includes raw counts.
    """
    if score is None:
        return []
    try:
        value = float(score)
    except (TypeError, ValueError):
        return []
    if value <= 0.0:
        return []

    band = _band_label(value)
    shown = _display_score(value)
    blurb = _DRIVER_BLURBS.get(str(driver or "").strip())
    if blurb is None:
        blurb = "shared agent capacity is under more contention than usual"

    return [
        f"Queue pressure: {shown}/10 ({band}) — {blurb}. "
        "Deep Cursor digs compete with that. Prefer hire when Mind says deep."
    ]
