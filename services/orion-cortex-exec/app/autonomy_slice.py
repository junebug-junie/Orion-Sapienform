from __future__ import annotations

import logging
from typing import Any, Dict

from orion.schemas.thought import AutonomySliceV1

logger = logging.getLogger("orion.cortex.autonomy_slice")

# Compact projection, not a full evidence dump: cap active_tensions hard.
_MAX_ACTIVE_TENSIONS = 3
# Minimum |after - before| pressure movement to call a trend rather than "stable".
_PRESSURE_TREND_EPSILON = 0.02


def _bounded_unique_tensions(values: Any, limit: int = _MAX_ACTIVE_TENSIONS) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for v in values:
        text = str(v or "").strip()
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _pressure_trend(movement_debug: Any) -> str | None:
    """Cheap before/after pressure comparison from ctx['chat_autonomy_movement_debug'].

    Never fabricates a trend: returns None unless both before/after pressure
    dicts are actually present (e.g. first-ever turn for a subject has no
    'before' pressures to compare against).
    """
    if not isinstance(movement_debug, dict):
        return None
    before = movement_debug.get("pressures_before")
    after = movement_debug.get("pressures_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    keys = set(before) | set(after)
    if not keys:
        return None
    try:
        delta_sum = sum(float(after.get(k, 0.0)) - float(before.get(k, 0.0)) for k in keys)
    except (TypeError, ValueError):
        return None
    if delta_sum > _PRESSURE_TREND_EPSILON:
        return "rising"
    if delta_sum < -_PRESSURE_TREND_EPSILON:
        return "falling"
    return "stable"


def build_autonomy_slice(ctx: Dict[str, Any]) -> AutonomySliceV1 | None:
    """Assemble the compact slice from the V2 reducer output already in ctx.

    Returns None (omit, not empty) when the reducer produced no meaningful
    signal -- never fabricates a dominant_drive or tension.
    """
    state = ctx.get("chat_autonomy_state_v2")
    if not isinstance(state, dict) or not state:
        return None

    delta = ctx.get("chat_autonomy_state_delta")
    if not isinstance(delta, dict):
        delta = {}

    dominant_drive = str(state.get("dominant_drive") or "").strip() or None

    # tension_kinds is the state's own current-tension set (same field
    # summarize_autonomy_state() uses to derive AutonomySummaryV1.active_tensions).
    # Falls back to this turn's newly-minted tensions if tension_kinds is absent.
    active_tensions = _bounded_unique_tensions(state.get("tension_kinds"))
    if not active_tensions:
        active_tensions = _bounded_unique_tensions(delta.get("new_tensions"))

    pressure_trend = _pressure_trend(ctx.get("chat_autonomy_movement_debug"))

    confidence = state.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = None

    # confidence deliberately excluded from this check: AutonomyStateV2.confidence
    # is a required field defaulting to 0.5, so it is present on essentially every
    # reducer output and would defeat omit-when-empty if allowed to count as
    # "signal" on its own -- only real content (drive/tensions/trend) justifies
    # emitting a slice.
    if dominant_drive is None and not active_tensions and pressure_trend is None:
        return None

    try:
        return AutonomySliceV1(
            dominant_drive=dominant_drive,
            active_tensions=active_tensions,
            pressure_trend=pressure_trend,
            confidence=confidence,
        )
    except Exception:
        logger.warning("autonomy_slice_build_failed", exc_info=True)
        return None
