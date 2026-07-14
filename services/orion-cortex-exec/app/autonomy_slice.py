from __future__ import annotations

import logging
from typing import Any, Dict

from orion.schemas.thought import AutonomySliceV1

logger = logging.getLogger("orion.cortex.autonomy_slice")

# Compact projection, not a full evidence dump: cap active_tensions hard.
_MAX_ACTIVE_TENSIONS = 3
# Minimum |after - before| pressure movement to call a trend rather than "stable".
_PRESSURE_TREND_EPSILON = 0.02
# Default cap for recent_actions when a caller doesn't pass max_recent_actions
# explicitly (e.g. direct/test callers). The real production call site
# (chat_stance.py) always passes its own _MAX_RECENT_DISPATCH_ACTIONS
# explicitly so the two stay in sync without a second hardcoded copy of "3"
# governing the live path.
_DEFAULT_MAX_RECENT_ACTIONS = 3
# Compact one-line "{kind}: {summary}" budget for recent_actions entries --
# tighter than the raw summary ceiling upstream (chat_stance.py's
# _DISPATCH_ACTION_SUMMARY_MAX_CHARS = 300, itself a defensive ceiling on top
# of the producer's own ACTION_OUTCOME_SUMMARY_MAX_CHARS = 280 in
# services/orion-execution-dispatch-runtime/app/worker.py) since this string
# renders directly into the stance LLM's advisory prompt block alongside the
# already-short dominant_drive/active_tensions strings.
_RECENT_ACTION_LINE_MAX_CHARS = 160


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


def _format_recent_actions(entries: Any, limit: int) -> list[str]:
    """Format successful Layer-9 dispatch-action outcomes into compact
    one-line strings for ``AutonomySliceV1.recent_actions``.

    ``entries`` is expected to be the list of ``{kind, summary, success,
    observed_at}`` dicts produced by
    ``chat_stance._project_recent_dispatch_actions`` (already newest-first),
    but this is defensive against any shape: only ``success is True`` entries
    with non-empty string ``kind``/``summary`` are included -- mirrors
    ``extract_tensions_from_action_outcome``'s existing success-only
    convention. Never raises; malformed input just yields fewer/no lines.
    """
    if not isinstance(entries, list) or limit <= 0:
        return []
    out: list[str] = []
    for item in entries:
        if len(out) >= limit:
            break
        if not isinstance(item, dict):
            continue
        if item.get("success") is not True:
            continue
        kind = str(item.get("kind") or "").strip()
        summary = str(item.get("summary") or "").strip()
        if not kind or not summary:
            continue
        line = f"{kind}: {summary}"
        if len(line) > _RECENT_ACTION_LINE_MAX_CHARS:
            line = line[:_RECENT_ACTION_LINE_MAX_CHARS]
        out.append(line)
    return out


def build_autonomy_slice(
    ctx: Dict[str, Any],
    max_recent_actions: int = _DEFAULT_MAX_RECENT_ACTIONS,
) -> AutonomySliceV1 | None:
    """Assemble the compact slice from the V2 reducer output already in ctx,
    plus recent successful Layer-9 dispatch-action outcomes (also read from
    ctx, under ``ctx["chat_recent_dispatch_actions"]`` -- populated by
    ``chat_stance._project_recent_dispatch_actions`` before this is called).

    Returns None (omit, not empty) when neither the reducer nor recent
    dispatch actions produced any meaningful signal -- never fabricates a
    dominant_drive, tension, or action.
    """
    state = ctx.get("chat_autonomy_state_v2")
    if not isinstance(state, dict):
        state = {}

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

    recent_actions = _format_recent_actions(ctx.get("chat_recent_dispatch_actions"), max_recent_actions)

    # confidence deliberately excluded from this check: AutonomyStateV2.confidence
    # is a required field defaulting to 0.5, so it is present on essentially every
    # reducer output and would defeat omit-when-empty if allowed to count as
    # "signal" on its own -- only real content (drive/tensions/trend/recent
    # actions) justifies emitting a slice. A turn with only recent-action
    # signal and no drive/tension/trend still emits a slice here.
    if dominant_drive is None and not active_tensions and pressure_trend is None and not recent_actions:
        return None

    try:
        return AutonomySliceV1(
            dominant_drive=dominant_drive,
            active_tensions=active_tensions,
            pressure_trend=pressure_trend,
            confidence=confidence,
            recent_actions=recent_actions,
        )
    except Exception:
        logger.warning("autonomy_slice_build_failed", exc_info=True)
        return None
