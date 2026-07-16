from __future__ import annotations

import logging
from typing import Any, Dict

from orion.schemas.thought import AutonomySliceV1

logger = logging.getLogger("orion.cortex.autonomy_slice")

# Compact projection, not a full evidence dump: cap active_tensions hard.
_MAX_ACTIVE_TENSIONS = 3
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
    """Assemble the compact slice from DriveEngine's ``drive_state`` projection
    (``ctx["chat_drive_state"]``, stashed unconditionally by chat_stance.py) --
    the single live signal per orion/autonomy/drives_and_autonomy_retrospective.md
    §9. The AutonomyStateV2 reducer this used to also read is retired; there is
    no fallback path anymore.

    ``pressure_trend`` and ``confidence`` are always ``None``: DriveEngine's
    state evolves on its own async tick cadence, not per chat turn, so there is
    no honest single-turn before/after to report a trend from, and its
    projection carries no confidence estimate. Never fabricated.

    Also folds in recent successful Layer-9 dispatch-action outcomes (read
    from ``ctx["chat_recent_dispatch_actions"]`` -- populated by
    ``chat_stance._project_recent_dispatch_actions`` before this is called).

    Returns None (omit, not empty) when neither drive_state nor recent
    dispatch actions produced any meaningful signal -- never fabricates a
    dominant_drive or tension.
    """
    recent_actions = _format_recent_actions(ctx.get("chat_recent_dispatch_actions"), max_recent_actions)

    drive_state = ctx.get("chat_drive_state")
    if not isinstance(drive_state, dict):
        drive_state = {}

    dominant_drive = str(drive_state.get("dominant_drive") or "").strip() or None
    active_tensions = _bounded_unique_tensions(drive_state.get("tension_kinds"))

    if dominant_drive is None and not active_tensions and not recent_actions:
        return None

    try:
        return AutonomySliceV1(
            dominant_drive=dominant_drive,
            active_tensions=active_tensions,
            pressure_trend=None,
            confidence=None,
            recent_actions=recent_actions,
        )
    except Exception:
        logger.warning("autonomy_slice_build_failed", exc_info=True)
        return None
