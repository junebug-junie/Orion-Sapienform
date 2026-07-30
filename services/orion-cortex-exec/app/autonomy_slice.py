from __future__ import annotations

import logging
from typing import Any, Dict

from orion.schemas.thought import AutonomySliceV1

logger = logging.getLogger("orion.cortex.autonomy_slice")

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
    """Assemble the compact slice from recent successful Layer-9 dispatch-action
    outcomes (read from ``ctx["chat_recent_dispatch_actions"]`` -- populated by
    ``chat_stance._project_recent_dispatch_actions`` before this is called).

    ``dominant_drive`` and ``active_tensions`` are always empty/``None`` as of
    2026-07-30 (chore/delete-orion-drives Wave 2a): the DriveEngine ``chat_
    drive_state`` projection (``ctx["chat_drive_state"]``) this used to read
    is gone -- its own source, the Postgres ``drive_audits`` table, lost its
    last producer in Wave 1 and chat_stance.py stopped populating this ctx
    key in Wave 2a. The AutonomyStateV2 reducer this replaced on 2026-07-16
    was already retired before that; there is no fallback path. ``pressure_
    trend`` and ``confidence`` remain always ``None`` for the same "no honest
    signal to report" reason they always were.

    Returns None (omit, not empty) when there are no recent dispatch actions
    -- never fabricates a dominant_drive or tension.
    """
    recent_actions = _format_recent_actions(ctx.get("chat_recent_dispatch_actions"), max_recent_actions)

    if not recent_actions:
        return None

    try:
        return AutonomySliceV1(
            dominant_drive=None,
            active_tensions=[],
            pressure_trend=None,
            confidence=None,
            recent_actions=recent_actions,
        )
    except Exception:
        logger.warning("autonomy_slice_build_failed", exc_info=True)
        return None
