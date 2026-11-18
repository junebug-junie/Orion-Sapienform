"""Intention ledger D3: stance sees Orion's open promises. Ships OFF; #3531 decides.

Design: docs/superpowers/specs/2026-11-11-intention-ledger-design.md, D3. On Juniper's own turns
(``stance_inputs.utterance_origin == "juniper"``) stance_react gets up to ``CUE_MAX`` open promises,
oldest first, each as the quote Orion wrote and how long ago:

    OPEN PROMISES (mine, not yet kept):
    - [act:act_1a2b] 2 days ago: "I'll bring what surprised me next time we're in a technical thread"

Stance may carry one into its imperative; it is not told to. Nothing else reads this cue (not the
motor, not outreach, not reverie). The read is orion/grammar/promise_view.py (the same open acts the
5b judge is shown), on a worker thread, bounded by ``FETCH_TIMEOUT_SEC``. Fail-open: any miss
renders "" and the prompt is exactly as before.

Why OFF: while the promise acts are shadow (orion/grammar/promise_acts.py SHADOW_ACT_SIGNALS), the
finder's live precision is not yet measured (#3531). A false promise shown to stance would push
Orion to follow up on something Orion never said. ``THOUGHT_OPEN_PROMISES_CUE_ENABLED=true`` turns it
on once #3531 posts live precision.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from orion.schemas.harness_finalize import OpenPromiseRefV1

from .settings import settings

logger = logging.getLogger("orion-thought.open_promises_cue")

CUE_MAX = 3
FETCH_TIMEOUT_SEC = 1.5
HEADER = "OPEN PROMISES (mine, not yet kept):"


def _ago(opened_at: Any, now: datetime) -> str:
    if not isinstance(opened_at, datetime):
        return "earlier"
    if opened_at.tzinfo is None:
        opened_at = opened_at.replace(tzinfo=timezone.utc)
    hours = max(0.0, (now - opened_at).total_seconds() / 3600.0)
    if hours < 1:
        return "less than an hour ago"
    if hours < 24:
        n = int(hours)
        return f"{n} hour{'s' if n != 1 else ''} ago"
    n = int(hours // 24)
    return f"{n} day{'s' if n != 1 else ''} ago"


def render_open_promises_cue(promises: Sequence[OpenPromiseRefV1], opened_at: dict[str, Any],
                             *, now: Optional[datetime] = None) -> str:
    """Pure. Up to CUE_MAX promises in the order given (oldest first); "" when none."""
    if not promises:
        return ""
    now = now or datetime.now(timezone.utc)
    lines = [HEADER]
    for p in list(promises)[:CUE_MAX]:
        lines.append(f'- [act:{p.id}] {_ago(opened_at.get(p.id), now)}: "{p.quote}"')
    return "\n".join(lines)


def is_juniper_turn(stance_inputs: Any) -> bool:
    return isinstance(stance_inputs, dict) and stance_inputs.get("utterance_origin") == "juniper"


async def build_open_promises_cue(correlation_id: str, stance_inputs: Any, *, enabled: Optional[bool] = None) -> str:
    """The cue text, or "" (flag off, not Juniper's turn, nothing open, or an unread view)."""
    if enabled is None:
        enabled = settings.open_promises_cue_enabled
    if not enabled or not is_juniper_turn(stance_inputs):
        return ""
    try:
        from orion.grammar.promise_view import fetch_promise_view

        view = await asyncio.wait_for(asyncio.to_thread(fetch_promise_view, None, correlation_id),
                                      timeout=FETCH_TIMEOUT_SEC)
    except Exception as exc:  # noqa: BLE001 -- incl. the timeout: nothing to say, never a guess
        logger.info("open_promises_cue_skipped corr=%s reason=%s", correlation_id, type(exc).__name__)
        return ""
    if view is None:
        logger.info("open_promises_cue_skipped corr=%s reason=unread", correlation_id)
        return ""
    text = render_open_promises_cue(view.open_promises, view.opened_at)
    if text:
        logger.info("open_promises_cue corr=%s shown=%d", correlation_id, min(len(view.open_promises), CUE_MAX))
    return text


__all__ = ["CUE_MAX", "HEADER", "build_open_promises_cue", "is_juniper_turn", "render_open_promises_cue"]
