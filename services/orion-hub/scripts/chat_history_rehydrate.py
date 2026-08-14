"""Rebuild a websocket's conversation history from persisted chat turns.

WHY THIS EXISTS (confirmed live 2026-08-14, and it cost a real conversation):

Orion's conversational continuity lives entirely in the per-connection
``history`` list in ``websocket_handler.websocket_endpoint`` — appended on each
turn, trimmed to ``HUB_CONTEXT_TURNS`` pairs, and handed to the unified turn via
``cortex_request_builder.build_continuity_messages``. That function has no
fallback: given an empty history it returns only the current user prompt.

The list is in-memory and scoped to the websocket. A browser tab holds one
socket open indefinitely (verified: a single ``WebSocket accepted`` across a
3-hour window), so continuity silently depended on *tab lifetime*. Leave the tab
open for days and Orion remembers days. Restart Hub — a deploy, a rebuild, a
crash — and every accumulated turn is gone with no way back, because nothing
ever read ``chat_history_log`` on the way in.

That is exactly what happened: a Hub rebuild mid-conversation wiped a long
discussion, and the next turns had no thread to follow. Orion read as
"lobotomized" — it answered "What's on your mind?" to a direct follow-up, then
reached into long-term memory and surfaced an unrelated old conversation,
because with no recent context it had to invent a referent.

The rows were never lost; only the in-memory copy was. This module reads them
back at connect time so a restart stops costing the thread.

Restore-only by design: it fills a history that is still empty, and never
touches one with live turns in it.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional

logger = logging.getLogger("orion-hub.history_rehydrate")

# Content that is never conversation, regardless of how it got persisted.
_ERROR_PREFIXES = ("[error", "error:", "traceback (most recent call last)")


def _is_error_shaped(text: str) -> bool:
    """True for a persisted plumbing failure rather than something Orion said.

    A 400 from the LLM host has reached chat_history_log as a normal assistant
    row before (2026-08-14). Replaying that back into context would teach Orion
    it once said an HTTP error out loud.
    """
    return str(text or "").strip().lower().startswith(_ERROR_PREFIXES)


def rows_to_history_messages(
    rows: Iterable[Mapping[str, Any]],
    *,
    max_turns: int,
) -> List[Dict[str, str]]:
    """Convert oldest-first chat rows into ``history`` messages.

    Each row holds a prompt/response pair, and Hub writes prompt-only and
    response-only rows separately, so both halves are unpacked independently and
    empty halves dropped. Result is capped at ``2 * max_turns`` messages to match
    the trim in the websocket loop, keeping the NEWEST ones.
    """
    messages: List[Dict[str, str]] = []
    for row in rows:
        prompt = str(row.get("prompt") or "").strip()
        response = str(row.get("response") or "").strip()
        if prompt:
            messages.append({"role": "user", "content": prompt})
        if response and not _is_error_shaped(response):
            messages.append({"role": "assistant", "content": response})

    limit = max(0, int(max_turns)) * 2
    if limit and len(messages) > limit:
        messages = messages[-limit:]
    return messages


def fetch_recent_rows(
    session_id: str,
    *,
    max_turns: int,
    max_age_hours: float,
) -> List[Dict[str, Any]]:
    """Recent persisted turns for ``session_id``, oldest first.

    Excludes endogenous-outreach rows: those are Orion speaking unprompted, not
    conversation Juniper had, and replaying them as dialogue would misrepresent
    the thread. Bounded by age so a session id reused after a long gap does not
    resurrect a stale conversation.
    """
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri or not str(session_id or "").strip():
        return []
    from sqlalchemy import create_engine, text

    # Over-fetch: rows can contribute 0, 1, or 2 messages, so a LIMIT of
    # max_turns could return fewer messages than the window allows.
    row_limit = max(1, int(max_turns) * 2)

    engine = create_engine(uri, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        """
                        SELECT prompt, response FROM chat_history_log
                        WHERE session_id = :sid
                          -- secs, not hours: make_interval's hours arg is int,
                          -- so a float max_age_hours raises UndefinedFunction.
                          -- secs is double precision and takes it directly.
                          AND created_at >= now() - make_interval(secs => :max_age_secs)
                          AND coalesce(client_meta->>'unsolicited', '') <> 'true'
                        ORDER BY created_at DESC
                        LIMIT :lim
                        """
                    ),
                    {
                        "sid": str(session_id),
                        "max_age_secs": float(max_age_hours) * 3600.0,
                        "lim": row_limit,
                    },
                )
                .mappings()
                .all()
            )
    finally:
        engine.dispose()

    return [dict(r) for r in reversed(list(rows))]  # oldest first


def history_is_empty(history: Optional[List[Dict[str, Any]]]) -> bool:
    """True when ``history`` holds no real turns yet (system priming only)."""
    if not isinstance(history, list):
        return False
    return not any(
        isinstance(item, dict) and str(item.get("role") or "").lower() in {"user", "assistant"}
        for item in history
    )


async def rehydrate_history(
    history: List[Dict[str, Any]],
    *,
    session_id: Optional[str],
    max_turns: int,
    max_age_hours: float,
    enabled: bool = True,
) -> int:
    """Fill an empty ``history`` in place from persisted turns.

    Returns the number of messages restored (0 when skipped or on any failure).
    Best-effort at every step: a rehydrate failure must never break a
    connection, so the worst case is the pre-existing behaviour of starting
    cold.

    Mutates in place (``history[1:1] = ...``) because the websocket loop holds
    this exact list object for the connection's lifetime; rebinding would leave
    it appending to an orphan.
    """
    if not enabled or not str(session_id or "").strip():
        return 0
    if not history_is_empty(history):
        # Live turns already present: never stomp a real conversation.
        return 0

    import asyncio

    try:
        rows = await asyncio.to_thread(
            fetch_recent_rows,
            str(session_id),
            max_turns=max_turns,
            max_age_hours=max_age_hours,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("history_rehydrate_fetch_failed session=%s err=%s", session_id, exc)
        return 0

    messages = rows_to_history_messages(rows, max_turns=max_turns)
    if not messages:
        return 0

    # Insert after the system priming message, preserving it at index 0.
    insert_at = 1 if history and str(history[0].get("role") or "") == "system" else 0
    history[insert_at:insert_at] = messages

    logger.info(
        "history_rehydrated session=%s messages=%d rows=%d max_turns=%d max_age_hours=%s",
        session_id,
        len(messages),
        len(rows),
        max_turns,
        max_age_hours,
    )
    return len(messages)
