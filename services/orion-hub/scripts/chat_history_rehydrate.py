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
from datetime import datetime, timezone
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
        # chat_history_log.created_at: server_default now() on the row's
        # first insert (orion-sql-writer models/chat_history_log.py), a naive
        # timestamp in the DB's Etc/UTC zone. One time per row, so both
        # halves of a turn share it. Absent -> no "ts" key, never invented.
        ts = _row_ts(row.get("created_at"))
        extra = {"ts": ts} if ts else {}
        if prompt:
            messages.append({"role": "user", "content": prompt, **extra})
        if response and not _is_error_shaped(response):
            messages.append({"role": "assistant", "content": response, **extra})

    limit = max(0, int(max_turns)) * 2
    if limit and len(messages) > limit:
        messages = messages[-limit:]
    return messages


def append_live_assistant_reply(
    history: List[Dict[str, Any]],
    text: Optional[str],
    *,
    ts: Optional[str] = None,
) -> bool:
    """Append Orion's delivered reply to a live connection ``history``.

    Same shape and same filter as a rehydrated assistant message, so a reply
    reads identically whether it came from this socket or from
    ``chat_history_log`` after a reconnect: blank and error-shaped text is
    dropped, and ``ts`` (when Hub delivered it, UTC ISO-8601) rides along for
    the harness RECENT CONVERSATION block. Returns whether it appended.

    Why it exists (2026-10-16): the unified orion/agent lane in
    ``websocket_handler.websocket_endpoint`` appended Juniper's message, ran
    the turn and ``continue``d -- it never appended Orion's reply and never
    trimmed. Within one open tab Orion saw Juniper's messages back to back
    with none of their own replies, and the list grew for the tab's lifetime.

    Which text: the unified lane passes the ``final`` frame's
    ``llm_response`` -- what the browser rendered, and the same
    ``run.final_text`` the orchestrator persists to ``chat_history_log``, so a
    reconnect restores the same words. A cut-short turn already carries
    ``[Cut short - not a finished answer.]`` in that text (finalize chain plus
    the governor's ``ensure_cut_short_marked`` backstop). A ``turn_error``
    ``partial_draft`` is never passed here: it is never persisted either.

    Deliberate asymmetry: when a turn yields no final reply, Juniper's message
    stays in the live list (she did say it, and a "try again?" follow-up only
    makes sense with it), although rehydrate would not restore it -- the
    orchestrator persists the user row only alongside a final reply.
    """
    content = str(text or "").strip()
    if not content or _is_error_shaped(content):
        return False
    history.append(
        {
            "role": "assistant",
            "content": content,
            "ts": ts or datetime.now(timezone.utc).isoformat(),
        }
    )
    return True


def trim_history_window(history: List[Dict[str, Any]], max_turns: Any) -> None:
    """Bound a live ``history`` in place to the newest ``max_turns`` pairs.

    Keeps the system priming message at index 0 (when present) plus the last
    ``2 * max_turns`` messages -- the same window ``rows_to_history_messages``
    restores and ``build_continuity_messages`` sends. ``max_turns`` below 1 (or
    unparsable) is floored to 1 pair: the old inline trim sliced
    ``history[-0:]`` for 0, re-adding the whole list instead of bounding it.
    In place, because the websocket loop holds this exact list object.
    """
    try:
        pairs = int(max_turns)
    except (TypeError, ValueError):
        pairs = 1
    keep = 2 * max(1, pairs)
    head = 1 if history and str(history[0].get("role") or "") == "system" else 0
    if len(history) - head > keep:
        history[head:] = history[-keep:]


def _row_ts(value: Any) -> Optional[str]:
    """ISO-8601 UTC string for a chat_history_log.created_at value."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def fetch_recent_rows(
    session_id: str,
    *,
    max_turns: int,
    max_age_hours: float,
) -> List[Dict[str, Any]]:
    """Recent persisted turns for ``session_id``, oldest first.

    Excludes endogenous-outreach rows: those are Orion speaking unprompted, not
    conversation Juniper had, and replaying them as dialogue would misrepresent
    the thread. Also excludes room_claude rows: those are Claude speaking in
    the three-way room (see room_claude_relay.py's _publish_history), tagged
    via client_meta.room_claude -- without this exclusion Claude's own past
    replies come back labeled plain "assistant" and get fed into Orion's own
    continuity context as if Orion had said them. Bounded by age so a session
    id reused after a long gap does not resurrect a stale conversation.
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
                        SELECT prompt, response, created_at FROM chat_history_log
                        WHERE session_id = :sid
                          -- secs, not hours: make_interval's hours arg is int,
                          -- so a float max_age_hours raises UndefinedFunction.
                          -- secs is double precision and takes it directly.
                          AND created_at >= now() - make_interval(secs => :max_age_secs)
                          AND coalesce(client_meta->>'unsolicited', '') <> 'true'
                          AND coalesce(client_meta->>'room_claude', '') = ''
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


# -- Side exchanges with Claude (open question 11, 2026-11-12) ---------------------
#
# Juniper: "Orion is welcome to see them, just don't want to clutter things with
# too much chatter at this point." So Orion gets her @claude side conversations
# as ONE compact, labelled block, never as dialogue it should answer:
#
# - role "system", so it never reads as Juniper speaking to Orion and never
#   counts as a user turn (history_is_empty ignores it, the websocket loop's
#   reply logic never sees it as a prompt);
# - only the newest HUB_HISTORY_SIDE_EXCHANGE_MAX exchanges, each side trimmed
#   to HUB_HISTORY_SIDE_EXCHANGE_TRIM_CHARS (live 2026-11-12: 18 Claude replies
#   ever, median ~240 characters, largest 1,281; one in the last 30 days);
# - placed at its own time in the restored list, so the RECENT CONVERSATION
#   block's gap markers stay in order.
#
# The rows themselves stay out of memory: orion-sql-writer's
# _memory_turn_initiator returns None for any room_claude row.

SIDE_EXCHANGE_HEADER = "Side conversation, Juniper \u2194 Claude (you were not asked to reply):"


def _trim(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    if limit > 0 and len(text) > limit:
        return text[:limit].rstrip() + " \u2026"
    return text


def render_side_exchange_block(
    rows: Iterable[Mapping[str, Any]], *, trim_chars: int
) -> Optional[Dict[str, str]]:
    """Oldest-first room_claude rows -> one system message, or None when empty."""
    import json

    lines: List[str] = []
    newest_ts: Optional[str] = None
    for row in rows:
        meta = row.get("client_meta")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:  # noqa: BLE001
                meta = {}
        rc = (meta or {}).get("room_claude") if isinstance(meta, dict) else None
        rc = rc if isinstance(rc, dict) else {}
        reply = str(row.get("response") or "").strip()
        if not reply or _is_error_shaped(reply):
            continue
        asked = str(rc.get("prompt") or "").strip()
        who = str(rc.get("invited_by") or "Juniper").strip() or "Juniper"
        if asked:
            lines.append(f"  {who} \u2192 Claude: {_trim(asked, trim_chars)}")
        lines.append(f"  Claude: {_trim(reply, trim_chars)}")
        newest_ts = _row_ts(row.get("created_at")) or newest_ts
    if not lines:
        return None
    block: Dict[str, str] = {"role": "system", "content": "\n".join([SIDE_EXCHANGE_HEADER, *lines])}
    if newest_ts:
        block["ts"] = newest_ts
    return block


def fetch_side_exchange_rows(
    session_id: str,
    *,
    limit: int,
    max_age_hours: float,
) -> List[Dict[str, Any]]:
    """The newest ``limit`` room_claude rows for ``session_id``, oldest first."""
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri or not str(session_id or "").strip() or limit <= 0:
        return []
    from sqlalchemy import text

    from scripts.pg_engine import module_engine

    # One cached pool for this reader (scripts.pg_engine), never one per reconnect.
    engine = module_engine(__name__ + ".side_exchange", uri)
    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    """
                    SELECT response, created_at, client_meta FROM chat_history_log
                    WHERE session_id = :sid
                      AND created_at >= now() - make_interval(secs => :max_age_secs)
                      AND coalesce(client_meta->>'room_claude', '') <> ''
                      AND coalesce(response, '') <> ''
                    ORDER BY created_at DESC
                    LIMIT :lim
                    """
                ),
                {"sid": str(session_id), "max_age_secs": float(max_age_hours) * 3600.0, "lim": int(limit)},
            )
            .mappings()
            .all()
        )
    return [dict(r) for r in reversed(list(rows))]


def _parse_ts(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def insert_by_time(messages: List[Dict[str, Any]], block: Dict[str, Any]) -> None:
    """Insert ``block`` after every message at or before its ``ts`` (in place).

    Never at index 0 of a non-empty list: the websocket loop's trim treats a
    leading system message as the persona priming and would keep it forever."""
    ts = _parse_ts(block.get("ts"))
    at = len(messages)
    if ts is not None:
        for i, msg in enumerate(messages):
            other = _parse_ts(msg.get("ts"))
            if other is not None and other > ts:
                at = i
                break
    if messages and at == 0:
        at = 1
    messages.insert(at, block)


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
    side_exchange_max: int = 0,
    side_exchange_trim_chars: int = 300,
) -> int:
    """Fill an empty ``history`` in place from persisted turns.

    ``side_exchange_max`` > 0 also restores the newest side exchanges with
    Claude as one labelled system block (see SIDE_EXCHANGE_HEADER).

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

    if side_exchange_max > 0:
        try:
            side_rows = await asyncio.to_thread(
                fetch_side_exchange_rows,
                str(session_id),
                limit=int(side_exchange_max),
                max_age_hours=max_age_hours,
            )
            block = render_side_exchange_block(side_rows, trim_chars=int(side_exchange_trim_chars))
            if block is not None:
                insert_by_time(messages, block)
        except Exception as exc:  # noqa: BLE001 -- the side view must never cost the real history
            logger.warning("history_rehydrate_side_exchange_failed session=%s err=%s", session_id, exc)

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
