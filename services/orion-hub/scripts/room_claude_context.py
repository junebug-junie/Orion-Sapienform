"""What Claude sees when invited into Hub chat: the whole conversation (Patch 1a).

Design: docs/superpowers/specs/2026-11-12-claude-hub-tools-orion-consent-design.md
section 1. Before this, Claude got the last 12 turns from the last 48 hours,
without Orion's unprompted messages, without its own earlier replies, and
with every line stamped "now".

What this builds, for one Hub ``session_id``:

- **The conversation**, newest rows first, whole rows only, until the next row
  would pass ``max_chars`` (``HUB_ROOM_CLAUDE_CONTEXT_MAX_CHARS``, 160,000).
  No age limit. Each line carries its real ``created_at`` and a short turn ref
  (first 8 characters of the correlation id). Orion's unprompted messages are
  labelled; Claude's own earlier side exchanges (Juniper's @claude prompt and
  Claude's reply, one ``room_claude`` row) are marked ``side_exchange``.
- **One line about what was left out**: ``[Earlier: N turns from A to B not
  shown.]``.
- **A short index of what is open in Hub now** (names and counts only).

Only text counts toward the cap; a single row over 12,000 characters keeps
its first 8,000 and last 3,000 with a cut marker (none exist live today; the
largest row on 2026-11-12 was 9,694).

Best-effort everywhere: a database or Redis failure gives Claude less context,
never a failed invite.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from scripts.chat_history_rehydrate import _is_error_shaped, _row_ts

logger = logging.getLogger("orion-hub.room_claude_context")

LONG_ROW_CHARS = 12_000
LONG_ROW_HEAD = 8_000
LONG_ROW_TAIL = 3_000
# Rows read per invite. The cap is in characters; this only bounds the query.
# The live session's shortest rows are a few hundred characters, so 1,500 rows
# is always more than 160,000 characters' worth.
FETCH_ROW_LIMIT = 1_500
HUB_INDEX_MAX_CHARS = 800
TAB_FRESH_MINUTES = 3

ORION_NAME = "Oríon"


def clip_long_text(text: str) -> str:
    """Keep a very long single row readable instead of dropping it whole."""
    if len(text) <= LONG_ROW_CHARS:
        return text
    cut = len(text) - LONG_ROW_HEAD - LONG_ROW_TAIL
    return f"{text[:LONG_ROW_HEAD]}\n[... {cut} characters cut ...]\n{text[-LONG_ROW_TAIL:]}"


def _meta(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:  # noqa: BLE001
            return {}
    return value if isinstance(value, dict) else {}


def row_entries(row: Mapping[str, Any], *, claude_name: str = "Claude") -> List[Dict[str, Any]]:
    """One ``chat_history_log`` row -> 0..2 transcript entries (RoomTranscriptEntryV1 shape)."""
    meta = _meta(row.get("client_meta"))
    corr = str(row.get("correlation_id") or row.get("id") or "").strip()
    ref = corr[:8] or None
    ts = _row_ts(row.get("created_at"))
    base: Dict[str, Any] = {"turn_ref": ref}
    if ts:
        base["created_at"] = ts
    prompt = str(row.get("prompt") or "").strip()
    response = str(row.get("response") or "").strip()
    out: List[Dict[str, Any]] = []

    room_claude = meta.get("room_claude")
    if room_claude:
        # A side exchange with Claude: one row, Claude's reply in `response`,
        # and (since 2026-11-12) the prompt that asked for it in client_meta.
        rc = room_claude if isinstance(room_claude, dict) else {}
        side_prompt = str(rc.get("prompt") or "").strip()
        asked_by = str(rc.get("invited_by") or "Juniper").strip() or "Juniper"
        if side_prompt:
            is_orion = asked_by.casefold() in ("orion", ORION_NAME.casefold())
            out.append({
                **base,
                "speaker_id": "orion" if is_orion else "juniper",
                "speaker_name": f"{asked_by} (to {claude_name})",
                "speaker_kind": "peer_ai" if is_orion else "human",
                "text": clip_long_text(side_prompt),
                "side_exchange": True,
            })
        if response:
            out.append({
                **base,
                "speaker_id": "claude",
                "speaker_name": claude_name,
                "speaker_kind": "peer_ai",
                "text": clip_long_text(response),
                "side_exchange": True,
            })
        return out

    if prompt:
        out.append({
            **base, "speaker_id": "juniper", "speaker_name": "Juniper", "speaker_kind": "human",
            "text": clip_long_text(prompt),
        })
    if response and not _is_error_shaped(response):
        out.append({
            **base, "speaker_id": "orion", "speaker_name": ORION_NAME, "speaker_kind": "peer_ai",
            "text": clip_long_text(response),
            "unprompted": meta.get("unsolicited") is True or str(meta.get("unsolicited")).lower() == "true",
        })
    return out


def select_window(
    rows_newest_first: Iterable[Mapping[str, Any]],
    *,
    max_chars: int,
    claude_name: str = "Claude",
) -> Tuple[List[Dict[str, Any]], int, Optional[Mapping[str, Any]]]:
    """Walk back from the newest row, adding whole rows until the next would pass the cap.

    Returns (entries oldest first, rows kept, the first row NOT kept or None).
    A row with no usable text costs nothing and is skipped."""
    kept: List[List[Dict[str, Any]]] = []
    used = 0
    for row in rows_newest_first:
        entries = row_entries(row, claude_name=claude_name)
        if not entries:
            continue
        size = sum(len(e["text"]) for e in entries)
        if used + size > max_chars:
            return [e for group in reversed(kept) for e in group], len(kept), row
        kept.append(entries)
        used += size
    return [e for group in reversed(kept) for e in group], len(kept), None


def transcript_note(omitted: int, first_at: Any, last_at: Any) -> Optional[str]:
    if omitted <= 0:
        return None
    a, b = _day(first_at), _day(last_at)
    span = f" from {a} to {b}" if a and b and a != b else (f" from {a}" if a else "")
    noun = "turn" if omitted == 1 else "turns"
    return f"[Earlier: {omitted} {noun}{span} not shown.]"


def _day(value: Any) -> Optional[str]:
    ts = _row_ts(value)
    return ts[:10] if ts else None


# -- database ------------------------------------------------------------------------

_SESSION_ROWS_SQL = """
SELECT id, correlation_id, prompt, response, created_at, client_meta
FROM chat_history_log
WHERE session_id = :sid
ORDER BY created_at DESC
LIMIT :lim
"""

_OLDER_SUMMARY_SQL = """
SELECT count(*) AS n, min(created_at) AS first_at, max(created_at) AS last_at
FROM chat_history_log
WHERE session_id = :sid AND created_at <= :before
  AND (coalesce(prompt, '') <> '' OR coalesce(response, '') <> '')
"""


def _engine():
    """One cached pool for this module (scripts.pg_engine); never disposed per call."""
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        return None
    from scripts.pg_engine import module_engine

    return module_engine(__name__, uri)


def fetch_conversation(
    session_id: str, *, max_chars: int, claude_name: str = "Claude"
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """(transcript entries oldest first, the 'Earlier' note or None). Sync; run in a thread."""
    engine = _engine()
    if engine is None or not str(session_id or "").strip():
        return [], None
    from sqlalchemy import text

    note: Optional[str] = None
    with engine.connect() as conn:
        rows = [dict(r) for r in conn.execute(
            text(_SESSION_ROWS_SQL), {"sid": str(session_id), "lim": FETCH_ROW_LIMIT}
        ).mappings().all()]
        entries, _n_kept, stopped = select_window(rows, max_chars=max_chars, claude_name=claude_name)
        # Older rows exist when the cap stopped the walk, or the fetch limit did.
        if stopped is not None or len(rows) >= FETCH_ROW_LIMIT:
            before = stopped.get("created_at") if stopped is not None else rows[-1].get("created_at")
            if before is not None:
                summary = conn.execute(
                    text(_OLDER_SUMMARY_SQL), {"sid": str(session_id), "before": before}
                ).mappings().first()
                if summary:
                    note = transcript_note(int(summary["n"] or 0), summary["first_at"], summary["last_at"])
    return entries, note


_INDEX_TAB_SQL = f"""
SELECT tab FROM hub_attention_minute
WHERE last_at >= now() - interval '{TAB_FRESH_MINUTES} minutes' AND tab IS NOT NULL
ORDER BY last_at DESC LIMIT 1
"""
_INDEX_CONSULTS_SQL = """
SELECT consult_id, status, turns FROM orion_consult
WHERE status IN ('pending', 'open') OR (status = 'brief_sent' AND decision IS NULL)
ORDER BY created_at DESC LIMIT 5
"""
_INDEX_ASKS_SQL = """
SELECT count(*) FROM orion_ask
WHERE status = 'open' AND (expires_at IS NULL OR expires_at > now())
"""


def fetch_hub_index_lines() -> List[str]:
    """Names and counts from Postgres, one try per part: a missing table drops
    that line, not the whole index. Sync; run in a thread."""
    engine = _engine()
    if engine is None:
        return []
    from sqlalchemy import text

    lines: List[str] = []
    with engine.connect() as conn:
        for name, sql in (("tab", _INDEX_TAB_SQL), ("consults", _INDEX_CONSULTS_SQL), ("asks", _INDEX_ASKS_SQL)):
            try:
                result = conn.execute(text(sql))
                if name == "tab":
                    row = result.first()
                    if row and row[0]:
                        lines.append(f"- Juniper is on the Hub tab: {row[0]}")
                elif name == "consults":
                    rows = result.all()
                    if rows:
                        parts = ", ".join(f"{r[0]} ({r[1]}, {r[2]} turns)" for r in rows)
                        lines.append(f"- Open consults: {parts}")
                    else:
                        lines.append("- Open consults: none")
                else:
                    lines.append(f"- Open asks waiting on Juniper: {int(result.scalar() or 0)}")
            except Exception:  # noqa: BLE001
                conn.rollback()
                logger.debug("room_claude_hub_index_part_failed part=%s", name, exc_info=True)
    return lines


async def prime_concern_count_line(bus: Any) -> Optional[str]:
    """How many concerns Prime holds: a count only (the text is Orion's)."""
    try:
        from orion.prime.block import PRIME_STATE_KEY, PrimeStateV1

        raw = await bus.redis.get(PRIME_STATE_KEY)
        if not raw:
            return None
        state = PrimeStateV1.model_validate_json(raw)
        return f"- Prime holds {len(state.concerns)} concern(s)"
    except Exception:  # noqa: BLE001
        logger.debug("room_claude_hub_index_prime_failed", exc_info=True)
        return None


def render_hub_index(lines: List[str]) -> Optional[str]:
    lines = [ln for ln in lines if ln]
    if not lines:
        return None
    out: List[str] = []
    for ln in lines:
        if len("\n".join(out + [ln])) > HUB_INDEX_MAX_CHARS:
            break
        out.append(ln)
    return "\n".join(out) or None


async def build_room_claude_context(
    session_id: Optional[str], *, max_chars: int, bus: Any = None, claude_name: str = "Claude"
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
    """(transcript, transcript_note, hub_index) for one invite. Never raises."""
    import asyncio

    entries: List[Dict[str, Any]] = []
    note: Optional[str] = None
    if session_id and max_chars > 0:
        try:
            entries, note = await asyncio.to_thread(
                fetch_conversation, session_id, max_chars=max_chars, claude_name=claude_name
            )
        except Exception:  # noqa: BLE001
            logger.warning("room_claude_context_fetch_failed session=%s", session_id, exc_info=True)
    index_lines: List[str] = []
    try:
        index_lines = await asyncio.to_thread(fetch_hub_index_lines)
    except Exception:  # noqa: BLE001
        logger.debug("room_claude_hub_index_failed", exc_info=True)
    if bus is not None:
        prime = await prime_concern_count_line(bus)
        if prime:
            index_lines.append(prime)
    logger.info(
        "room_claude_context session=%s lines=%d chars=%d earlier=%s index_lines=%d",
        session_id, len(entries), sum(len(e["text"]) for e in entries), bool(note), len(index_lines),
    )
    return entries, note, render_hub_index(index_lines)


__all__ = [
    "build_room_claude_context",
    "clip_long_text",
    "fetch_conversation",
    "render_hub_index",
    "row_entries",
    "select_window",
    "transcript_note",
]

