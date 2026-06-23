"""Hydrate consolidation suggest drafts with turn text from chat_history_log."""

from __future__ import annotations

from typing import Any, Mapping

import asyncpg

_MAX_TURN_FIELD_CHARS = 800


def _clip(text: str) -> str:
    s = (text or "").strip()
    if len(s) <= _MAX_TURN_FIELD_CHARS:
        return s
    return s[: _MAX_TURN_FIELD_CHARS - 3] + "..."


def format_turn_utterance_text(prompt: str | None, response: str | None) -> str:
    p = _clip(str(prompt or ""))
    r = _clip(str(response or ""))
    if p and r:
        return f"User: {p}\nOrion: {r}"
    return p or r


def hydrate_draft_utterance_text(
    draft: dict[str, Any],
    *,
    turn_correlation_ids: list[str],
    turns_by_correlation: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Map utterance_ids to turn text by index-aligned correlation ids."""
    if not isinstance(draft, dict):
        return draft
    uids = draft.get("utterance_ids") if isinstance(draft.get("utterance_ids"), list) else []
    corr_ids = [str(c).strip() for c in (turn_correlation_ids or []) if str(c).strip()]
    text_map = (
        dict(draft.get("utterance_text_by_id") or {})
        if isinstance(draft.get("utterance_text_by_id"), dict)
        else {}
    )
    changed = False
    for i, uid in enumerate(uids):
        key = str(uid).strip()
        if not key or str(text_map.get(key) or "").strip():
            continue
        corr = corr_ids[i] if i < len(corr_ids) else ""
        turn = turns_by_correlation.get(corr) if corr else None
        if turn:
            text = format_turn_utterance_text(turn.get("prompt"), turn.get("response"))
            if text:
                text_map[key] = text
                changed = True
    if not changed:
        return draft
    out = dict(draft)
    out["utterance_text_by_id"] = text_map
    return out


async def fetch_turn_text_by_correlation(
    pool: asyncpg.Pool,
    correlation_ids: list[str],
) -> dict[str, dict[str, Any]]:
    ids = [str(c).strip() for c in correlation_ids if str(c).strip()]
    if not ids:
        return {}
    rows = await pool.fetch(
        """
        SELECT correlation_id::text AS correlation_id, prompt, response
        FROM chat_history_log
        WHERE correlation_id = ANY($1::text[])
        """,
        ids,
    )
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        cid = str(row["correlation_id"] or "").strip()
        if cid:
            out[cid] = {"prompt": row["prompt"], "response": row["response"]}
    return out


async def hydrate_consolidation_draft_dict(
    pool: asyncpg.Pool,
    draft: dict[str, Any],
    turn_correlation_ids: list[str],
) -> dict[str, Any]:
    turns = await fetch_turn_text_by_correlation(pool, turn_correlation_ids)
    return hydrate_draft_utterance_text(
        draft,
        turn_correlation_ids=turn_correlation_ids,
        turns_by_correlation=turns,
    )
