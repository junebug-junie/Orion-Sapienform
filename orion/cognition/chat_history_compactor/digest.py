from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_URL, uuid5

from orion.cognition.chat_history_compactor.constants import (
    CARD_SUMMARY_MAX_CHARS,
    COMPACTOR_MAX_TURNS,
    DIGEST_TURN_PROMPT_MAX_CHARS,
    DIGEST_TURN_RESPONSE_MAX_CHARS,
    JOURNAL_TITLE_MAX_CHARS,
)
from orion.cognition.compactor.budget import fit_fields_within_budget
from orion.cognition.compactor.chunking import chunk_items_by_char_budget
from orion.cognition.compactor.constants import DIGEST_INPUT_CHAR_BUDGET
from orion.cognition.compactor.digest import parse_compactor_digest_json
from orion.cognition.compactor.truncate import truncate_at_word_boundary
from orion.schemas.actions.chat_history_compactor import ChatHistoryCompactorDigestV1
from orion.schemas.discussion_window import DiscussionWindowResultV1

_COMPACTOR_JOURNAL_ENTRY_NS = NAMESPACE_URL


def _compact_turn(turn) -> tuple[dict[str, Any], bool]:
    prompt, prompt_truncated = truncate_at_word_boundary(str(turn.prompt or ""), DIGEST_TURN_PROMPT_MAX_CHARS)
    response, response_truncated = truncate_at_word_boundary(str(turn.response or ""), DIGEST_TURN_RESPONSE_MAX_CHARS)
    truncated = prompt_truncated or response_truncated
    compact = {
        "created_at": turn.created_at.isoformat() if turn.created_at else None,
        "correlation_id": turn.correlation_id,
        "user_id": turn.user_id,
        "source": turn.source,
        "prompt": prompt,
        "response": response,
    }
    if truncated:
        compact["truncated"] = True
    return compact, truncated


def build_chat_history_compactor_digest_inputs(
    window: DiscussionWindowResultV1,
    *,
    budget_chars: int = DIGEST_INPUT_CHAR_BUDGET,
    fetch_limit: int = COMPACTOR_MAX_TURNS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every turn in the window -> one or more digest inputs, each within `budget_chars`.

    Returns ``(inputs, stats)``. One input when the day fits one call; several
    when it does not (map step, then the orch merges the partial digests). No
    turn is dropped here. ``input_truncated`` is true when a turn hit the
    per-turn safety cap, or when the window fetch returned exactly
    ``fetch_limit`` turns (the skill may have had more).
    """
    turns = list(window.turns or [])
    compact_turns: list[dict[str, Any]] = []
    any_turn_truncated = False
    for turn in turns:
        compact, truncated = _compact_turn(turn)
        any_turn_truncated = any_turn_truncated or truncated
        compact_turns.append(compact)
    chunks = chunk_items_by_char_budget(compact_turns, budget_chars=budget_chars) if compact_turns else []
    inputs: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        payload: dict[str, Any] = {
            "window_start_utc": window.window_start_utc.isoformat(),
            "window_end_utc": window.window_end_utc.isoformat(),
            "turn_count": len(compact_turns),
            "selection_strategy": window.selection_strategy,
            "turns": chunk,
        }
        if len(chunks) > 1:
            payload["chunk_index"] = index
            payload["chunk_count"] = len(chunks)
        if any(t.get("truncated") for t in chunk):
            payload["turn_content_truncated"] = True
        inputs.append(payload)
    fetch_capped = fetch_limit > 0 and len(turns) >= fetch_limit
    stats = {
        "total_count": len(turns),
        "covered_count": sum(len(chunk) for chunk in chunks),
        "input_truncated": bool(any_turn_truncated or fetch_capped),
        "turn_content_truncated": any_turn_truncated,
        "fetch_limit_hit": fetch_capped,
        "chunk_count": len(chunks),
    }
    return inputs, stats


def build_chat_history_compactor_merge_input(
    *,
    base_input: dict[str, Any],
    partial_digests: list[ChatHistoryCompactorDigestV1],
) -> dict[str, Any]:
    payload = {
        key: base_input.get(key)
        for key in ("window_start_utc", "window_end_utc", "turn_count", "selection_strategy")
        if base_input.get(key) is not None
    }
    payload["partial_digests"] = [
        {
            "chunk_index": index,
            "card_summary": digest.card_summary,
            "journal_title": digest.journal_title,
            "journal_body": digest.journal_body,
            "turn_refs": list(digest.turn_refs or []),
        }
        for index, digest in enumerate(partial_digests, start=1)
    ]
    return payload


def concatenate_chat_partial_digests(
    partial_digests: list[ChatHistoryCompactorDigestV1],
    *,
    window_label: str,
) -> ChatHistoryCompactorDigestV1:
    """Deterministic reduce used only when every merge call failed (recorded as merge_mode=concatenated)."""
    refs: list[str] = []
    for digest in partial_digests:
        for ref in digest.turn_refs or []:
            if ref not in refs:
                refs.append(ref)
    body = "\n\n".join(
        f"Part {index} of {len(partial_digests)}\n\n{(digest.journal_body or digest.card_summary).strip()}"
        for index, digest in enumerate(partial_digests, start=1)
    )
    summary = " ".join(d.card_summary.strip() for d in partial_digests if d.card_summary.strip())
    return ChatHistoryCompactorDigestV1(
        card_summary=summary or f"Chat digest — {window_label}",
        journal_title=f"Chat digest — {window_label}",
        journal_body=body,
        turn_refs=refs,
    )


def fit_chat_compactor_digest_within_budget(
    digest: ChatHistoryCompactorDigestV1,
) -> tuple[ChatHistoryCompactorDigestV1, list[str]]:
    """Trim the memory-card fields (card_summary, journal_title) to their caps.

    ``journal_body`` is deliberately NOT trimmed (stored in full for the daily
    letter). Returns ``(digest, trimmed_field_names)``.
    """
    fitted, trimmed = fit_fields_within_budget(
        {
            "card_summary": (digest.card_summary, CARD_SUMMARY_MAX_CHARS),
            "journal_title": (digest.journal_title or "", JOURNAL_TITLE_MAX_CHARS),
        }
    )
    if not trimmed:
        return digest, []
    return digest.model_copy(update=fitted), trimmed


def build_quiet_day_chat_digest(*, window_label: str) -> ChatHistoryCompactorDigestV1:
    label = (window_label or "window").strip()
    return ChatHistoryCompactorDigestV1(
        card_summary=f"No Hub chat turns in {label}.",
        journal_title=f"Chat digest — {label} (quiet)",
        journal_body=(
            f"No chat_history_log turns were found for {label}. "
            "No indexed chat digest memory card was written."
        ),
        turn_refs=[],
    )


def parse_chat_history_compactor_digest_json(raw: str) -> ChatHistoryCompactorDigestV1:
    return parse_compactor_digest_json(raw, ChatHistoryCompactorDigestV1)


def stable_chat_compactor_journal_entry_id(*, workflow_id: str, compactor_index: str) -> str:
    payload = "|".join([workflow_id.strip(), compactor_index.strip()])
    return str(uuid5(_COMPACTOR_JOURNAL_ENTRY_NS, payload))
