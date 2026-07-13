from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_URL, uuid5

from orion.cognition.chat_history_compactor.constants import (
    CARD_SUMMARY_MAX_CHARS,
    DEFAULT_MAX_TURNS,
    DIGEST_TURN_PROMPT_MAX_CHARS,
    DIGEST_TURN_RESPONSE_MAX_CHARS,
    JOURNAL_BODY_MAX_CHARS,
    JOURNAL_TITLE_MAX_CHARS,
)
from orion.cognition.compactor.budget import assert_fields_within_budget
from orion.cognition.compactor.digest import parse_compactor_digest_json
from orion.schemas.actions.chat_history_compactor import ChatHistoryCompactorDigestV1
from orion.schemas.discussion_window import DiscussionWindowResultV1

_COMPACTOR_JOURNAL_ENTRY_NS = NAMESPACE_URL


def trim_chat_history_compactor_input(
    window: DiscussionWindowResultV1,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> dict[str, Any]:
    turns = list(window.turns or [])
    total = len(turns)
    # Keep newest suffix (discussion window already contiguous-suffix oriented)
    selected = turns[-max_turns:] if max_turns > 0 else []
    compact_turns = []
    for turn in selected:
        prompt = str(turn.prompt or "")
        response = str(turn.response or "")
        if len(prompt) > DIGEST_TURN_PROMPT_MAX_CHARS:
            prompt = prompt[:DIGEST_TURN_PROMPT_MAX_CHARS].rstrip() + "…"
        if len(response) > DIGEST_TURN_RESPONSE_MAX_CHARS:
            response = response[:DIGEST_TURN_RESPONSE_MAX_CHARS].rstrip() + "…"
        compact_turns.append(
            {
                "created_at": turn.created_at.isoformat() if turn.created_at else None,
                "correlation_id": turn.correlation_id,
                "user_id": turn.user_id,
                "source": turn.source,
                "prompt": prompt,
                "response": response,
            }
        )
    payload: dict[str, Any] = {
        "window_start_utc": window.window_start_utc.isoformat(),
        "window_end_utc": window.window_end_utc.isoformat(),
        "turn_count": int(window.turn_count),
        "selection_strategy": window.selection_strategy,
        "turns": compact_turns,
    }
    if total > len(selected):
        payload["turns_truncated_for_digest"] = True
        payload["turns_total"] = total
    return payload


def assert_chat_compactor_digest_within_budget(digest: ChatHistoryCompactorDigestV1) -> None:
    assert_fields_within_budget(
        {
            "card_summary": (digest.card_summary, CARD_SUMMARY_MAX_CHARS),
            "journal_title": (digest.journal_title or "", JOURNAL_TITLE_MAX_CHARS),
            "journal_body": (digest.journal_body or "", JOURNAL_BODY_MAX_CHARS),
        }
    )


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
