from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.cognition.chat_history_compactor.constants import CARD_SUMMARY_MAX_CHARS
from orion.cognition.chat_history_compactor.digest import (
    assert_chat_compactor_digest_within_budget,
    build_quiet_day_chat_digest,
    parse_chat_history_compactor_digest_json,
    stable_chat_compactor_journal_entry_id,
    trim_chat_history_compactor_input,
)
from orion.schemas.actions.chat_history_compactor import ChatHistoryCompactorDigestV1
from orion.schemas.discussion_window import DiscussionWindowResultV1, DiscussionWindowTurnV1


def test_chat_history_compactor_digest_v1_rejects_empty_card_summary() -> None:
    with pytest.raises(ValidationError):
        ChatHistoryCompactorDigestV1(
            card_summary="",
            journal_title="Title",
            journal_body="Body",
            turn_refs=["corr-1"],
        )


def test_assert_chat_compactor_digest_within_budget() -> None:
    digest = ChatHistoryCompactorDigestV1(
        card_summary="x" * (CARD_SUMMARY_MAX_CHARS + 1),
        journal_title="t",
        journal_body="b",
        turn_refs=[],
    )
    with pytest.raises(ValueError, match="compactor_output_over_budget:card_summary"):
        assert_chat_compactor_digest_within_budget(digest)


def test_build_quiet_day_chat_digest() -> None:
    digest = build_quiet_day_chat_digest(window_label="2026-07-08")
    assert "No Hub chat turns" in digest.card_summary or "quiet" in digest.card_summary.lower() or "No" in digest.card_summary
    assert digest.turn_refs == []


def test_stable_chat_compactor_journal_entry_id_is_deterministic() -> None:
    a = stable_chat_compactor_journal_entry_id(
        workflow_id="chat_history_compactor_pass",
        compactor_index="chat_compactor:day:2026-07-08",
    )
    b = stable_chat_compactor_journal_entry_id(
        workflow_id="chat_history_compactor_pass",
        compactor_index="chat_compactor:day:2026-07-08",
    )
    assert a == b


def test_parse_chat_history_compactor_digest_json() -> None:
    raw = '{"card_summary":"Talked about memory cards.","journal_title":"Chat digest","journal_body":"Details.","turn_refs":["c1"]}'
    digest = parse_chat_history_compactor_digest_json(raw)
    assert digest.card_summary.startswith("Talked")
    assert digest.turn_refs == ["c1"]


def test_trim_chat_history_compactor_input_bounds_turns() -> None:
    turns = [
        DiscussionWindowTurnV1(
            created_at=datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc),
            correlation_id=f"c{i}",
            prompt="p" * 2000,
            response="r" * 2000,
        )
        for i in range(5)
    ]
    window = DiscussionWindowResultV1(
        window_start_utc=datetime(2026, 7, 8, 0, 0, tzinfo=timezone.utc),
        window_end_utc=datetime(2026, 7, 8, 23, 59, tzinfo=timezone.utc),
        turn_count=5,
        turns=turns,
        transcript_text="ignored",
    )
    trimmed = trim_chat_history_compactor_input(window, max_turns=3)
    assert len(trimmed["turns"]) == 3
    assert trimmed["turn_count"] == 5
    assert trimmed["turns_truncated_for_digest"] is True
    assert len(trimmed["turns"][0]["prompt"]) <= 401
