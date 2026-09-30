from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from orion.cognition.chat_history_compactor.constants import (
    CARD_SUMMARY_MAX_CHARS,
    DIGEST_TURN_PROMPT_MAX_CHARS,
    JOURNAL_TITLE_MAX_CHARS,
)
from orion.cognition.chat_history_compactor.digest import (
    build_chat_history_compactor_digest_inputs,
    build_chat_history_compactor_merge_input,
    concatenate_chat_partial_digests,
    fit_chat_compactor_digest_within_budget,
    build_quiet_day_chat_digest,
    parse_chat_history_compactor_digest_json,
    stable_chat_compactor_journal_entry_id,
)
from orion.cognition.compactor.constants import DIGEST_INPUT_CHAR_BUDGET
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


def test_fit_chat_compactor_digest_within_budget_repairs_over_limit() -> None:
    digest = ChatHistoryCompactorDigestV1(
        card_summary="x" * (CARD_SUMMARY_MAX_CHARS + 1),
        journal_title="t",
        journal_body="b",
        turn_refs=["corr-1"],
    )
    fitted, trimmed = fit_chat_compactor_digest_within_budget(digest)
    assert trimmed == ["card_summary"]
    assert len(fitted.card_summary) == CARD_SUMMARY_MAX_CHARS
    assert fitted.turn_refs == ["corr-1"]


def test_fit_chat_compactor_digest_within_budget_passes_in_limit_through() -> None:
    digest = ChatHistoryCompactorDigestV1(
        card_summary="x" * CARD_SUMMARY_MAX_CHARS,
        journal_title="t",
        journal_body="b",
        turn_refs=[],
    )
    fitted, trimmed = fit_chat_compactor_digest_within_budget(digest)
    assert trimmed == []
    assert fitted is digest


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


def _window_of(turns: list[DiscussionWindowTurnV1]) -> DiscussionWindowResultV1:
    return DiscussionWindowResultV1(
        window_start_utc=datetime(2026, 7, 8, 0, 0, tzinfo=timezone.utc),
        window_end_utc=datetime(2026, 7, 8, 23, 59, tzinfo=timezone.utc),
        turn_count=len(turns),
        turns=turns,
        transcript_text="ignored",
    )


def test_digest_inputs_cover_all_120_turns() -> None:
    turns = [
        DiscussionWindowTurnV1(
            created_at=datetime(2026, 7, 8, 1, 0, tzinfo=timezone.utc) + timedelta(minutes=i),
            correlation_id=f"c{i}",
            prompt="p " * 600,
            response="r " * 1200,
        )
        for i in range(120)
    ]
    inputs, stats = build_chat_history_compactor_digest_inputs(_window_of(turns))
    refs = [t["correlation_id"] for ci in inputs for t in ci["turns"]]
    assert refs == [f"c{i}" for i in range(120)]  # every turn, in order
    assert stats["total_count"] == stats["covered_count"] == 120
    assert stats["input_truncated"] is False
    assert len(inputs) > 1
    for ci in inputs:
        assert len(json.dumps(ci["turns"])) <= DIGEST_INPUT_CHAR_BUDGET
        assert ci["turn_count"] == 120
        assert ci["chunk_count"] == len(inputs)


def test_digest_inputs_flag_fetch_limit_and_runaway_turn() -> None:
    turns = [
        DiscussionWindowTurnV1(
            created_at=datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc),
            correlation_id=f"c{i}",
            prompt="p" * (DIGEST_TURN_PROMPT_MAX_CHARS + 100) if i == 0 else "short",
            response="ok",
        )
        for i in range(3)
    ]
    inputs, stats = build_chat_history_compactor_digest_inputs(_window_of(turns), fetch_limit=3)
    assert stats["fetch_limit_hit"] is True
    assert stats["turn_content_truncated"] is True
    assert stats["input_truncated"] is True
    assert inputs[0]["turns"][0]["truncated"] is True
    assert len(inputs[0]["turns"][0]["prompt"]) <= DIGEST_TURN_PROMPT_MAX_CHARS + 1


def test_digest_inputs_preserve_short_turns_untruncated() -> None:
    turns = [
        DiscussionWindowTurnV1(
            created_at=datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc),
            correlation_id="c1",
            prompt="Short prompt, well within budget.",
            response="Short response, well within budget.",
        )
    ]
    inputs, stats = build_chat_history_compactor_digest_inputs(_window_of(turns))
    assert len(inputs) == 1
    assert "chunk_count" not in inputs[0]
    assert "truncated" not in inputs[0]["turns"][0]
    assert "turn_content_truncated" not in inputs[0]
    assert stats["input_truncated"] is False


def test_fit_chat_digest_never_trims_journal_body() -> None:
    long_body = "b" * 50_000  # old cap was 4000
    digest = ChatHistoryCompactorDigestV1(
        card_summary="x" * (CARD_SUMMARY_MAX_CHARS + 5),
        journal_title="t" * (JOURNAL_TITLE_MAX_CHARS + 5),
        journal_body=long_body,
        turn_refs=[],
    )
    fitted, trimmed = fit_chat_compactor_digest_within_budget(digest)
    assert trimmed == ["card_summary", "journal_title"]
    assert fitted.journal_body == long_body


def test_chat_merge_input_and_concatenation_keep_every_ref() -> None:
    parts = [
        ChatHistoryCompactorDigestV1(card_summary="a", journal_body="body a", turn_refs=["c1"]),
        ChatHistoryCompactorDigestV1(card_summary="b", journal_body="", turn_refs=["c2", "c1"]),
    ]
    merged_in = build_chat_history_compactor_merge_input(
        base_input={"window_start_utc": "s", "window_end_utc": "e", "turn_count": 2, "turns": []},
        partial_digests=parts,
    )
    assert "turns" not in merged_in
    assert len(merged_in["partial_digests"]) == 2
    joined = concatenate_chat_partial_digests(parts, window_label="2026-07-08")
    assert joined.turn_refs == ["c1", "c2"]
    assert "body a" in joined.journal_body and "b" in joined.journal_body
