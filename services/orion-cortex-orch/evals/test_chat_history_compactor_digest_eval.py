"""Deterministic evals for the chat history compactor digest seam.

These measure the input/output budget contract the digest LLM lives inside:
adversarial windows must trim to a bounded prompt payload, digest outputs must
respect card/journal budgets, and quiet windows must never fabricate substance.
Digest *quality* (does the summary reflect the transcript) still needs an
LLM-in-the-loop eval; that gap is tracked in the PR report.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from orion.cognition.chat_history_compactor.constants import (
    CARD_SUMMARY_MAX_CHARS,
    DEFAULT_MAX_TURNS,
    DIGEST_TURN_PROMPT_MAX_CHARS,
    DIGEST_TURN_RESPONSE_MAX_CHARS,
    JOURNAL_BODY_MAX_CHARS,
    JOURNAL_TITLE_MAX_CHARS,
)
from orion.cognition.chat_history_compactor.digest import (
    assert_chat_compactor_digest_within_budget,
    build_quiet_day_chat_digest,
    parse_chat_history_compactor_digest_json,
    trim_chat_history_compactor_input,
)
from orion.schemas.discussion_window import DiscussionWindowResultV1, DiscussionWindowTurnV1

# 30 turns x (400 + 600) chars plus JSON/field overhead; the digest prompt
# payload must never grow past this regardless of raw window size.
DIGEST_INPUT_MAX_SERIALIZED_CHARS = 45_000


def _window(turns: list[DiscussionWindowTurnV1]) -> DiscussionWindowResultV1:
    start = datetime(2026, 7, 9, 4, 0, tzinfo=timezone.utc)
    return DiscussionWindowResultV1(
        window_start_utc=start,
        window_end_utc=start + timedelta(hours=6),
        turn_count=len(turns),
        turns=turns,
        transcript_text="\n".join(f"user: {t.prompt}\norion: {t.response}" for t in turns),
        selection_strategy="time_bound_then_contiguous_suffix",
    )


def test_eval_adversarial_window_trims_to_bounded_digest_input() -> None:
    turns = [
        DiscussionWindowTurnV1(
            created_at=datetime(2026, 7, 9, 4, 0, tzinfo=timezone.utc) + timedelta(seconds=i),
            correlation_id=f"corr-{i}",
            prompt="p" * 10_000,
            response="r" * 10_000,
        )
        for i in range(500)
    ]
    payload = trim_chat_history_compactor_input(_window(turns))

    assert len(payload["turns"]) == DEFAULT_MAX_TURNS
    assert payload["turns_truncated_for_digest"] is True
    assert payload["turns_total"] == 500
    for turn in payload["turns"]:
        assert len(turn["prompt"]) <= DIGEST_TURN_PROMPT_MAX_CHARS + 1  # +ellipsis
        assert len(turn["response"]) <= DIGEST_TURN_RESPONSE_MAX_CHARS + 1
    # Newest suffix wins: the last raw turn must survive the trim.
    assert payload["turns"][-1]["correlation_id"] == "corr-499"
    assert len(json.dumps(payload)) <= DIGEST_INPUT_MAX_SERIALIZED_CHARS


def test_eval_small_window_passes_through_untrimmed() -> None:
    turns = [
        DiscussionWindowTurnV1(
            created_at=datetime(2026, 7, 9, 5, 0, tzinfo=timezone.utc),
            correlation_id="corr-a",
            prompt="short prompt",
            response="short response",
        )
    ]
    payload = trim_chat_history_compactor_input(_window(turns))
    assert payload["turn_count"] == 1
    assert "turns_truncated_for_digest" not in payload
    assert payload["turns"][0]["prompt"] == "short prompt"
    assert payload["turns"][0]["response"] == "short response"


def test_eval_quiet_window_digest_is_honest_and_within_budget() -> None:
    digest = build_quiet_day_chat_digest(window_label="2026-07-08")
    assert_chat_compactor_digest_within_budget(digest)
    assert digest.turn_refs == []
    # The quiet digest must say nothing was written, not fake substance.
    assert "No indexed chat digest memory card was written" in digest.journal_body
    assert len(digest.card_summary) <= CARD_SUMMARY_MAX_CHARS
    assert len(digest.journal_title or "") <= JOURNAL_TITLE_MAX_CHARS
    assert len(digest.journal_body or "") <= JOURNAL_BODY_MAX_CHARS


def test_eval_digest_json_round_trip_and_rejection() -> None:
    digest = {
        "card_summary": "Discussed indexed compactor upserts.",
        "journal_title": "Chat digest — 2026-07-08",
        "journal_body": "Talked through upsert semantics.",
        "turn_refs": ["corr-a", "corr-b"],
    }
    parsed = parse_chat_history_compactor_digest_json(json.dumps(digest))
    assert parsed.card_summary == digest["card_summary"]
    assert parsed.turn_refs == digest["turn_refs"]

    with pytest.raises(ValueError, match="compactor_digest_not_object"):
        parse_chat_history_compactor_digest_json(json.dumps([digest]))
    with pytest.raises(ValueError):
        parse_chat_history_compactor_digest_json("not json at all")


def test_eval_over_budget_digest_fails_loud() -> None:
    over = parse_chat_history_compactor_digest_json(
        json.dumps(
            {
                "card_summary": "x" * (CARD_SUMMARY_MAX_CHARS + 1),
                "journal_title": "Title",
                "journal_body": "Body",
                "turn_refs": [],
            }
        )
    )
    with pytest.raises(ValueError, match="compactor_output_over_budget:card_summary"):
        assert_chat_compactor_digest_within_budget(over)
