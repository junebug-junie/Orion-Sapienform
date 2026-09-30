"""Deterministic evals for the chat history compactor digest seam.

These measure the input/output budget contract the digest LLM lives inside:
adversarial windows must split into bounded per-call payloads WITHOUT dropping
any turn, digest outputs must respect card budgets (journal body is stored in
full), and quiet windows must never fabricate substance.
Digest *quality* (does the summary reflect the transcript) still needs an
LLM-in-the-loop eval; that gap is tracked in the PR report.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from orion.cognition.chat_history_compactor.constants import (
    CARD_SUMMARY_MAX_CHARS,
    DIGEST_TURN_PROMPT_MAX_CHARS,
    DIGEST_TURN_RESPONSE_MAX_CHARS,
    JOURNAL_TITLE_MAX_CHARS,
)
from orion.cognition.chat_history_compactor.digest import (
    build_chat_history_compactor_digest_inputs,
    fit_chat_compactor_digest_within_budget,
    build_quiet_day_chat_digest,
    parse_chat_history_compactor_digest_json,
)
from orion.schemas.discussion_window import DiscussionWindowResultV1, DiscussionWindowTurnV1

# Fixed, independent of DIGEST_TURN_*_MAX_CHARS on purpose: the point of this
# ceiling is to catch a *future* widening of those per-turn caps pushing the
# real per-call digest payload past what's safe for the smallest context an
# agent-class call can land on (chat's card, 65k tokens/slot) -- deriving it
# from the same constants it's meant to guard would make it tautological.
# ~150k chars of serialized JSON (incl. window metadata) stays under that with
# a 16k-token completion; revisit only with a real gateway context limit in
# hand, not to make an over-budget test pass.
DIGEST_INPUT_MAX_SERIALIZED_CHARS = 150_000


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


def test_eval_adversarial_window_splits_into_bounded_calls_without_dropping_turns() -> None:
    # Realistic-length ids/metadata (real UUID4s, a user_id, a source label)
    # rather than short synthetic strings like "corr-0" -- a fixture with
    # trivially short ids under-counts the real serialized payload size.
    turns = [
        DiscussionWindowTurnV1(
            created_at=datetime(2026, 7, 9, 4, 0, tzinfo=timezone.utc) + timedelta(seconds=i),
            correlation_id=str(uuid.uuid4()),
            user_id=str(uuid.uuid4()),
            source="hub_ws",
            prompt="p" * 10_000,
            response="r" * 20_000,
        )
        for i in range(500)
    ]
    inputs, stats = build_chat_history_compactor_digest_inputs(_window(turns))

    covered = [t["correlation_id"] for payload in inputs for t in payload["turns"]]
    assert covered == [t.correlation_id for t in turns]  # every turn, in order
    assert stats["covered_count"] == stats["total_count"] == 500
    assert stats["input_truncated"] is True  # runaway turns hit the safety caps
    for payload in inputs:
        for turn in payload["turns"]:
            assert len(turn["prompt"]) <= DIGEST_TURN_PROMPT_MAX_CHARS + 1  # +ellipsis
            assert len(turn["response"]) <= DIGEST_TURN_RESPONSE_MAX_CHARS + 1
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
    inputs, _stats = build_chat_history_compactor_digest_inputs(_window(turns))
    assert len(inputs) == 1
    payload = inputs[0]
    assert payload["turn_count"] == 1
    assert "chunk_count" not in payload
    assert payload["turns"][0]["prompt"] == "short prompt"
    assert payload["turns"][0]["response"] == "short response"


def test_eval_quiet_window_digest_is_honest_and_within_budget() -> None:
    digest = build_quiet_day_chat_digest(window_label="2026-07-08")
    fitted, trimmed = fit_chat_compactor_digest_within_budget(digest)
    # The quiet digest is authored in-code, so it must already fit -- nothing to trim.
    assert trimmed == []
    assert fitted is digest
    assert digest.turn_refs == []
    # The quiet digest must say nothing was written, not fake substance.
    assert "No indexed chat digest memory card was written" in digest.journal_body
    assert len(digest.card_summary) <= CARD_SUMMARY_MAX_CHARS
    assert len(digest.journal_title or "") <= JOURNAL_TITLE_MAX_CHARS


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


def test_eval_over_budget_digest_is_repaired_not_discarded() -> None:
    """Quality bar: an over-budget digest still yields a usable card.

    The cap is a storage/display bound on already-validated content, so the
    graceful outcome is a summary trimmed to the bound -- not a discarded digest
    and a failed daily run.
    """
    over = parse_chat_history_compactor_digest_json(
        json.dumps(
            {
                "card_summary": "x" * (CARD_SUMMARY_MAX_CHARS + 1),
                "journal_title": "Title",
                "journal_body": "Body",
                "turn_refs": ["corr-a"],
            }
        )
    )
    fitted, trimmed = fit_chat_compactor_digest_within_budget(over)
    assert trimmed == ["card_summary"]
    assert len(fitted.card_summary) == CARD_SUMMARY_MAX_CHARS
    # Everything the digest actually asserted about the window survives intact.
    assert fitted.journal_title == "Title"
    assert fitted.journal_body == "Body"
    assert fitted.turn_refs == ["corr-a"]
