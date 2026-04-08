from __future__ import annotations

import pytest

from orion.schemas.chat_response_feedback import ChatResponseFeedbackV1, MAX_FREE_TEXT_CHARS


def test_feedback_schema_accepts_valid_thumbs_down_payload() -> None:
    payload = ChatResponseFeedbackV1(
        feedback_id="fb-1",
        target_turn_id="turn-1",
        target_message_id="turn-1:assistant",
        target_correlation_id="turn-1",
        session_id="sid-1",
        feedback_value="down",
        categories=["missed_relevant_context", "other"],
        free_text="It ignored my earlier constraints.",
    )
    assert payload.feedback_value == "down"
    assert payload.categories == ["missed_relevant_context", "other"]
    assert payload.target_key
    assert payload.submission_fingerprint


def test_feedback_schema_rejects_unknown_category_for_sentiment() -> None:
    with pytest.raises(ValueError):
        ChatResponseFeedbackV1(
            feedback_id="fb-2",
            target_turn_id="turn-2",
            feedback_value="up",
            categories=["made_up_facts"],
        )


def test_feedback_schema_requires_target_identifier() -> None:
    with pytest.raises(ValueError):
        ChatResponseFeedbackV1(
            feedback_id="fb-3",
            feedback_value="up",
            categories=["helpful_actionable"],
        )


def test_feedback_schema_rejects_oversized_free_text() -> None:
    with pytest.raises(ValueError):
        ChatResponseFeedbackV1(
            feedback_id="fb-4",
            target_turn_id="turn-4",
            feedback_value="up",
            categories=["helpful_actionable"],
            free_text="x" * (MAX_FREE_TEXT_CHARS + 1),
        )


def test_feedback_schema_trims_blank_linkage_fields_before_required_check() -> None:
    with pytest.raises(ValueError):
        ChatResponseFeedbackV1(
            feedback_id="fb-5",
            target_turn_id="   ",
            target_message_id="",
            target_correlation_id="  ",
            feedback_value="down",
            categories=["missed_relevant_context"],
        )


def test_feedback_schema_rejects_duplicate_categories() -> None:
    with pytest.raises(ValueError):
        ChatResponseFeedbackV1(
            feedback_id="fb-6",
            target_turn_id="turn-6",
            feedback_value="down",
            categories=["missed_relevant_context", "missed_relevant_context"],
        )
