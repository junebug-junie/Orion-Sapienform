from __future__ import annotations

from app.recall_utils import plan_ctx_latest_user_text


def test_plan_ctx_prefers_raw_user_text() -> None:
    assert plan_ctx_latest_user_text({"raw_user_text": "  hi  ", "user_message": "ignored"}) == "hi"


def test_plan_ctx_falls_back_to_user_message() -> None:
    assert plan_ctx_latest_user_text({"raw_user_text": "", "user_message": "from user_message"}) == "from user_message"


def test_plan_ctx_falls_back_to_last_user_in_messages() -> None:
    ctx = {
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "last turn"},
        ]
    }
    assert plan_ctx_latest_user_text(ctx) == "last turn"
