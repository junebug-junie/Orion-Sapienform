from __future__ import annotations

import pytest
from pydantic import ValidationError

from orion.schemas.actions.chat_history_compactor import ChatHistoryCompactorDigestV1


def test_chat_history_compactor_digest_v1_rejects_empty_card_summary() -> None:
    with pytest.raises(ValidationError):
        ChatHistoryCompactorDigestV1(
            card_summary="",
            journal_title="Title",
            journal_body="Body",
            turn_refs=["corr-1"],
        )
