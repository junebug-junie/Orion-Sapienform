from __future__ import annotations

import pytest
from pydantic import ValidationError

from orion.harness.tests.fixtures import make_thought
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.context_exec import ContextExecPermissionV1
from orion.schemas.harness_finalize import HARNESS_RECENT_TURNS_MAX, HarnessRunRequestV1
from orion.schemas.pre_turn_appraisal import TurnWindowMessageV1


def _build_request(**overrides: object) -> HarnessRunRequestV1:
    base: dict[str, object] = {
        "correlation_id": "c-1",
        "thought_event": make_thought(),
        "user_message": "hello",
        "permissions": ContextExecPermissionV1(),
        "answer_contract": AnswerContract(),
    }
    base.update(overrides)
    return HarnessRunRequestV1.model_validate(base)


def test_recent_turns_defaults_to_empty() -> None:
    req = _build_request()
    assert req.recent_turns == []


def test_recent_turns_accepts_up_to_the_cap() -> None:
    turns = [
        TurnWindowMessageV1(role="user", content=f"turn {i}")
        for i in range(HARNESS_RECENT_TURNS_MAX)
    ]
    req = _build_request(recent_turns=turns)
    assert len(req.recent_turns) == HARNESS_RECENT_TURNS_MAX


def test_recent_turns_rejects_more_than_the_cap() -> None:
    """The schema's own invariant must hold regardless of caller -- a caller
    that forgets to cap should get a validation error, not a silently
    unbounded list (see the evidence_event_ids unbounded-growth precedent
    this field's docstring cites)."""
    turns = [
        TurnWindowMessageV1(role="user", content=f"turn {i}")
        for i in range(HARNESS_RECENT_TURNS_MAX + 1)
    ]
    with pytest.raises(ValidationError):
        _build_request(recent_turns=turns)
