"""Bounded-retrieval contract fields (2026-09-29 design, Phase 2).

RecallQueryV1 and RecallDecisionV1 are extra="forbid", so every new field is
additive and optional: an old caller payload must still validate, and a new
payload must round-trip through JSON unchanged.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orion.core.contracts.recall import RecallDecisionV1, RecallQueryV1
from orion.schemas.registry import resolve


def test_query_new_fields_round_trip() -> None:
    q = RecallQueryV1(
        fragment="long turn text",
        retrieval_query="gpu1 p4 lane",
        deadline_ms=90000,
        mode="context_only",
    )
    again = RecallQueryV1.model_validate_json(q.model_dump_json())
    assert again == q
    assert again.retrieval_query == "gpu1 p4 lane"
    assert again.deadline_ms == 90000
    assert again.mode == "context_only"


def test_query_old_payload_still_validates_with_defaults() -> None:
    q = RecallQueryV1.model_validate({"fragment": "hi", "verb": "chat_general"})
    assert q.retrieval_query is None
    assert q.deadline_ms is None
    assert q.mode == "retrieve"


@pytest.mark.parametrize(
    "bad",
    [
        {"retrieval_query": "x" * 1001},
        {"deadline_ms": 0},
        {"deadline_ms": -5},
        {"mode": "everything"},
    ],
)
def test_query_new_fields_reject_bad_values(bad) -> None:
    with pytest.raises(ValidationError):
        RecallQueryV1.model_validate({"fragment": "hi", **bad})


def test_decision_new_fields_round_trip() -> None:
    d = RecallDecisionV1(
        corr_id="c",
        query="q",
        query_chars=1,
        retrieval_query_source="caller",
        sub_query_count=3,
        candidates_fetched=40,
        candidates_kept=10,
        deadline_hit=False,
        timings_ms={"intake": 0, "total": 12},
    )
    again = RecallDecisionV1.model_validate(d.model_dump(mode="json"))
    assert again.model_dump() == d.model_dump()


def test_decision_old_payload_still_validates() -> None:
    d = RecallDecisionV1.model_validate({"corr_id": "c", "query": "q"})
    assert d.retrieval_query_source is None
    assert d.deadline_hit is None
    assert d.timings_ms == {}


def test_decision_rejects_unknown_source() -> None:
    with pytest.raises(ValidationError):
        RecallDecisionV1(corr_id="c", query="q", retrieval_query_source="guessed")


def test_registry_resolves_models_with_new_fields() -> None:
    assert "retrieval_query" in resolve("RecallQueryV1").model_fields
    assert "timings_ms" in resolve("RecallDecisionV1").model_fields
