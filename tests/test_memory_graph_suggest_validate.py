from __future__ import annotations

import json
from pathlib import Path

from orion.memory_graph.suggest_validate import validate_for_escalation


def _fixture() -> dict:
    raw = Path("tests/fixtures/memory_graph/joey_cats_draft.json").read_text(encoding="utf-8")
    return json.loads(raw)


def test_valid_fixture_does_not_escalate() -> None:
    data = _fixture()
    should, errors = validate_for_escalation(
        data,
        utterance_text="Joey angered Juniper last week about cats",
    )
    assert should is False
    assert errors == []


def test_unknown_predicate_escalates() -> None:
    data = _fixture()
    data["edges"][0]["p"] = "orionmem:notReal"
    should, errors = validate_for_escalation(data, utterance_text="Joey")
    assert should is True
    assert any("unknown_predicate" in e for e in errors)


def test_missing_entities_when_subjects_expected_escalates() -> None:
    data = _fixture()
    data["entities"] = []
    should, errors = validate_for_escalation(
        data,
        utterance_text="Joey angered Juniper last week",
    )
    assert should is True
    assert "no_entities_when_subjects_expected" in errors
