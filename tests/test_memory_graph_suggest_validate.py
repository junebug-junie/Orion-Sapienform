from __future__ import annotations

import json
from pathlib import Path

from orion.memory_graph.suggest_validate import (
    extract_selected_role_evidence,
    role_grounded_extraction_expected,
    validate_for_escalation,
)

SHOWER_UTTERANCE_TEXT = """Structured transcript evidence for memory graph extraction (do not invent turns).

--- turn 1 id=u1 role=user ---
k, off to shower. Be back soon!

--- turn 2 id=a1 role=assistant ---
Shower well. I'll be here when you're back.

Emit utterance_ids matching the ids above; fill utterance_text_by_id with excerpts."""

BIKES_UTTERANCE_TEXT = """Structured transcript evidence for memory graph extraction (do not invent turns).

--- turn 1 id=u2 role=user ---
I'm going to ride bikes with the kids.

Emit utterance_ids matching the ids above; fill utterance_text_by_id with excerpts."""


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


def test_empty_predicate_escalates() -> None:
    data = _fixture()
    data["edges"][0]["p"] = ""
    should, errors = validate_for_escalation(data, utterance_text="Joey")
    assert should is True
    assert "edge_missing_predicate" in errors


def test_invalid_confidence_escalates() -> None:
    data = _fixture()
    data["edges"][0]["confidence"] = 1.5
    should, errors = validate_for_escalation(data, utterance_text="Joey")
    assert should is True
    assert "invalid_confidence" in errors


def test_low_confidence_does_not_escalate() -> None:
    data = _fixture()
    data["edges"][0]["confidence"] = 0.1
    should, errors = validate_for_escalation(
        data,
        utterance_text="Joey angered Juniper last week about cats",
    )
    assert should is False


def test_missing_entities_when_subjects_expected_escalates() -> None:
    data = _fixture()
    data["entities"] = []
    should, errors = validate_for_escalation(
        data,
        utterance_text="Joey angered Juniper last week",
    )
    assert should is True
    assert "no_entities_when_subjects_expected" in errors


def test_shower_empty_draft_escalates_role_grounded() -> None:
    empty = {
        "ontology_version": "orionmem-2026-05",
        "utterance_ids": ["u1", "a1"],
        "entities": [],
        "situations": [],
        "edges": [],
        "dispositions": [],
    }
    should, errors = validate_for_escalation(empty, utterance_text=SHOWER_UTTERANCE_TEXT)
    assert should is True
    assert "no_entities_when_role_grounded_subjects_expected" in errors
    assert "no_situations_when_role_grounded_context_expected" in errors
    assert "missing_user_role_entity" in errors
    assert "missing_assistant_role_entity" in errors


def test_shower_minimal_graph_does_not_escalate() -> None:
    path = Path("tests/fixtures/memory_graph/shower_role_grounded_draft.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    should, errors = validate_for_escalation(data, utterance_text=SHOWER_UTTERANCE_TEXT)
    assert should is False
    assert errors == []


def test_user_only_bikes_empty_escalates() -> None:
    empty = {
        "ontology_version": "orionmem-2026-05",
        "utterance_ids": ["u2"],
        "entities": [],
        "situations": [],
        "edges": [],
        "dispositions": [],
    }
    should, errors = validate_for_escalation(empty, utterance_text=BIKES_UTTERANCE_TEXT)
    assert should is True
    assert "no_entities_when_role_grounded_subjects_expected" in errors
    assert "no_situations_when_role_grounded_context_expected" in errors


def test_user_only_bikes_minimal_graph_passes() -> None:
    minimal = {
        "ontology_version": "orionmem-2026-05",
        "utterance_ids": ["u2"],
        "entities": [
            {
                "id": "urn:uuid:f0000001-0000-4000-8000-000000000001",
                "label": "User",
                "entityKind": "person",
                "surfaceForms": ["I"],
                "generalizes_to": None,
            },
            {
                "id": "urn:uuid:f0000002-0000-4000-8000-000000000002",
                "label": "bikes",
                "entityKind": "abstract",
                "surfaceForms": ["bikes"],
                "generalizes_to": None,
            },
        ],
        "situations": [
            {
                "id": "urn:uuid:f0000003-0000-4000-8000-000000000003",
                "utterance_ids": ["u2"],
                "label": "User plans to ride bikes with the kids",
                "stimulus_entity_id": "urn:uuid:f0000001-0000-4000-8000-000000000001",
                "about_entity_ids": ["urn:uuid:f0000002-0000-4000-8000-000000000002"],
                "target_entity_ids": [],
                "affectLabel": "neutral",
                "timeQualitative": "today",
                "occurredAt": None,
            }
        ],
        "edges": [
            {
                "s": "urn:uuid:f0000001-0000-4000-8000-000000000001",
                "p": "orionmem:inSituation",
                "o": "urn:uuid:f0000003-0000-4000-8000-000000000003",
                "confidence": 0.8,
            }
        ],
        "dispositions": [],
    }
    should, errors = validate_for_escalation(minimal, utterance_text=BIKES_UTTERANCE_TEXT)
    assert should is False
    assert errors == []


def test_extract_selected_role_evidence_shower() -> None:
    ev = extract_selected_role_evidence(SHOWER_UTTERANCE_TEXT)
    assert ev["has_user_turn"] is True
    assert ev["has_assistant_turn"] is True
    assert ev["has_nonempty_text"] is True
    assert ev["has_extractable_relation"] is True
    assert role_grounded_extraction_expected(SHOWER_UTTERANCE_TEXT) is True
