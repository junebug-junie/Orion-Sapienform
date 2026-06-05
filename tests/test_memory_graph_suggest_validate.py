from __future__ import annotations

import json
from pathlib import Path

from orion.memory_graph.suggest_validate import (
    collect_topical_spine_warnings,
    extract_selected_role_evidence,
    repair_role_grounded_suggest_draft,
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


def test_repair_injects_orion_when_model_omits_assistant_role_entity() -> None:
    partial = {
        "ontology_version": "orionmem-2026-05",
        "utterance_ids": ["u1", "a1"],
        "entities": [
            {
                "id": "urn:uuid:f0000001-0000-4000-8000-000000000001",
                "label": "User",
                "entityKind": "person",
                "surfaceForms": ["I"],
                "generalizes_to": None,
            }
        ],
        "situations": [
            {
                "id": "urn:uuid:f0000003-0000-4000-8000-000000000003",
                "utterance_ids": ["u1"],
                "label": "User leaves to shower",
                "stimulus_entity_id": "urn:uuid:f0000001-0000-4000-8000-000000000001",
                "about_entity_ids": [],
                "target_entity_ids": [],
                "affectLabel": "neutral",
                "timeQualitative": "today",
                "occurredAt": None,
            }
        ],
        "edges": [],
        "dispositions": [],
    }
    should_before, errors_before = validate_for_escalation(partial, utterance_text=SHOWER_UTTERANCE_TEXT)
    assert should_before is True
    assert "missing_assistant_role_entity" in errors_before

    repaired = repair_role_grounded_suggest_draft(partial, utterance_text=SHOWER_UTTERANCE_TEXT)
    should_after, errors_after = validate_for_escalation(repaired, utterance_text=SHOWER_UTTERANCE_TEXT)
    assert should_after is False
    assert "missing_assistant_role_entity" not in errors_after
    labels = {str(ent.get("label") or "").lower() for ent in repaired.get("entities") or []}
    assert "orion" in labels


def test_assistant_role_detected_via_assistant_turn_situation_without_orion_label() -> None:
    draft = {
        "ontology_version": "orionmem-2026-05",
        "utterance_ids": ["u1", "a1"],
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
                "label": "Speaker B",
                "entityKind": "abstract",
                "surfaceForms": ["here"],
                "generalizes_to": None,
            },
        ],
        "situations": [
            {
                "id": "urn:uuid:f0000003-0000-4000-8000-000000000003",
                "utterance_ids": ["a1"],
                "label": "Speaker remains available",
                "stimulus_entity_id": "urn:uuid:f0000002-0000-4000-8000-000000000002",
                "about_entity_ids": [],
                "target_entity_ids": [],
                "affectLabel": "neutral",
                "timeQualitative": "today",
                "occurredAt": None,
            }
        ],
        "edges": [],
        "dispositions": [],
    }
    should, errors = validate_for_escalation(draft, utterance_text=SHOWER_UTTERANCE_TEXT)
    assert should is False
    assert "missing_assistant_role_entity" not in errors


PRAGMATIC_UTTERANCE_TEXT = """Structured transcript evidence for memory graph extraction (do not invent turns).

--- turn 1 id=u-prag role=user ---
thanks for sharing. Certainly a POV. You didn't undersell the pragmatic take.

--- turn 2 id=a-prag role=assistant ---
Appreciate that. The whole point was to keep it usable, not decorative.

Emit utterance_ids matching the ids above; fill utterance_text_by_id with excerpts."""


def test_pragmatic_take_fixture_does_not_escalate() -> None:
    path = Path("tests/fixtures/memory_graph/pragmatic_take_draft.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    should, errors = validate_for_escalation(data, utterance_text=PRAGMATIC_UTTERANCE_TEXT)
    assert should is False
    assert errors == []


def test_pragmatic_take_fixture_has_no_topical_warnings() -> None:
    path = Path("tests/fixtures/memory_graph/pragmatic_take_draft.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    warnings = collect_topical_spine_warnings(data, utterance_text=PRAGMATIC_UTTERANCE_TEXT)
    assert warnings == []


def test_dyad_only_pragmatic_emits_topical_warnings() -> None:
    dyad_only = {
        "ontology_version": "orionmem-2026-05",
        "utterance_ids": ["u-prag", "a-prag"],
        "entities": [
            {
                "id": "urn:uuid:12345678-1234-1234-1234-1234567890ab",
                "label": "User",
                "entityKind": "person",
                "surfaceForms": ["User"],
                "generalizes_to": None,
            },
            {
                "id": "urn:uuid:12345678-1234-1234-1234-1234567890ac",
                "label": "Orion",
                "entityKind": "person",
                "surfaceForms": ["Orion"],
                "generalizes_to": None,
            },
        ],
        "situations": [
            {
                "id": "urn:uuid:12345678-1234-1234-1234-1234567890ad",
                "utterance_ids": ["u-prag"],
                "label": "User acknowledges Orion's pragmatic approach",
                "stimulus_entity_id": "urn:uuid:12345678-1234-1234-1234-1234567890ac",
                "about_entity_ids": ["urn:uuid:12345678-1234-1234-1234-1234567890ab"],
                "target_entity_ids": [],
                "affectLabel": "affection",
                "timeQualitative": "recent",
                "occurredAt": None,
                "participants": [],
            }
        ],
        "edges": [],
        "dispositions": [],
        "utterance_text_by_id": {
            "u-prag": "You didn't undersell the pragmatic take.",
            "a-prag": "keep it usable, not decorative.",
        },
    }
    warnings = collect_topical_spine_warnings(dyad_only, utterance_text=PRAGMATIC_UTTERANCE_TEXT)
    assert any("topical_spine_missing" in w for w in warnings)
    assert any("pragmatic take" in w for w in warnings)
