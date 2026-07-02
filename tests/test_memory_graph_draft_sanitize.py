from __future__ import annotations

import json

from orion.memory_graph.dto import SuggestDraftV1
from orion.memory_graph.draft_sanitize import is_resolvable_entity_ref, sanitize_suggest_draft_dict
from orion.memory_graph.json_to_rdf import draft_to_graph


def test_sanitize_strips_null_string_refs_and_approve_graph_builds() -> None:
    data = {
        "ontology_version": "orionmem-2026-05",
        "utterance_ids": ["hub-utterance:u1"],
        "entities": [
            {
                "id": "urn:uuid:a0000001-0000-4000-8000-000000000001",
                "label": "User",
                "entityKind": "person",
                "surfaceForms": ["I"],
                "generalizes_to": "null",
            },
            {
                "id": "urn:uuid:a0000002-0000-4000-8000-000000000002",
                "label": "Orion",
                "entityKind": "person",
                "surfaceForms": ["Orion"],
            },
        ],
        "situations": [
            {
                "id": "urn:uuid:b0000001-0000-4000-8000-000000000001",
                "utterance_ids": ["hub-utterance:u1"],
                "label": "User states age",
                "stimulus_entity_id": "null",
                "about_entity_ids": [],
                "target_entity_ids": [],
                "affectLabel": "neutral",
                "timeQualitative": "recent",
                "participants": [
                    {"entity_id": "urn:uuid:a0000001-0000-4000-8000-000000000001", "role": "agent"}
                ],
            }
        ],
        "edges": [
            {
                "s": "hub-utterance:u1",
                "p": "orionmem:inSituation",
                "o": "hub-utterance:u1",
                "confidence": 0.9,
            }
        ],
        "dispositions": [],
        "utterance_text_by_id": {"hub-utterance:u1": "I'm 43"},
    }
    cleaned = sanitize_suggest_draft_dict(data)
    assert cleaned["entities"][0]["generalizes_to"] is None
    assert cleaned["situations"][0]["stimulus_entity_id"] is None
    assert any(e["p"] == "orionmem:inSituation" for e in cleaned["edges"])
    assert cleaned["entities"][0]["label"] == "Juniper"
    draft = SuggestDraftV1.model_validate(cleaned)
    g = draft_to_graph(draft)
    assert len(g) > 10


def test_sanitize_rebuilds_structural_edges_from_situation() -> None:
    data = {
        "ontology_version": "orionmem-2026-05",
        "utterance_ids": ["t1"],
        "entities": [
            {
                "id": "urn:uuid:a0000001-0000-4000-8000-000000000001",
                "label": "User",
                "entityKind": "person",
                "surfaceForms": ["User"],
            },
            {
                "id": "urn:uuid:a0000002-0000-4000-8000-000000000002",
                "label": "Orion",
                "entityKind": "person",
                "surfaceForms": ["Orion"],
            },
        ],
        "situations": [
            {
                "id": "urn:uuid:b0000001-0000-4000-8000-000000000001",
                "utterance_ids": ["t1"],
                "label": "Orion responds",
                "stimulus_entity_id": "urn:uuid:a0000002-0000-4000-8000-000000000002",
                "about_entity_ids": ["urn:uuid:a0000001-0000-4000-8000-000000000001"],
                "target_entity_ids": [],
                "affectLabel": "neutral",
                "timeQualitative": "recent",
                "participants": [],
            }
        ],
        "edges": [],
        "dispositions": [],
    }
    cleaned = sanitize_suggest_draft_dict(data)
    preds = {e["p"] for e in cleaned["edges"]}
    assert "orionmem:stimulusEntity" in preds
    assert "schema:about" in preds


def test_sanitize_repairs_malformed_sequential_urn_uuid_suffixes() -> None:
    """LLMs often copy test-style ids (…90ab, …90ac) and roll past hex f into …90ag."""
    bad_situation_id = "urn:uuid:12345678-1234-1234-1234-1234567890ag"
    data = {
        "ontology_version": "orionmem-2026-05",
        "utterance_ids": ["u1", "a1"],
        "entities": [
            {
                "id": "urn:uuid:12345678-1234-1234-1234-1234567890ab",
                "label": "Juniper",
                "entityKind": "person",
                "surfaceForms": ["Juniper", "I"],
            },
            {
                "id": "urn:uuid:12345678-1234-1234-1234-1234567890ac",
                "label": "Orion",
                "entityKind": "person",
                "surfaceForms": ["Orion"],
            },
            {
                "id": "urn:uuid:12345678-1234-1234-1234-1234567890ad",
                "label": "Mom",
                "entityKind": "person",
                "surfaceForms": ["mom", "my mom"],
            },
            {
                "id": "urn:uuid:12345678-1234-1234-1234-1234567890ae",
                "label": "Naturalization Ceremony",
                "entityKind": "abstract",
                "surfaceForms": ["naturalization ceremony"],
            },
            {
                "id": "urn:uuid:12345678-1234-1234-1234-1234567890af",
                "label": "US Citizen",
                "entityKind": "abstract",
                "surfaceForms": ["US citizen"],
            },
        ],
        "situations": [
            {
                "id": bad_situation_id,
                "utterance_ids": ["u1"],
                "label": "User shares news about mom's naturalization ceremony",
                "stimulus_entity_id": "urn:uuid:12345678-1234-1234-1234-1234567890ae",
                "about_entity_ids": [
                    "urn:uuid:12345678-1234-1234-1234-1234567890ad",
                    "urn:uuid:12345678-1234-1234-1234-1234567890ae",
                ],
                "target_entity_ids": [],
                "affectLabel": "affection",
                "timeQualitative": "recent",
                "participants": [],
            },
            {
                "id": "urn:uuid:12345678-1234-1234-1234-1234567890ah",
                "utterance_ids": ["a1"],
                "label": "Orion acknowledges the ceremony and asks about next steps",
                "stimulus_entity_id": "urn:uuid:12345678-1234-1234-1234-1234567890ab",
                "about_entity_ids": ["urn:uuid:12345678-1234-1234-1234-1234567890ae"],
                "target_entity_ids": [],
                "affectLabel": "neutral",
                "timeQualitative": "recent",
                "participants": [],
            },
        ],
        "edges": [],
        "dispositions": [],
        "utterance_text_by_id": {
            "u1": "My mom had her naturalization ceremony today!",
            "a1": "That's wonderful — what happens next for her as a new US citizen?",
        },
    }
    cleaned = sanitize_suggest_draft_dict(data)
    assert bad_situation_id not in json.dumps(cleaned)
    for sit in cleaned["situations"]:
        assert is_resolvable_entity_ref(sit["id"])
    draft = SuggestDraftV1.model_validate(cleaned)
    g = draft_to_graph(draft)
    assert len(g) > 10


def test_sanitize_repairs_short_local_entity_and_situation_ids() -> None:
    data = {
        "ontology_version": "orionmem-2026-05",
        "utterance_ids": ["hub-utterance:u1"],
        "entities": [
            {
                "id": "e_user",
                "label": "User",
                "entityKind": "person",
                "surfaceForms": ["I"],
            },
            {
                "id": "e_orion",
                "label": "Orion",
                "entityKind": "person",
                "surfaceForms": ["Orion"],
            },
        ],
        "situations": [
            {
                "id": "s1",
                "utterance_ids": ["hub-utterance:u1"],
                "label": "User shares update",
                "stimulus_entity_id": "e_user",
                "about_entity_ids": ["e_orion"],
                "target_entity_ids": [],
                "affectLabel": "neutral",
                "timeQualitative": "recent",
                "participants": [{"entity_id": "e_user", "role": "agent"}],
            }
        ],
        "edges": [
            {"s": "e_user", "p": "orionmem:inSituation", "o": "s1", "confidence": 0.9}
        ],
        "dispositions": [],
        "utterance_text_by_id": {"hub-utterance:u1": "Hey Orion"},
    }
    cleaned = sanitize_suggest_draft_dict(data)
    assert "e_user" not in json.dumps(cleaned)
    assert "s1" not in json.dumps(cleaned)
    for ent in cleaned["entities"]:
        assert is_resolvable_entity_ref(ent["id"])
    for sit in cleaned["situations"]:
        assert is_resolvable_entity_ref(sit["id"])
    draft = SuggestDraftV1.model_validate(cleaned)
    g = draft_to_graph(draft)
    assert len(g) > 10


def test_sanitize_replaces_stale_occurred_at_with_today() -> None:
    from datetime import date

    data = {
        "ontology_version": "orionmem-2026-05",
        "utterance_ids": ["t1"],
        "entities": [],
        "situations": [
            {
                "id": "urn:uuid:b0000001-0000-4000-8000-000000000001",
                "utterance_ids": ["t1"],
                "label": "old date",
                "occurredAt": "2023-01-15",
                "timeQualitative": "unknown",
            }
        ],
        "edges": [],
        "dispositions": [],
    }
    cleaned = sanitize_suggest_draft_dict(data)
    assert cleaned["situations"][0]["occurredAt"] == date.today().isoformat()
    assert cleaned["situations"][0]["timeQualitative"] == "today"
