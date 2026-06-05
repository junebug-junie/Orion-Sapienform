from __future__ import annotations

import json

from orion.memory_graph.dto import SuggestDraftV1
from orion.memory_graph.draft_sanitize import sanitize_suggest_draft_dict
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
