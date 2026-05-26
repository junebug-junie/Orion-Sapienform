from __future__ import annotations

from orion.memory_graph.dto import SuggestDraftV1
from orion.memory_graph.json_to_rdf import draft_to_graph
from orion.memory_graph.utterance_text import ensure_draft_utterance_text


def test_ensure_merges_supplemental_and_writes_schema_text() -> None:
    draft = SuggestDraftV1.model_validate(
        {
            "ontology_version": "orionmem-2026-05",
            "utterance_ids": [
                "hub-utterance:2aca6dcc-a02a-438d-b2b8-8e0b5fed60ad",
                "0bc6c4c4-9bf5-4c50-9a7e-a8d338caea09",
            ],
            "entities": [],
            "situations": [],
            "edges": [],
            "dispositions": [],
            "utterance_text_by_id": {},
        }
    )
    enriched = ensure_draft_utterance_text(
        draft,
        supplemental={
            "hub-utterance:2aca6dcc-a02a-438d-b2b8-8e0b5fed60ad": "cool confirmed resolved!",
            "0bc6c4c4-9bf5-4c50-9a7e-a8d338caea09": "Got it — resolution confirmed.",
        },
    )
    from rdflib import Namespace

    schema = Namespace("https://schema.org/")
    g = draft_to_graph(enriched)
    texts = list(g.objects(None, schema.text))
    assert any("cool confirmed resolved!" in str(t) for t in texts)


def test_ensure_fuzzy_matches_uuid_tail_keys() -> None:
    draft = SuggestDraftV1.model_validate(
        {
            "ontology_version": "orionmem-2026-05",
            "utterance_ids": ["0bc6c4c4-9bf5-4c50-9a7e-a8d338caea09"],
            "entities": [],
            "situations": [],
            "edges": [],
            "dispositions": [],
            "utterance_text_by_id": {},
        }
    )
    enriched = ensure_draft_utterance_text(
        draft,
        supplemental={"0bc6c4c4-9bf5-4c50-9a7e-a8d338caea09": "assistant line"},
    )
    assert enriched.utterance_text_by_id["0bc6c4c4-9bf5-4c50-9a7e-a8d338caea09"] == "assistant line"
