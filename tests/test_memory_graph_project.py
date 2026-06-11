from __future__ import annotations

import json
from pathlib import Path

from orion.core.contracts.memory_cards import MemoryCardCreateV1
from orion.memory_graph.dto import CardProjectionDefaultsV1, SuggestDraftV1
from orion.memory_graph.json_to_rdf import draft_to_graph
from orion.memory_graph.project import apply_card_projection_defaults, project_graph_to_cards


def test_project_situation_centric_yields_rich_active_cards() -> None:
    raw = json.loads(Path("tests/fixtures/memory_graph/joey_cats_draft.json").read_text(encoding="utf-8"))
    draft = SuggestDraftV1.model_validate(raw)
    g = draft_to_graph(draft)
    pack = project_graph_to_cards(g, draft)

    anchor_cards = [c for c in pack.creates if "anchor" in c.types]
    assert anchor_cards == []

    situation_cards = [c for c in pack.creates if c.title.startswith("Joey angered")]
    assert len(situation_cards) == 1
    card = MemoryCardCreateV1.model_validate(situation_cards[0].model_dump(mode="json"))
    assert card.status == "active"
    assert card.title
    assert card.summary
    assert card.still_true
    assert card.anchors
    assert "Joey" in card.anchors
    assert card.evidence
    assert card.subschema.get("memory_graph")
    assert card.priority == "high_recall"

    belief_cards = [c for c in pack.creates if "belief" in c.types]
    assert len(belief_cards) == 1
    assert "trust" in belief_cards[0].summary.lower() or "breed" in belief_cards[0].title.lower()

    assert pack.edge_indices


def test_project_situation_centric_multi_situation_no_entity_stubs() -> None:
    raw = json.loads(Path("tests/fixtures/memory_graph/pragmatic_take_draft.json").read_text(encoding="utf-8"))
    draft = SuggestDraftV1.model_validate(raw)
    g = draft_to_graph(draft)
    pack = project_graph_to_cards(g, draft)
    assert not [c for c in pack.creates if "anchor" in c.types]
    event_cards = [c for c in pack.creates if "event" in c.types]
    assert len(event_cards) == 2
    assert all(c.status == "active" for c in pack.creates)
    assert all(c.evidence for c in event_cards)


def test_project_entity_per_card_legacy_mode() -> None:
    raw = json.loads(Path("tests/fixtures/memory_graph/joey_cats_draft.json").read_text(encoding="utf-8"))
    draft = SuggestDraftV1.model_validate(raw)
    g = draft_to_graph(draft)
    mapping = {
        "projection_mode": "entity_per_card",
        "graph_approve_card_status": "pending_review",
        "entity_iri_base": "https://orion.example/ns/mem/entity/",
        "rdf_type_to_card_types": {
            "https://orion.local/ns/mem/v2026-05#Situation": ["event"],
            "https://orion.local/ns/mem/v2026-05#TypedEntity": ["anchor"],
            "https://orion.local/ns/mem/v2026-05#AffectiveDisposition": ["belief"],
        },
        "entity_kind_to_anchor_class": {"person": "person", "animal": "concept"},
        "predicate_to_edge_type": {},
    }
    pack = project_graph_to_cards(g, draft, mapping=mapping)
    assert any("anchor" in c.types for c in pack.creates)
    assert any(c.status == "pending_review" for c in pack.creates)


def test_apply_card_projection_defaults_overrides_confidence() -> None:
    raw = json.loads(Path("tests/fixtures/memory_graph/joey_cats_draft.json").read_text(encoding="utf-8"))
    draft = SuggestDraftV1.model_validate(raw)
    g = draft_to_graph(draft)
    pack = project_graph_to_cards(g, draft)
    defaults = CardProjectionDefaultsV1(confidence="certain", priority="always_inject")
    merged = apply_card_projection_defaults(pack.creates, defaults)
    assert merged
    assert all(c.confidence == "certain" for c in merged)
    assert all(c.priority == "always_inject" for c in merged)
