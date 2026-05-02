from __future__ import annotations

import json
from pathlib import Path

from orion.core.contracts.memory_cards import MemoryCardCreateV1
from orion.memory_graph.dto import SuggestDraftV1
from orion.memory_graph.json_to_rdf import draft_to_graph
from orion.memory_graph.project import project_graph_to_cards


def test_project_yields_valid_create_payload() -> None:
    raw = json.loads(Path("tests/fixtures/memory_graph/joey_cats_draft.json").read_text(encoding="utf-8"))
    draft = SuggestDraftV1.model_validate(raw)
    g = draft_to_graph(draft)
    pack = project_graph_to_cards(g, draft)
    situation_cards = [c for c in pack.creates if c.title.startswith("Joey angered")]
    assert situation_cards
    card = MemoryCardCreateV1.model_validate(situation_cards[0].model_dump(mode="json"))
    assert card.title
