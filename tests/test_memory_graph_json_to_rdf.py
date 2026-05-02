from __future__ import annotations

import json
from pathlib import Path

from rdflib.namespace import RDF

from orion.memory_graph.dto import SuggestDraftV1
from orion.memory_graph.json_to_rdf import ORIONMEM, draft_to_graph


def test_draft_graph_has_situation_and_min_triples() -> None:
    raw = json.loads(Path("tests/fixtures/memory_graph/joey_cats_draft.json").read_text(encoding="utf-8"))
    draft = SuggestDraftV1.model_validate(raw)
    g = draft_to_graph(draft)
    assert len(g) >= 24
    sits = list(g.subjects(RDF.type, ORIONMEM.Situation))
    assert len(sits) >= 1
