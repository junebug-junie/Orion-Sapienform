from __future__ import annotations

import json
from pathlib import Path

from rdflib import Literal
from rdflib.namespace import RDF

from orion.memory_graph.dto import SuggestDraftV1
from orion.memory_graph.json_to_rdf import ORIONMEM, draft_to_graph, entity_uri


def test_draft_graph_has_situation_and_min_triples() -> None:
    raw = json.loads(Path("tests/fixtures/memory_graph/joey_cats_draft.json").read_text(encoding="utf-8"))
    draft = SuggestDraftV1.model_validate(raw)
    g = draft_to_graph(draft)
    assert len(g) >= 24
    sits = list(g.subjects(RDF.type, ORIONMEM.Situation))
    assert len(sits) >= 1


def test_revision_batch_tags_edge_endpoints() -> None:
    raw = json.loads(Path("tests/fixtures/memory_graph/joey_cats_draft.json").read_text(encoding="utf-8"))
    draft = SuggestDraftV1.model_validate(raw)
    rid = "batch-xyz"
    g = draft_to_graph(draft, revision_batch=rid)
    lit = Literal(rid)
    for e in draft.edges:
        snode = entity_uri(e.s)
        onode = entity_uri(e.o)
        assert (snode, ORIONMEM.revisionBatch, lit) in g
        assert (onode, ORIONMEM.revisionBatch, lit) in g
