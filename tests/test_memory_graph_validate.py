from __future__ import annotations

import json
from pathlib import Path

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS

from orion.memory_graph.dto import SuggestDraftV1
from orion.memory_graph.json_to_rdf import ORIONMEM, draft_to_graph
from orion.memory_graph.validate import validate_graph


def test_fixture_passes_shacl() -> None:
    raw = json.loads(Path("tests/fixtures/memory_graph/joey_cats_draft.json").read_text(encoding="utf-8"))
    draft = SuggestDraftV1.model_validate(raw)
    g = draft_to_graph(draft)
    viol = validate_graph(g)
    assert viol == []


def test_situation_missing_derivation_fails() -> None:
    g = Graph()
    sit = URIRef("https://orion.example/ns/mem/entity/b0000001-0000-4000-8000-000000000001")
    g.add((sit, RDF.type, ORIONMEM.Situation))
    g.add((sit, RDFS.label, Literal("orphan situation")))
    viol = validate_graph(g)
    assert viol
