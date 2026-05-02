from __future__ import annotations

import json
from pathlib import Path

from orion.memory_graph.dto import SuggestDraftV1


def test_suggest_draft_parses_fixture() -> None:
    raw = json.loads(Path("tests/fixtures/memory_graph/joey_cats_draft.json").read_text(encoding="utf-8"))
    m = SuggestDraftV1.model_validate(raw)
    assert m.ontology_version == "orionmem-2026-05"
    assert m.utterance_ids == ["t42"]
