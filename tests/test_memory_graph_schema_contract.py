"""Tests for SuggestDraftV1 compact JSON schema contract."""

from __future__ import annotations

import json
from pathlib import Path

from orion.memory_graph.dto import SuggestDraftV1
from orion.memory_graph.schema_contract import (
    compact_suggest_draft_json_schema,
    suggest_draft_json_schema,
)


def test_compact_schema_top_level_required_keys() -> None:
    schema = compact_suggest_draft_json_schema()
    required = set(schema.get("required") or [])
    assert "ontology_version" in required
    assert "utterance_ids" in required
    assert "entities" in required
    assert "situations" in required
    assert "edges" in required
    assert "dispositions" in required


def test_compact_schema_disallows_extra_top_level() -> None:
    schema = compact_suggest_draft_json_schema()
    assert schema.get("additionalProperties") is False


def test_compact_schema_has_graph_arrays() -> None:
    props = compact_suggest_draft_json_schema().get("properties") or {}
    for key in ("entities", "situations", "edges", "dispositions"):
        assert key in props
        assert props[key].get("type") == "array"


def test_suggest_draft_fixture_still_validates() -> None:
    raw = Path("tests/fixtures/memory_graph/shower_role_grounded_draft.json").read_text(
        encoding="utf-8"
    )
    data = json.loads(raw)
    draft = SuggestDraftV1.model_validate(data)
    assert draft.ontology_version == "orionmem-2026-05"


def test_full_pydantic_schema_has_title() -> None:
    full = suggest_draft_json_schema()
    assert full.get("title") == "SuggestDraftV1" or "SuggestDraftV1" in str(full)
