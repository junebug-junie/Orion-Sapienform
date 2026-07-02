from __future__ import annotations

import json

from orion.memory_graph.json_extract import salvage_truncated_json_object_text
from orion.memory_graph.suggest_validate import parse_json_object


def test_salvage_truncated_json_object_text() -> None:
    partial = (
        '{"ontology_version":"orionmem-2026-05","utterance_ids":["u1"],'
        '"entities":[{"id":"e_user","label":"User","entityKind":"person","surfaceForms":["I"]},'
        '"situations":[{"id":"s1","utterance_ids":["u1"],"label":"User leaves'
    )
    salvaged = salvage_truncated_json_object_text(partial)
    assert salvaged is not None
    data, err = parse_json_object(salvaged)
    assert err is None
    assert isinstance(data, dict)
    assert data.get("ontology_version") == "orionmem-2026-05"
    assert isinstance(data.get("entities"), list)


def test_parse_json_object_uses_salvage_for_unclosed_object() -> None:
    partial = '{"ontology_version":"orionmem-2026-05","utterance_ids":["u1"],"entities":['
    data, err = parse_json_object(partial)
    assert err is None
    assert isinstance(data, dict)
    assert data["utterance_ids"] == ["u1"]
