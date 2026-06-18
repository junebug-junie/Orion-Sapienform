from __future__ import annotations

from orion.memory_graph.suggest_runner import build_memory_graph_suggest_options


def test_build_memory_graph_suggest_options() -> None:
    opts = build_memory_graph_suggest_options()
    assert opts["no_write"] is True
    assert opts["skip_autonomy_context"] is True
    assert opts["structured_output_schema_name"] == "SuggestDraftV1"
    assert isinstance(opts["structured_output_schema"], dict)
    assert opts["structured_output_method"] == "json_object_schema"
    assert opts["max_tokens"] >= 8192
