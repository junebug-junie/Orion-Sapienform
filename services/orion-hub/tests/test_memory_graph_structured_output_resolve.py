"""Resolve memory-graph structured-output method defaults."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]


def _ensure_imports() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    for p in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))


def test_none_env_maps_to_json_object_schema() -> None:
    _ensure_imports()
    from scripts.memory_graph_structured_output import resolve_memory_graph_structured_output_method

    settings = SimpleNamespace(MEMORY_GRAPH_SUGGEST_STRUCTURED_OUTPUT_METHOD="none")
    assert resolve_memory_graph_structured_output_method(settings) == "json_object_schema"


def test_unset_defaults_to_json_object_schema(monkeypatch) -> None:
    _ensure_imports()
    from scripts.memory_graph_structured_output import resolve_memory_graph_structured_output_method

    monkeypatch.delenv("MEMORY_GRAPH_SUGGEST_STRUCTURED_OUTPUT_METHOD", raising=False)
    monkeypatch.delenv("LLM_STRUCTURED_OUTPUT_METHOD", raising=False)
    settings = SimpleNamespace(MEMORY_GRAPH_SUGGEST_STRUCTURED_OUTPUT_METHOD="")
    assert resolve_memory_graph_structured_output_method(settings) == "json_object_schema"
