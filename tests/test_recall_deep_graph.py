from __future__ import annotations

import importlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def _purge_module(prefix: str) -> None:
    for key in list(sys.modules.keys()):
        if key == prefix or key.startswith(f"{prefix}."):
            sys.modules.pop(key, None)


def _import_app_module(service_root: Path, module: str):
    sys.path.insert(0, str(service_root))
    try:
        return importlib.import_module(module)
    finally:
        if sys.path[0] == str(service_root):
            sys.path.pop(0)


def test_deep_graph_verb_profile_binding() -> None:
    service_root = ROOT / "services" / "orion-cortex-exec"
    _purge_module("app")
    module = _import_app_module(service_root, "app.verb_adapters")
    profile = module._load_verb_recall_profile("chat_deep_graph")
    assert profile == "deep.graph.v1"


def test_rdf_adapter_builds_neighbor_query() -> None:
    recall_root = ROOT / "services" / "orion-recall"
    _purge_module("app")
    module = _import_app_module(recall_root, "app.storage.rdf_adapter")
    sparql = module._build_sparql_query(["orion", "graph"], max_nodes=2, max_results=8)
    assert "UNION" in sparql
    assert 'CONTAINS(LCASE(STR(?node)), "orion")' in sparql
    assert "?neighbor ?p ?node" in sparql


def test_rdf_enabled_for_deep_graph_profile() -> None:
    recall_root = ROOT / "services" / "orion-recall"
    _purge_module("app")
    module = _import_app_module(recall_root, "app.worker")
    assert module._rdf_enabled({"profile": "deep.graph.v1", "rdf_top_k": 12}) is True
