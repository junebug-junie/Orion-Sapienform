from __future__ import annotations

import os
import sys
import importlib.util
from pathlib import Path

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
hub_scripts_pkg = HUB_ROOT / "scripts" / "__init__.py"
if (
    "scripts" not in sys.modules
    or not str(getattr(sys.modules.get("scripts"), "__file__", "")).startswith(str(HUB_ROOT))
):
    spec = importlib.util.spec_from_file_location(
        "scripts",
        str(hub_scripts_pkg),
        submodule_search_locations=[str(HUB_ROOT / "scripts")],
    )
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules["scripts"] = module
        spec.loader.exec_module(module)

from scripts import api_routes  # noqa: E402

from orion.substrate.store import InMemorySubstrateGraphStore  # noqa: E402


def test_seed_golden_concepts_at_startup_seeds_three_concepts(monkeypatch) -> None:
    fresh_store = InMemorySubstrateGraphStore()
    monkeypatch.setattr(api_routes, "SUBSTRATE_SEMANTIC_STORE", fresh_store)

    count = api_routes.seed_golden_concepts_at_startup()

    assert count == 3
    result = fresh_store.query_concept_region(limit_nodes=32, limit_edges=64)
    labels = {node.label for node in result.slice.nodes}
    assert labels == {"Orion", "Juniper", "Orion-Juniper relationship"}


def test_seed_golden_concepts_at_startup_is_idempotent(monkeypatch) -> None:
    fresh_store = InMemorySubstrateGraphStore()
    monkeypatch.setattr(api_routes, "SUBSTRATE_SEMANTIC_STORE", fresh_store)

    first = api_routes.seed_golden_concepts_at_startup()
    second = api_routes.seed_golden_concepts_at_startup()

    assert first == 3
    assert second == 3
    result = fresh_store.query_concept_region(limit_nodes=32, limit_edges=64)
    assert len(result.slice.nodes) == 3  # no duplicates from calling twice


def test_seed_golden_concepts_at_startup_degrades_gracefully_on_loader_failure(monkeypatch) -> None:
    fresh_store = InMemorySubstrateGraphStore()
    monkeypatch.setattr(api_routes, "SUBSTRATE_SEMANTIC_STORE", fresh_store)

    def _raise(_store):
        raise RuntimeError("boom")

    monkeypatch.setattr("orion.substrate.seed.load_seed_concepts_into_store", _raise)

    count = api_routes.seed_golden_concepts_at_startup()

    assert count == 0
