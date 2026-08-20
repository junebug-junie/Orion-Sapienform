"""AI Town's own concept graph store singleton
(``services/orion-hub/scripts/api_routes.py``'s ``SUBSTRATE_SEMANTIC_STORE_AITOWN``
/ ``_build_aitown_substrate_store_from_env()``).

Mirrors the import-path setup already established in
``test_substrate_concept_seed_startup.py`` for tests that need the real
``scripts.api_routes`` module (not the isolated-router pattern
``concept_atlas_routes.py``'s own tests use, since this module's own
top-level singleton construction is exactly what's under test here).
"""

from __future__ import annotations

import importlib.util
import os
import sys
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


def test_singleton_exists_and_is_a_real_store() -> None:
    """Real assertion, not just "doesn't crash on import": the singleton
    must exist and be a usable store object (this test's own env has no
    SUBSTRATE_STORE_BACKEND set, so it degrades to in-memory -- same
    default-v1-safety fallback build_substrate_store_from_env() already
    uses for the primary store)."""
    assert api_routes.SUBSTRATE_SEMANTIC_STORE_AITOWN is not None
    # snapshot()/query_concept_region() are the two methods every consumer
    # of this store (concept_atlas_routes.py) actually calls.
    assert hasattr(api_routes.SUBSTRATE_SEMANTIC_STORE_AITOWN, "snapshot")
    assert hasattr(api_routes.SUBSTRATE_SEMANTIC_STORE_AITOWN, "query_concept_region")


def test_singleton_is_a_distinct_object_from_the_orion_store() -> None:
    """The load-bearing guarantee: two independent stores, not one store
    aliased under two names (which would silently merge AI Town concepts
    into Orion's own graph)."""
    assert api_routes.SUBSTRATE_SEMANTIC_STORE_AITOWN is not api_routes.SUBSTRATE_SEMANTIC_STORE


def test_builder_falls_back_to_in_memory_when_backend_not_falkor(monkeypatch) -> None:
    monkeypatch.delenv("SUBSTRATE_STORE_BACKEND", raising=False)
    result = api_routes._build_aitown_substrate_store_from_env()
    assert isinstance(result, InMemorySubstrateGraphStore)


def test_builder_delegates_to_aitown_falkor_builder_when_backend_is_falkor(monkeypatch) -> None:
    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "falkor")

    sentinel = object()
    called = {}

    def fake_builder():
        called["invoked"] = True
        return sentinel

    monkeypatch.setattr(
        "orion.substrate.falkor_store.build_aitown_falkor_substrate_store_from_env", fake_builder
    )

    result = api_routes._build_aitown_substrate_store_from_env()

    assert called.get("invoked") is True
    assert result is sentinel
