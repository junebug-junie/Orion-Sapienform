"""The dead `self_study` belief producer is gone from every registry.

It queried a GraphDB/RDF named graph via SPARQL. `SELF_STUDY_NAMED_GRAPH` was
empty in every env, and RDF/Fuseki is retired repo-wide, so the adapter
returned None on every chat turn while still costing a cold-pull thread.
Its replacement is the `self_definition` producer (PR #2158). Kill means
kill: the registry entries, the adapter module, its package re-exports and
its env key are all removed, and this test refuses any of them coming back.
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import pytest

_SERVICE_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))


def _ids(registry) -> list[str]:
    return [p.producer_id for p in registry.producers]


def test_chat_stance_registry_has_no_self_study_but_keeps_self_definition() -> None:
    from app.chat_stance import _build_unification_registry

    ids = _ids(_build_unification_registry())
    assert "self_study" not in ids
    assert "self_definition" in ids


def test_projection_builder_registry_has_no_self_study() -> None:
    from orion.cognition.projection_builder import build_projection_unification_registry

    assert "self_study" not in _ids(build_projection_unification_registry())


def test_adapter_module_and_reexports_are_gone() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("orion.substrate.relational.adapters.self_study")
    for pkg in ("orion.substrate", "orion.substrate.relational", "orion.substrate.relational.adapters"):
        mod = importlib.import_module(pkg)
        assert not hasattr(mod, "map_self_study_to_substrate"), pkg
        assert "map_self_study_to_substrate" not in getattr(mod, "__all__", ()), pkg


def test_env_example_no_longer_carries_the_dead_key() -> None:
    text = (_SERVICE_ROOT / ".env_example").read_text()
    assert "SELF_STUDY_NAMED_GRAPH" not in text
    assert "SELF_STUDY_GRAPHDB" not in text
