"""build_falkor_substrate_store_from_env()'s new graph-name parameterization
and build_aitown_falkor_substrate_store_from_env() -- AI Town's own concept
graph (docs/superpowers/specs/2026-08-18-aitown-concept-graph-split-and-
atlas-readability-design.md, "AI Town's own concept graph").

FalkorSubstrateStore's real constructor eagerly hydrates from a live Falkor
connection (hydrate=True by default), so these tests replace the class
itself with a lightweight recorder rather than exercising a real network
call -- consistent with this being a pure "which graph name got resolved"
concern, not a store-behavior test (those live in test_falkor_store.py).
"""

from __future__ import annotations

from orion.substrate import falkor_store as fs
from orion.substrate.store import InMemorySubstrateGraphStore


class _RecordingFakeStore:
    """Captures the FalkorSubstrateStoreConfig it was constructed with."""

    last_cfg = None

    def __init__(self, cfg):
        _RecordingFakeStore.last_cfg = cfg


def test_default_builder_resolves_orion_substrate_graph_by_default(monkeypatch) -> None:
    monkeypatch.setenv("FALKORDB_URI", "redis://fake-host:6379")
    monkeypatch.delenv("FALKORDB_SUBSTRATE_GRAPH", raising=False)
    monkeypatch.setattr(fs, "FalkorSubstrateStore", _RecordingFakeStore)

    result = fs.build_falkor_substrate_store_from_env()

    assert isinstance(result, _RecordingFakeStore)
    assert _RecordingFakeStore.last_cfg.graph_name == "orion_substrate"
    assert _RecordingFakeStore.last_cfg.uri == "redis://fake-host:6379"


def test_default_builder_honors_falkordb_substrate_graph_env(monkeypatch) -> None:
    monkeypatch.setenv("FALKORDB_URI", "redis://fake-host:6379")
    monkeypatch.setenv("FALKORDB_SUBSTRATE_GRAPH", "some_custom_graph")
    monkeypatch.setattr(fs, "FalkorSubstrateStore", _RecordingFakeStore)

    fs.build_falkor_substrate_store_from_env()

    assert _RecordingFakeStore.last_cfg.graph_name == "some_custom_graph"


def test_aitown_builder_resolves_orion_substrate_aitown_graph_by_default(monkeypatch) -> None:
    monkeypatch.setenv("FALKORDB_URI", "redis://fake-host:6379")
    monkeypatch.delenv("FALKORDB_AITOWN_SUBSTRATE_GRAPH", raising=False)
    monkeypatch.setattr(fs, "FalkorSubstrateStore", _RecordingFakeStore)

    result = fs.build_aitown_falkor_substrate_store_from_env()

    assert isinstance(result, _RecordingFakeStore)
    assert _RecordingFakeStore.last_cfg.graph_name == "orion_substrate_aitown"
    # Same FalkorDB instance/URI as the primary graph -- a second graph name
    # on the same connection, not a separate infra dependency.
    assert _RecordingFakeStore.last_cfg.uri == "redis://fake-host:6379"


def test_aitown_builder_honors_its_own_env_override(monkeypatch) -> None:
    monkeypatch.setenv("FALKORDB_URI", "redis://fake-host:6379")
    monkeypatch.setenv("FALKORDB_AITOWN_SUBSTRATE_GRAPH", "custom_aitown_graph")
    monkeypatch.setattr(fs, "FalkorSubstrateStore", _RecordingFakeStore)

    fs.build_aitown_falkor_substrate_store_from_env()

    assert _RecordingFakeStore.last_cfg.graph_name == "custom_aitown_graph"


def test_aitown_builder_is_independent_of_the_primary_graph_env(monkeypatch) -> None:
    """The two builders must resolve independently -- setting only the
    primary graph's env var must not leak into the AI Town graph name."""
    monkeypatch.setenv("FALKORDB_URI", "redis://fake-host:6379")
    monkeypatch.setenv("FALKORDB_SUBSTRATE_GRAPH", "orion_substrate")
    monkeypatch.delenv("FALKORDB_AITOWN_SUBSTRATE_GRAPH", raising=False)
    monkeypatch.setattr(fs, "FalkorSubstrateStore", _RecordingFakeStore)

    fs.build_aitown_falkor_substrate_store_from_env()

    assert _RecordingFakeStore.last_cfg.graph_name == "orion_substrate_aitown"


def test_default_builder_falls_back_to_in_memory_when_uri_missing(monkeypatch) -> None:
    monkeypatch.delenv("FALKORDB_URI", raising=False)
    result = fs.build_falkor_substrate_store_from_env()
    assert isinstance(result, InMemorySubstrateGraphStore)


def test_aitown_builder_falls_back_to_in_memory_when_uri_missing(monkeypatch) -> None:
    monkeypatch.delenv("FALKORDB_URI", raising=False)
    result = fs.build_aitown_falkor_substrate_store_from_env()
    assert isinstance(result, InMemorySubstrateGraphStore)
