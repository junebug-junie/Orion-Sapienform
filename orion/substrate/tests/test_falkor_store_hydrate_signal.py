"""Hydrate-outcome signal and optional client socket timeouts (additive,
fix/recall-pcr-block-timing review, 2026-09-30).

orion-recall caches its substrate store for the process lifetime and never
refreshes it on the recall path, so it must be able to tell "hydrated" from
"hydrate query raised and was swallowed" (empty cache). These pin the signal
and that the new timeout knobs default to the old behaviour.
"""

from __future__ import annotations

import pytest

from orion.graph import falkor_client as falkor_client_mod
from orion.substrate import falkor_store as falkor_store_mod
from orion.substrate.falkor_store import (
    FalkorSubstrateStore,
    FalkorSubstrateStoreConfig,
    RecordingFalkorClient,
)
from orion.substrate.graphdb_store import build_substrate_store_from_env
from orion.substrate.tests.test_falkor_store import _hydrated_node_row


class _RaisingClient:
    def graph_query(self, cypher, params=None):
        raise ConnectionError("falkor down")


def _cfg(**kw) -> FalkorSubstrateStoreConfig:
    return FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate", **kw)


def test_hydrate_success_sets_ok_and_node_count() -> None:
    client = RecordingFalkorClient(
        hydrate_node_rows=[_hydrated_node_row("concept-a", "concept:a"), _hydrated_node_row("concept-b", "concept:b")]
    )
    store = FalkorSubstrateStore(_cfg(), client=client)
    assert store.last_hydrate_ok is True
    assert store.last_hydrate_node_count == 2


def test_hydrate_empty_graph_is_ok_with_zero_nodes() -> None:
    store = FalkorSubstrateStore(_cfg(), client=RecordingFalkorClient())
    assert store.last_hydrate_ok is True
    assert store.last_hydrate_node_count == 0


def test_hydrate_failure_sets_ok_false() -> None:
    store = FalkorSubstrateStore(_cfg(), client=_RaisingClient())
    assert store.last_hydrate_ok is False
    assert store.last_hydrate_node_count == 0


def test_no_hydrate_leaves_signal_unset() -> None:
    store = FalkorSubstrateStore(_cfg(), client=RecordingFalkorClient(), hydrate=False)
    assert store.last_hydrate_ok is None


class _RecordingRedis:
    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _RecordingRedis.instances.append(self)


def _patch_redis(monkeypatch) -> None:
    import redis
    import redis.commands.graph as graph_mod

    _RecordingRedis.instances = []
    monkeypatch.setattr(redis, "Redis", _RecordingRedis)
    monkeypatch.setattr(graph_mod, "Graph", lambda r, name: object())


def test_redis_client_default_passes_no_socket_timeouts(monkeypatch) -> None:
    _patch_redis(monkeypatch)
    falkor_client_mod.RedisGraphQueryClient(uri="redis://h:6379/0", graph_name="g")
    kwargs = _RecordingRedis.instances[-1].kwargs
    assert "socket_timeout" not in kwargs and "socket_connect_timeout" not in kwargs


def test_redis_client_passes_socket_timeouts_when_set(monkeypatch) -> None:
    _patch_redis(monkeypatch)
    falkor_client_mod.RedisGraphQueryClient(
        uri="redis://h:6379/0", graph_name="g", socket_timeout=30, socket_connect_timeout=5
    )
    kwargs = _RecordingRedis.instances[-1].kwargs
    assert kwargs["socket_timeout"] == 30.0 and kwargs["socket_connect_timeout"] == 5.0


def test_env_builder_threads_timeouts_to_the_store_client(monkeypatch) -> None:
    seen: list = []

    class _Client:
        def __init__(self, **kwargs):
            seen.append(kwargs)

        def graph_query(self, cypher, params=None):
            return []

    monkeypatch.setattr(falkor_store_mod, "RedisGraphQueryClient", _Client)
    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "falkor")
    monkeypatch.setenv("FALKORDB_URI", "redis://h:6379")

    # Each build makes the store's own client first, then a short-lived,
    # always-bounded client for the node_id index bootstrap.
    build_substrate_store_from_env(falkor_socket_timeout_s=30.0, falkor_socket_connect_timeout_s=5.0)
    store_client, index_client = seen[-2], seen[-1]
    assert store_client["socket_timeout"] == 30.0 and store_client["socket_connect_timeout"] == 5.0
    assert index_client["socket_timeout"] == falkor_store_mod.ENSURE_INDEX_SOCKET_TIMEOUT_S

    seen.clear()
    build_substrate_store_from_env()
    store_client, index_client = seen
    assert "socket_timeout" not in store_client and "socket_connect_timeout" not in store_client
    assert index_client["socket_connect_timeout"] == falkor_store_mod.ENSURE_INDEX_CONNECT_TIMEOUT_S
