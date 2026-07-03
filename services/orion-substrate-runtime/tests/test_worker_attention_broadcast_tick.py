"""Unit tests for the continuous attention broadcast tick (rung 3).

Verifies the worker's `_attention_broadcast_tick` is a no-op when disabled,
runs the workspace competition over the shared graph store snapshot, persists
the winning coalition as a projection, and fails open on errors.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from app.worker import BiometricsSubstrateWorker


def _make_worker(monkeypatch, *, broadcast_enabled: bool = True) -> BiometricsSubstrateWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv(
        "ORION_ATTENTION_BROADCAST_ENABLED", "true" if broadcast_enabled else "false"
    )
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = settings_mod.get_settings()
    worker._substrate_graph_store = None
    worker._store = MagicMock()
    return worker


def _graph_node(node_id: str, label: str, pressure: float) -> SimpleNamespace:
    return SimpleNamespace(
        node_id=node_id,
        label=label,
        metadata={"dynamic_pressure": pressure},
        signals=SimpleNamespace(confidence=0.8),
    )


def test_broadcast_disabled_is_noop(monkeypatch):
    worker = _make_worker(monkeypatch, broadcast_enabled=False)
    with patch("orion.substrate.graphdb_store.build_substrate_store_from_env") as build:
        worker._attention_broadcast_tick()
    build.assert_not_called()
    worker._store.save_attention_broadcast.assert_not_called()


def test_broadcast_persists_winning_coalition(monkeypatch):
    worker = _make_worker(monkeypatch, broadcast_enabled=True)
    fake_store = MagicMock()
    fake_store.snapshot.return_value = SimpleNamespace(
        nodes={
            "node:hot": _graph_node("node:hot", "unresolved contradiction", 0.9),
            "node:calm": _graph_node("node:calm", "calm concept", 0.05),
        }
    )
    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ):
        worker._attention_broadcast_tick()

    projection = worker._store.save_attention_broadcast.call_args.args[0]
    assert projection.projection_id == "substrate.attention.broadcast.v1"
    assert projection.selected_description == "unresolved contradiction"
    assert projection.attended_node_ids == ["node:hot"]


def test_broadcast_fails_open_on_store_init_error(monkeypatch):
    worker = _make_worker(monkeypatch, broadcast_enabled=True)
    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        side_effect=RuntimeError("fuseki down"),
    ):
        worker._attention_broadcast_tick()  # must not raise
    worker._store.save_attention_broadcast.assert_not_called()


def test_broadcast_fails_open_on_snapshot_error(monkeypatch):
    worker = _make_worker(monkeypatch, broadcast_enabled=True)
    fake_store = MagicMock()
    fake_store.snapshot.side_effect = RuntimeError("boom")
    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ):
        worker._attention_broadcast_tick()  # must not raise
    worker._store.save_attention_broadcast.assert_not_called()


def test_broadcast_tick_writes_dwell_log_row(monkeypatch):
    """Each broadcast tick appends a dwell row alongside the projection."""
    worker = _make_worker(monkeypatch, broadcast_enabled=True)
    fake_store = MagicMock()
    fake_store.snapshot.return_value = SimpleNamespace(
        nodes={"node:hot": _graph_node("node:hot", "unresolved contradiction", 0.9)}
    )
    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ):
        worker._attention_broadcast_tick()

    worker._store.save_attention_broadcast.assert_called_once()
    worker._store.save_coalition_dwell.assert_called_once()
    projection = worker._store.save_coalition_dwell.call_args.args[0]
    assert projection is worker._store.save_attention_broadcast.call_args.args[0]


def test_broadcast_tick_dwell_failure_does_not_break_tick(monkeypatch):
    worker = _make_worker(monkeypatch, broadcast_enabled=True)
    fake_store = MagicMock()
    fake_store.snapshot.return_value = SimpleNamespace(
        nodes={"node:hot": _graph_node("node:hot", "unresolved contradiction", 0.9)}
    )
    worker._store.save_coalition_dwell.side_effect = RuntimeError("db down")
    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ):
        worker._attention_broadcast_tick()  # must not raise

    worker._store.save_attention_broadcast.assert_called_once()
