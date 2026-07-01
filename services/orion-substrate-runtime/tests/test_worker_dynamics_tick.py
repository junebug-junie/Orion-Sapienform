"""Unit tests for the periodic dynamics-engine tick loop (closes PR #766 gap).

Verifies the worker's `_dynamics_tick` reuses the shared substrate graph
store, is a no-op when disabled, and fails open on both store-init and
engine errors.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from app.worker import BiometricsSubstrateWorker


def _make_worker(monkeypatch, *, dynamics_tick_enabled: bool = True) -> BiometricsSubstrateWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv(
        "SUBSTRATE_DYNAMICS_TICK_ENABLED", "true" if dynamics_tick_enabled else "false"
    )
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = settings_mod.get_settings()
    worker._substrate_graph_store = None
    return worker


def test_dynamics_tick_disabled_is_noop(monkeypatch):
    worker = _make_worker(monkeypatch, dynamics_tick_enabled=False)
    with patch("orion.substrate.graphdb_store.build_substrate_store_from_env") as build:
        worker._dynamics_tick()
    build.assert_not_called()


def test_dynamics_tick_calls_engine_against_shared_store(monkeypatch):
    worker = _make_worker(monkeypatch, dynamics_tick_enabled=True)
    fake_store = MagicMock()
    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ) as build, patch("orion.substrate.dynamics.SubstrateDynamicsEngine") as engine_cls:
        worker._dynamics_tick()
        worker._dynamics_tick()  # second call must reuse the cached store

    build.assert_called_once()
    engine_cls.assert_called_with(store=fake_store)
    assert worker._substrate_graph_store is fake_store


def test_dynamics_tick_fails_open_on_store_init_error(monkeypatch):
    worker = _make_worker(monkeypatch, dynamics_tick_enabled=True)
    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        side_effect=RuntimeError("fuseki down"),
    ):
        worker._dynamics_tick()  # must not raise


def test_dynamics_tick_fails_open_on_engine_error(monkeypatch):
    worker = _make_worker(monkeypatch, dynamics_tick_enabled=True)
    fake_store = MagicMock()
    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ), patch("orion.substrate.dynamics.SubstrateDynamicsEngine") as engine_cls:
        engine_cls.return_value.tick.side_effect = RuntimeError("boom")
        worker._dynamics_tick()  # must not raise
