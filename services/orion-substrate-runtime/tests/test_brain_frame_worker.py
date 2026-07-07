from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from app.worker import BiometricsSubstrateWorker


def _node(node_id, kind, activation, pressure=0.0):
    return SimpleNamespace(
        node_id=node_id, node_kind=kind, label=kind,
        activation=activation, metadata={"dynamic_pressure": pressure},
    )


def _worker():
    w = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    w._settings = SimpleNamespace(
        brain_frame_enabled=True,
        brain_frame_sample_nodes=40,
        brain_frame_sample_edges=60,
        brain_frame_firing_threshold=0.5,
        brain_frame_starving_threshold=0.1,
        brain_frame_self_state_cadence_sec=30.0,
        brain_frame_spotlight_cadence_sec=30.0,
        brain_frame_retention_hours=24,
    )
    w._store = MagicMock()
    w._brain_frame_seq = 0
    return w


def test_brain_frame_tick_assembles_and_persists(monkeypatch):
    w = _worker()
    graph_store = MagicMock()
    graph_store.snapshot.return_value = SimpleNamespace(
        nodes={"t1": _node("t1", "tension", 0.9, 0.9)}, edges=[]
    )
    monkeypatch.setattr(w, "_get_substrate_graph_store", lambda **k: graph_store)
    monkeypatch.setattr(w, "_brain_frame_lane_health", lambda: {"cursor_lag_by_reducer": {}, "pending_backlog_by_reducer": {}, "quarantine_by_reducer": {}})
    monkeypatch.setattr(w, "_brain_frame_self_state", lambda: None)
    w._store.load_attention_broadcast.return_value = None

    frame = w._brain_frame_tick()
    assert frame is not None
    assert frame.phase == "live"
    assert w._store.save_brain_frame.called
    assert w._brain_frame_seq == 1


def test_brain_frame_tick_skips_when_disabled(monkeypatch):
    w = _worker()
    w._settings.brain_frame_enabled = False
    assert w._brain_frame_tick() is None
    assert not w._store.save_brain_frame.called
