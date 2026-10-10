"""World-first wiring in the substrate broadcast tick (spec 2026-10-07 A)."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, SUBSTRATE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.worker import BiometricsSubstrateWorker  # noqa: E402


def _worker(monkeypatch, *, world_first: bool) -> BiometricsSubstrateWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("ORION_ATTENTION_BROADCAST_ENABLED", "true")
    monkeypatch.setenv("SUBSTRATE_PE_HISTORY_ENABLED", "false")
    monkeypatch.setenv("ATTENTION_WORLD_FIRST_ENABLED", "true" if world_first else "false")
    monkeypatch.setenv("ORION_ATTENTION_TOPDOWN_ENABLED", "false")
    import app.settings as settings_mod

    settings_mod._settings = None
    w = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    w._settings = settings_mod.get_settings()
    w._substrate_graph_store = None
    w._store = MagicMock()
    return w


def _snapshot():
    node = SimpleNamespace(
        node_id="node:concept.hot", label="unresolved contradiction",
        metadata={"dynamic_pressure": 0.9}, signals=SimpleNamespace(confidence=0.8),
    )
    return SimpleNamespace(nodes={"node:concept.hot": node})


def _tick(worker):
    store = MagicMock()
    store.snapshot.return_value = _snapshot()
    with patch("orion.substrate.graphdb_store.build_substrate_store_from_env", return_value=store), patch(
        "orion.substrate.attention_broadcast.load_terminal_verdict_loop_ids", return_value=set()
    ):
        worker._attention_broadcast_tick()
    return worker._store.save_attention_broadcast.call_args.args[0]


def test_flag_on_reads_chat_and_attends_the_busy_world(monkeypatch) -> None:
    w = _worker(monkeypatch, world_first=True)
    now = datetime.now(timezone.utc)
    w._store.fetch_chat_turn_times.return_value = [
        now - timedelta(days=d, minutes=40) for d in range(1, 7)
    ] + [now - timedelta(minutes=1)]
    proj = _tick(w)
    w._store.fetch_chat_turn_times.assert_called_once()
    assert proj.attended_node_ids == ["world:chat"]
    # The uncalibrated concept node lost to the world instead of winning on pressure.
    assert proj.frame.debug["world_first"]["uncalibrated"] == ["node:concept.hot"]


def test_flag_on_chat_read_failure_is_absent_and_no_winner(monkeypatch) -> None:
    w = _worker(monkeypatch, world_first=True)
    w._store.fetch_chat_turn_times.side_effect = RuntimeError("db down")
    proj = _tick(w)
    assert proj.attended_node_ids == []
    chat = next(c for c in proj.frame.debug["world_first"]["candidates"] if c["source_id"] == "world:chat")
    assert chat["absent"] is True


def test_flag_off_never_reads_chat_and_keeps_the_pressure_winner(monkeypatch) -> None:
    w = _worker(monkeypatch, world_first=False)
    proj = _tick(w)
    w._store.fetch_chat_turn_times.assert_not_called()
    assert proj.attended_node_ids == ["node:concept.hot"]
    assert "world_first" not in proj.frame.debug
