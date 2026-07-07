"""Unit tests for the Orion embodiment perception->substrate ingest consumer.

Verifies the substrate worker folds a decoded ``WorldPerceptionV1`` into the
shared graph store as real proximity nodes when enabled, writes nothing when
disabled, and fails open on both undecodable messages and store-write errors.
Constructs the worker via ``__new__`` (mirrors ``test_embodiment_c_hook``).
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

from orion.core.schemas.cognitive_substrate import ConceptNodeV1

_PERCEPTION = {
    "player_id": "orion",
    "position": {"x": 0.0, "y": 0.0},
    "nearby_players": [
        {"player_id": "j", "name": "Juniper", "position": {"x": 1.0, "y": 0.0}, "distance": 1.0}
    ],
}


def _make_worker(monkeypatch, *, ingest_enabled: bool) -> BiometricsSubstrateWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv(
        "EMBODIMENT_PERCEPTION_SUBSTRATE_ENABLED", "true" if ingest_enabled else "false"
    )
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = settings_mod.get_settings()
    worker._bus = MagicMock()
    worker._substrate_graph_store = None
    return worker


def _set_decode(worker, *, ok: bool, payload: dict | None = None) -> None:
    decoded = MagicMock()
    decoded.ok = ok
    decoded.error = None if ok else "boom"
    decoded.envelope = MagicMock(payload=payload or {})
    worker._bus.codec.decode.return_value = decoded


def test_flag_on_writes_real_proximity_node(monkeypatch):
    worker = _make_worker(monkeypatch, ingest_enabled=True)
    _set_decode(worker, ok=True, payload=_PERCEPTION)
    fake_store = MagicMock()
    worker._substrate_graph_store = fake_store

    worker._ingest_perception_message({"data": b"..."})

    assert fake_store.upsert_node.call_count == 1
    kwargs = fake_store.upsert_node.call_args.kwargs
    assert kwargs["identity_key"] == "town_perception|j"
    node = kwargs["node"]
    # A real node, not an empty/placeholder projection.
    assert isinstance(node, ConceptNodeV1)
    assert "Juniper" in node.label
    assert node.metadata.get("player_id") == "j"
    assert node.signals.salience > 0.0


def test_flag_off_writes_nothing(monkeypatch):
    worker = _make_worker(monkeypatch, ingest_enabled=False)
    _set_decode(worker, ok=True, payload=_PERCEPTION)
    fake_store = MagicMock()
    worker._substrate_graph_store = fake_store

    worker._ingest_perception_message({"data": b"..."})

    fake_store.upsert_node.assert_not_called()


def test_undecodable_message_fails_open(monkeypatch):
    worker = _make_worker(monkeypatch, ingest_enabled=True)
    _set_decode(worker, ok=False)
    fake_store = MagicMock()
    worker._substrate_graph_store = fake_store

    # Must not raise; nothing written.
    worker._ingest_perception_message({"data": b"garbage"})

    fake_store.upsert_node.assert_not_called()


def test_invalid_payload_fails_open(monkeypatch):
    worker = _make_worker(monkeypatch, ingest_enabled=True)
    _set_decode(worker, ok=True, payload={"not": "a perception"})
    fake_store = MagicMock()
    worker._substrate_graph_store = fake_store

    # Malformed-but-decodable payload must not propagate.
    worker._ingest_perception_message({"data": b"..."})

    fake_store.upsert_node.assert_not_called()


def test_store_write_exception_fails_open(monkeypatch):
    worker = _make_worker(monkeypatch, ingest_enabled=True)
    _set_decode(worker, ok=True, payload=_PERCEPTION)
    fake_store = MagicMock()
    fake_store.upsert_node.side_effect = RuntimeError("fuseki down")
    worker._substrate_graph_store = fake_store

    # Store-write error must be swallowed, not propagated.
    worker._ingest_perception_message({"data": b"..."})

    assert fake_store.upsert_node.call_count == 1


def test_no_store_fails_open(monkeypatch):
    worker = _make_worker(monkeypatch, ingest_enabled=True)
    _set_decode(worker, ok=True, payload=_PERCEPTION)
    worker._substrate_graph_store = None

    # Build path raises -> _get_substrate_graph_store returns None -> skip, no raise.
    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        side_effect=RuntimeError("no store"),
    ):
        worker._ingest_perception_message({"data": b"..."})
