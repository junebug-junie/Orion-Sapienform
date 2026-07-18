"""Unit tests for the Rung 1 bridge: worker writes prediction_error onto a
durable substrate node (default-off, fail-open)."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from app.worker import BiometricsSubstrateWorker, _PREDICTION_ERROR_NODE_FLAG


class _FakeStore:
    """Records upsert_node calls without touching a real graph store."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def upsert_node(self, *, identity_key, node) -> None:
        self.calls.append({"identity_key": identity_key, "node": node})


class _RaisingStore:
    def __init__(self) -> None:
        self.calls = 0

    def upsert_node(self, *, identity_key, node) -> None:
        self.calls += 1
        raise RuntimeError("boom")


class _RecordingStore:
    """Fake store that actually persists nodes by node_id, so get_node_by_id
    carry-forward (existing.metadata) can be exercised across repeat writes
    to the same fixed node_id, the way the real graph store does."""

    def __init__(self) -> None:
        self.nodes: dict = {}
        self.upsert_calls = 0

    def get_node_by_id(self, node_id):
        return self.nodes.get(node_id)

    def upsert_node(self, *, identity_key, node) -> None:
        self.upsert_calls += 1
        self.nodes[node.node_id] = node


def _make_worker(store) -> BiometricsSubstrateWorker:
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._substrate_graph_store = store
    return worker


def test_write_prediction_error_node_upserts_when_flag_on(monkeypatch, caplog) -> None:
    monkeypatch.setenv(_PREDICTION_ERROR_NODE_FLAG, "true")
    store = _FakeStore()
    worker = _make_worker(store)
    now = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)

    with caplog.at_level("INFO"):
        worker._write_prediction_error_node(
            node_id="node:substrate.execution",
            error=0.42,
            now=now,
            reducer_key="execution_trajectory",
        )

    assert len(store.calls) == 1
    call = store.calls[0]
    assert call["identity_key"] == "substrate_prediction_error|node:substrate.execution"
    node = call["node"]
    assert node.metadata["prediction_error"] == 0.42
    assert node.anchor_scope == "orion"
    assert node.subject_ref == "entity:orion"
    assert node.temporal.observed_at == now
    assert node.node_id == "node:substrate.execution"
    assert node.metadata["reducer_key"] == "execution_trajectory"

    # Success-path visibility: one INFO line proving the write actually happened.
    written_records = [
        r for r in caplog.records if r.message.startswith("substrate_prediction_error_node_written")
    ]
    assert len(written_records) == 1
    assert written_records[0].levelname == "INFO"
    assert "node:substrate.execution" in written_records[0].message
    assert "0.420" in written_records[0].message
    assert "execution_trajectory" in written_records[0].message


def test_write_prediction_error_node_warns_when_store_init_failed(monkeypatch, caplog) -> None:
    monkeypatch.setenv(_PREDICTION_ERROR_NODE_FLAG, "true")
    worker = _make_worker(None)
    # Simulate store init failing/unavailable without depending on real env resolution.
    monkeypatch.setattr(worker, "_get_substrate_graph_store", lambda **_: None)

    with caplog.at_level("WARNING"):
        worker._write_prediction_error_node(
            node_id="node:substrate.transport",
            error=0.5,
            now=datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc),
            reducer_key="transport_bus",
        )

    skipped_records = [
        r for r in caplog.records if r.message.startswith("substrate_prediction_error_node_skipped_no_store")
    ]
    assert len(skipped_records) == 1
    assert skipped_records[0].levelname == "WARNING"
    assert "node:substrate.transport" in skipped_records[0].message
    assert "transport_bus" in skipped_records[0].message


def test_write_prediction_error_node_noop_when_flag_off(monkeypatch) -> None:
    monkeypatch.delenv(_PREDICTION_ERROR_NODE_FLAG, raising=False)
    store = _FakeStore()
    worker = _make_worker(store)
    now = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)

    worker._write_prediction_error_node(
        node_id="node:substrate.execution",
        error=0.42,
        now=now,
        reducer_key="execution_trajectory",
    )

    assert store.calls == []


def test_write_prediction_error_node_fail_open_on_raise(monkeypatch) -> None:
    monkeypatch.setenv(_PREDICTION_ERROR_NODE_FLAG, "1")
    store = _RaisingStore()
    worker = _make_worker(store)
    now = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Must not propagate.
    worker._write_prediction_error_node(
        node_id="node:substrate.transport",
        error=0.9,
        now=now,
        reducer_key="transport_bus",
    )

    assert store.calls == 1


def test_contributing_turn_ids_accumulate_across_repeat_writes(monkeypatch) -> None:
    monkeypatch.setenv(_PREDICTION_ERROR_NODE_FLAG, "true")
    store = _RecordingStore()
    worker = _make_worker(store)
    now = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)

    worker._write_prediction_error_node(
        node_id="node:substrate.harness_closure",
        error=0.4,
        now=now,
        reducer_key="post_turn_closure",
        contributing_id="corr-1",
    )
    worker._write_prediction_error_node(
        node_id="node:substrate.harness_closure",
        error=0.4,
        now=now,
        reducer_key="post_turn_closure",
        contributing_id="corr-2",
    )

    node = store.nodes["node:substrate.harness_closure"]
    assert node.metadata["contributing_turn_ids"] == ["corr-1", "corr-2"]


def test_contributing_turn_ids_dedupes_same_id_on_refire(monkeypatch) -> None:
    monkeypatch.setenv(_PREDICTION_ERROR_NODE_FLAG, "true")
    store = _RecordingStore()
    worker = _make_worker(store)
    now = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)

    for _ in range(3):
        worker._write_prediction_error_node(
            node_id="node:substrate.harness_closure",
            error=0.4,
            now=now,
            reducer_key="post_turn_closure",
            contributing_id="corr-1",
        )

    node = store.nodes["node:substrate.harness_closure"]
    assert node.metadata["contributing_turn_ids"] == ["corr-1"]


def test_contributing_turn_ids_caps_and_drops_oldest(monkeypatch) -> None:
    monkeypatch.setenv(_PREDICTION_ERROR_NODE_FLAG, "true")
    store = _RecordingStore()
    worker = _make_worker(store)
    now = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)

    for i in range(21):
        worker._write_prediction_error_node(
            node_id="node:substrate.harness_closure",
            error=0.4,
            now=now,
            reducer_key="post_turn_closure",
            contributing_id=f"corr-{i}",
        )

    node = store.nodes["node:substrate.harness_closure"]
    ids = node.metadata["contributing_turn_ids"]
    assert len(ids) == 20
    # Oldest (corr-0) dropped, most recent (corr-20) retained, order preserved.
    assert ids[0] == "corr-1"
    assert ids[-1] == "corr-20"


def test_contributing_turn_ids_tolerates_malformed_existing_value(monkeypatch) -> None:
    """Fail-open: a non-list stored value must not raise, just be discarded."""
    monkeypatch.setenv(_PREDICTION_ERROR_NODE_FLAG, "true")
    store = _RecordingStore()
    worker = _make_worker(store)
    now = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)

    worker._write_prediction_error_node(
        node_id="node:substrate.harness_closure",
        error=0.4,
        now=now,
        reducer_key="post_turn_closure",
        contributing_id="corr-1",
    )
    # Corrupt the stored value directly to simulate a malformed carried-forward field.
    bad_metadata = dict(store.nodes["node:substrate.harness_closure"].metadata)
    bad_metadata["contributing_turn_ids"] = "not-a-list"
    store.nodes["node:substrate.harness_closure"] = store.nodes[
        "node:substrate.harness_closure"
    ].model_copy(update={"metadata": bad_metadata})

    worker._write_prediction_error_node(
        node_id="node:substrate.harness_closure",
        error=0.4,
        now=now,
        reducer_key="post_turn_closure",
        contributing_id="corr-2",
    )

    node = store.nodes["node:substrate.harness_closure"]
    assert node.metadata["contributing_turn_ids"] == ["corr-2"]


def test_no_contributing_turn_ids_key_when_contributing_id_never_provided(monkeypatch) -> None:
    monkeypatch.setenv(_PREDICTION_ERROR_NODE_FLAG, "true")
    store = _RecordingStore()
    worker = _make_worker(store)
    now = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)

    worker._write_prediction_error_node(
        node_id="node:substrate.execution",
        error=0.4,
        now=now,
        reducer_key="execution_trajectory",
    )

    node = store.nodes["node:substrate.execution"]
    assert "contributing_turn_ids" not in node.metadata
