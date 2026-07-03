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
