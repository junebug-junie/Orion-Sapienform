"""PE retire Patch 2: the alarm writers leave the prediction-error slot.

_write_signal_node writes signal_value / signal_kind and no prediction_error key (so the first
write clears the old key on the graph); _write_prediction_error_node refuses an alarm id; the store
never puts an alarm row in the PE table; the history fetch reads the alarms read-both."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, SUBSTRATE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.worker import _PREDICTION_ERROR_NODE_FLAG  # noqa: E402
from test_store_signal_history import TS, _Engine, _store  # noqa: E402
from test_worker_prediction_error_node import _FakeStore, _make_worker  # noqa: E402

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)
ALARMS = ("node:substrate.cabinet", "node:substrate.strain", "node:substrate.street")


@pytest.fixture(autouse=True)
def _nodes_on(monkeypatch):
    monkeypatch.setenv(_PREDICTION_ERROR_NODE_FLAG, "true")


@pytest.mark.parametrize("node_id", ALARMS)
def test_signal_node_carries_signal_value_and_no_prediction_error(node_id):
    store = _FakeStore()
    assert _make_worker(store)._write_signal_node(node_id=node_id, value=0.7, now=NOW, reducer_key="k")
    (call,) = store.calls
    md = call["node"].metadata
    assert md["signal_value"] == 0.7 and md["signal_kind"] == "alarm"
    assert "prediction_error" not in md
    # Same node identity as before the move, so the write updates (and clears) the old node.
    assert call["identity_key"] == f"substrate_prediction_error|{node_id}"
    assert "prediction error" not in call["node"].label.lower()


@pytest.mark.parametrize("node_id", ALARMS)
def test_prediction_error_writer_refuses_an_alarm(node_id):
    store = _FakeStore()
    assert _make_worker(store)._write_prediction_error_node(node_id=node_id, error=1.0, now=NOW) is False
    assert store.calls == []


def test_signal_writer_refuses_a_non_moved_node():
    store = _FakeStore()
    w = _make_worker(store)
    assert w._write_signal_node(node_id="node:substrate.perception", value=0.2, now=NOW) is False
    assert w._write_prediction_error_node(node_id="node:prediction.body", error=0.2, now=NOW) is True
    assert "prediction_error" in store.calls[0]["node"].metadata


def test_codec_round_trips_signal_value_and_clears_prediction_error():
    from orion.substrate.falkor_codec import (
        DYNAMICS_METADATA_KEYS,
        EXTERNALLY_OWNED_METADATA_KEYS,
        _dynamics_metadata_from_row,
        encode_node_properties,
    )

    store = _FakeStore()
    _make_worker(store)._write_signal_node(node_id="node:substrate.strain", value=1.0, now=NOW)
    props = encode_node_properties(store.calls[0]["node"], "substrate_prediction_error|node:substrate.strain")
    assert props["signal_value"] == 1.0 and props["signal_kind"] == "alarm"
    assert props["prediction_error"] is None  # ON MATCH SET clears the old key
    back = _dynamics_metadata_from_row(props)
    assert back["signal_value"] == 1.0 and back["signal_kind"] == "alarm" and "prediction_error" not in back
    assert {"signal_value", "signal_kind"} <= set(DYNAMICS_METADATA_KEYS) & EXTERNALLY_OWNED_METADATA_KEYS


def test_store_never_writes_an_alarm_row_to_the_pe_table(caplog):
    eng = _Engine()
    with caplog.at_level("WARNING"):
        n = _store(eng).save_prediction_error_history_samples(
            [("node:substrate.strain", TS, 1.0), ("node:prediction.body", TS, 0.2)]
        )
    assert n == 1
    (_, params), = eng.executed
    assert [p["node_id"] for p in params] == ["node:prediction.body"]
    assert "substrate_pe_history_moved_node_refused" in caplog.text
    eng2 = _Engine()
    assert _store(eng2).save_prediction_error_history_samples([("node:substrate.cabinet", TS, 0.0)]) == 0
    assert eng2.executed == []


def test_history_seed_reads_the_alarms_read_both():
    eng = _Engine(rows=[("node:substrate.strain", TS, 1.0)])
    rows = _store(eng).fetch_prediction_error_history(since=TS)
    sql, params = eng.executed[0]
    assert "substrate_node_signal_history" in sql and "DISTINCT ON (node_id, observed_at)" in sql
    assert params == {"since": TS} and rows == [("node:substrate.strain", TS, 1.0)]


