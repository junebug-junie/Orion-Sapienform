"""Field-channel raw-substrate corpus collector (Item 1 v2, roadmap item 1
correction, 2026-07-13) -- mirrors the shape of orion-spark-introspector's
services/orion-spark-introspector/tests/test_mood_arc_corpus.py.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import app.worker as worker
from app.tensor.field_state import empty_field_state
from test_worker import _make_worker


def _read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip("\n").splitlines()
    return [json.loads(line) for line in lines if line]


def test_field_channel_corpus_appends_with_real_channel_pressures(monkeypatch, tmp_path) -> None:
    corpus_path = tmp_path / "field_channel.jsonl"
    monkeypatch.setattr(
        worker, "_FIELD_CHANNEL_SINK", worker.InnerStateCorpusSink(str(corpus_path)), raising=False
    )

    field_worker = _make_worker(monkeypatch, idle_tick_enabled=True)
    field_worker._store.fetch_new_receipts.return_value = []
    existing = empty_field_state(
        lattice=field_worker._lattice, now=datetime.now(timezone.utc), tick_id="tick_old"
    )
    field_worker._store.load_latest_field.return_value = existing

    field_worker._tick()

    rows = _read_jsonl(corpus_path)
    assert len(rows) == 1

    saved_state = field_worker._store.save_field.call_args.args[0]
    expected_channels, _provenance = worker.collect_field_channel_pressures(saved_state)
    # DEFAULT_NODE_VECTOR/DEFAULT_CAPABILITY_VECTOR seed real, non-empty
    # pressure channels (e.g. availability=1.0, confidence=1.0, plus every
    # zero-valued PRESSURE_CHANNELS entry) -- this is not an empty-shell row.
    assert expected_channels
    row = rows[0]
    assert row["tick_id"] == saved_state.tick_id
    assert row["channels"] == expected_channels


def test_field_channel_corpus_disabled_when_path_empty(monkeypatch, tmp_path) -> None:
    corpus_path = tmp_path / "field_channel.jsonl"
    # Default construction: empty path -> disabled, no-op.
    monkeypatch.setattr(worker, "_FIELD_CHANNEL_SINK", worker.InnerStateCorpusSink(""), raising=False)

    field_worker = _make_worker(monkeypatch, idle_tick_enabled=True)
    field_worker._store.fetch_new_receipts.return_value = []
    existing = empty_field_state(
        lattice=field_worker._lattice, now=datetime.now(timezone.utc), tick_id="tick_old"
    )
    field_worker._store.load_latest_field.return_value = existing

    field_worker._tick()

    assert not corpus_path.exists()
    field_worker._store.save_field.assert_called_once()


def test_field_channel_corpus_construction_failure_does_not_abort_tick(monkeypatch, tmp_path) -> None:
    # Mirrors test_mood_arc_corpus_construction_failure_does_not_abort_tick:
    # a pydantic ValidationError (or any other non-OSError exception) raised
    # while building FieldChannelCorpusRowV1 must not propagate past _tick()
    # and abort the real digestion/save-field work that follows.
    corpus_path = tmp_path / "field_channel.jsonl"
    monkeypatch.setattr(
        worker, "_FIELD_CHANNEL_SINK", worker.InnerStateCorpusSink(str(corpus_path)), raising=False
    )

    def _raise(*args, **kwargs):
        raise ValueError("simulated corpus-row construction failure")

    monkeypatch.setattr(worker, "FieldChannelCorpusRowV1", _raise, raising=False)

    field_worker = _make_worker(monkeypatch, idle_tick_enabled=True)
    field_worker._store.fetch_new_receipts.return_value = []
    existing = empty_field_state(
        lattice=field_worker._lattice, now=datetime.now(timezone.utc), tick_id="tick_old"
    )
    field_worker._store.load_latest_field.return_value = existing

    field_worker._tick()

    assert not corpus_path.exists(), "the simulated failure should have prevented any write"
    field_worker._store.save_field.assert_called_once(), (
        "a field-channel corpus construction failure must not abort the rest of _tick()"
    )
