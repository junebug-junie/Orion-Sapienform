"""Unit tests for the AST/HOT self-model live tick's two new store methods
(docs/superpowers/specs/2026-07-29-ast-hot-reducer-live-ticking-design.md):
`get_latest_field_attention_frame` (read-only, cross-service) and
`save_attention_self_model` (this tick's sole write path, into a table
nothing else in the repo touches).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from orion.schemas.attention_self_model import AttentionSelfModelV1
from orion.schemas.field_attention_frame import FieldAttentionFrameV1


class _RecordingEngine:
    def __init__(self, first_row=None):
        self.executed = []
        self._first_row = first_row

    @contextmanager
    def begin(self):
        conn = MagicMock()

        def execute(stmt, params=None):
            self.executed.append((str(stmt), params))
            return MagicMock()

        conn.execute.side_effect = execute
        yield conn

    @contextmanager
    def connect(self):
        conn = MagicMock()

        def execute(stmt, params=None):
            self.executed.append((str(stmt), params))
            m = MagicMock()
            m.mappings.return_value.first.return_value = self._first_row
            return m

        conn.execute.side_effect = execute
        yield conn


def _store_with(engine):
    from app.store import BiometricsSubstrateStore

    store = BiometricsSubstrateStore.__new__(BiometricsSubstrateStore)
    store._engine = engine
    return store


def _field_frame(ts) -> FieldAttentionFrameV1:
    return FieldAttentionFrameV1(
        frame_id="f1",
        source_field_tick_id="t1",
        source_field_generated_at=ts,
        generated_at=ts,
        overall_salience=0.5,
    )


def test_get_latest_field_attention_frame_parses_row():
    ts = datetime(2026, 7, 29, tzinfo=timezone.utc)
    row = {"frame_json": _field_frame(ts).model_dump(mode="json")}
    eng = _RecordingEngine(first_row=row)
    store = _store_with(eng)

    frame = store.get_latest_field_attention_frame()

    assert frame is not None
    assert frame.frame_id == "f1"
    sql = eng.executed[0][0]
    assert "SELECT frame_json FROM substrate_attention_frames" in sql
    assert "ORDER BY generated_at DESC" in sql


def test_get_latest_field_attention_frame_returns_none_when_empty():
    eng = _RecordingEngine(first_row=None)
    store = _store_with(eng)
    assert store.get_latest_field_attention_frame() is None


def test_get_latest_field_attention_frame_fails_open_on_error():
    eng = MagicMock()
    eng.connect.side_effect = RuntimeError("db down")
    store = _store_with(eng)
    assert store.get_latest_field_attention_frame() is None  # must not raise


def test_save_attention_self_model_inserts_and_prunes():
    eng = _RecordingEngine()
    store = _store_with(eng)
    ts = datetime(2026, 7, 29, tzinfo=timezone.utc)
    model = AttentionSelfModelV1(generated_at=ts, attention_reason="bottom_up_salience")

    store.save_attention_self_model(model, retention_hours=168.0)

    sqls = " ".join(s for s, _ in eng.executed)
    assert "INSERT INTO substrate_attention_self_model" in sqls
    assert "DELETE FROM substrate_attention_self_model" in sqls
    insert_params = eng.executed[0][1]
    assert insert_params["generated_at"] == ts
    assert insert_params["model_id"].startswith("self-model-")
