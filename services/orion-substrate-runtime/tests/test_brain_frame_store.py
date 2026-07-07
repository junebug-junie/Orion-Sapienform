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

from orion.schemas.brain_frame import SubstrateBrainFrameV1


class _RecordingEngine:
    def __init__(self, tail_rows=None, range_rows=None):
        self.executed = []
        self._tail_rows = tail_rows or []
        self._range_rows = range_rows or []

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
            rows = self._range_rows if "BETWEEN" in str(stmt) or "generated_at >=" in str(stmt) else self._tail_rows
            m.mappings.return_value.all.return_value = rows
            return m

        conn.execute.side_effect = execute
        yield conn


def _store_with(engine):
    from app.store import BiometricsSubstrateStore

    store = BiometricsSubstrateStore.__new__(BiometricsSubstrateStore)
    store._engine = engine
    return store


def _frame(seq, ts):
    return SubstrateBrainFrameV1(frame_id=f"f{seq}", generated_at=ts, tick_seq=seq, phase="live")


def test_save_brain_frame_inserts_and_prunes():
    eng = _RecordingEngine()
    store = _store_with(eng)
    store.save_brain_frame(_frame(1, datetime(2026, 7, 7, tzinfo=timezone.utc)), retention_hours=24)
    sqls = " ".join(s for s, _ in eng.executed)
    assert "INSERT INTO substrate_brain_frame_log" in sqls
    assert "DELETE FROM substrate_brain_frame_log" in sqls
    # params carried the frame id + json
    insert_params = eng.executed[0][1]
    assert insert_params["frame_id"] == "f1"
    assert insert_params["tick_seq"] == 1


def test_load_tail_returns_ascending():
    ts = datetime(2026, 7, 7, tzinfo=timezone.utc)
    rows = [
        {"frame_json": _frame(3, ts).model_dump(mode="json")},
        {"frame_json": _frame(2, ts).model_dump(mode="json")},
    ]  # DB returns DESC
    eng = _RecordingEngine(tail_rows=rows)
    store = _store_with(eng)
    frames = store.load_brain_frames_tail(limit=2)
    assert [f["tick_seq"] for f in frames] == [2, 3]  # reversed to ascending
