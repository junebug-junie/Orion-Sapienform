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


def test_load_range_downsamples_to_max_and_ascending():
    ts = datetime(2026, 7, 7, tzinfo=timezone.utc)
    start = datetime(2026, 7, 6, tzinfo=timezone.utc)
    end = datetime(2026, 7, 8, tzinfo=timezone.utc)

    big_rows = [{"frame_json": _frame(i, ts).model_dump(mode="json")} for i in range(1000)]
    eng = _RecordingEngine(range_rows=big_rows)
    store = _store_with(eng)
    result = store.load_brain_frames_range(start, end, max_frames=240)
    assert len(result) == 240
    seqs = [f["tick_seq"] for f in result]
    assert seqs == sorted(seqs)  # ascending
    assert seqs[0] == 0  # index 0 always kept

    small_rows = [{"frame_json": _frame(i, ts).model_dump(mode="json")} for i in range(10)]
    eng_small = _RecordingEngine(range_rows=small_rows)
    store_small = _store_with(eng_small)
    result_small = store_small.load_brain_frames_range(start, end, max_frames=240)
    assert [f["tick_seq"] for f in result_small] == list(range(10))


class _WindowEngine:
    """Fake engine returning queued ``.mappings().first()`` values in call order."""

    def __init__(self, first_values):
        self._first_values = list(first_values)

    @contextmanager
    def connect(self):
        engine = self

        def execute(stmt, params=None):
            m = MagicMock()
            m.mappings.return_value.first.return_value = engine._first_values.pop(0)
            return m

        conn = MagicMock()
        conn.execute.side_effect = execute
        yield conn


def test_brain_frame_window_reports_bounds_and_phase():
    earliest = datetime(2026, 7, 6, 1, 0, tzinfo=timezone.utc)
    latest = datetime(2026, 7, 7, 2, 0, tzinfo=timezone.utc)
    eng = _WindowEngine(
        [
            {"earliest": earliest, "latest": latest, "n": 5},
            {"phase": "live"},
        ]
    )
    store = _store_with(eng)
    window = store.brain_frame_window()
    assert window["earliest"] == earliest.isoformat()
    assert window["latest"] == latest.isoformat()
    assert window["frame_count"] == 5
    assert window["phase"] == "live"
