from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from orion.schemas.system_one_appraisal import (
    SystemOneAnswerV1,
    SystemOneAppraisalFrameV1,
    SystemOneInputStateV1,
    SystemOneQuestionV1,
)


NOW = datetime(2026, 9, 23, tzinfo=timezone.utc)


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
            result = MagicMock()
            result.mappings.return_value.first.return_value = self._first_row
            return result

        conn.execute.side_effect = execute
        yield conn


def _store(engine):
    from app.store import BiometricsSubstrateStore

    store = BiometricsSubstrateStore.__new__(BiometricsSubstrateStore)
    store._engine = engine
    return store


def _frame() -> SystemOneAppraisalFrameV1:
    question = SystemOneQuestionV1(
        type="score",
        instructions="Rate",
        criteria=["low", "medium", "high"],
    )
    answer = SystemOneAnswerV1(
        question_id="reverie_fit",
        type="score",
        score=1.2,
        confidence=0.5,
        probabilities={"0": 0.1, "1": 0.6, "2": 0.3},
    )
    state = SystemOneInputStateV1(
        source_broadcast_projection_id="substrate.attention.broadcast.v1",
        source_broadcast_generated_at=NOW,
        selected_action_type="reflect",
        coalition_stability_score=0.7,
    )
    return SystemOneAppraisalFrameV1(
        frame_id="frame-1",
        question_set_id="test.v1",
        generated_at=NOW,
        expires_at=NOW + timedelta(seconds=90),
        provider="kev",
        model_id="kev-latest",
        input_state=state,
        questions={"reverie_fit": question},
        answers={"reverie_fit": answer},
    )


def test_save_system_one_appraisal_is_append_only_and_prunes() -> None:
    engine = _RecordingEngine()
    store = _store(engine)

    store.save_system_one_appraisal(_frame(), retention_hours=168.0)

    sql = " ".join(statement for statement, _ in engine.executed)
    assert "INSERT INTO substrate_system_one_appraisal" in sql
    assert "ON CONFLICT (frame_id) DO NOTHING" in sql
    assert "DELETE FROM substrate_system_one_appraisal" in sql
    params = engine.executed[0][1]
    assert params["frame_id"] == "frame-1"
    assert params["source_broadcast_projection_id"] == (
        "substrate.attention.broadcast.v1"
    )


def test_load_latest_system_one_appraisal_parses_typed_frame() -> None:
    frame = _frame()
    engine = _RecordingEngine(
        first_row={"frame_json": frame.model_dump(mode="json")}
    )
    store = _store(engine)

    loaded = store.load_latest_system_one_appraisal()

    assert loaded is not None
    assert loaded.frame_id == "frame-1"
    assert loaded.answers["reverie_fit"].probabilities["2"] == 0.3
    assert "ORDER BY generated_at DESC" in engine.executed[0][0]


def test_load_latest_system_one_appraisal_fails_open() -> None:
    engine = MagicMock()
    engine.connect.side_effect = RuntimeError("db unavailable")
    store = _store(engine)

    assert store.load_latest_system_one_appraisal() is None
