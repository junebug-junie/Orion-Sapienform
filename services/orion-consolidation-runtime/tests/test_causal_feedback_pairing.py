from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from app.store import ConsolidationRuntimeStore
from app.worker import ConsolidationRuntimeWorker
from orion.consolidation.windows import stable_consolidation_frame_id
from orion.schemas.feedback_frame import FeedbackFrameV1


class _ScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar_one(self):
        return self.value


class _CaptureConnection:
    def __init__(self, engine):
        self.engine = engine

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, statement, params=None):
        self.engine.calls.append((" ".join(str(statement).split()), params))
        return _ScalarResult(datetime(2026, 11, 3, 0, 45, tzinfo=timezone.utc))


class _CaptureEngine:
    def __init__(self):
        self.calls = []

    def connect(self):
        return _CaptureConnection(self)


def test_completion_watermark_covers_invisible_and_visible_dispatches():
    store = ConsolidationRuntimeStore.__new__(ConsolidationRuntimeStore)
    store._engine = _CaptureEngine()

    store.load_feedback_completion_watermark(source_visibility_lag_sec=900)

    sql, params = store._engine.calls[0]
    assert "LEAST(" in sql
    assert "WHERE feedback_pending" in sql
    assert "FROM substrate_execution_dispatch_inflight" in sql
    assert "now() - make_interval(secs => :visibility_lag_sec)" in sql
    assert params == {"visibility_lag_sec": 900}


def test_feedback_is_selected_by_source_dispatch_window_not_processing_time():
    store = ConsolidationRuntimeStore.__new__(ConsolidationRuntimeStore)
    calls = []

    def capture(sql, window_start, window_end, limit, model, column):
        calls.append((" ".join(sql.split()), window_start, window_end, limit, model, column))
        return []

    store._load_rows = capture
    start = datetime(2026, 11, 3, 5, tzinfo=timezone.utc)
    end = datetime(2026, 11, 3, 6, tzinfo=timezone.utc)

    store.load_window_data(start, end, 500)

    feedback_sql, *_ = next(call for call in calls if call[4] is FeedbackFrameV1)
    assert "WITH window_dispatches AS" in feedback_sql
    assert "FROM substrate_execution_dispatch_frames" in feedback_sql
    assert "f.source_execution_dispatch_frame_id = d.frame_id" in feedback_sql
    assert "WHERE generated_at >= :window_start" in feedback_sql
    assert "FROM substrate_feedback_frames WHERE generated_at" not in feedback_sql
    assert "LIMIT :max_per_source" in feedback_sql


class _WatermarkStore:
    def __init__(self, watermark):
        self.watermark = watermark
        self.requested_frame_id = None
        self.visibility_lag_sec = None

    def load_feedback_completion_watermark(self, *, source_visibility_lag_sec):
        self.visibility_lag_sec = source_visibility_lag_sec
        return self.watermark

    def load_consolidation_frame_for_window(self, frame_id):
        self.requested_frame_id = frame_id
        return object()  # stop after proving which stable window was selected


def _window_requested_at(watermark):
    worker = ConsolidationRuntimeWorker.__new__(ConsolidationRuntimeWorker)
    worker._settings = SimpleNamespace(enable_consolidation_runtime=True)
    worker._policy = SimpleNamespace(
        policy_id="consolidation_policy.v1",
        window=SimpleNamespace(
            lookback_minutes=60,
            max_frames_per_source=500,
            source_visibility_lag_sec=900,
        ),
    )
    worker._store = _WatermarkStore(watermark)
    worker._tick()
    assert worker._store.visibility_lag_sec == 900
    return worker._store.requested_frame_id


def test_pending_dispatch_holds_its_hour_open_until_feedback_completes():
    pending_at = datetime(2026, 11, 3, 0, 59, 55, tzinfo=timezone.utc)
    held_frame = stable_consolidation_frame_id(
        window_start=datetime(2026, 11, 2, 23, tzinfo=timezone.utc),
        window_end=datetime(2026, 11, 3, 0, tzinfo=timezone.utc),
        policy_id="consolidation_policy.v1",
    )
    assert _window_requested_at(pending_at) == held_frame

    crossed_at = datetime(2026, 11, 3, 1, 0, tzinfo=timezone.utc)
    newly_closed_frame = stable_consolidation_frame_id(
        window_start=datetime(2026, 11, 3, 0, tzinfo=timezone.utc),
        window_end=crossed_at,
        policy_id="consolidation_policy.v1",
    )
    assert _window_requested_at(crossed_at) == newly_closed_frame
