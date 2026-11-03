from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.worker import (
    ControlReductionResult,
    FeedbackRuntimeWorker,
    _is_structurally_empty_dispatch,
)
from orion.autonomy.contrast import ControlCell
from orion.autonomy.prediction import EffectPosterior
from orion.feedback.outcome_resolution import ControlObservation, OutcomeResolution
from orion.schemas.execution_dispatch_frame import ExecutionDispatchFrameV1
from orion.schemas.feedback_frame import FeedbackFrameV1


def _empty_dispatch(**overrides) -> ExecutionDispatchFrameV1:
    values = {
        "frame_id": "dispatch-empty-1",
        "generated_at": datetime.now(timezone.utc) - timedelta(minutes=5),
        "source_policy_frame_id": "policy-1",
        "source_proposal_frame_id": "proposal-1",
    }
    values.update(overrides)
    return ExecutionDispatchFrameV1(**values)


def _resolution() -> OutcomeResolution:
    cell = ControlCell(
        posterior=EffectPosterior(mean=0.01, variance=0.1, n=4),
        moved_n=1,
        move_rate=0.4,
    )
    observation = ControlObservation(
        signal_id="resource_pressure",
        arm="no_action",
        baseline_bin=2,
        observed_delta=0.01,
        moved=True,
    )
    return OutcomeResolution(
        records=[],
        posteriors={},
        control_posteriors={("resource_pressure", "no_action", 2): cell},
        control_observations=[observation],
        skipped={},
    )


class _Store:
    def __init__(self, dispatch):
        self.dispatch = dispatch
        self.reductions = []

    def reconcile_feedback_pending(self):
        return 0

    def load_latest_dispatch_frame_without_feedback(self):
        return self.dispatch

    def load_feedback_frame_for_dispatch(self, _frame_id):
        return None

    def load_cortex_result_evidence(self, _dispatch):
        raise AssertionError("a structurally empty dispatch has no result identity to query")

    def load_action_scoring_window(self, _generated_at, *, settle_sec):
        return object(), object()

    def load_effect_posteriors(self):
        return {}

    def load_control_posteriors(self):
        return {}

    def save_control_reduction(self, dispatch, *, control_cells, observation_count):
        self.reductions.append((dispatch.frame_id, control_cells, observation_count))
        return True


def _worker(store) -> FeedbackRuntimeWorker:
    worker = FeedbackRuntimeWorker.__new__(FeedbackRuntimeWorker)
    worker._settings = SimpleNamespace(
        enable_feedback_runtime=True,
        action_settle_sec=15.0,
        action_settle_max_sec=180.0,
    )
    worker._store = store
    worker._policy = SimpleNamespace(windows=SimpleNamespace(field_after_window_sec=30.0))
    worker._visual_parked = {}
    return worker


def test_a_measurable_empty_dispatch_is_reduced_without_a_feedback_frame(monkeypatch):
    store = _Store(_empty_dispatch())
    worker = _worker(store)
    monkeypatch.setattr("app.worker.resolve_action_outcomes", lambda **_kw: _resolution())
    monkeypatch.setattr(
        "app.worker.build_feedback_frame",
        lambda **_kw: (_ for _ in ()).throw(AssertionError("full frame path was entered")),
    )

    result = worker._tick()

    assert result == ControlReductionResult("dispatch-empty-1", 1, 1)
    assert len(store.reductions) == 1


def test_empty_classification_fails_closed_on_inconsistent_counts():
    assert _is_structurally_empty_dispatch(_empty_dispatch()) is True
    assert _is_structurally_empty_dispatch(_empty_dispatch(blocked_count=1)) is False
    assert _is_structurally_empty_dispatch(_empty_dispatch(dispatch_attempted=True)) is False
    assert _is_structurally_empty_dispatch(_empty_dispatch(warnings=["proposal_unavailable"])) is False

    holdback = SimpleNamespace(
        candidates=[],
        blocked_candidates=[object()],
        dispatched_candidates=[],
        dispatch_attempted=False,
        dispatch_count=0,
        blocked_count=1,
        warnings=[],
    )
    assert _is_structurally_empty_dispatch(holdback) is False


class _FullFrameStore(_Store):
    def __init__(self, dispatch):
        super().__init__(dispatch)
        self.feedback = []

    def load_cortex_result_evidence(self, _dispatch):
        return []

    def load_field_for_tick(self, _field_tick_id):
        return None

    def load_latest_field_after(self, _generated_at, *, window_sec):
        return None

    def load_policy_frame(self, _frame_id):
        return None

    def load_proposal_frame(self, _frame_id):
        return None

    def save_feedback_frame(self, frame, **_kwargs):
        self.feedback.append(frame)


def _feedback_for(dispatch):
    return FeedbackFrameV1(
        frame_id=f"feedback:{dispatch.frame_id}",
        generated_at=datetime.now(timezone.utc),
        source_execution_dispatch_frame_id=dispatch.frame_id,
        outcome_status="unknown",
        outcome_score=0.0,
        confidence_score=0.0,
    )


def test_missing_control_window_falls_back_to_full_feedback(monkeypatch):
    dispatch = _empty_dispatch()
    store = _FullFrameStore(dispatch)
    worker = _worker(store)
    empty_resolution = OutcomeResolution([], {}, {}, [], {})
    monkeypatch.setattr("app.worker.resolve_action_outcomes", lambda **_kw: empty_resolution)
    monkeypatch.setattr("app.worker.build_feedback_frame", lambda **_kw: _feedback_for(dispatch))

    result = worker._tick()

    assert isinstance(result, FeedbackFrameV1)
    assert result.source_execution_dispatch_frame_id == dispatch.frame_id
    assert store.reductions == []
    assert len(store.feedback) == 1


def test_warning_only_unevaluable_dispatch_stays_full_evidence(monkeypatch):
    dispatch = _empty_dispatch(warnings=["source_proposal_unavailable"])
    store = _FullFrameStore(dispatch)
    worker = _worker(store)
    monkeypatch.setattr(
        "app.worker.resolve_action_outcomes", lambda **_kw: OutcomeResolution([], {}, {}, [], {})
    )
    monkeypatch.setattr("app.worker.build_feedback_frame", lambda **_kw: _feedback_for(dispatch))

    result = worker._tick()

    assert isinstance(result, FeedbackFrameV1)
    assert store.reductions == []
    assert len(store.feedback) == 1


def test_poll_batch_is_serial_and_bounded_to_three_items():
    worker = FeedbackRuntimeWorker.__new__(FeedbackRuntimeWorker)
    frame1 = FeedbackFrameV1(
        frame_id="feedback-1",
        generated_at=datetime.now(timezone.utc),
        source_execution_dispatch_frame_id="dispatch-1",
        outcome_status="unknown",
        outcome_score=0.0,
        confidence_score=0.0,
    )
    frame2 = frame1.model_copy(
        update={"frame_id": "feedback-2", "source_execution_dispatch_frame_id": "dispatch-2"}
    )
    pending = [ControlReductionResult("empty", 1, 1), frame1, frame2, frame1]
    published = []
    worker._tick = lambda: pending.pop(0)

    async def publish(frame):
        published.append(frame.frame_id)

    worker._publish_feedback_frame = publish

    assert asyncio.run(worker._process_pending_batch()) == 3
    assert published == ["feedback-1", "feedback-2"]
    assert len(pending) == 1


def test_poll_batch_stops_when_the_fifo_cannot_advance():
    worker = FeedbackRuntimeWorker.__new__(FeedbackRuntimeWorker)
    pending = [ControlReductionResult("empty", 1, 1), None, ControlReductionResult("later", 1, 1)]
    worker._tick = lambda: pending.pop(0)
    worker._publish_feedback_frame = lambda _frame: None

    assert asyncio.run(worker._process_pending_batch()) == 1
    assert len(pending) == 1
