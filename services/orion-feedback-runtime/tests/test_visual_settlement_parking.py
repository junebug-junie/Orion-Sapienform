"""Durable render_scene results settle after the dispatch frame is written.

execution-dispatch stores a durable render as `settlement.state="pending"` with
visual_outcome "unknown", and settles the SAME row once the reverie.visual run
ends. Scoring the frame before that folds "unknown" for an image that may still
arrive, so the frame is parked -- without holding the ~95%-utilized FIFO head --
until nothing in it is pending or FEEDBACK_VISUAL_SETTLE_MAX_SEC has passed.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import app.worker as worker_mod
from app.store import FeedbackRuntimeStore
from app.worker import FeedbackRuntimeWorker


# --- store: which row is scored, and what it carries --------------------------------------


def _evidence(monkeypatch, rows):
    store = FeedbackRuntimeStore("postgresql://test:test@localhost/test")
    engine = MagicMock()
    conn = engine.connect.return_value.__enter__.return_value
    conn.execute.return_value.mappings.return_value.all.return_value = rows
    monkeypatch.setattr(store, "_engine", engine)
    dispatch = SimpleNamespace(
        candidates=[SimpleNamespace(dispatch_id="dispatch:1")],
        blocked_candidates=[],
        dispatched_candidates=[],
    )
    return store.load_cortex_result_evidence(dispatch)


def _row(result_id, state, *, outcome="unknown", latency=None, status="success"):
    payload = {"visual_outcome": outcome, "evidence_refs": [result_id]}
    if state is not None:
        payload["settlement"] = {"state": state, "durable_run_id": "run-1"}
    return {
        "result_id": result_id,
        "dispatch_id": "dispatch:1",
        "status": status,
        "result_json": payload,
        "latency_ms": latency,
    }


class TestTheStoreScoresTheLatestSettledRow:
    def test_a_pending_render_carries_its_settlement_state(self, monkeypatch) -> None:
        evidence = _evidence(monkeypatch, [_row("r:new", "pending", status="pending")])
        assert evidence == [
            {
                "result_id": "r:new",
                "dispatch_id": "dispatch:1",
                "status": "pending",
                "evidence_refs": ["r:new"],
                "visual_outcome": "unknown",
                "settlement_state": "pending",
            }
        ]

    def test_an_older_settled_row_beats_a_newer_pending_one(self, monkeypatch) -> None:
        evidence = _evidence(
            monkeypatch,
            [
                _row("r:new", "pending", status="pending"),
                _row("r:old", "settled", outcome="produced", latency=61_000.0),
            ],
        )
        assert [e["result_id"] for e in evidence] == ["r:old"]
        assert evidence[0]["settlement_state"] == "settled"
        assert evidence[0]["visual_outcome"] == "produced"
        assert evidence[0]["latency_ms"] == 61_000.0

    def test_an_older_settled_row_beats_a_newer_unconfirmed_kickoff(self, monkeypatch) -> None:
        evidence = _evidence(
            monkeypatch,
            [_row("r:new", "not_submitted", status="empty"),
             _row("r:old", "settled", outcome="produced", latency=61_000.0)],
        )
        assert [e["result_id"] for e in evidence] == ["r:old"]

    def test_the_newest_settled_row_wins(self, monkeypatch) -> None:
        evidence = _evidence(
            monkeypatch,
            [_row("r:new", "settled", outcome="produced"), _row("r:old", "pending")],
        )
        assert [e["result_id"] for e in evidence] == ["r:new"]

    def test_a_direct_path_result_has_no_settlement_state(self, monkeypatch) -> None:
        evidence = _evidence(monkeypatch, [_row("r:direct", None, outcome="produced", latency=9.0)])
        assert "settlement_state" not in evidence[0]


class _Result:
    def __init__(self, row=None):
        self._row = row

    def mappings(self):
        return self

    def first(self):
        return self._row


class TestTheExcludingLookup:
    def _store(self, calls):
        store = FeedbackRuntimeStore("postgresql://test:test@localhost/test")
        engine = MagicMock()
        conn = engine.connect.return_value.__enter__.return_value

        def execute(stmt, params=None):
            calls.append((" ".join(str(stmt).split()), params))
            return _Result(None)

        conn.execute.side_effect = execute
        store._engine = engine
        return store

    def test_no_exclusions_keeps_the_plain_marker_scan(self) -> None:
        calls: list = []
        assert self._store(calls).load_latest_dispatch_frame_without_feedback() is None
        (sql, params), = calls
        assert "NOT IN" not in sql and params is None

    def test_parked_frames_are_excluded_and_order_is_unchanged(self) -> None:
        calls: list = []
        store = self._store(calls)
        assert store.load_latest_dispatch_frame_without_feedback(exclude_frame_ids=["f1", "f2"]) is None
        (sql, params), = calls
        assert "d.feedback_pending" in sql and "d.frame_id NOT IN" in sql
        assert "ORDER BY d.generated_at ASC" in sql and "LIMIT 1" in sql
        assert params == {"excluded": ["f1", "f2"]}


# --- worker: park, move on, come back ------------------------------------------------------


class _Settings:
    action_settle_sec = 15.0
    action_settle_max_sec = 180.0
    enable_feedback_runtime = True
    feedback_visual_settle_max_sec = 900.0


class _Dispatch:
    def __init__(self, frame_id: str, age_sec: float) -> None:
        self.frame_id = frame_id
        self.generated_at = datetime.now(timezone.utc) - timedelta(seconds=age_sec)
        self.source_policy_frame_id = "p"
        self.source_proposal_frame_id = "q"
        self.source_field_tick_id = "tick-1"


class _Store:
    """Oldest-first FIFO over `frames`, honoring exclusions like the real SQL."""

    def __init__(self, frames, evidence):
        self.frames = list(frames)
        self.evidence = evidence
        self.lookups: list[list[str] | None] = []
        self.saved: list[str] = []
        self.cleared: list[str] = []

    def reconcile_feedback_pending(self):
        return None

    def load_latest_dispatch_frame_without_feedback(self, *, exclude_frame_ids=None):
        self.lookups.append(list(exclude_frame_ids) if exclude_frame_ids else None)
        for frame in self.frames:
            if frame.frame_id in self.saved or frame.frame_id in (exclude_frame_ids or []):
                continue
            return frame
        return None

    def load_feedback_frame_for_dispatch(self, _frame_id):
        return None

    def load_policy_frame(self, _fid):
        return None

    def load_proposal_frame(self, _fid):
        return None

    def load_cortex_result_evidence(self, dispatch):
        return self.evidence[dispatch.frame_id]

    def load_field_for_tick(self, _tick):
        return None

    def load_latest_field_after(self, _at, window_sec=30):
        return None

    def load_action_scoring_window(self, _at, *, settle_sec):
        return None, None

    def load_effect_posteriors(self):
        return {}

    def load_control_posteriors(self):
        return {}

    def clear_feedback_pending(self, frame_id):
        self.cleared.append(frame_id)

    def save_feedback_frame(self, frame, *, control_frame_id=None, **_kw):
        self.saved.append(control_frame_id)


def _worker(store, **overrides) -> FeedbackRuntimeWorker:
    w = FeedbackRuntimeWorker.__new__(FeedbackRuntimeWorker)
    settings = _Settings()
    for key, value in overrides.items():
        setattr(settings, key, value)
    w._settings = settings
    w._policy = SimpleNamespace(windows=SimpleNamespace(field_after_window_sec=30))
    w._store = store
    return w


@pytest.fixture
def built(monkeypatch):
    """Records the cortex evidence each built frame was scored with."""
    seen: list[list[dict]] = []

    class _Frame:
        frame_id = "feedback-frame"
        outcome_status = "unknown"
        observations: list = []

    def _build(**kw):
        seen.append(kw["cortex_results"])
        return _Frame()

    monkeypatch.setattr(worker_mod, "build_feedback_frame", _build)
    return seen


def _pending(dispatch_id="d-render"):
    return {"dispatch_id": dispatch_id, "status": "pending", "visual_outcome": "unknown",
            "settlement_state": "pending"}


def _settled(dispatch_id="d-render"):
    return {"dispatch_id": dispatch_id, "status": "success", "visual_outcome": "produced",
            "settlement_state": "settled", "latency_ms": 1_000.0}


class TestAPendingRenderParksTheFrame:
    def test_it_is_not_scored_and_its_marker_is_not_cleared(self, built) -> None:
        store = _Store([_Dispatch("f-render", age_sec=60.0)], {"f-render": [_pending()]})
        w = _worker(store)

        assert w._tick() is None
        assert store.saved == [] and store.cleared == [] and built == []
        assert "f-render" in w._visual_parked_frames()

    def test_the_fifo_moves_on_to_the_next_frame(self, built) -> None:
        store = _Store(
            [_Dispatch("f-render", age_sec=60.0), _Dispatch("f-next", age_sec=50.0)],
            {"f-render": [_pending()], "f-next": [{"dispatch_id": "d-other", "status": "success"}]},
        )
        w = _worker(store)

        w._tick()
        w._tick()

        assert store.lookups == [None, ["f-render"]]
        assert store.saved == ["f-next"], "a parked render must not hold the head"

    def test_it_comes_back_after_its_recheck_and_scores_the_settled_row(
        self, built, monkeypatch
    ) -> None:
        clock = [1_000.0]
        monkeypatch.setattr(worker_mod.time, "monotonic", lambda: clock[0])
        store = _Store([_Dispatch("f-render", age_sec=60.0)], {"f-render": [_pending()]})
        w = _worker(store)

        w._tick()
        clock[0] += 10.0
        w._tick()  # still inside the recheck interval: excluded, nothing else to do
        assert store.lookups[-1] == ["f-render"] and store.saved == []

        store.evidence["f-render"] = [_settled()]
        clock[0] += worker_mod._VISUAL_PARK_RECHECK_SEC
        w._tick()

        assert store.saved == ["f-render"]
        assert built[-1][0]["visual_outcome"] == "produced"
        assert "f-render" not in w._visual_parked_frames()


class TestTheParkIsBounded:
    def test_past_the_bound_it_scores_the_visual_as_is(self, built, caplog) -> None:
        store = _Store([_Dispatch("f-render", age_sec=901.0)], {"f-render": [_pending()]})
        w = _worker(store)

        with caplog.at_level("WARNING"):
            w._tick()

        assert store.saved == ["f-render"]
        assert built[-1][0]["visual_outcome"] == "unknown"
        assert any("feedback_visual_settlement_bound_expired" in r.getMessage() for r in caplog.records)

    def test_zero_disables_the_park(self, built) -> None:
        store = _Store([_Dispatch("f-render", age_sec=30.0)], {"f-render": [_pending()]})
        _worker(store, feedback_visual_settle_max_sec=0.0)._tick()
        assert store.saved == ["f-render"]

    def test_a_future_dated_frame_is_not_parked_forever(self, built) -> None:
        store = _Store([_Dispatch("f-render", age_sec=-3600.0)], {"f-render": [_pending()]})
        _worker(store)._tick()
        assert store.saved == ["f-render"]

    def test_the_recheck_never_overshoots_the_bound(self, built, monkeypatch) -> None:
        monkeypatch.setattr(worker_mod.time, "monotonic", lambda: 1_000.0)
        store = _Store([_Dispatch("f-render", age_sec=890.0)], {"f-render": [_pending()]})
        w = _worker(store)
        w._tick()
        assert w._visual_parked_frames()["f-render"] == pytest.approx(1_010.0, abs=0.5)

    def test_a_full_park_defers_in_place_without_growing(self, built, monkeypatch) -> None:
        monkeypatch.setattr(worker_mod, "_VISUAL_PARK_MAX_FRAMES", 1)
        monkeypatch.setattr(worker_mod.time, "monotonic", lambda: 1_000.0)
        store = _Store(
            [_Dispatch("f-a", age_sec=60.0), _Dispatch("f-b", age_sec=50.0)],
            {"f-a": [_pending("d-a")], "f-b": [_pending("d-b")]},
        )
        w = _worker(store)
        w._tick()
        w._tick()
        assert list(w._visual_parked_frames()) == ["f-a"]
        assert store.saved == []

    def test_an_unconfirmed_kickoff_parks_too_so_a_late_image_still_counts(self, built) -> None:
        unconfirmed = {"dispatch_id": "d-render", "status": "empty", "visual_outcome": "unknown",
                       "settlement_state": "not_submitted"}
        store = _Store([_Dispatch("f-render", age_sec=60.0)], {"f-render": [unconfirmed]})
        w = _worker(store)
        assert w._tick() is None
        assert store.saved == [] and "f-render" in w._visual_parked_frames()

    def test_unsettled_states_match_dispatch(self) -> None:
        from app.store import UNSETTLED_RENDER_STATES
        from orion.execution_dispatch.visual_settlement import SETTLEABLE_STATES

        assert UNSETTLED_RENDER_STATES == frozenset(SETTLEABLE_STATES)

    def test_a_direct_path_render_is_not_parked(self, built) -> None:
        store = _Store(
            [_Dispatch("f-render", age_sec=60.0)],
            {"f-render": [{"dispatch_id": "d-render", "status": "success", "visual_outcome": "produced"}]},
        )
        _worker(store)._tick()
        assert store.saved == ["f-render"]



# --- the scoring window must contain the image, not just its GPU seconds -------------------


class TestASettledRenderWindowReachesTheRunsEnd:
    def _settled_at(self, dispatched_at, finished_after_sec, gpu_ms=60_000.0):
        return [{
            "dispatch_id": "d-render", "status": "success", "visual_outcome": "produced",
            "settlement_state": "settled", "latency_ms": gpu_ms,
            "settled_finished_at": (dispatched_at + timedelta(seconds=finished_after_sec)).isoformat(),
        }]

    def test_a_render_that_queued_first_clamps_instead_of_scoring_early(self) -> None:
        """8 min in the diffusion queue + 60s of GPU: sized from GPU seconds the
        'after' sample lands at +75s, minutes before the image existed."""
        at = datetime.now(timezone.utc)
        w = _worker(_Store([], {}))
        assert w._scoring_settle_sec(self._settled_at(at, 540.0), dispatched_at=at) == (180.0, True)

    def test_a_prompt_render_is_windowed_from_its_real_end(self) -> None:
        at = datetime.now(timezone.utc)
        w = _worker(_Store([], {}))
        settle, clamped = w._scoring_settle_sec(self._settled_at(at, 100.0, gpu_ms=40_000.0), dispatched_at=at)
        assert (settle, clamped) == (pytest.approx(115.0), False)

    def test_gpu_seconds_still_win_when_they_are_longer(self) -> None:
        at = datetime.now(timezone.utc)
        w = _worker(_Store([], {}))
        settle, _ = w._scoring_settle_sec(self._settled_at(at, 10.0, gpu_ms=50_000.0), dispatched_at=at)
        assert settle == pytest.approx(65.0)

    def test_the_store_carries_the_runs_end(self, monkeypatch) -> None:
        row = _row("r:new", "settled", outcome="produced", latency=60_000.0)
        row["result_json"]["settlement"]["finished_at"] = "2026-09-28T12:09:00+00:00"
        evidence = _evidence(monkeypatch, [row])
        assert evidence[0]["settled_finished_at"] == "2026-09-28T12:09:00+00:00"

    def test_tick_asks_for_the_widened_window(self, built) -> None:
        dispatch = _Dispatch("f-render", age_sec=400.0)
        store = _Store([dispatch], {"f-render": self._settled_at(dispatch.generated_at, 100.0, gpu_ms=40_000.0)})
        windows: list[float] = []
        store.load_action_scoring_window = lambda _at, *, settle_sec: windows.append(settle_sec) or (None, None)
        _worker(store)._tick()
        assert windows == [pytest.approx(115.0)]
