"""The scoring window has to CONTAIN the action, and actions no longer share one duration.

`store.load_action_scoring_window`'s own docstring documents this defect being
found and fixed once, on 2026-08-22, for a population of 1.2-5.4s actions. The
fix was a fixed 15s `action_settle_sec`. A constant cannot follow an action, so
when `express` shipped at ~50s the same defect came straight back: three
consecutive live outcomes on 2026-08-31 with `baseline == observed_after` to
four decimal places and `latency_ms` around 50,000 -- the "after" sample taken
35 seconds before the action finished.

That is not a weak measurement. It is an unbiased estimate of a quantity that is
null by construction, and its variance collapses toward zero with confidence, so
the action gets retired below the allocator's information floor for a reason
that was never measured. These tests pin the two properties that stop it:

  * the window follows the action's MEASURED latency, and
  * a frame whose window has not closed yet is DEFERRED, not consumed --
    because scoring it early skips every candidate as `missing_field_window`
    while still clearing `feedback_pending`, which loses the measurement for good.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.worker import FeedbackRuntimeWorker


class _Settings:
    action_settle_sec = 15.0
    action_settle_max_sec = 180.0
    enable_feedback_runtime = True


def _worker(**overrides) -> FeedbackRuntimeWorker:
    w = FeedbackRuntimeWorker.__new__(FeedbackRuntimeWorker)
    settings = _Settings()
    for key, value in overrides.items():
        setattr(settings, key, value)
    w._settings = settings
    w._policy = SimpleNamespace(windows=SimpleNamespace(field_after_window_sec=30))
    return w


def _results(*latencies_ms: float) -> list[dict[str, object]]:
    return [
        {"dispatch_id": f"d{i}", "latency_ms": ms} for i, ms in enumerate(latencies_ms)
    ]


class TestTheWindowFollowsTheAction:
    def test_a_fifty_second_action_gets_a_window_that_outlasts_it(self) -> None:
        """THE regression test. At the old fixed 15s the sample landed 35s
        before express finished."""
        settle = _worker()._scoring_settle_sec(_results(50_882.0))[0]

        assert settle == pytest.approx(65.882)
        assert settle > 50.882, "the window closes before the action finishes"

    def test_it_is_the_frame_wide_max_not_the_first_or_the_mean(self) -> None:
        """The field delta is frame-wide, so a window containing only the
        fastest action in the frame attributes a shared delta from a sample
        taken while the others were still running."""
        settle = _worker()._scoring_settle_sec(_results(1_200.0, 50_000.0, 3_400.0))[0]

        assert settle == pytest.approx(65.0)

    def test_a_fast_action_is_barely_widened(self) -> None:
        # 1.2-5.4s actions were fine before and must stay tight: widening every
        # window to the slowest action's would pull ~70 unrelated dispatches
        # into each fast action's measurement.
        assert _worker()._scoring_settle_sec(_results(1_200.0))[0] == pytest.approx(16.2)

    def test_absent_latency_falls_back_to_the_constant_not_to_zero(self) -> None:
        """Coercing a missing latency to 0.0 reads as "this action was free" and
        silently rebuilds the too-narrow window this exists to fix."""
        assert _worker()._scoring_settle_sec([{"dispatch_id": "d0"}]) == (15.0, False)
        assert _worker()._scoring_settle_sec([]) == (15.0, False)
        assert _worker()._scoring_settle_sec(None) == (15.0, False)

    def test_a_pathological_latency_cannot_park_the_fifo(self) -> None:
        # The FIFO head is deferred until its window closes; an absurd reading
        # must not hold the whole queue for hours.
        assert _worker()._scoring_settle_sec(_results(9_000_000.0)) == (180.0, True)

    def test_the_clamp_reports_itself_so_the_caller_can_refuse(self) -> None:
        """Review finding: 180s does NOT cover every configured action --
        builder_prune and prune_dangling_images allow rpc_timeout_sec 720, and
        live max success latency is 107.5s. A clamped window is KNOWN to be
        shorter than the action, so scoring it rebuilds the null-by-construction
        estimate this patch exists to stop. Silently returning the ceiling made
        that invisible."""
        settle, clamped = _worker()._scoring_settle_sec(_results(600_000.0))
        assert (settle, clamped) == (180.0, True)
        # And a window that genuinely fits must NOT be flagged.
        assert _worker()._scoring_settle_sec(_results(50_000.0))[1] is False


class _Dispatch:
    def __init__(self, age_sec: float) -> None:
        self.frame_id = "dispatch-frame-1"
        self.generated_at = datetime.now(timezone.utc) - timedelta(seconds=age_sec)
        self.source_policy_frame_id = "p"
        self.source_proposal_frame_id = "q"
        self.source_field_tick_id = "tick-1"


class _Store:
    """Records what _tick did, so 'deferred' can be distinguished from 'scored
    nothing' -- which is the whole point: they look identical from outside."""

    def __init__(self, dispatch, latency_ms: float) -> None:
        self.dispatch = dispatch
        self.latency_ms = latency_ms
        self.cleared: list[str] = []
        self.saved: list[object] = []
        self.window_settles: list[float] = []

    def reconcile_feedback_pending(self):
        return None

    def load_latest_dispatch_frame_without_feedback(self):
        return self.dispatch

    def load_feedback_frame_for_dispatch(self, _frame_id):
        return None

    def load_policy_frame(self, _fid):
        return None

    def load_proposal_frame(self, _fid):
        return None

    def load_cortex_result_evidence(self, _dispatch):
        return [{"dispatch_id": "d0", "latency_ms": self.latency_ms}]

    def load_field_for_tick(self, _tick):
        return None

    def load_latest_field_after(self, _at, window_sec=30):
        return None

    def load_action_scoring_window(self, _at, *, settle_sec):
        # Records rather than raising: _tick wraps the whole scoring block in
        # `except Exception` on purpose (a scoring bug must not stall a pipeline
        # that has run for months), so an exception sentinel here is swallowed
        # and the test sees a pass either way. Found the hard way.
        self.window_settles.append(settle_sec)
        return None, None

    def clear_feedback_pending(self, frame_id):
        self.cleared.append(frame_id)

    def save_feedback_frame(self, frame, **_kw):
        self.saved.append(frame)


def _stub_frame_build(monkeypatch):
    """Frame construction is a separate concern with its own coverage; these
    tests are about which WINDOW gets asked for, and building a real frame here
    would need the whole policy/proposal graph."""
    import app.worker as worker_mod

    class _Frame:
        frame_id = "feedback-frame-1"
        outcome_status = "completed"
        observations: list = []

    monkeypatch.setattr(worker_mod, "build_feedback_frame", lambda **_kw: _Frame())


class TestAFrameIsDeferredNotConsumed:
    def test_a_frame_younger_than_its_window_is_left_pending(self, monkeypatch) -> None:
        """Measured live: frame-scoring lag is p50 94.5s but min 0.1s over
        10,261 frames. The fast tail already loses measurements at settle=15;
        widening the window without deferring turns that tail into the norm."""
        store = _Store(_Dispatch(age_sec=5.0), latency_ms=50_000.0)  # settle 65s
        w = _worker()
        w._store = store
        # Stubbed even though the guard should stop us first: without it,
        # killing the guard fails this test with an incidental AttributeError
        # from the real frame builder rather than on any of its four
        # assertions -- so it could not actually tell "deferred" from "scored
        # nothing", which is its entire stated purpose (review finding).
        _stub_frame_build(monkeypatch)

        assert w._tick() is None
        assert store.saved == [], "a deferred frame must not be scored empty"
        assert store.cleared == [], (
            "clearing feedback_pending here loses the measurement permanently -- "
            "the dispatch is never rescored"
        )
        assert store.window_settles == [], "it must not even look for the window yet"

    def test_it_proceeds_once_the_window_has_closed(self, monkeypatch) -> None:
        store = _Store(_Dispatch(age_sec=90.0), latency_ms=50_000.0)  # settle 65s
        w = _worker()
        w._store = store
        _stub_frame_build(monkeypatch)

        w._tick()

        # Asking for the window at all is the assertion: it proves _tick got
        # past the defer, which a bare `is not None` could not distinguish from
        # returning early for an unrelated reason.
        assert store.window_settles, "the frame was still deferred at 90s with a 65s settle"
        assert store.saved, "a frame past its window must be consumed, not held forever"
        # And the priors must NOT have been folded from a half-window.
        assert store.window_settles == [pytest.approx(65.0)]

    def test_the_window_it_asks_for_is_the_computed_one_not_the_constant(self, monkeypatch) -> None:
        """End to end: the measured settle has to actually reach the query. A
        correct _scoring_settle_sec whose value is dropped at the call site
        would leave the live bug exactly where it was."""
        store = _Store(_Dispatch(age_sec=200.0), latency_ms=50_000.0)
        w = _worker()
        w._store = store
        _stub_frame_build(monkeypatch)

        w._tick()

        assert store.window_settles == [pytest.approx(65.0)]
        assert store.window_settles != [15.0], (
            "the measured settle was computed and then dropped at the call site -- "
            "the live bug would be exactly where it was"
        )


class TestTheDeferCannotBecomeAStuckHead:
    def test_a_future_dated_frame_is_not_deferred_forever(self, monkeypatch) -> None:
        """The defer clears itself only because wall-clock age GROWS. A negative
        age never reaches the settle, so a backwards clock step, a restore, or a
        naive datetime (legal -- the schema field is a bare `datetime`) would
        park the FIFO head permanently at one INFO line per 2s poll. That is the
        2026-07-22 stuck-head incident with a different cause."""
        store = _Store(_Dispatch(age_sec=-3600.0), latency_ms=50_000.0)
        w = _worker()
        w._store = store
        _stub_frame_build(monkeypatch)

        w._tick()

        assert store.saved, (
            "a future-dated frame was deferred instead of retired -- the head "
            "never advances and every frame behind it starves"
        )


class TestAClampedWindowIsNotScored:
    def test_an_action_outlasting_the_ceiling_is_refused_not_mis_scored(
        self, monkeypatch, caplog
    ) -> None:
        # 600s action -> wants 615s, clamped to 180s. The window cannot contain
        # it, so folding a posterior from it would be a confident wrong belief.
        store = _Store(_Dispatch(age_sec=400.0), latency_ms=600_000.0)
        w = _worker()
        w._store = store
        _stub_frame_build(monkeypatch)

        with caplog.at_level("WARNING"):
            w._tick()

        assert store.window_settles == [], (
            "a window known to be too short must not be queried and folded"
        )
        assert any(
            "feedback_scoring_window_clamped" in r.getMessage() for r in caplog.records
        ), "a refusal that is not logged is indistinguishable from a silent skip"
        assert store.saved, "the frame must still be consumed; only SCORING is refused"
