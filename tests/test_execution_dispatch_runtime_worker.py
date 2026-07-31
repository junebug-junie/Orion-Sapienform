from __future__ import annotations

import asyncio
import math
import sys
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

REPO = Path(__file__).resolve().parents[1]
SVC = REPO / "services" / "orion-execution-dispatch-runtime"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SVC))

import app.worker as worker_mod  # noqa: E402
from app.worker import ExecutionDispatchRuntimeWorker  # noqa: E402
from orion.schemas.execution_dispatch_frame import (  # noqa: E402
    ExecutionDispatchCandidateV1,
    ExecutionDispatchFrameV1,
)
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1, PolicyDecisionV1  # noqa: E402
from orion.schemas.proposal_frame import ProposalFrameV1  # noqa: E402

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _proposal() -> ProposalFrameV1:
    return ProposalFrameV1(
        frame_id="proposal.frame:test:proposal_policy.v1",
        generated_at=NOW,
        source_field_tick_id="tick:test",
        source_field_generated_at=NOW,
        source_attention_frame_id="frame:test",
        overall_action_pressure=0.4,
        overall_risk=0.1,
        candidates=[],
    )


def _policy_frame() -> PolicyDecisionFrameV1:
    return PolicyDecisionFrameV1(
        frame_id="policy.frame:proposal.frame:test:substrate_policy.v1",
        generated_at=NOW,
        source_proposal_frame_id="proposal.frame:test:proposal_policy.v1",
        source_field_tick_id="tick:test",
        overall_risk=0.0,
    )


def test_worker_skips_when_no_policy_pending(monkeypatch) -> None:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    # 2026-07-30: staleness-discard override disabled (very large) -- this
    # test is about "no pending policy frame at all," not about staleness
    # behavior, same reasoning as the two tests below.
    monkeypatch.setenv("EXECUTION_DISPATCH_STALENESS_OVERRIDE_SEC", "99999999999")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = ExecutionDispatchRuntimeWorker()
    worker._store.load_oldest_policy_frames_without_dispatch = MagicMock(return_value=[])
    worker._store.load_freshest_policy_frame_without_dispatch = MagicMock(return_value=None)
    worker._store.load_latest_staleness_discard_baseline = MagicMock(return_value=None)
    worker._store.save_dispatch_frame = MagicMock()

    worker._tick()

    worker._store.save_dispatch_frame.assert_not_called()


def test_worker_records_unevaluable_frame_when_proposal_missing(monkeypatch) -> None:
    # 2026-07-12: a naive skip-and-return would retry the same oldest
    # undispatched policy frame forever, blocking every policy frame queued
    # behind it. The worker must record an honest "could not evaluate" frame
    # so the FIFO queue advances.
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    # 2026-07-30: _policy_frame()'s generated_at is a fixed historical
    # constant (NOW, 2026-05-24) -- real wall-clock age against that would
    # always exceed even the widest real staleness window and get fast-
    # discarded before this test's actual target behavior (proposal-missing
    # handling) ever runs. This test is about that handling, not staleness,
    # so disable the discard path here (dedicated staleness tests below
    # exercise it directly with controlled ages).
    monkeypatch.setenv("EXECUTION_DISPATCH_STALENESS_OVERRIDE_SEC", "99999999999")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = ExecutionDispatchRuntimeWorker()
    policy_frame = _policy_frame()
    worker._store.load_oldest_policy_frames_without_dispatch = MagicMock(return_value=[policy_frame])
    worker._store.load_latest_staleness_discard_baseline = MagicMock(return_value=None)
    worker._store.load_proposal_frame = MagicMock(return_value=None)
    worker._store.save_dispatch_frame = MagicMock()

    worker._tick()

    worker._store.save_dispatch_frame.assert_called_once()
    saved_frame = worker._store.save_dispatch_frame.call_args[0][0]
    assert saved_frame.source_policy_frame_id == policy_frame.frame_id
    assert saved_frame.dispatch_attempted is False
    assert saved_frame.candidates == []
    assert any("proposal_frame" in w for w in saved_frame.warnings)


def test_worker_saves_dispatch_frame_for_pending_policy(monkeypatch) -> None:
    # 2026-07-22 (SelfStateV1 burn): build_execution_dispatch_frame now takes
    # field_tick_id straight off policy_frame -- no separate self-state load
    # that could fail and stall the FIFO queue.
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    # 2026-07-30: see test_worker_records_unevaluable_frame_when_proposal_
    # missing's comment above -- same fixed-historical-generated_at issue.
    monkeypatch.setenv("EXECUTION_DISPATCH_STALENESS_OVERRIDE_SEC", "99999999999")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = ExecutionDispatchRuntimeWorker()
    policy_frame = _policy_frame()
    worker._store.load_oldest_policy_frames_without_dispatch = MagicMock(return_value=[policy_frame])
    worker._store.load_latest_staleness_discard_baseline = MagicMock(return_value=None)
    worker._store.load_proposal_frame = MagicMock(return_value=_proposal())
    worker._store.save_dispatch_frame = MagicMock()

    worker._tick()

    worker._store.save_dispatch_frame.assert_called_once()
    saved = worker._store.save_dispatch_frame.call_args[0][0]
    assert isinstance(saved, ExecutionDispatchFrameV1)
    assert saved.source_policy_frame_id == policy_frame.frame_id
    assert saved.source_field_tick_id == policy_frame.source_field_tick_id


def _policy_frame_at(generated_at: datetime, *, frame_id: str = "policy.frame:staleness-test") -> PolicyDecisionFrameV1:
    return PolicyDecisionFrameV1(
        frame_id=frame_id,
        generated_at=generated_at,
        source_proposal_frame_id=f"proposal.frame:{frame_id}",
        source_field_tick_id="tick:staleness-test",
        overall_risk=0.0,
    )


def _staleness_worker(monkeypatch, *, override_sec: float | None = None) -> ExecutionDispatchRuntimeWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    if override_sec is not None:
        monkeypatch.setenv("EXECUTION_DISPATCH_STALENESS_OVERRIDE_SEC", str(override_sec))
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = ExecutionDispatchRuntimeWorker()
    worker._store.load_latest_staleness_discard_baseline = MagicMock(return_value=None)
    worker._store.save_dispatch_frame = MagicMock()
    worker._store.load_proposal_frame = MagicMock(return_value=_proposal())
    # Default: nothing fresher available either (2026-07-30 fresh-priority
    # fallback) -- tests that specifically exercise the fallback override
    # this explicitly.
    worker._store.load_freshest_policy_frame_without_dispatch = MagicMock(return_value=None)
    return worker


def test_staleness_threshold_respects_override(monkeypatch) -> None:
    worker = _staleness_worker(monkeypatch, override_sec=42.0)
    for _ in range(5):
        assert worker._staleness_threshold_sec() == 42.0


def test_staleness_threshold_falls_in_configured_range_without_override(monkeypatch) -> None:
    worker = _staleness_worker(monkeypatch)
    lo = worker._settings.execution_dispatch_staleness_min_sec
    hi = worker._settings.execution_dispatch_staleness_max_sec
    seen = {worker._staleness_threshold_sec() for _ in range(50)}
    assert all(lo <= v <= hi for v in seen)
    # 50 draws from a continuous uniform range landing on fewer than 2
    # distinct values would indicate this isn't actually randomized.
    assert len(seen) > 1


def test_worker_discards_a_lone_stale_policy_frame_without_dispatching(monkeypatch) -> None:
    worker = _staleness_worker(monkeypatch, override_sec=120.0)
    stale_frame = _policy_frame_at(datetime.now(timezone.utc) - timedelta(hours=6))
    worker._store.load_oldest_policy_frames_without_dispatch = MagicMock(
        return_value=[stale_frame]
    )

    worker._tick()

    # Saved twice: once during the drain (carrying the pre-tick baseline
    # forward, unknown final count yet) and once more to re-stamp this
    # tick's now-final EWMA update onto that same frame_id (idempotent
    # upsert, not a duplicate row -- see _tick()'s own comment). The content
    # that matters is the LAST call, same as what the next tick's
    # load_latest_staleness_discard_baseline() would actually read back.
    assert worker._store.save_dispatch_frame.call_count == 2
    saved = worker._store.save_dispatch_frame.call_args[0][0]
    assert saved.source_policy_frame_id == stale_frame.frame_id
    assert saved.dispatch_attempted is False
    assert any("stale_backlog_discarded" in w for w in saved.warnings)
    worker._store.load_proposal_frame.assert_not_called()


def test_worker_materializes_per_candidate_discard_summary(monkeypatch) -> None:
    """The discard is not a silent drop -- each candidate's template and
    decision are preserved in the saved frame's warnings, real forensic
    content, not just a bare 'stale' flag."""
    worker = _staleness_worker(monkeypatch, override_sec=120.0)
    stale_frame = _policy_frame_at(datetime.now(timezone.utc) - timedelta(hours=6)).model_copy(
        update={
            "decisions": [
                PolicyDecisionV1(
                    decision_id="policy.decision:1",
                    proposal_id="proposal:inspect_bus_channel_catalog:tick_x:attention.frame:tick_x:field_attention_policy.v1",
                    decision="approved_read_only",
                    policy_gate="read_only",
                    risk_score=0.1,
                    reversibility_score=1.0,
                    confidence_score=0.8,
                )
            ]
        }
    )
    worker._store.load_oldest_policy_frames_without_dispatch = MagicMock(
        return_value=[stale_frame]
    )

    worker._tick()

    saved = worker._store.save_dispatch_frame.call_args[0][0]
    assert "stale_discard:inspect_bus_channel_catalog:approved_read_only" in saved.warnings


def test_worker_drains_stale_frames_then_processes_a_fresh_one(monkeypatch) -> None:
    worker = _staleness_worker(monkeypatch, override_sec=120.0)
    stale_1 = _policy_frame_at(datetime.now(timezone.utc) - timedelta(hours=6), frame_id="policy.frame:stale-1")
    stale_2 = _policy_frame_at(datetime.now(timezone.utc) - timedelta(hours=3), frame_id="policy.frame:stale-2")
    fresh = _policy_frame_at(datetime.now(timezone.utc) - timedelta(seconds=1), frame_id="policy.frame:fresh")
    worker._store.load_oldest_policy_frames_without_dispatch = MagicMock(
        return_value=[stale_1, stale_2, fresh]
    )

    worker._tick()

    assert worker._store.save_dispatch_frame.call_count == 3
    worker._store.load_proposal_frame.assert_called_once_with(fresh.source_proposal_frame_id)
    final_saved = worker._store.save_dispatch_frame.call_args_list[-1][0][0]
    assert final_saved.source_policy_frame_id == fresh.frame_id
    assert final_saved.dispatch_attempted is False or final_saved.candidates == []
    # The FIFO drain found a fresh frame on its own -- the newest-first
    # fallback must never be consulted in that case.
    worker._store.load_freshest_policy_frame_without_dispatch.assert_not_called()


def test_worker_falls_back_to_freshest_when_drain_finds_nothing(monkeypatch) -> None:
    """Regression guard for the live incident (2026-07-30, docs/superpowers/
    specs/2026-07-30-execution-dispatch-staleness-discard-design.md): a deep
    backlog made the FIFO drain spend its entire per-tick cap on ancient
    frames without ever reaching one recent enough to process -- zero real
    dispatches for 6+ minutes after the staleness-discard patch shipped.
    When the drain hits its cap without finding anything fresh, _tick() must
    still check for -- and process -- a genuinely current proposal directly,
    regardless of how deep the old backlog behind it is."""
    worker = _staleness_worker(monkeypatch, override_sec=120.0)
    ancient = _policy_frame_at(datetime.now(timezone.utc) - timedelta(days=2))
    worker._store.load_oldest_policy_frames_without_dispatch = MagicMock(return_value=[ancient])
    fresh = _policy_frame_at(datetime.now(timezone.utc) - timedelta(seconds=1), frame_id="policy.frame:fresh")
    worker._store.load_freshest_policy_frame_without_dispatch = MagicMock(return_value=fresh)

    worker._tick()

    worker._store.load_proposal_frame.assert_called_once_with(fresh.source_proposal_frame_id)
    final_saved = worker._store.save_dispatch_frame.call_args_list[-1][0][0]
    assert final_saved.source_policy_frame_id == fresh.frame_id


def test_worker_fallback_correctly_finds_nothing_when_freshest_is_also_stale(monkeypatch) -> None:
    """If even the single newest unprocessed policy frame is already past
    the staleness window, production itself has stalled -- there is
    genuinely nothing current to dispatch this tick, not a backlog-depth
    artifact. Must not fabricate a real-dispatch attempt in that case."""
    worker = _staleness_worker(monkeypatch, override_sec=120.0)
    ancient = _policy_frame_at(datetime.now(timezone.utc) - timedelta(days=2))
    worker._store.load_oldest_policy_frames_without_dispatch = MagicMock(return_value=[ancient])
    also_stale_but_newest = _policy_frame_at(
        datetime.now(timezone.utc) - timedelta(hours=1), frame_id="policy.frame:newest-but-stale"
    )
    worker._store.load_freshest_policy_frame_without_dispatch = MagicMock(
        return_value=also_stale_but_newest
    )

    worker._tick()

    worker._store.load_proposal_frame.assert_not_called()


def test_max_stale_discards_per_tick_caps_the_drain(monkeypatch) -> None:
    """2026-07-30 perf fix: the cap is now enforced by the LIMIT passed to
    the single batch query (load_oldest_policy_frames_without_dispatch), not
    by an application-level while loop counting single-row fetches -- there
    is no longer any worker-side backstop against the store ever returning
    more than the requested limit (that trust boundary moved entirely to
    Postgres honoring the bound :limit parameter). This test verifies what's
    actually left on the worker side: it requests exactly MAX_STALE_
    DISCARDS_PER_TICK, and correctly processes a full-size all-stale batch
    in full (every item discarded, none skipped) rather than stopping short.
    It does NOT and cannot prove the cap holds if the store ever returned an
    oversized batch -- that would need a store/SQL-level test, not this
    one."""
    worker = _staleness_worker(monkeypatch, override_sec=120.0)
    ancient_batch = [
        _policy_frame_at(
            datetime.now(timezone.utc) - timedelta(days=2), frame_id=f"policy.frame:ancient-{i}"
        )
        for i in range(worker_mod.MAX_STALE_DISCARDS_PER_TICK)
    ]
    worker._store.load_oldest_policy_frames_without_dispatch = MagicMock(return_value=ancient_batch)

    worker._tick()

    worker._store.load_oldest_policy_frames_without_dispatch.assert_called_once_with(
        worker_mod.MAX_STALE_DISCARDS_PER_TICK
    )
    # +1: the final re-stamp save of the last discard frame with this tick's
    # now-final EWMA count, same reasoning as test_worker_discards_a_lone_
    # stale_policy_frame_without_dispatching above.
    assert worker._store.save_dispatch_frame.call_count == worker_mod.MAX_STALE_DISCARDS_PER_TICK + 1
    worker._store.load_proposal_frame.assert_not_called()


def test_staleness_discard_ewma_advances_and_persists(monkeypatch) -> None:
    worker = _staleness_worker(monkeypatch, override_sec=120.0)
    worker._store.load_latest_staleness_discard_baseline = MagicMock(
        return_value={
            "staleness_discard_count_ewma": 3.0,
            "staleness_discard_count_ewma_var": 1.0,
            "staleness_discard_count_ewma_n": 5,
        }
    )
    stale_frame = _policy_frame_at(datetime.now(timezone.utc) - timedelta(hours=6))
    worker._store.load_oldest_policy_frames_without_dispatch = MagicMock(
        return_value=[stale_frame]
    )

    worker._tick()

    saved = worker._store.save_dispatch_frame.call_args[0][0]
    assert saved.staleness_discard_count_ewma_n == 6
    # value=1.0 (one real discard this tick) pulls the ewma up from its
    # prior 3.0 baseline toward 1.0 -- direction, not exact magnitude
    # (compute_ewma_update's own math is tested elsewhere), is what this
    # regression guards.
    assert saved.staleness_discard_count_ewma < 3.0


def test_worker_fresh_policy_frame_never_discarded(monkeypatch) -> None:
    """A policy frame well within the staleness window is processed
    normally, not discarded -- the staleness path must not fire on healthy,
    real-time traffic."""
    worker = _staleness_worker(monkeypatch, override_sec=300.0)
    fresh = _policy_frame_at(datetime.now(timezone.utc))
    worker._store.load_oldest_policy_frames_without_dispatch = MagicMock(return_value=[fresh])

    worker._tick()

    worker._store.save_dispatch_frame.assert_called_once()
    saved = worker._store.save_dispatch_frame.call_args[0][0]
    assert not any("stale_backlog_discarded" in w for w in saved.warnings)
    worker._store.load_proposal_frame.assert_called_once()


def test_worker_handles_naive_generated_at_without_crashing(monkeypatch) -> None:
    """Regression guard (code review, 2026-07-30): PolicyDecisionFrameV1.
    generated_at is a plain datetime, not enforced tz-aware. A naive value
    subtracted from datetime.now(timezone.utc) without normalizing first
    raises TypeError -- caught by _poll_loop's broad except, but that exact
    policy frame would then be the permanent FIFO head forever (never
    resolves, never gets marked processed), the same queue-blocking failure
    shape build_unevaluable_execution_dispatch_frame exists to prevent for a
    missing proposal. _drain_stale_policy_frames must normalize before
    subtracting."""
    worker = _staleness_worker(monkeypatch, override_sec=120.0)
    naive_stale = _policy_frame_at(datetime.now() - timedelta(hours=6))
    assert naive_stale.generated_at.tzinfo is None
    worker._store.load_oldest_policy_frames_without_dispatch = MagicMock(
        return_value=[naive_stale]
    )

    worker._tick()  # must not raise

    saved = worker._store.save_dispatch_frame.call_args[0][0]
    assert any("stale_backlog_discarded" in w for w in saved.warnings)


def _make_worker(monkeypatch) -> ExecutionDispatchRuntimeWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = ExecutionDispatchRuntimeWorker()
    # Default: no baseline history anywhere and no historical closed day to
    # seed from -- _derive_daily_risk_cap degrades to n==0's static fallback
    # (settings.orion_dispatch_max_risk_per_day), preserving every existing
    # risk-budget test's behavior unchanged. Tests that specifically exercise
    # the EWMA baseline override these explicitly.
    worker._store.load_latest_daily_risk_baseline = MagicMock(return_value=None)
    worker._store.most_recent_closed_day_with_data = MagicMock(return_value=None)
    worker._store.sum_uncapped_risk_for_day = MagicMock(return_value=0.0)
    return worker


def _candidate(
    dispatch_id: str, status: str = "prepared_for_dispatch", risk_score: float = 0.05
) -> ExecutionDispatchCandidateV1:
    return ExecutionDispatchCandidateV1(
        dispatch_id=dispatch_id,
        source_decision_id=f"pd:{dispatch_id}",
        source_proposal_id=f"proposal:{dispatch_id}",
        dispatch_status=status,
        dispatch_mode="dispatch_read_only",
        dispatch_kind="inspect",
        target_id="capability:orchestration",
        target_kind="capability",
        cortex_verb="substrate.inspect",
        cortex_mode="brain",
        request_envelope={"context": {"target_id": "capability:orchestration"}},
        risk_score=risk_score,
        confidence_score=0.9,
    )


def _frame_with_candidates(
    *candidates: ExecutionDispatchCandidateV1, generated_at: datetime = NOW
) -> ExecutionDispatchFrameV1:
    return ExecutionDispatchFrameV1(
        frame_id="execution.dispatch.frame:test:execution_dispatch_policy.v1",
        generated_at=generated_at,
        source_policy_frame_id="policy.frame:test",
        source_proposal_frame_id="proposal.frame:test",
        source_field_tick_id="field.tick:test",
        dispatch_mode="dispatch_read_only",
        candidates=list(candidates),
    )


class _FakeClient:
    """Stand-in for ExecutionDispatchCortexClient -- returns canned results
    or raises, keyed by dispatch_id, without touching the real bus."""

    def __init__(self, *_, **__) -> None:
        pass

    async def dispatch(self, *, verb, mode, context, dispatch_id, timeout_sec=None):
        outcome = _FAKE_CLIENT_OUTCOMES[dispatch_id]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


_FAKE_CLIENT_OUTCOMES: dict[str, object] = {}


def _patch_bus_and_client(monkeypatch, outcomes: dict[str, object]) -> MagicMock:
    _FAKE_CLIENT_OUTCOMES.clear()
    _FAKE_CLIENT_OUTCOMES.update(outcomes)
    fake_bus = MagicMock()
    fake_bus.connect = AsyncMock()
    fake_bus.close = AsyncMock()
    fake_bus.publish = AsyncMock()
    monkeypatch.setattr(worker_mod, "OrionBusAsync", MagicMock(return_value=fake_bus))
    monkeypatch.setattr(worker_mod, "ExecutionDispatchCortexClient", _FakeClient)
    return fake_bus


# 2026-07-31 (docs/superpowers/specs/2026-07-30-execution-dispatch-
# staleness-discard-design.md's Part 2): a real, measurable delay per call
# -- concurrency tests need actual wall-clock overlap to prove anything,
# not just absence-of-crash.
_SLOW_CLIENT_DELAY_SEC = 0.15


class _SlowFakeClient:
    """Like _FakeClient, but dispatch() awaits a real delay first."""

    def __init__(self, *_, **__) -> None:
        pass

    async def dispatch(self, *, verb, mode, context, dispatch_id, timeout_sec=None):
        await asyncio.sleep(_SLOW_CLIENT_DELAY_SEC)
        outcome = _FAKE_CLIENT_OUTCOMES[dispatch_id]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _patch_bus_and_slow_client(monkeypatch, outcomes: dict[str, object]) -> MagicMock:
    _FAKE_CLIENT_OUTCOMES.clear()
    _FAKE_CLIENT_OUTCOMES.update(outcomes)
    fake_bus = MagicMock()
    fake_bus.connect = AsyncMock()
    fake_bus.close = AsyncMock()
    fake_bus.publish = AsyncMock()
    monkeypatch.setattr(worker_mod, "OrionBusAsync", MagicMock(return_value=fake_bus))
    monkeypatch.setattr(worker_mod, "ExecutionDispatchCortexClient", _SlowFakeClient)
    return fake_bus


@pytest.mark.asyncio
async def test_send_prepared_candidates_promotes_on_success(monkeypatch) -> None:
    worker = _make_worker(monkeypatch)
    worker._store.sum_risk_dispatched_today = MagicMock(return_value=0.0)
    worker._store.latest_bus_synaptic_prediction_error = MagicMock(return_value=0.1675)
    worker._store.save_dispatch_result = MagicMock()
    worker._store.load_dispatch_result_by_dispatch_id = MagicMock(return_value=None)
    fake_bus = _patch_bus_and_client(
        monkeypatch,
        {"dispatch:1": {"result": {"final_text": '{"observation": "steady", "confidence": 0.8}'}}},
    )
    frame = _frame_with_candidates(_candidate("dispatch:1"))

    updated = await worker._send_prepared_candidates(frame)

    # Regression guard: bus.connect() must be called before any RPC is attempted --
    # a missing call here degrades every real send to "OrionBusAsync not connected"
    # while every candidate still burns the daily dispatch budget as status=failed.
    fake_bus.connect.assert_awaited_once()
    assert updated.candidates == []
    assert len(updated.dispatched_candidates) == 1
    promoted = updated.dispatched_candidates[0]
    assert promoted.dispatch_status == "dispatched"
    assert promoted.dispatched_at is not None
    assert promoted.result_ref == "result:dispatch:1"
    assert promoted.dispatch_error is None
    assert updated.dispatch_count == 1
    assert updated.dispatch_attempted is True
    worker._store.save_dispatch_result.assert_called_once()
    assert worker._store.save_dispatch_result.call_args.kwargs["status"] == "success"
    assert worker._store.save_dispatch_result.call_args.kwargs["raw_len"] == 6


@pytest.mark.asyncio
async def test_send_one_emits_action_outcome_on_success(monkeypatch) -> None:
    worker = _make_worker(monkeypatch)
    worker._store.sum_risk_dispatched_today = MagicMock(return_value=0.0)
    worker._store.latest_bus_synaptic_prediction_error = MagicMock(return_value=0.1675)
    worker._store.save_dispatch_result = MagicMock()
    worker._store.load_dispatch_result_by_dispatch_id = MagicMock(return_value=None)
    fake_bus = _patch_bus_and_client(
        monkeypatch,
        {"dispatch:1": {"result": {"final_text": '{"observation": "steady state observed"}'}}},
    )
    frame = _frame_with_candidates(_candidate("dispatch:1"))

    await worker._send_prepared_candidates(frame)

    fake_bus.publish.assert_awaited_once()
    channel, env = fake_bus.publish.await_args.args
    assert channel == worker._settings.action_outcome_channel
    assert env.kind == "action.outcome.emit.v1"
    assert env.payload["subject"] == "orion"
    assert env.payload["action_id"] == "dispatch:1"
    assert env.payload["kind"] == "inspect"
    assert env.payload["summary"] == "steady state observed"
    assert env.payload["success"] is True
    assert env.payload["surprise"] == 0.1675


@pytest.mark.asyncio
async def test_send_one_action_outcome_surprise_falls_back_to_zero_when_unavailable(
    monkeypatch,
) -> None:
    worker = _make_worker(monkeypatch)
    worker._store.sum_risk_dispatched_today = MagicMock(return_value=0.0)
    worker._store.latest_bus_synaptic_prediction_error = MagicMock(return_value=None)
    worker._store.save_dispatch_result = MagicMock()
    worker._store.load_dispatch_result_by_dispatch_id = MagicMock(return_value=None)
    fake_bus = _patch_bus_and_client(
        monkeypatch,
        {"dispatch:1": {"result": {"final_text": '{"observation": "steady"}'}}},
    )
    frame = _frame_with_candidates(_candidate("dispatch:1"))

    await worker._send_prepared_candidates(frame)

    _, env = fake_bus.publish.await_args.args
    assert env.payload["surprise"] == 0.0


@pytest.mark.asyncio
async def test_send_one_action_outcome_surprise_falls_back_to_zero_on_fetch_error(
    monkeypatch,
) -> None:
    worker = _make_worker(monkeypatch)
    worker._store.sum_risk_dispatched_today = MagicMock(return_value=0.0)
    worker._store.latest_bus_synaptic_prediction_error = MagicMock(
        side_effect=RuntimeError("db unreachable")
    )
    worker._store.save_dispatch_result = MagicMock()
    worker._store.load_dispatch_result_by_dispatch_id = MagicMock(return_value=None)
    fake_bus = _patch_bus_and_client(
        monkeypatch,
        {"dispatch:1": {"result": {"final_text": '{"observation": "steady"}'}}},
    )
    frame = _frame_with_candidates(_candidate("dispatch:1"))

    await worker._send_prepared_candidates(frame)

    _, env = fake_bus.publish.await_args.args
    assert env.payload["surprise"] == 0.0


@pytest.mark.asyncio
async def test_send_one_emits_action_outcome_on_empty_observation(monkeypatch) -> None:
    worker = _make_worker(monkeypatch)
    worker._store.sum_risk_dispatched_today = MagicMock(return_value=0.0)
    worker._store.latest_bus_synaptic_prediction_error = MagicMock(return_value=0.1675)
    worker._store.save_dispatch_result = MagicMock()
    worker._store.load_dispatch_result_by_dispatch_id = MagicMock(return_value=None)
    fake_bus = _patch_bus_and_client(monkeypatch, {"dispatch:1": {"result": {"final_text": ""}}})
    frame = _frame_with_candidates(_candidate("dispatch:1"))

    await worker._send_prepared_candidates(frame)

    fake_bus.publish.assert_awaited_once()
    _, env = fake_bus.publish.await_args.args
    assert env.payload["success"] is False
    assert "no observation" in env.payload["summary"]


@pytest.mark.asyncio
async def test_send_one_emits_action_outcome_on_rpc_failure(monkeypatch) -> None:
    worker = _make_worker(monkeypatch)
    worker._store.sum_risk_dispatched_today = MagicMock(return_value=0.0)
    worker._store.latest_bus_synaptic_prediction_error = MagicMock(return_value=0.1675)
    worker._store.save_dispatch_result = MagicMock()
    worker._store.load_dispatch_result_by_dispatch_id = MagicMock(return_value=None)
    fake_bus = _patch_bus_and_client(monkeypatch, {"dispatch:1": RuntimeError("rpc timed out")})
    frame = _frame_with_candidates(_candidate("dispatch:1"))

    await worker._send_prepared_candidates(frame)

    fake_bus.publish.assert_awaited_once()
    _, env = fake_bus.publish.await_args.args
    assert env.payload["success"] is False
    assert "send failed" in env.payload["summary"]


@pytest.mark.asyncio
async def test_send_one_re_emits_action_outcome_on_idempotent_replay(monkeypatch) -> None:
    # Re-emitting on replay is safe (action_outcomes.action_id is the SQL
    # primary key, sql-writer's route upserts by merge() -- a repeat emit
    # idempotently overwrites the same row, it does not duplicate). NOT
    # re-emitting would risk permanently losing the outcome if the process
    # died between the original save_dispatch_result and its emit, or if
    # that emit itself failed transiently -- every later tick also hits
    # this same replay branch, so there'd be no other chance to retry it.
    worker = _make_worker(monkeypatch)
    worker._store.sum_risk_dispatched_today = MagicMock(return_value=0.0)
    worker._store.latest_bus_synaptic_prediction_error = MagicMock(return_value=0.1675)
    worker._store.save_dispatch_result = MagicMock()
    worker._store.load_dispatch_result_by_dispatch_id = MagicMock(
        return_value={
            "result_id": "result:dispatch:1",
            "status": "success",
            "result_json": {"observation": "steady state"},
            "raw_len": 6,
        }
    )
    fake_bus = _patch_bus_and_client(monkeypatch, {})
    frame = _frame_with_candidates(_candidate("dispatch:1"))

    await worker._send_prepared_candidates(frame)

    fake_bus.publish.assert_awaited_once()
    _, env = fake_bus.publish.await_args.args
    assert env.payload["summary"] == "steady state"
    assert env.payload["success"] is True
    # The dispatch result itself is NOT re-saved -- only the bus emit repeats.
    worker._store.save_dispatch_result.assert_not_called()


@pytest.mark.asyncio
async def test_send_one_action_outcome_publish_failure_does_not_raise(monkeypatch) -> None:
    worker = _make_worker(monkeypatch)
    worker._store.sum_risk_dispatched_today = MagicMock(return_value=0.0)
    worker._store.latest_bus_synaptic_prediction_error = MagicMock(return_value=0.1675)
    worker._store.save_dispatch_result = MagicMock()
    worker._store.load_dispatch_result_by_dispatch_id = MagicMock(return_value=None)
    fake_bus = _patch_bus_and_client(
        monkeypatch,
        {"dispatch:1": {"result": {"final_text": '{"observation": "steady"}'}}},
    )
    fake_bus.publish = AsyncMock(side_effect=RuntimeError("bus unreachable"))
    frame = _frame_with_candidates(_candidate("dispatch:1"))

    # Must not raise -- save_dispatch_result already durably recorded the
    # result; an unreachable bus must not lose that or crash the tick.
    updated = await worker._send_prepared_candidates(frame)

    assert updated.dispatched_candidates[0].result_ref == "result:dispatch:1"


@pytest.mark.asyncio
async def test_send_one_action_outcome_summary_is_truncated(monkeypatch) -> None:
    worker = _make_worker(monkeypatch)
    worker._store.sum_risk_dispatched_today = MagicMock(return_value=0.0)
    worker._store.latest_bus_synaptic_prediction_error = MagicMock(return_value=0.1675)
    worker._store.save_dispatch_result = MagicMock()
    worker._store.load_dispatch_result_by_dispatch_id = MagicMock(return_value=None)
    long_observation = "x" * 5000
    fake_bus = _patch_bus_and_client(
        monkeypatch,
        {"dispatch:1": {"result": {"final_text": f'{{"observation": "{long_observation}"}}'}}},
    )
    frame = _frame_with_candidates(_candidate("dispatch:1"))

    await worker._send_prepared_candidates(frame)

    _, env = fake_bus.publish.await_args.args
    assert len(env.payload["summary"]) == worker_mod.ACTION_OUTCOME_SUMMARY_MAX_CHARS


@pytest.mark.asyncio
async def test_send_prepared_candidates_records_failure_on_rpc_exception(monkeypatch) -> None:
    worker = _make_worker(monkeypatch)
    worker._store.sum_risk_dispatched_today = MagicMock(return_value=0.0)
    worker._store.latest_bus_synaptic_prediction_error = MagicMock(return_value=0.1675)
    worker._store.save_dispatch_result = MagicMock()
    worker._store.load_dispatch_result_by_dispatch_id = MagicMock(return_value=None)
    _patch_bus_and_client(monkeypatch, {"dispatch:1": RuntimeError("rpc timed out")})
    frame = _frame_with_candidates(_candidate("dispatch:1"))

    updated = await worker._send_prepared_candidates(frame)

    promoted = updated.dispatched_candidates[0]
    assert promoted.dispatch_status == "dispatched"
    assert promoted.result_ref is None
    assert promoted.dispatch_error == "rpc timed out"
    assert worker._store.save_dispatch_result.call_args.kwargs["status"] == "failed"
    assert worker._store.save_dispatch_result.call_args.kwargs["raw_len"] == 0


@pytest.mark.asyncio
async def test_send_prepared_candidates_empty_observation_status_empty(monkeypatch) -> None:
    worker = _make_worker(monkeypatch)
    worker._store.sum_risk_dispatched_today = MagicMock(return_value=0.0)
    worker._store.latest_bus_synaptic_prediction_error = MagicMock(return_value=0.1675)
    worker._store.save_dispatch_result = MagicMock()
    worker._store.load_dispatch_result_by_dispatch_id = MagicMock(return_value=None)
    _patch_bus_and_client(monkeypatch, {"dispatch:1": {"result": {"final_text": ""}}})
    frame = _frame_with_candidates(_candidate("dispatch:1"))

    updated = await worker._send_prepared_candidates(frame)

    # Empty observation is still an evidenced, real attempt -- promoted to
    # dispatched with a result_ref, never fabricated as a non-attempt.
    promoted = updated.dispatched_candidates[0]
    assert promoted.dispatch_status == "dispatched"
    assert promoted.result_ref == "result:dispatch:1"
    assert worker._store.save_dispatch_result.call_args.kwargs["status"] == "empty"
    assert worker._store.save_dispatch_result.call_args.kwargs["raw_len"] == 0


@pytest.mark.asyncio
async def test_send_prepared_candidates_respects_per_tick_budget(monkeypatch) -> None:
    worker = _make_worker(monkeypatch)
    worker._policy = worker._policy.model_copy(
        update={"limits": worker._policy.limits.model_copy(update={"max_dispatches_per_tick": 1})}
    )
    worker._store.sum_risk_dispatched_today = MagicMock(return_value=0.0)
    worker._store.latest_bus_synaptic_prediction_error = MagicMock(return_value=0.1675)
    worker._store.save_dispatch_result = MagicMock()
    worker._store.load_dispatch_result_by_dispatch_id = MagicMock(return_value=None)
    _patch_bus_and_client(
        monkeypatch,
        {
            "dispatch:1": {"result": {"final_text": '{"observation": "a"}'}},
            "dispatch:2": {"result": {"final_text": '{"observation": "b"}'}},
        },
    )
    frame = _frame_with_candidates(_candidate("dispatch:1"), _candidate("dispatch:2"))

    updated = await worker._send_prepared_candidates(frame)

    assert len(updated.dispatched_candidates) == 1
    assert len(updated.candidates) == 1
    assert updated.candidates[0].dispatch_status == "prepared_for_dispatch"
    worker._store.save_dispatch_result.assert_called_once()


@pytest.mark.asyncio
async def test_send_prepared_candidates_stops_at_first_candidate_exceeding_remaining_risk_budget(
    monkeypatch,
) -> None:
    """The real, new behavior this patch adds: selection is risk-weighted,
    not a blind count. A candidate whose own risk_score would push
    cumulative spend over what's left of today's budget is left prepared
    (not sent, and not skipped-past to try a smaller one later in priority
    order -- matches the existing simple sequential-take style)."""
    worker = _make_worker(monkeypatch)
    worker._settings.orion_dispatch_max_risk_per_day = 0.10
    worker._store.sum_risk_dispatched_today = MagicMock(return_value=0.0)
    worker._store.latest_bus_synaptic_prediction_error = MagicMock(return_value=0.1675)
    worker._store.save_dispatch_result = MagicMock()
    worker._store.load_dispatch_result_by_dispatch_id = MagicMock(return_value=None)
    _patch_bus_and_client(
        monkeypatch,
        {
            "dispatch:1": {"result": {"final_text": '{"observation": "a"}'}},
            "dispatch:2": {"result": {"final_text": '{"observation": "b"}'}},
        },
    )
    # risk_score 0.06 + 0.06 = 0.12 > the 0.10 budget -- the second must not send.
    frame = _frame_with_candidates(
        _candidate("dispatch:1", risk_score=0.06), _candidate("dispatch:2", risk_score=0.06)
    )

    updated = await worker._send_prepared_candidates(frame)

    assert len(updated.dispatched_candidates) == 1
    assert updated.dispatched_candidates[0].dispatch_id == "dispatch:1"
    assert len(updated.candidates) == 1
    assert updated.candidates[0].dispatch_id == "dispatch:2"
    assert updated.candidates[0].dispatch_status == "prepared_for_dispatch"
    worker._store.save_dispatch_result.assert_called_once()


@pytest.mark.asyncio
async def test_send_prepared_candidates_sends_all_when_cumulative_risk_fits_budget(
    monkeypatch,
) -> None:
    worker = _make_worker(monkeypatch)
    worker._policy = worker._policy.model_copy(
        update={"limits": worker._policy.limits.model_copy(update={"max_dispatches_per_tick": 5})}
    )
    worker._settings.orion_dispatch_max_risk_per_day = 1.0
    worker._store.sum_risk_dispatched_today = MagicMock(return_value=0.0)
    worker._store.latest_bus_synaptic_prediction_error = MagicMock(return_value=0.1675)
    worker._store.save_dispatch_result = MagicMock()
    worker._store.load_dispatch_result_by_dispatch_id = MagicMock(return_value=None)
    _patch_bus_and_client(
        monkeypatch,
        {
            "dispatch:1": {"result": {"final_text": '{"observation": "a"}'}},
            "dispatch:2": {"result": {"final_text": '{"observation": "b"}'}},
            "dispatch:3": {"result": {"final_text": '{"observation": "c"}'}},
        },
    )
    frame = _frame_with_candidates(
        _candidate("dispatch:1", risk_score=0.05),
        _candidate("dispatch:2", risk_score=0.05),
        _candidate("dispatch:3", risk_score=0.05),
    )

    updated = await worker._send_prepared_candidates(frame)

    assert len(updated.dispatched_candidates) == 3
    assert updated.candidates == []
    assert worker._store.save_dispatch_result.call_count == 3


@pytest.mark.asyncio
async def test_send_prepared_candidates_skips_non_prepared_without_consuming_per_tick_budget(
    monkeypatch,
) -> None:
    """Defensive coverage for a shape not currently reachable in production
    (build_execution_dispatch_frame routes anything not prepared_for_dispatch
    into blocked_candidates, never candidates -- confirmed via review) but
    real code path nonetheless: a non-prepared candidate earlier in
    frame.candidates must not count against max_dispatches_per_tick before
    the loop reaches a real prepared one."""
    worker = _make_worker(monkeypatch)
    worker._policy = worker._policy.model_copy(
        update={"limits": worker._policy.limits.model_copy(update={"max_dispatches_per_tick": 1})}
    )
    worker._settings.orion_dispatch_max_risk_per_day = 1.0
    worker._store.sum_risk_dispatched_today = MagicMock(return_value=0.0)
    worker._store.latest_bus_synaptic_prediction_error = MagicMock(return_value=0.1675)
    worker._store.save_dispatch_result = MagicMock()
    worker._store.load_dispatch_result_by_dispatch_id = MagicMock(return_value=None)
    _patch_bus_and_client(
        monkeypatch, {"dispatch:2": {"result": {"final_text": '{"observation": "b"}'}}}
    )
    frame = _frame_with_candidates(
        _candidate("dispatch:1", status="blocked"),
        _candidate("dispatch:2"),
    )

    updated = await worker._send_prepared_candidates(frame)

    assert len(updated.dispatched_candidates) == 1
    assert updated.dispatched_candidates[0].dispatch_id == "dispatch:2"
    worker._store.save_dispatch_result.assert_called_once()


@pytest.mark.asyncio
async def test_send_prepared_candidates_zero_risk_candidate_still_spends_the_floor(
    monkeypatch,
) -> None:
    """MINIMUM_REAL_RISK_FLOOR closes a real, disclosed gap: a risk_score=0.0
    candidate (not reachable with today's live proposal templates, but not
    prevented at the schema level either) must still cost something against
    the daily budget, or the budget provides no ceiling at all for it."""
    worker = _make_worker(monkeypatch)
    worker._policy = worker._policy.model_copy(
        update={"limits": worker._policy.limits.model_copy(update={"max_dispatches_per_tick": 5})}
    )
    worker._settings.orion_dispatch_max_risk_per_day = worker_mod.MINIMUM_REAL_RISK_FLOOR
    worker._store.sum_risk_dispatched_today = MagicMock(return_value=0.0)
    worker._store.latest_bus_synaptic_prediction_error = MagicMock(return_value=0.1675)
    worker._store.save_dispatch_result = MagicMock()
    worker._store.load_dispatch_result_by_dispatch_id = MagicMock(return_value=None)
    _patch_bus_and_client(
        monkeypatch,
        {
            "dispatch:1": {"result": {"final_text": '{"observation": "a"}'}},
            "dispatch:2": {"result": {"final_text": '{"observation": "b"}'}},
        },
    )
    frame = _frame_with_candidates(
        _candidate("dispatch:1", risk_score=0.0), _candidate("dispatch:2", risk_score=0.0)
    )

    updated = await worker._send_prepared_candidates(frame)

    # Budget is exactly one floor-spend's worth -- the first candidate fits
    # (spends the floor, hitting the budget exactly), the second does not.
    assert len(updated.dispatched_candidates) == 1
    assert updated.dispatched_candidates[0].dispatch_id == "dispatch:1"
    assert len(updated.candidates) == 1
    assert updated.candidates[0].dispatch_id == "dispatch:2"


@pytest.mark.asyncio
async def test_send_one_replays_existing_result_without_resending(monkeypatch) -> None:
    # Crash-recovery scenario: dispatch_id is deterministic, so if a prior
    # tick already sent this exact candidate and recorded a result (but the
    # process died before save_dispatch_frame persisted that), the next
    # tick must NOT fire a second real cortex-exec RPC for it.
    worker = _make_worker(monkeypatch)
    worker._store.sum_risk_dispatched_today = MagicMock(return_value=0.0)
    worker._store.latest_bus_synaptic_prediction_error = MagicMock(return_value=0.1675)
    worker._store.save_dispatch_result = MagicMock()
    worker._store.load_dispatch_result_by_dispatch_id = MagicMock(
        return_value={
            "result_id": "result:dispatch:1",
            "status": "success",
            "result_json": {"observation": "steady", "evidence_refs": ["result:dispatch:1"]},
            "raw_len": 6,
        }
    )
    _patch_bus_and_client(monkeypatch, {})  # no outcome registered -- a real send would KeyError
    frame = _frame_with_candidates(_candidate("dispatch:1"))

    updated = await worker._send_prepared_candidates(frame)

    promoted = updated.dispatched_candidates[0]
    assert promoted.dispatch_status == "dispatched"
    assert promoted.result_ref == "result:dispatch:1"
    worker._store.save_dispatch_result.assert_not_called()


@pytest.mark.asyncio
async def test_send_one_replays_existing_failed_result_without_resending(monkeypatch) -> None:
    worker = _make_worker(monkeypatch)
    worker._store.sum_risk_dispatched_today = MagicMock(return_value=0.0)
    worker._store.latest_bus_synaptic_prediction_error = MagicMock(return_value=0.1675)
    worker._store.save_dispatch_result = MagicMock()
    worker._store.load_dispatch_result_by_dispatch_id = MagicMock(
        return_value={
            "result_id": "result:dispatch:1",
            "status": "failed",
            "result_json": {"error": "rpc timed out"},
            "raw_len": 0,
        }
    )
    _patch_bus_and_client(monkeypatch, {})
    frame = _frame_with_candidates(_candidate("dispatch:1"))

    updated = await worker._send_prepared_candidates(frame)

    promoted = updated.dispatched_candidates[0]
    assert promoted.dispatch_status == "dispatched"
    assert promoted.result_ref is None
    assert promoted.dispatch_error == "rpc timed out"
    worker._store.save_dispatch_result.assert_not_called()


@pytest.mark.asyncio
async def test_send_prepared_candidates_skips_when_daily_risk_cap_reached_and_enforced(
    monkeypatch,
) -> None:
    worker = _make_worker(monkeypatch)
    worker._settings.orion_dispatch_risk_cap_advisory_only = False
    worker._store.sum_risk_dispatched_today = MagicMock(
        return_value=worker._settings.orion_dispatch_max_risk_per_day
    )
    worker._store.save_dispatch_result = MagicMock()
    _patch_bus_and_client(monkeypatch, {})
    frame = _frame_with_candidates(_candidate("dispatch:1"))

    updated = await worker._send_prepared_candidates(frame)

    assert updated.candidates == [frame.candidates[0]]
    assert updated.dispatched_candidates == []
    worker._store.save_dispatch_result.assert_not_called()


@pytest.mark.asyncio
async def test_send_prepared_candidates_dispatches_past_reached_cap_when_advisory_only(
    monkeypatch,
) -> None:
    """2026-07-29: advisory_only is no longer the default (enforcement is back
    on, now against a derived EWMA ceiling instead of the old hand-picked
    static number) -- it's an explicit operator override an operator must opt
    into. When set, it must still behave exactly as before: log that the cap
    was reached, but never withhold a real dispatch on it."""
    worker = _make_worker(monkeypatch)
    assert worker._settings.orion_dispatch_risk_cap_advisory_only is False
    worker._settings.orion_dispatch_risk_cap_advisory_only = True
    worker._store.sum_risk_dispatched_today = MagicMock(
        return_value=worker._settings.orion_dispatch_max_risk_per_day
    )
    worker._store.latest_bus_synaptic_prediction_error = MagicMock(return_value=0.1675)
    worker._store.save_dispatch_result = MagicMock()
    worker._store.load_dispatch_result_by_dispatch_id = MagicMock(return_value=None)
    fake_bus = _patch_bus_and_client(
        monkeypatch,
        {"dispatch:1": {"result": {"final_text": '{"observation": "steady"}'}}},
    )
    frame = _frame_with_candidates(_candidate("dispatch:1"))

    updated = await worker._send_prepared_candidates(frame)

    assert len(updated.dispatched_candidates) == 1
    fake_bus.connect.assert_awaited_once()
    worker._store.save_dispatch_result.assert_called_once()


@pytest.mark.asyncio
async def test_send_prepared_candidates_skips_when_tripwire_active(monkeypatch) -> None:
    worker = _make_worker(monkeypatch)
    # Tripwire window is now in-process (worker.py, fixed 2026-07-25 -- a live
    # Postgres query let stale pre-restart rows defeat "restart to re-arm";
    # see the theater-tripwire-age-gate fix), not backed by the store.
    worker._recent_dispatch_statuses = deque(
        ["empty"] * 6 + ["success"] * 4, maxlen=worker_mod.THEATER_TRIPWIRE_WINDOW
    )
    worker._store.sum_risk_dispatched_today = MagicMock(return_value=0.0)
    worker._store.latest_bus_synaptic_prediction_error = MagicMock(return_value=0.1675)
    worker._store.save_dispatch_result = MagicMock()
    worker._notify.send = MagicMock()
    _patch_bus_and_client(monkeypatch, {})
    frame = _frame_with_candidates(_candidate("dispatch:1"))

    updated = await worker._send_prepared_candidates(frame)

    assert worker.theater_tripwire_active is True
    assert updated.candidates == [frame.candidates[0]]
    worker._store.save_dispatch_result.assert_not_called()
    worker._notify.send.assert_called_once()


@pytest.mark.asyncio
async def test_send_prepared_candidates_tripwire_stays_tripped_across_calls(monkeypatch) -> None:
    worker = _make_worker(monkeypatch)
    worker._recent_dispatch_statuses = deque(
        ["empty"] * 6 + ["success"] * 4, maxlen=worker_mod.THEATER_TRIPWIRE_WINDOW
    )
    worker._store.sum_risk_dispatched_today = MagicMock(return_value=0.0)
    worker._store.latest_bus_synaptic_prediction_error = MagicMock(return_value=0.1675)
    worker._store.save_dispatch_result = MagicMock()
    worker._notify.send = MagicMock()
    _patch_bus_and_client(monkeypatch, {})
    frame = _frame_with_candidates(_candidate("dispatch:1"))

    await worker._send_prepared_candidates(frame)
    await worker._send_prepared_candidates(_frame_with_candidates(_candidate("dispatch:2")))

    # Notified once on the transition into tripped, not on every subsequent tick.
    worker._notify.send.assert_called_once()


@pytest.mark.asyncio
async def test_send_prepared_candidates_noop_when_nothing_prepared(monkeypatch) -> None:
    worker = _make_worker(monkeypatch)
    worker._store.sum_risk_dispatched_today = MagicMock(return_value=0.0)
    worker._store.latest_bus_synaptic_prediction_error = MagicMock(return_value=0.1675)
    frame = _frame_with_candidates(_candidate("dispatch:1", status="blocked"))

    updated = await worker._send_prepared_candidates(frame)

    # No longer strict identity: every return path (including this no-op one)
    # now stamps the current daily_risk_baseline_* fields via model_copy so
    # the next tick's baseline lookup always finds the latest state. With no
    # baseline history (the _make_worker default), those fields are just the
    # cold-start defaults, so the frame is value-equal, not the same object.
    assert updated == frame.model_copy(
        update={
            "daily_risk_baseline_ewma": 0.0,
            "daily_risk_baseline_ewma_var": 0.0,
            "daily_risk_baseline_ewma_n": 0,
            "daily_risk_baseline_last_day": None,
        }
    )
    assert updated.dispatch_attempted is False


# ---------------------------------------------------------------------------
# Self-calibrating daily risk ceiling (2026-07-29)
# ---------------------------------------------------------------------------


def test_derive_daily_risk_cap_falls_back_to_static_when_no_history_at_all(monkeypatch) -> None:
    """Truly first-ever tick: no baseline row anywhere, and no historical
    closed day to seed from either. Must fall back to the static
    orion_dispatch_max_risk_per_day setting, not error or default to 0."""
    worker = _make_worker(monkeypatch)
    generated_at = datetime(2026, 7, 29, 3, 0, tzinfo=timezone.utc)

    cap, fields = worker._derive_daily_risk_cap(generated_at)

    assert fields["daily_risk_baseline_ewma_n"] == 0
    assert fields["daily_risk_baseline_last_day"] is None
    assert cap == worker._settings.orion_dispatch_max_risk_per_day


def test_derive_daily_risk_cap_seeds_from_historical_closed_day(monkeypatch) -> None:
    """Cold-start seeding: n==0, last_day=None, but a real historical closed
    day (2026-07-28's real 817.65) exists. Sample #1 comes from that real
    uncapped total, not a hardcoded starting constant, and the interim cap
    (n==1, below DAILY_RISK_BASELINE_MIN_SAMPLES) is the disclosed 2x
    margin."""
    worker = _make_worker(monkeypatch)
    generated_at = datetime(2026, 7, 29, 3, 0, tzinfo=timezone.utc)
    worker._store.most_recent_closed_day_with_data = MagicMock(
        return_value=("2026-07-28", 817.65)
    )

    cap, fields = worker._derive_daily_risk_cap(generated_at)

    worker._store.most_recent_closed_day_with_data.assert_called_once_with(
        datetime(2026, 7, 29, 0, 0, tzinfo=timezone.utc)
    )
    assert fields["daily_risk_baseline_ewma"] == pytest.approx(817.65)
    assert fields["daily_risk_baseline_ewma_var"] == 0.0
    assert fields["daily_risk_baseline_ewma_n"] == 1
    # last_day is set to the day of THIS tick, not the seeded day -- see
    # _derive_daily_risk_cap's own docstring on why (avoiding a real
    # double-count on the very next tick of the same day).
    assert fields["daily_risk_baseline_last_day"] == "2026-07-29"
    assert cap == pytest.approx(817.65 * 2.0)


def test_derive_daily_risk_cap_carries_forward_within_same_day(monkeypatch) -> None:
    """Multiple ticks the same UTC day must not re-derive/re-absorb anything
    -- sum_uncapped_risk_for_day must not even be called."""
    worker = _make_worker(monkeypatch)
    generated_at = datetime(2026, 7, 29, 15, 0, tzinfo=timezone.utc)
    worker._store.load_latest_daily_risk_baseline = MagicMock(
        return_value={
            "daily_risk_baseline_ewma": 140.0,
            "daily_risk_baseline_ewma_var": 4000.0,
            "daily_risk_baseline_ewma_n": 2,
            "daily_risk_baseline_last_day": "2026-07-29",
        }
    )

    cap, fields = worker._derive_daily_risk_cap(generated_at)

    worker._store.sum_uncapped_risk_for_day.assert_not_called()
    assert fields["daily_risk_baseline_ewma_n"] == 2
    assert fields["daily_risk_baseline_ewma"] == 140.0
    assert cap == pytest.approx(140.0 + 3.0 * math.sqrt(4000.0))


def test_derive_daily_risk_cap_rolls_over_exactly_once_per_day_boundary(monkeypatch) -> None:
    """The real day-boundary update: absorbs the prior day's real uncapped
    total into the EWMA exactly once, then a second call for the same
    "today" (simulating the next tick reading back what the first tick would
    have saved) must NOT re-absorb it a second time."""
    worker = _make_worker(monkeypatch)
    generated_at = datetime(2026, 7, 29, 3, 0, tzinfo=timezone.utc)
    worker._store.load_latest_daily_risk_baseline = MagicMock(
        return_value={
            "daily_risk_baseline_ewma": 100.0,
            "daily_risk_baseline_ewma_var": 0.0,
            "daily_risk_baseline_ewma_n": 1,
            "daily_risk_baseline_last_day": "2026-07-28",
        }
    )
    worker._store.sum_uncapped_risk_for_day = MagicMock(return_value=200.0)

    cap, fields = worker._derive_daily_risk_cap(generated_at)

    worker._store.sum_uncapped_risk_for_day.assert_called_once_with(
        datetime(2026, 7, 28, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 7, 29, 0, 0, tzinfo=timezone.utc),
    )
    assert fields["daily_risk_baseline_ewma_n"] == 2
    assert fields["daily_risk_baseline_last_day"] == "2026-07-29"
    assert fields["daily_risk_baseline_ewma"] == pytest.approx(140.0)  # 0.4*200 + 0.6*100
    assert fields["daily_risk_baseline_ewma_var"] == pytest.approx(4000.0)  # 0.4*(100**2)
    assert cap == pytest.approx(140.0 + 3.0 * math.sqrt(4000.0))

    # Second tick, same "today" -- baseline now reflects what the first
    # tick's frame would have persisted. Must be a pure carry-forward.
    worker._store.load_latest_daily_risk_baseline = MagicMock(return_value=fields)
    worker._store.sum_uncapped_risk_for_day.reset_mock()

    cap2, fields2 = worker._derive_daily_risk_cap(generated_at)

    worker._store.sum_uncapped_risk_for_day.assert_not_called()
    assert fields2 == fields
    assert cap2 == cap


def test_derive_daily_risk_cap_uses_domain_variance_floor_not_shared_default(monkeypatch) -> None:
    """A tiny real variance (e.g. two nearly-identical closed days) must be
    floored at DAILY_RISK_BASELINE_MIN_VARIANCE (1.0), not orion/bus/
    ewma.py's own shared default (1e-6) -- confirming the domain-specific
    floor is actually wired through, not silently bypassed."""
    worker = _make_worker(monkeypatch)
    generated_at = datetime(2026, 7, 29, 3, 0, tzinfo=timezone.utc)
    worker._store.load_latest_daily_risk_baseline = MagicMock(
        return_value={
            "daily_risk_baseline_ewma": 800.0,
            "daily_risk_baseline_ewma_var": 1e-9,
            "daily_risk_baseline_ewma_n": 2,
            "daily_risk_baseline_last_day": "2026-07-29",
        }
    )

    cap, _ = worker._derive_daily_risk_cap(generated_at)

    assert cap == pytest.approx(
        800.0 + 3.0 * math.sqrt(worker_mod.DAILY_RISK_BASELINE_MIN_VARIANCE)
    )


@pytest.mark.asyncio
async def test_send_prepared_candidates_enforces_against_derived_cap(monkeypatch) -> None:
    """The real, new behavior this patch adds at the integration level: with
    the new default (advisory_only=False), a tick whose cumulative risk
    would exceed the *derived* EWMA cap actually blocks -- mirrors
    test_send_prepared_candidates_stops_at_first_candidate_exceeding_remaining_risk_budget
    but sourced from a baseline instead of the static setting."""
    worker = _make_worker(monkeypatch)
    assert worker._settings.orion_dispatch_risk_cap_advisory_only is False
    # n==1 -> derived cap is ewma*2.0 == 0.10, same numeric scenario as the
    # static-setting version of this test.
    worker._store.load_latest_daily_risk_baseline = MagicMock(
        return_value={
            "daily_risk_baseline_ewma": 0.05,
            "daily_risk_baseline_ewma_var": 0.0,
            "daily_risk_baseline_ewma_n": 1,
            "daily_risk_baseline_last_day": NOW.date().isoformat(),
        }
    )
    worker._store.sum_risk_dispatched_today = MagicMock(return_value=0.0)
    worker._store.latest_bus_synaptic_prediction_error = MagicMock(return_value=0.1675)
    worker._store.save_dispatch_result = MagicMock()
    worker._store.load_dispatch_result_by_dispatch_id = MagicMock(return_value=None)
    _patch_bus_and_client(
        monkeypatch,
        {
            "dispatch:1": {"result": {"final_text": '{"observation": "a"}'}},
            "dispatch:2": {"result": {"final_text": '{"observation": "b"}'}},
        },
    )
    frame = _frame_with_candidates(
        _candidate("dispatch:1", risk_score=0.06), _candidate("dispatch:2", risk_score=0.06)
    )

    updated = await worker._send_prepared_candidates(frame)

    assert len(updated.dispatched_candidates) == 1
    assert updated.dispatched_candidates[0].dispatch_id == "dispatch:1"
    assert len(updated.candidates) == 1
    assert updated.candidates[0].dispatch_id == "dispatch:2"
    assert updated.daily_risk_baseline_ewma_n == 1
    worker._store.save_dispatch_result.assert_called_once()


@pytest.mark.asyncio
async def test_send_prepared_candidates_stamps_baseline_fields_on_early_return_paths(
    monkeypatch,
) -> None:
    """Every early-return path (theater tripwire here) still has to carry the
    current baseline state forward onto the saved frame -- otherwise the
    next tick's load_latest_daily_risk_baseline() would silently regress to
    stale state whenever a tick happens to hit an early return."""
    worker = _make_worker(monkeypatch)
    worker._recent_dispatch_statuses = deque(
        ["empty"] * 6 + ["success"] * 4, maxlen=worker_mod.THEATER_TRIPWIRE_WINDOW
    )
    worker._store.load_latest_daily_risk_baseline = MagicMock(
        return_value={
            "daily_risk_baseline_ewma": 140.0,
            "daily_risk_baseline_ewma_var": 4000.0,
            "daily_risk_baseline_ewma_n": 2,
            "daily_risk_baseline_last_day": NOW.date().isoformat(),
        }
    )
    worker._store.sum_risk_dispatched_today = MagicMock(return_value=0.0)
    worker._notify.send = MagicMock()
    _patch_bus_and_client(monkeypatch, {})
    frame = _frame_with_candidates(_candidate("dispatch:1"))

    updated = await worker._send_prepared_candidates(frame)

    assert updated.daily_risk_baseline_ewma == 140.0
    assert updated.daily_risk_baseline_ewma_n == 2


@pytest.mark.asyncio
async def test_send_prepared_candidates_sends_concurrently_not_sequentially(monkeypatch) -> None:
    """2026-07-31 Part 2 (docs/superpowers/specs/2026-07-30-execution-
    dispatch-staleness-discard-design.md): the whole point of this patch --
    real RPC wait time must actually overlap across candidates, not just
    avoid raising. 3 candidates, each with a _SLOW_CLIENT_DELAY_SEC-second
    simulated RPC: sequential would take >= 3x that; concurrent should land
    well under 2x a single delay even with real scheduling overhead."""
    worker = _make_worker(monkeypatch)
    worker._store.sum_risk_dispatched_today = MagicMock(return_value=0.0)
    worker._store.latest_bus_synaptic_prediction_error = MagicMock(return_value=0.1)
    worker._store.save_dispatch_result = MagicMock()
    worker._store.load_dispatch_result_by_dispatch_id = MagicMock(return_value=None)
    _patch_bus_and_slow_client(
        monkeypatch,
        {
            "dispatch:1": {"result": {"final_text": '{"observation": "a"}'}},
            "dispatch:2": {"result": {"final_text": '{"observation": "b"}'}},
            "dispatch:3": {"result": {"final_text": '{"observation": "c"}'}},
        },
    )
    frame = _frame_with_candidates(
        _candidate("dispatch:1", risk_score=0.01),
        _candidate("dispatch:2", risk_score=0.01),
        _candidate("dispatch:3", risk_score=0.01),
    )

    start = time.monotonic()
    updated = await worker._send_prepared_candidates(frame)
    elapsed = time.monotonic() - start

    assert len(updated.dispatched_candidates) == 3
    assert elapsed < _SLOW_CLIENT_DELAY_SEC * 2


@pytest.mark.asyncio
async def test_send_one_wrapper_converts_unexpected_failure_to_dispatch_error(monkeypatch) -> None:
    """_send_one must be a total function -- an unexpected failure anywhere
    in _send_one_inner (not just the RPC call) degrades to a dispatch_error
    candidate, never raises. Required for asyncio.gather(return_exceptions=
    True) to never see a raw exception object in its results list, which
    ExecutionDispatchFrameV1.dispatched_candidates: list[
    ExecutionDispatchCandidateV1] cannot hold."""
    worker = _make_worker(monkeypatch)
    candidate = _candidate("dispatch:boom")
    frame = _frame_with_candidates(candidate)

    async def _raise(*_args, **_kwargs):
        raise RuntimeError("simulated unexpected failure")

    worker._send_one_inner = _raise

    result = await worker._send_one(client=None, bus=None, frame=frame, candidate=candidate)

    assert result.dispatch_status == "dispatched"
    assert result.dispatch_error is not None
    assert "simulated unexpected failure" in result.dispatch_error


def test_send_prepared_candidates_uses_return_exceptions_true(monkeypatch) -> None:
    """Regression guard for the real risk found on the third adversarial
    pass at Part 2's design: bare asyncio.gather() (no return_exceptions=
    True) lets one candidate's failure cancel a still-in-flight sibling.
    Confirmed live by direct test before this shipped, not assumed -- and
    that cancellation is specifically a consequence of asyncio.run()'s own
    shutdown path when its top-level coroutine exits via an exception, NOT
    of gather() itself (gather()'s own documented behavior, without
    return_exceptions, is to propagate the first exception WITHOUT
    cancelling siblings -- they only get cancelled if whatever awaited
    gather() then also exits, and something -- asyncio.run() here -- reacts
    to that exit by tearing down remaining tasks). Real production code
    hits exactly this shape: _tick() calls asyncio.run(self.
    _send_prepared_candidates(frame)) from a background thread
    (asyncio.to_thread). This test intentionally skips @pytest.mark.asyncio
    and calls asyncio.run() directly for that reason -- a pytest-asyncio-
    managed loop does not tear down the same way between test statements,
    so it would not actually reproduce the real risk being guarded against.

    Also bypasses _send_one's own hardening on purpose (mocks _send_one
    directly, not _send_one_inner), so this test proves gather()'s own
    return_exceptions=True usage in isolation, independent of _send_one's
    separate defense-in-depth (tested separately, test_send_one_wrapper_
    converts_unexpected_failure_to_dispatch_error). If a future change ever
    "simplified" the gather(...) call back to bare gather(), this test must
    fail."""
    worker = _make_worker(monkeypatch)
    worker._store.sum_risk_dispatched_today = MagicMock(return_value=0.0)
    worker._store.latest_bus_synaptic_prediction_error = MagicMock(return_value=0.1)
    _patch_bus_and_client(monkeypatch, {})

    slow_completed = False

    async def _fake_send_one(client, bus, frame, candidate):
        nonlocal slow_completed
        if candidate.dispatch_id == "dispatch:bad":
            raise RuntimeError("simulated failure bypassing _send_one's own hardening")
        await asyncio.sleep(_SLOW_CLIENT_DELAY_SEC)
        slow_completed = True
        return candidate.model_copy(
            update={
                "dispatch_status": "dispatched",
                "dispatched_at": NOW,
                "result_ref": "result:dispatch:good",
            }
        )

    worker._send_one = _fake_send_one
    frame = _frame_with_candidates(
        _candidate("dispatch:bad", risk_score=0.01),
        _candidate("dispatch:good", risk_score=0.01),
    )

    # Deliberately not asserting on the returned frame's content: this test
    # bypasses _send_one's own hardening on purpose, so newly_dispatched can
    # legitimately contain a raw RuntimeError alongside the real candidate
    # -- that malformed-frame consequence is exactly why _send_one's
    # hardening is required in real production code, tested separately.
    # Swallow whatever asyncio.run() raises/returns here; only slow_
    # completed matters to this test.
    try:
        asyncio.run(worker._send_prepared_candidates(frame))
    except Exception:
        pass

    # If return_exceptions=True is ever removed, the "bad" coroutine's raw
    # exception propagates out of gather() and out of _send_prepared_
    # candidates, making the top-level coroutine given to asyncio.run() exit
    # via that exception -- asyncio.run()'s own shutdown then cancels the
    # still-sleeping "good" coroutine before it can set slow_completed. This
    # assertion is the real, direct proof, run the same way production
    # actually invokes this code.
    assert slow_completed is True
