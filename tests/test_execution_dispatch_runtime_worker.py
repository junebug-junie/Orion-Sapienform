from __future__ import annotations

import sys
from datetime import datetime, timezone
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
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1  # noqa: E402
from orion.schemas.proposal_frame import ProposalFrameV1  # noqa: E402
from orion.schemas.self_state import SelfStateV1  # noqa: E402

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _self_state() -> SelfStateV1:
    return SelfStateV1(
        self_state_id="self.state:test",
        generated_at=NOW,
        source_field_tick_id="tick:test",
        source_field_generated_at=NOW,
        source_attention_frame_id="frame:test",
        source_attention_generated_at=NOW,
        overall_intensity=0.5,
        overall_confidence=0.9,
    )


def _proposal() -> ProposalFrameV1:
    return ProposalFrameV1(
        frame_id="proposal.frame:test:proposal_policy.v1",
        generated_at=NOW,
        source_self_state_id="self.state:test",
        source_self_state_generated_at=NOW,
        source_attention_frame_id="frame:test",
        source_field_tick_id="tick:test",
        overall_action_pressure=0.4,
        overall_risk=0.1,
        candidates=[],
    )


def _policy_frame() -> PolicyDecisionFrameV1:
    return PolicyDecisionFrameV1(
        frame_id="policy.frame:proposal.frame:test:substrate_policy.v1",
        generated_at=NOW,
        source_proposal_frame_id="proposal.frame:test:proposal_policy.v1",
        source_self_state_id="self.state:test",
        overall_risk=0.0,
    )


def test_worker_skips_when_no_policy_pending(monkeypatch) -> None:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = ExecutionDispatchRuntimeWorker()
    worker._store.load_latest_policy_frame_without_dispatch = MagicMock(return_value=None)
    worker._store.save_dispatch_frame = MagicMock()

    worker._tick()

    worker._store.save_dispatch_frame.assert_not_called()


def test_worker_records_unevaluable_frame_when_proposal_missing(monkeypatch) -> None:
    # 2026-07-12: a naive skip-and-return would retry the same oldest
    # undispatched policy frame forever, blocking every policy frame queued
    # behind it. The worker must record an honest "could not evaluate" frame
    # so the FIFO queue advances.
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = ExecutionDispatchRuntimeWorker()
    policy_frame = _policy_frame()
    worker._store.load_latest_policy_frame_without_dispatch = MagicMock(return_value=policy_frame)
    worker._store.load_proposal_frame = MagicMock(return_value=None)
    worker._store.save_dispatch_frame = MagicMock()

    worker._tick()

    worker._store.save_dispatch_frame.assert_called_once()
    saved_frame = worker._store.save_dispatch_frame.call_args[0][0]
    assert saved_frame.source_policy_frame_id == policy_frame.frame_id
    assert saved_frame.dispatch_attempted is False
    assert saved_frame.candidates == []
    assert any("proposal_frame" in w for w in saved_frame.warnings)


def test_worker_records_unevaluable_frame_when_self_state_missing(monkeypatch) -> None:
    # Same reasoning as the missing-proposal case above -- live incident,
    # 2026-07-12 (a schema change made an old self-state row permanently
    # unloadable, stalling this queue for it).
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = ExecutionDispatchRuntimeWorker()
    policy_frame = _policy_frame()
    worker._store.load_latest_policy_frame_without_dispatch = MagicMock(return_value=policy_frame)
    worker._store.load_proposal_frame = MagicMock(return_value=_proposal())
    worker._store.load_self_state = MagicMock(return_value=None)
    worker._store.save_dispatch_frame = MagicMock()

    worker._tick()

    worker._store.save_dispatch_frame.assert_called_once()
    saved_frame = worker._store.save_dispatch_frame.call_args[0][0]
    assert saved_frame.source_policy_frame_id == policy_frame.frame_id
    assert saved_frame.dispatch_attempted is False
    assert any("self_state" in w for w in saved_frame.warnings)


def test_worker_saves_dispatch_frame_for_pending_policy(monkeypatch) -> None:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = ExecutionDispatchRuntimeWorker()
    policy_frame = _policy_frame()
    worker._store.load_latest_policy_frame_without_dispatch = MagicMock(return_value=policy_frame)
    worker._store.load_proposal_frame = MagicMock(return_value=_proposal())
    worker._store.load_self_state = MagicMock(return_value=_self_state())
    worker._store.save_dispatch_frame = MagicMock()

    worker._tick()

    worker._store.save_dispatch_frame.assert_called_once()
    saved = worker._store.save_dispatch_frame.call_args[0][0]
    assert isinstance(saved, ExecutionDispatchFrameV1)
    assert saved.source_policy_frame_id == policy_frame.frame_id


def _make_worker(monkeypatch) -> ExecutionDispatchRuntimeWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import app.settings as settings_mod

    settings_mod._settings = None
    return ExecutionDispatchRuntimeWorker()


def _candidate(dispatch_id: str, status: str = "prepared_for_dispatch") -> ExecutionDispatchCandidateV1:
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
        risk_score=0.05,
        confidence_score=0.9,
    )


def _frame_with_candidates(*candidates: ExecutionDispatchCandidateV1) -> ExecutionDispatchFrameV1:
    return ExecutionDispatchFrameV1(
        frame_id="execution.dispatch.frame:test:execution_dispatch_policy.v1",
        generated_at=NOW,
        source_policy_frame_id="policy.frame:test",
        source_proposal_frame_id="proposal.frame:test",
        source_self_state_id="self.state:test",
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
    fake_bus.close = AsyncMock()
    fake_bus.publish = AsyncMock()
    monkeypatch.setattr(worker_mod, "OrionBusAsync", MagicMock(return_value=fake_bus))
    monkeypatch.setattr(worker_mod, "ExecutionDispatchCortexClient", _FakeClient)
    return fake_bus


@pytest.mark.asyncio
async def test_send_prepared_candidates_promotes_on_success(monkeypatch) -> None:
    worker = _make_worker(monkeypatch)
    worker._store.recent_dispatch_result_statuses = MagicMock(return_value=[])
    worker._store.count_dispatches_today = MagicMock(return_value=0)
    worker._store.save_dispatch_result = MagicMock()
    worker._store.load_dispatch_result_by_dispatch_id = MagicMock(return_value=None)
    _patch_bus_and_client(
        monkeypatch,
        {"dispatch:1": {"result": {"final_text": '{"observation": "steady", "confidence": 0.8}'}}},
    )
    frame = _frame_with_candidates(_candidate("dispatch:1"))

    updated = await worker._send_prepared_candidates(frame)

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
    worker._store.recent_dispatch_result_statuses = MagicMock(return_value=[])
    worker._store.count_dispatches_today = MagicMock(return_value=0)
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


@pytest.mark.asyncio
async def test_send_one_emits_action_outcome_on_empty_observation(monkeypatch) -> None:
    worker = _make_worker(monkeypatch)
    worker._store.recent_dispatch_result_statuses = MagicMock(return_value=[])
    worker._store.count_dispatches_today = MagicMock(return_value=0)
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
    worker._store.recent_dispatch_result_statuses = MagicMock(return_value=[])
    worker._store.count_dispatches_today = MagicMock(return_value=0)
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
    worker._store.recent_dispatch_result_statuses = MagicMock(return_value=[])
    worker._store.count_dispatches_today = MagicMock(return_value=0)
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
    worker._store.recent_dispatch_result_statuses = MagicMock(return_value=[])
    worker._store.count_dispatches_today = MagicMock(return_value=0)
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
    worker._store.recent_dispatch_result_statuses = MagicMock(return_value=[])
    worker._store.count_dispatches_today = MagicMock(return_value=0)
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
    worker._store.recent_dispatch_result_statuses = MagicMock(return_value=[])
    worker._store.count_dispatches_today = MagicMock(return_value=0)
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
    worker._store.recent_dispatch_result_statuses = MagicMock(return_value=[])
    worker._store.count_dispatches_today = MagicMock(return_value=0)
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
    worker._store.recent_dispatch_result_statuses = MagicMock(return_value=[])
    worker._store.count_dispatches_today = MagicMock(return_value=0)
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
async def test_send_one_replays_existing_result_without_resending(monkeypatch) -> None:
    # Crash-recovery scenario: dispatch_id is deterministic, so if a prior
    # tick already sent this exact candidate and recorded a result (but the
    # process died before save_dispatch_frame persisted that), the next
    # tick must NOT fire a second real cortex-exec RPC for it.
    worker = _make_worker(monkeypatch)
    worker._store.recent_dispatch_result_statuses = MagicMock(return_value=[])
    worker._store.count_dispatches_today = MagicMock(return_value=0)
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
    worker._store.recent_dispatch_result_statuses = MagicMock(return_value=[])
    worker._store.count_dispatches_today = MagicMock(return_value=0)
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
async def test_send_prepared_candidates_skips_when_daily_cap_reached(monkeypatch) -> None:
    worker = _make_worker(monkeypatch)
    worker._store.recent_dispatch_result_statuses = MagicMock(return_value=[])
    worker._store.count_dispatches_today = MagicMock(
        return_value=worker._settings.orion_dispatch_max_per_day
    )
    worker._store.save_dispatch_result = MagicMock()
    _patch_bus_and_client(monkeypatch, {})
    frame = _frame_with_candidates(_candidate("dispatch:1"))

    updated = await worker._send_prepared_candidates(frame)

    assert updated.candidates == [frame.candidates[0]]
    assert updated.dispatched_candidates == []
    worker._store.save_dispatch_result.assert_not_called()


@pytest.mark.asyncio
async def test_send_prepared_candidates_skips_when_tripwire_active(monkeypatch) -> None:
    worker = _make_worker(monkeypatch)
    worker._store.recent_dispatch_result_statuses = MagicMock(
        return_value=["empty"] * 6 + ["success"] * 4
    )
    worker._store.count_dispatches_today = MagicMock(return_value=0)
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
    worker._store.recent_dispatch_result_statuses = MagicMock(
        return_value=["empty"] * 6 + ["success"] * 4
    )
    worker._store.count_dispatches_today = MagicMock(return_value=0)
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
    worker._store.recent_dispatch_result_statuses = MagicMock(return_value=[])
    worker._store.count_dispatches_today = MagicMock(return_value=0)
    frame = _frame_with_candidates(_candidate("dispatch:1", status="blocked"))

    updated = await worker._send_prepared_candidates(frame)

    assert updated is frame
