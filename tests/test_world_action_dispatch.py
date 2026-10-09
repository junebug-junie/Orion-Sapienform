"""shed_background_gpu through execution dispatch: world gate, precommit, holdback, shed RPC, settle.

Every pool interaction is a fake (acceptance check 4 / D5 "tests hitting production"): the bus is a
MagicMock and the shed RPC is answered in-process. Nothing here can reach a live GPU pool."""
from __future__ import annotations

import sys
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
from orion.execution_dispatch.policy import SHED_EXECUTOR_VERB  # noqa: E402
from orion.execution_dispatch.shed_settlement import SHED_PENDING, settle_shed_result  # noqa: E402
from orion.schemas.action_prediction import ExpectedEffectV1  # noqa: E402
from orion.schemas.execution_dispatch_frame import ExecutionDispatchCandidateV1, ExecutionDispatchFrameV1  # noqa: E402
from orion.schemas.gpu_pool import GpuPoolShedResultV1  # noqa: E402

NOW = datetime.now(timezone.utc)


def _worker(monkeypatch, **env) -> ExecutionDispatchRuntimeWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import app.settings as settings_mod

    settings_mod._settings = None
    w = ExecutionDispatchRuntimeWorker()
    s = w._store
    s.load_latest_daily_risk_baseline = MagicMock(return_value=None)
    s.most_recent_closed_day_with_data = MagicMock(return_value=None)
    s.sum_uncapped_risk_for_day = MagicMock(return_value=0.0)
    s.sum_risk_dispatched_today = MagicMock(return_value=0.0)
    s.latest_bus_synaptic_prediction_error = MagicMock(return_value=0.1)
    s.save_dispatch_result = MagicMock()
    s.load_dispatch_result_by_dispatch_id = MagicMock(return_value=None)
    s.insert_world_episode = MagicMock(return_value=True)
    s.record_world_settlement = MagicMock()
    s.load_world_admission_inputs = MagicMock(return_value=([], []))
    s.world_episode_arm = MagicMock(return_value="treated")
    w._derive_motor_budget = MagicMock(return_value=None)
    fake_bus = MagicMock()
    fake_bus.connect, fake_bus.close, fake_bus.publish = AsyncMock(), AsyncMock(), AsyncMock()
    fake_bus.rpc_request = AsyncMock(side_effect=AssertionError("tests never reach a real pool"))
    monkeypatch.setattr(worker_mod, "OrionBusAsync", MagicMock(return_value=fake_bus))
    w.fake_bus = fake_bus
    return w


def _shed_candidate(dispatch_id="dispatch:shed:1", *, holdback=0.5, evaluated_at=None) -> ExecutionDispatchCandidateV1:
    return ExecutionDispatchCandidateV1(
        dispatch_id=dispatch_id, source_decision_id="pd:1", source_proposal_id="proposal:shed_background_gpu:t:b",
        dispatch_status="prepared_for_dispatch", dispatch_mode="dispatch_read_only", dispatch_kind="self_regulate",
        target_id="pool:background_gpu", target_kind="system", cortex_verb=SHED_EXECUTOR_VERB, cortex_mode="brain",
        risk_score=0.1, confidence_score=1.0,
        expected_effect=ExpectedEffectV1(signal_id="cabinet_heat_pressure", direction="decrease", predicted_delta=0.0,
                                         predictor_variance=0.25, predictor_n=0, cold_start=True),
        world_action={"template": "shed_background_gpu", "holdback_fraction": holdback,
                      "attention_winner": {"broadcast_log_id": "broadcast-x", "open_loop_id": "open-loop-c",
                                           "node_id": "node:substrate.cabinet"},
                      "eligibility": {"eligible": True, "evaluated_at": (evaluated_at or NOW).isoformat(),
                                      "hardware_watch": {"open_incident_ids": []}}})


def _frame(*cands) -> ExecutionDispatchFrameV1:
    return ExecutionDispatchFrameV1(frame_id="execution.dispatch.frame:t", generated_at=NOW,
                                    source_policy_frame_id="p", source_proposal_frame_id="q",
                                    source_field_tick_id="f", dispatch_mode="dispatch_read_only", candidates=list(cands))


def _active(dispatch_id="dispatch:shed:1") -> GpuPoolShedResultV1:
    return GpuPoolShedResultV1(ok=True, shed_id="oshed_1", dispatch_id=dispatch_id, state="active", ttl_sec=900,
                               started_at=NOW, valid_until=NOW + timedelta(seconds=900))


@pytest.mark.asyncio
async def test_treated_precommits_before_the_rpc_and_records_the_shed(monkeypatch):
    w = _worker(monkeypatch)
    order: list[str] = []
    w._store.insert_world_episode.side_effect = lambda row: order.append(f"episode:{row['arm']}") or True
    w._store.save_dispatch_result.side_effect = lambda **kw: order.append(f"result:{kw['result_json']['settlement'].get('shed_id')}")

    async def rpc(bus, req):
        order.append("rpc")
        assert req.reason == "orion_self_shed" and req.action == "set"
        assert req.correlation["open_loop_id"] == "open-loop-c"
        return _active()

    w._shed_rpc = rpc
    w._stable_draw = lambda dispatch_id: 0.99   # not held back
    out = await w._send_prepared_candidates(_frame(_shed_candidate()))
    assert order == ["episode:treated", "result:None", "rpc", "result:oshed_1"]   # precommit first
    final = w._store.save_dispatch_result.call_args.kwargs
    assert final["status"] == "pending" and final["result_json"]["settlement"]["state"] == SHED_PENDING
    assert final["latency_ms"] is not None and final["latency_ms"] < 5000     # motor cost = RPC wall time
    assert out.dispatched_candidates[0].dispatch_id == "dispatch:shed:1"
    assert not w.fake_bus.publish.await_count                                # outcome only at settlement


@pytest.mark.asyncio
async def test_holdback_writes_a_control_episode_and_sends_nothing(monkeypatch):
    w = _worker(monkeypatch)
    w._shed_rpc = AsyncMock(side_effect=AssertionError("a control must not shed"))
    w._stable_draw = lambda dispatch_id: 0.1    # < 0.5 -> held back
    out = await w._send_prepared_candidates(_frame(_shed_candidate()))
    row = w._store.insert_world_episode.call_args.args[0]
    assert row["arm"] == "control" and row["open_loop_id"] == "open-loop-c"
    assert row["eligibility"]["eligible"] is True
    assert row["scoring_due_at"] - row["decided_at"] == timedelta(seconds=1200)
    assert out.blocked_candidates[0].blocked_by == ["world_action_holdback"]
    assert not out.dispatched_candidates


@pytest.mark.asyncio
async def test_stale_eligibility_is_blocked_not_acted_on(monkeypatch):
    w = _worker(monkeypatch)
    w._shed_rpc = AsyncMock(side_effect=AssertionError("stale view must not act"))
    out = await w._send_prepared_candidates(_frame(_shed_candidate(evaluated_at=NOW - timedelta(minutes=10))))
    assert out.blocked_candidates[0].blocked_by == ["world_eligibility_stale"]
    w._store.insert_world_episode.assert_not_called()


@pytest.mark.asyncio
async def test_no_precommit_no_action(monkeypatch):
    w = _worker(monkeypatch)
    w._store.insert_world_episode.side_effect = RuntimeError("ledger down")
    w._shed_rpc = AsyncMock(side_effect=AssertionError("must not shed without a precommit"))
    w._stable_draw = lambda dispatch_id: 0.99
    out = await w._send_prepared_candidates(_frame(_shed_candidate()))
    assert "world_episode_precommit_failed" in out.dispatched_candidates[0].dispatch_error
    w._store.save_dispatch_result.assert_not_called()


@pytest.mark.asyncio
async def test_rpc_failure_keeps_the_row_pending_for_the_ledger_to_settle(monkeypatch):
    w = _worker(monkeypatch)
    w._shed_rpc = AsyncMock(side_effect=TimeoutError("no reply"))
    w._stable_draw = lambda dispatch_id: 0.99
    out = await w._send_prepared_candidates(_frame(_shed_candidate()))
    final = w._store.save_dispatch_result.call_args.kwargs["result_json"]["settlement"]
    assert final["state"] == SHED_PENDING and "rpc_error" in final
    assert out.dispatched_candidates[0].dispatch_error


@pytest.mark.asyncio
async def test_replay_sends_nothing_twice(monkeypatch):
    w = _worker(monkeypatch)
    w._store.load_dispatch_result_by_dispatch_id = MagicMock(return_value={"result_id": "result:dispatch:shed:1",
                                                                          "status": "pending", "result_json": {}})
    w._shed_rpc = AsyncMock(side_effect=AssertionError("replay must not resend"))
    w._stable_draw = lambda dispatch_id: 0.99
    await w._send_prepared_candidates(_frame(_shed_candidate()))
    w._store.insert_world_episode.assert_not_called()


@pytest.mark.asyncio
async def test_settlement_publishes_once_and_records_the_terminal(monkeypatch):
    w = _worker(monkeypatch)
    decided = NOW - timedelta(minutes=16)
    w._store.load_pending_shed_settlements = MagicMock(return_value=[{
        "result_id": "result:d", "dispatch_id": "d", "dispatch_kind": "self_regulate", "created_at": decided,
        "result_json": {"settlement": {"state": SHED_PENDING, "decided_at": decided.isoformat(), "ttl_sec": 900}},
        "pool_row": {"shed_id": "oshed_1", "state": "expired", "started_at": decided.isoformat(),
                     "ended_at": (decided + timedelta(seconds=900)).isoformat(), "drained_at": None,
                     "grants_withheld": 2, "delayed_grant_sec": 300.0}}])
    w._store.settle_shed_dispatch_result = MagicMock(return_value=True)
    assert await w._reconcile_shed_settlements(now=NOW) == 1
    kw = w._store.settle_shed_dispatch_result.call_args.kwargs
    assert kw["status"] == "success" and kw["result_json"]["settlement"]["terminal"] == "expired"
    assert kw["result_json"]["settlement"]["manipulation_check"]["drain"] == "none"
    assert w._store.record_world_settlement.call_args.kwargs["state"] == "expired"
    assert w.fake_bus.publish.await_count == 1


def test_settle_rules_terminal_states_and_orphan():
    decided = NOW - timedelta(minutes=30)
    pending = {"settlement": {"state": SHED_PENDING, "decided_at": decided.isoformat(), "ttl_sec": 900}}
    assert settle_shed_result(result_json=pending, pool_row={"state": "active"}, now=decided + timedelta(minutes=5)) is None
    for state, status in (("expired", "success"), ("cancelled", "empty"), ("preempted_by_reflex", "empty")):
        s = settle_shed_result(result_json=pending, pool_row={"state": state}, now=NOW)
        assert s.terminal == state and s.status == status
    s = settle_shed_result(result_json=pending, pool_row={"state": "refused", "refusal": "daily_cap"}, now=NOW)
    assert s.terminal == "refused:daily_cap" and not s.success
    orphan = settle_shed_result(result_json=pending, pool_row=None, now=decided + timedelta(seconds=1201))
    assert orphan.terminal == "settlement_timeout"
    assert settle_shed_result(result_json=pending, pool_row=None, now=decided + timedelta(seconds=1100)) is None


def test_the_draw_is_stable_per_decision_and_roughly_half():
    draws = [ExecutionDispatchRuntimeWorker._stable_draw(f"dispatch:{i}") for i in range(2000)]
    assert ExecutionDispatchRuntimeWorker._stable_draw("dispatch:7") == draws[7]
    assert 0.45 < sum(d < 0.5 for d in draws) / len(draws) < 0.55


@pytest.mark.asyncio
@pytest.mark.parametrize("inputs,refusal", [
    ((["dispatch:old"], []), "world_action_in_flight"),
    (([], [{"decided_at": NOW - timedelta(minutes=20), "settlement_state": "expired", "settlement": {}}]), "pool_gap"),
    (([], [{"decided_at": NOW - timedelta(hours=h), "settlement_state": "expired", "settlement": {}} for h in (2, 4, 6, 8)]),
     "pool_daily_cap"),
    (([], [{"decided_at": NOW - timedelta(minutes=40), "settlement_state": "refused:disabled", "settlement": {}}]),
     "pool_refusing:disabled"),
])
async def test_admission_is_checked_before_the_draw_so_neither_arm_is_recorded(monkeypatch, inputs, refusal):
    w = _worker(monkeypatch)
    w._store.load_world_admission_inputs = MagicMock(return_value=inputs)
    w._shed_rpc = AsyncMock(side_effect=AssertionError("must not act"))
    w._stable_draw = lambda dispatch_id: 0.1
    out = await w._send_prepared_candidates(_frame(_shed_candidate()))
    assert out.blocked_candidates[0].blocked_by == [refusal]
    w._store.insert_world_episode.assert_not_called()


@pytest.mark.asyncio
async def test_two_world_candidates_in_one_tick_only_one_decides(monkeypatch):
    w = _worker(monkeypatch)
    w._shed_rpc = AsyncMock(return_value=_active("dispatch:shed:1"))
    w._stable_draw = lambda dispatch_id: 0.99
    out = await w._send_prepared_candidates(_frame(_shed_candidate("dispatch:shed:1"), _shed_candidate("dispatch:shed:2")))
    assert [c.blocked_by for c in out.blocked_candidates] == [["world_action_in_flight"]]
    assert w._shed_rpc.await_count == 1


@pytest.mark.asyncio
async def test_replay_of_a_control_decision_never_sheds(monkeypatch):
    w = _worker(monkeypatch)
    w._store.insert_world_episode = MagicMock(return_value=False)
    w._store.world_episode_arm = MagicMock(return_value="control")
    w._shed_rpc = AsyncMock(side_effect=AssertionError("never shed on a control row"))
    w._stable_draw = lambda dispatch_id: 0.99
    out = await w._send_prepared_candidates(_frame(_shed_candidate()))
    assert out.dispatched_candidates[0].dispatch_error == "world_episode_recorded_as_control"
