"""AI Town sessions (Juniper 2026-10-30): the motor allocator admits or refuses one town session.

Mirror of tests/test_rung3_retro_leisure.py (the same dispatch shape). Pins:
- the route declares the aitown.session `play` act with a dispatch object, the act's own duration
  as its cost, and the controller's 1800 s session cap as the cold ceiling; the template is a
  no-dimension express proposal;
- cold (no session ever closed): charged the ceiling, admitted on the allocator's cold-start rule;
  measured: charged the measured session time; unreadable: refused `no_cost_estimate`, never the ceiling;
- the kill switch and a held doer refusal are refused before the allocator, and the frame stamps
  them `allocator:<reason>`;
- an open play act holds the route (one session at a time);
- the start RPC: started -> result row with the session, no latency; refused -> held for the controller's
  retry_after_sec and the dispatch carries doer_refused; no answer -> held 300 s.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

REPO = Path(__file__).resolve().parents[1]
SVC = REPO / "services" / "orion-execution-dispatch-runtime"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SVC))

import app.worker as worker_mod  # noqa: E402
from app.aitown_session_dispatch import FAILURE_HOLD_SEC  # noqa: E402
from app.worker import ExecutionDispatchRuntimeWorker  # noqa: E402
from orion.execution_dispatch.policy import AITOWN_SESSION_EXECUTOR_VERB, load_execution_dispatch_policy  # noqa: E402
from orion.schemas.execution_dispatch_frame import ExecutionDispatchCandidateV1, ExecutionDispatchFrameV1  # noqa: E402
from orion.schemas.aitown_session import AITownSessionStartResultV1, aitown_session_object_ref  # noqa: E402

NOW = datetime(2026, 10, 22, 18, 0, tzinfo=timezone.utc)
DISPATCH = "dispatch:proposal:aitown_session:t1"
MOTOR = SimpleNamespace(remaining_sec=1e6)


def test_route_declares_the_play_act_and_its_cost_rule() -> None:
    policy = load_execution_dispatch_policy(REPO / "config/execution_dispatch/execution_dispatch_policy.v1.yaml")
    ref = policy.act_refs_by_cortex_verb()[AITOWN_SESSION_EXECUTOR_VERB]
    assert (ref.verb, ref.actor, ref.signal_id, ref.object_source, ref.cost_source) == (
        "play", "aitown.session", "artifact_produced", "dispatch", "act_duration")
    # horizon = the controller's 1800 s cap + 600 s for its save and export; ceiling = the cap.
    assert (ref.horizon_sec, ref.cold_cost_ceiling_sec) == (2400.0, 1800.0)
    route = policy.template_to_cortex["aitown_session"]
    assert (route.dispatch_kind, route.allowed_scope) == ("express", "express_bounded")


def test_template_is_a_no_dimension_express_proposal() -> None:
    from orion.proposals.policy import load_proposal_policy

    t = load_proposal_policy(REPO / "config/proposals/proposal_policy.v1.yaml").proposal_templates[
        "aitown_session"]
    assert (t.kind, t.target_id, t.dimensions) == ("express", "world:aitown", {})


def _cand(dispatch_id: str = DISPATCH) -> ExecutionDispatchCandidateV1:
    return ExecutionDispatchCandidateV1(
        dispatch_id=dispatch_id, source_decision_id=f"pd:{dispatch_id}", source_proposal_id=f"p:{dispatch_id}",
        dispatch_status="prepared_for_dispatch", dispatch_mode="dispatch_read_only", dispatch_kind="express",
        target_id="world:aitown", target_kind="system", cortex_verb=AITOWN_SESSION_EXECUTOR_VERB,
        cortex_mode="brain", risk_score=0.05, confidence_score=0.9)


def _frame(*cands) -> ExecutionDispatchFrameV1:
    return ExecutionDispatchFrameV1(
        frame_id="execution.dispatch.frame:test", source_policy_frame_id="policy.frame:test",
        source_proposal_frame_id="proposal.frame:test", generated_at=NOW,
        dispatch_mode="dispatch_read_only", candidates=list(cands))


def _worker(monkeypatch, *, enabled=True, measured=None):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = ExecutionDispatchRuntimeWorker()
    monkeypatch.setattr(worker._settings, "orion_dispatch_act_candidates_enabled", True, raising=False)
    monkeypatch.setattr(worker._settings, "orion_dispatch_aitown_session_enabled", enabled, raising=False)
    store = MagicMock()
    store.load_action_cost_estimates.return_value = {}
    store.load_effect_posteriors.return_value = {}
    store.act_duration_p50_sec.return_value = measured
    store.act_in_flight.return_value = False
    store.act_episode_in_flight.return_value = False
    store.act_control_stats.return_value = (0, 0)
    store.act_arm_stats.return_value = (0, 0)
    store.load_dispatch_result_by_dispatch_id.return_value = None
    worker._store = store
    return worker


def test_cold_session_is_charged_the_cap_and_admitted_cold(monkeypatch) -> None:
    worker = _worker(monkeypatch)
    out = worker._log_allocator_preview(_frame(_cand()), MOTOR)
    assert [c.dispatch_id for c in out.admitted] == [DISPATCH]
    assert out.admitted[0].cost_sec == 1800.0 and out.admitted[0].cold_start
    worker._store.act_duration_p50_sec.assert_called_once_with(
        actor="aitown.session", verb="play", signal_id="artifact_produced", lookback_hours=720)
    assert "act_expectation:play:aitown.session:artifact_produced" in worker._last_act_reasons[DISPATCH]


def test_measured_session_time_replaces_the_ceiling(monkeypatch) -> None:
    worker = _worker(monkeypatch, measured=1320.0)
    out = worker._log_allocator_preview(_frame(_cand()), MOTOR)
    assert out.admitted[0].cost_sec == 1320.0


def test_unreadable_cost_is_refused_not_charged_the_ceiling(monkeypatch) -> None:
    worker = _worker(monkeypatch)
    worker._store.act_duration_p50_sec.side_effect = RuntimeError("db down")
    out = worker._log_allocator_preview(_frame(_cand()), MOTOR)
    assert out.refusals_by_reason() == {"no_cost_estimate": 1}


def test_kill_switch_and_doer_hold_refuse_before_the_allocator(monkeypatch) -> None:
    off = _worker(monkeypatch, enabled=False)
    assert off._log_allocator_preview(_frame(_cand()), MOTOR).refusals_by_reason() == {"aitown_session_disabled": 1}
    held = _worker(monkeypatch)
    held._aitown_session_hold = (datetime.now(timezone.utc) + timedelta(seconds=300), "manual_session_open")
    assert held._log_allocator_preview(_frame(_cand()), MOTOR).refusals_by_reason() == {
        "doer_refused:manual_session_open": 1}


def test_an_open_play_act_holds_the_route(monkeypatch) -> None:
    worker = _worker(monkeypatch)
    worker._store.act_in_flight.return_value = True
    out = worker._log_allocator_preview(_frame(_cand()), MOTOR)
    assert out.refusals_by_reason() == {"act_in_flight": 1}
    assert worker._store.act_in_flight.call_args.kwargs["include_open_acts"] is True


def _send_worker(monkeypatch, *, reply=None, error=None, enabled=True):
    worker = _worker(monkeypatch, enabled=enabled)
    worker._derive_motor_budget = MagicMock(return_value=SimpleNamespace(
        remaining_sec=1e6, mode="enforcing", exhausted=False, spent_sec=0.0, allowance_sec=1e6, pace=1.0,
        projected_day_sec=0.0))
    worker._derive_daily_risk_cap = MagicMock(return_value=(1e9, {}))
    worker._store.sum_risk_dispatched_today.return_value = 0.0
    monkeypatch.setattr(worker._settings, "orion_dispatch_allocator_enforce", True, raising=False)
    fake_bus = MagicMock(connect=AsyncMock(), close=AsyncMock(), publish=AsyncMock())
    monkeypatch.setattr(worker_mod, "OrionBusAsync", MagicMock(return_value=fake_bus))
    monkeypatch.setattr(worker_mod.RPC_HEALTH_SINK, "absorb_bus", MagicMock())
    calls = []

    async def rpc(bus, req):
        calls.append(req)
        if error is not None:
            raise error
        return reply

    worker._aitown_session_rpc = rpc
    worker._record_dispatch_status = MagicMock()
    worker._store.insert_act_episode.return_value = True
    return worker, calls


def _reply(state, **kw):
    return AITownSessionStartResultV1(dispatch_id=DISPATCH, state=state, **kw)


@pytest.mark.asyncio
async def test_a_disabled_route_is_stamped_allocator_reason_on_the_frame(monkeypatch) -> None:
    worker, calls = _send_worker(monkeypatch, enabled=False)
    out = await worker._send_prepared_candidates(_frame(_cand()))
    assert calls == [] and out.dispatched_candidates == []
    reasons = [r for c in out.blocked_candidates for r in c.reasons]
    assert "allocator:aitown_session_disabled" in reasons


@pytest.mark.asyncio
async def test_a_started_session_saves_its_result_without_latency(monkeypatch) -> None:
    ref = aitown_session_object_ref(DISPATCH)
    worker, calls = _send_worker(monkeypatch, reply=_reply("started", session_id="s-1", object_ref=ref))
    out = await worker._send_prepared_candidates(_frame(_cand()))
    assert [c.dispatch_id for c in calls] == [DISPATCH]
    assert [c.dispatch_id for c in out.dispatched_candidates] == [DISPATCH]
    saved = worker._store.save_dispatch_result.call_args.kwargs
    assert saved["latency_ms"] is None
    assert saved["result_json"]["settlement"] | {"rpc_ms": 0} == {
        "state": "started", "kind": "aitown_session", "session_id": "s-1", "object_ref": ref,
        "refusal": None, "retry_after_sec": None, "rpc_ms": 0}
    assert getattr(worker, "_aitown_session_hold", None) is None


@pytest.mark.asyncio
async def test_a_refused_start_holds_the_route_for_the_doer_clock(monkeypatch) -> None:
    worker, _ = _send_worker(monkeypatch, reply=_reply("refused", refusal="daily_cap", retry_after_sec=600.0))
    out = await worker._send_prepared_candidates(_frame(_cand()))
    assert out.dispatched_candidates[0].dispatch_error == "doer_refused:daily_cap"
    until, why = worker._aitown_session_hold
    assert why == "daily_cap"
    assert 590 <= (until - datetime.now(timezone.utc)).total_seconds() <= 600
    # The next tick is refused before the allocator, with no RPC.
    assert worker._log_allocator_preview(_frame(_cand("d:2")), MOTOR).refusals_by_reason() == {
        "doer_refused:daily_cap": 1}


@pytest.mark.asyncio
async def test_no_answer_holds_the_route_five_minutes(monkeypatch) -> None:
    worker, _ = _send_worker(monkeypatch, error=TimeoutError("no reply"))
    out = await worker._send_prepared_candidates(_frame(_cand()))
    assert out.dispatched_candidates[0].dispatch_error.startswith("TimeoutError")
    until, why = worker._aitown_session_hold
    assert why == "start_failed"
    assert FAILURE_HOLD_SEC - 10 <= (until - datetime.now(timezone.utc)).total_seconds() <= FAILURE_HOLD_SEC
    assert worker._store.save_dispatch_result.call_args.kwargs["status"] == "failed"


def test_no_answer_still_names_the_proven_episode_object(monkeypatch) -> None:
    """The controller may have started the session: the deterministic object lets the act's close resolve it."""
    worker = _worker(monkeypatch)
    worker._settle_aitown_session_episode(DISPATCH, None)
    worker._store.set_act_episode_object.assert_called_once_with(DISPATCH, aitown_session_object_ref(DISPATCH))
    worker._store.void_act_episode.assert_not_called()


def test_a_started_session_holds_the_route_before_its_act_is_stored(monkeypatch) -> None:
    worker = _worker(monkeypatch)
    worker._log_allocator_preview(_frame(_cand()), MOTOR)
    assert worker._store.act_in_flight.call_args.kwargs["started_within_sec"] == 2400.0


def test_a_proven_decision_is_named_or_voided_by_the_doer(monkeypatch) -> None:
    worker = _worker(monkeypatch)
    ref = aitown_session_object_ref(DISPATCH)
    worker._settle_aitown_session_episode(DISPATCH, _reply("started", session_id="s", object_ref=ref))
    worker._store.set_act_episode_object.assert_called_once_with(DISPATCH, ref)
    worker._settle_aitown_session_episode(DISPATCH, _reply("refused", refusal="session_open"))
    worker._store.void_act_episode.assert_called_once_with(DISPATCH, "doer_refused:session_open")
