"""Stage 4.4: Hub accepts a durable run's GPU pool hold ref (``CuriosityTurnRequestV1.gpu_lease``).

Hub fences it with the pool's ``status`` verb (not durable-runs /leases/validate), runs the turn
under it (every LLM call attaches to the hold), and picks it up for Door-A outreach. Coexists with
the old ``lease`` until 4.6. Spec: docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from orion.gpu_pool.client import LeaseUnavailable
from orion.schemas.durable_run import CuriosityTurnRequestV1
from orion.schemas.gpu_pool import GpuLeaseRefV1
from scripts import curiosity_investigation as ci
from test_curiosity_admission import lease
from test_curiosity_investigation import _CortexBus, _loop


def ref(run="run-one", generation=1, role="agent-gpu2", holder=None):
    return GpuLeaseRefV1(lease_id=f"hold-{run}", generation=generation, role=role,
                         holder=holder or f"durable-runs:{run}")


def request(run="run-one", generation=1, **kw):
    return CuriosityTurnRequestV1(run_id=run, correlation_id=run, prompt="Study a real question", timeout_sec=42,
                                  gpu_lease=ref(run, generation), **kw)


def test_hold_ref_is_validated_with_the_pool_and_the_turn_runs_under_it(monkeypatch):
    validate = AsyncMock()
    monkeypatch.setattr(ci, "validate_hold_ref", validate)
    old = AsyncMock(side_effect=AssertionError("a pool ref must not hit durable-runs /leases/validate"))
    monkeypatch.setattr(ci, "validate_resource_lease", old)
    loop = _loop(_CortexBus(), kickoff_via_cortex=True)
    loop._generate = AsyncMock(return_value=("grounded finding", {}))
    result = asyncio.run(loop._turn_result_for(request(), hold_lock=False))
    assert result.ok
    args, kwargs = validate.await_args
    assert args[1] == ref() and kwargs["expected_holder"] == "durable-runs:run-one"
    gen = loop._generate.await_args.kwargs
    assert gen["gpu_lease"] == ref() and gen["timeout_sec"] == 42
    # The role (agent-gpu2) is not a route: FCC names the hold's work-class route.
    assert gen["fcc_model_label"] == "llamacpp/agent"
    assert "resource_lease" not in gen


def test_rejected_hold_refuses_the_turn_before_any_generation(monkeypatch):
    monkeypatch.setattr(ci, "validate_hold_ref", AsyncMock(side_effect=LeaseUnavailable("gpu_lease_stale_generation")))
    loop = _loop(_CortexBus(), kickoff_via_cortex=True)
    loop._generate = AsyncMock(side_effect=AssertionError("must not run"))
    with pytest.raises(ValueError, match="gpu_lease_stale_generation"):
        asyncio.run(loop._turn_result_for(request(), hold_lock=False))


def test_real_pool_status_refuses_another_runs_hold():
    """No mock on the validator: the holder check needs no pool round trip."""
    loop = _loop(_CortexBus(), kickoff_via_cortex=True)
    loop._generate = AsyncMock(side_effect=AssertionError("must not run"))
    bad = CuriosityTurnRequestV1(run_id="run-one", correlation_id="c", prompt="p", timeout_sec=42,
                                 gpu_lease=ref(holder="durable-runs:someone-else"))
    with pytest.raises(ValueError, match="holder_mismatch"):
        asyncio.run(loop._turn_result_for(bad, hold_lock=False))


def test_fence_rechecked_on_cache_and_new_generation_is_a_new_execution(monkeypatch):
    validate = AsyncMock()
    monkeypatch.setattr(ci, "validate_hold_ref", validate)
    loop = _loop(_CortexBus(), kickoff_via_cortex=True)
    loop._generate = AsyncMock(return_value=("grounded finding", {}))

    async def scenario():
        await loop._turn_result_for(request(), hold_lock=False)
        await loop._turn_result_for(request(), hold_lock=False)
        assert loop._generate.await_count == 1
        await loop._turn_result_for(request(generation=2), hold_lock=False)
        assert loop._generate.await_count == 2
    asyncio.run(scenario())
    assert validate.await_count == 3


def test_old_lease_and_new_ref_coexist(monkeypatch):
    monkeypatch.setattr(ci, "validate_hold_ref", AsyncMock())
    monkeypatch.setattr(ci, "validate_resource_lease", AsyncMock())
    loop = _loop(_CortexBus(), kickoff_via_cortex=True)
    loop._generate = AsyncMock(return_value=("grounded finding", {}))
    both = CuriosityTurnRequestV1(run_id="run-one", correlation_id="run-one", prompt="p", timeout_sec=42,
                                  assigned_lane="chat", lease=lease("run-one", "chat"), gpu_lease=ref())
    asyncio.run(loop._turn_result_for(both, hold_lock=False))
    gen = loop._generate.await_args.kwargs
    assert gen["fcc_model_label"] == "llamacpp/chat"  # the old lease's lane still wins until 4.6
    assert gen["resource_lease"].lane == "chat" and gen["gpu_lease"] == ref()


def test_generate_puts_the_ref_and_timeout_in_the_unified_turn_payload(monkeypatch):
    from orion.hub import turn_orchestrator
    captured = {}

    async def turn(**kwargs):
        captured.update(kwargs)
        return [{"type": "final", "llm_response": "grounded result", "harness_step_count": 14}]
    monkeypatch.setattr(turn_orchestrator, "execute_unified_turn", turn)
    loop = _loop(_CortexBus())
    del loop._generate
    text, _ = asyncio.run(loop._generate("prompt", "corr", fcc_model_label="llamacpp/agent", timeout_sec=42,
                                         gpu_lease=ref()))
    assert text == "grounded result"
    assert captured["payload"]["gpu_lease"] == ref().model_dump(mode="json")
    assert captured["payload"]["inference_timeout_sec"] == 42
    assert "resource_lease" not in captured["payload"]


def test_turn_request_schema_accepts_the_ref_consumer_first():
    wire = request().model_dump(mode="json")
    assert CuriosityTurnRequestV1.model_validate(wire).gpu_lease == ref()
    assert CuriosityTurnRequestV1(run_id="r", correlation_id="c", prompt="p", timeout_sec=1).gpu_lease is None


def test_door_a_outreach_composes_under_the_ref_and_releases(monkeypatch):
    loop = _loop(_CortexBus(), kickoff_via_cortex=True)
    inner = AsyncMock(return_value="sent")
    release = AsyncMock()
    monkeypatch.setattr(loop, "_maybe_reach_out_inner", inner)
    monkeypatch.setattr(loop, "_release_outreach_lease", release)
    outcome = ci.TurnOutcome(run_id="run-one", continue_line=False, continue_note="", reach_out=True, reach_out_why="x")
    asyncio.run(loop._maybe_reach_out(outcome=outcome, finding_text="f", run_id="run-one", gpu_lease=ref()))
    assert inner.await_args.kwargs["gpu_lease"] == ref()
    release.assert_awaited_once_with("run-one")


def test_completed_run_state_hands_the_door_a_ref_to_outreach():
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    bus = _CortexBus()
    loop = _loop(bus, kickoff_via_cortex=True)
    seen = {}

    async def fake_reach_out(**kwargs):
        seen.update(kwargs)
    loop._maybe_reach_out = fake_reach_out  # type: ignore[assignment]
    state = {"run_id": "run-one", "workflow": "curiosity.investigate", "thread_id": "run-one", "node": "finish",
             "status": "completed", "correlation_id": "c",
             "detail": {"reach_out": True, "reach_out_why": "worth it", "finding_text": "f",
                        "gpu_lease": ref().model_dump(mode="json")}}
    env = BaseEnvelope(kind="durable.run.state.v1", source=ServiceRef(name="t"), payload=state)
    asyncio.run(loop._handle_run_state({"data": bus.codec.encode(env)}))
    assert seen["gpu_lease"] == ref() and seen["resource_lease"] is None


class _Outreach:
    def blocked_reason(self, **_kw):
        return None


def _door_a_loop(monkeypatch):
    loop = _loop(_CortexBus(), kickoff_via_cortex=True)
    loop.outreach_enabled = True
    loop._outreach_provider = lambda: _Outreach()
    skips = []
    monkeypatch.setattr(loop, "_record_outreach_skip",
                        lambda outreach, reason, **kw: skips.append((reason, kw["run_id"])))
    return loop, skips


def test_door_a_refuses_a_hold_the_pool_no_longer_grants(monkeypatch):
    validate = AsyncMock(side_effect=LeaseUnavailable("gpu_lease_unknown_lease"))
    monkeypatch.setattr(ci, "validate_hold_ref", validate)
    loop, skips = _door_a_loop(monkeypatch)
    loop._generate = AsyncMock(side_effect=AssertionError("must not compose"))
    outcome = ci.TurnOutcome(run_id="run-one", continue_line=False, continue_note="", reach_out=True, reach_out_why="x")
    result = asyncio.run(loop._maybe_reach_out_inner(
        outcome=outcome, finding_text="f", run_id="run-one", hop_notes=[], line=ci.LINE_INVESTIGATE,
        resource_lease=None, correlation_id="c", gpu_lease=ref()))
    assert result == "gpu_lease_invalid" and skips == [("gpu_lease_invalid", "run-one")]
    assert validate.await_args.kwargs["expected_holder"] == "durable-runs:run-one"


def test_door_a_malformed_ref_skips_and_still_releases(monkeypatch):
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    bus = _CortexBus()
    loop = _loop(bus, kickoff_via_cortex=True)
    skips, released = [], []
    monkeypatch.setattr(loop, "_record_outreach_skip", lambda outreach, reason, **kw: skips.append(reason))

    async def release(run_id):
        released.append(run_id)
    monkeypatch.setattr(loop, "_release_outreach_lease", release)
    loop._maybe_reach_out = AsyncMock(side_effect=AssertionError("never compose without the ref"))
    state = {"run_id": "run-one", "workflow": "curiosity.investigate", "thread_id": "run-one", "node": "finish",
             "status": "completed", "correlation_id": "c",
             "detail": {"reach_out": True, "finding_text": "f", "gpu_lease": {"lease_id": "hold-1"}}}
    env = BaseEnvelope(kind="durable.run.state.v1", source=ServiceRef(name="t"), payload=state)
    asyncio.run(loop._handle_run_state({"data": bus.codec.encode(env)}))
    assert skips == ["gpu_lease_malformed"] and released == ["run-one"]
