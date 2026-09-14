"""The live Hub adapter preserves leases, timeouts and lane concurrency."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from orion.schemas.durable_run import CuriosityTurnRequestV1
from orion.schemas.resource_admission import ResourceLeaseV1
from scripts import curiosity_investigation as ci
from test_curiosity_investigation import _CortexBus, _loop


def lease(run="run-one", lane="agent", generation=1):
    now = datetime.now(timezone.utc)
    return ResourceLeaseV1(
        run_id=run, demand_id=f"{run}:turn", lease_id=f"lease-{run}", resource_key=f"llm.route.{lane}",
        lane=lane, backend_key=f"http://{lane}:8000", generation=generation,
        granted_at=now, heartbeat_at=now, expires_at=now + timedelta(seconds=60),
    )


def request(run="run-one", lane="agent", generation=1):
    return CuriosityTurnRequestV1(
        run_id=run, correlation_id=run, prompt="Study a real question", timeout_sec=42,
        assigned_lane=lane, fcc_model_label=f"orion-{lane}", lease=lease(run, lane, generation),
    )


@pytest.mark.parametrize("elastic", [False, True])
def test_admission_kickoff_declares_resource_and_keeps_ambiguous_run_queued(elastic):
    bus = _CortexBus(raise_on_rpc=True)
    loop = _loop(bus, kickoff_via_cortex=True, durable_admission_enabled=True, elastic_activation_enabled=elastic, llm_route="agent")
    assert asyncio.run(loop.tick()) is None
    assert not bus.journal
    assert bus.redis.values.get("orion:curiosity:last_investigation_at") is not None
    durable = bus.rpc_calls[0][1].payload["context"]["metadata"]["durable_run"]
    assert durable["admission"]["resource"] == "llm.route.agent"
    assert durable["admission"]["mode"] == "exclusive"
    assert durable["admission"]["allow_elastic_activation"] is elastic


def test_admitted_turns_use_assigned_route_and_do_not_hold_legacy_lock(monkeypatch):
    monkeypatch.setattr(ci, "validate_resource_lease", AsyncMock())

    async def scenario():
        loop = _loop(_CortexBus(), kickoff_via_cortex=True)
        entered = set()
        both = asyncio.Event()
        release = asyncio.Event()
        async def generate(prompt, corr, **kwargs):
            entered.add(corr)
            assert kwargs["timeout_sec"] == 42
            assert kwargs["fcc_model_label"] == f"llamacpp/{kwargs['resource_lease'].lane}"
            if len(entered) == 2:
                both.set()
            await release.wait()
            return "a grounded finding", {}
        loop._generate = generate
        await loop._run_lock.acquire()
        a = asyncio.create_task(loop._turn_result_for(request(), hold_lock=False))
        b = asyncio.create_task(loop._turn_result_for(request("run-two", "chat"), hold_lock=False))
        await asyncio.wait_for(both.wait(), 1)
        release.set()
        assert all(item.ok for item in await asyncio.gather(a, b))
        loop._run_lock.release()
    asyncio.run(scenario())


def test_fencing_rechecked_before_serving_cache_and_new_generation_is_new_execution(monkeypatch):
    validate = AsyncMock()
    monkeypatch.setattr(ci, "validate_resource_lease", validate)
    loop = _loop(_CortexBus(), kickoff_via_cortex=True)
    loop._generate = AsyncMock(return_value=("grounded finding", {}))
    async def scenario():
        await loop._turn_result_for(request(), hold_lock=False)
        await loop._turn_result_for(request(), hold_lock=False)
        assert loop._generate.await_count == 1
        await loop._turn_result_for(request(generation=2), hold_lock=False)
        assert loop._generate.await_count == 2
        validate.side_effect = RuntimeError("stale lease")
        with pytest.raises(RuntimeError, match="stale lease"):
            await loop._turn_result_for(request(generation=2), hold_lock=False)
    asyncio.run(scenario())
    assert validate.await_count == 4


def test_real_generate_places_request_lease_and_timeout_in_unified_payload(monkeypatch):
    from orion.hub import turn_orchestrator
    captured = {}
    async def turn(**kwargs):
        captured.update(kwargs)
        return [{"type": "final", "llm_response": "grounded result", "harness_step_count": 14}]
    monkeypatch.setattr(turn_orchestrator, "execute_unified_turn", turn)
    loop = _loop(_CortexBus())
    del loop._generate
    token = lease()
    text, _ = asyncio.run(loop._generate("prompt", "corr", fcc_model_label="orion-agent", timeout_sec=42, resource_lease=token))
    assert text == "grounded result"
    assert captured["payload"]["resource_lease"] == token.model_dump(mode="json")
    assert captured["payload"]["inference_timeout_sec"] == 42
    assert captured["payload"]["fcc_model_label"] == "orion-agent"
