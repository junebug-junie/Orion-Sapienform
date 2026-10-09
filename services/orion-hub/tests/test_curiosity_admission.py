"""The live Hub adapter: admitted kickoff, held-turn timeouts/concurrency, Door-A release URL.

GPU pool stage 4.6 deleted the durable lease token (``CuriosityTurnRequestV1.lease``,
``validate_resource_lease``); an admitted turn now carries only the pool hold ref (``gpu_lease``).
The hold fencing itself is covered in test_curiosity_gpu_lease.py.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from orion.schemas.durable_run import CuriosityTurnRequestV1
from orion.schemas.gpu_pool import GpuLeaseRefV1
from scripts import curiosity_investigation as ci
from test_curiosity_investigation import _CortexBus, _loop


def request(run="run-one", generation=1):
    return CuriosityTurnRequestV1(
        run_id=run, correlation_id=run, prompt="Study a real question", timeout_sec=42,
        gpu_lease=GpuLeaseRefV1(lease_id=f"hold-{run}", generation=generation, role="agent",
                                holder=f"durable-runs:{run}"),
    )


def test_admission_kickoff_declares_resource_and_keeps_ambiguous_run_queued():
    bus = _CortexBus(raise_on_rpc=True)
    loop = _loop(bus, kickoff_via_cortex=True, durable_admission_enabled=True, llm_route="agent")
    assert asyncio.run(loop.tick()) is None
    assert not bus.journal
    assert bus.redis.values.get("orion:curiosity:last_investigation_at") is not None
    durable = bus.rpc_calls[0][1].payload["context"]["metadata"]["durable_run"]
    assert durable["admission"]["resource"] == "llm.route.agent"
    assert durable["admission"]["mode"] == "exclusive"
    # The broker-only lane-choice fields were deleted in stage 4.6; the pool places the run.
    for gone in ("allow_elastic_activation", "alternatives", "pinned_lane", "operator_override"):
        assert gone not in durable["admission"]


def test_held_turns_keep_their_timeout_and_do_not_hold_the_legacy_lock(monkeypatch):
    monkeypatch.setattr(ci, "validate_hold_ref", AsyncMock())

    async def scenario():
        loop = _loop(_CortexBus(), kickoff_via_cortex=True)
        entered = set()
        both = asyncio.Event()
        release = asyncio.Event()

        async def generate(prompt, corr, **kwargs):
            entered.add(corr)
            assert 41.0 < kwargs["timeout_sec"] <= 42  # counted from receipt
            assert "resource_lease" not in kwargs
            if len(entered) == 2:
                both.set()
            await release.wait()
            return "a grounded finding", {}
        loop._generate = generate
        await loop._run_lock.acquire()
        a = asyncio.create_task(loop._turn_result_for(request(), hold_lock=False))
        b = asyncio.create_task(loop._turn_result_for(request("run-two"), hold_lock=False))
        await asyncio.wait_for(both.wait(), 1)
        release.set()
        assert all(item.ok for item in await asyncio.gather(a, b))
        loop._run_lock.release()
    asyncio.run(scenario())


@pytest.mark.parametrize("base,expected", [
    ("http://127.0.0.1:8124", "http://127.0.0.1:8124/runs/run-one/release-outreach-lease"),
    ("http://127.0.0.1:8124/", "http://127.0.0.1:8124/runs/run-one/release-outreach-lease"),
    ("", ""),
])
def test_door_a_release_url_comes_from_the_durable_runs_base_url(base, expected):
    loop = _loop(_CortexBus(), kickoff_via_cortex=True, durable_runs_url=base)
    assert loop._outreach_lease_release_url("run-one") == expected


def test_door_a_release_url_default_matches_the_pre_4_6_derived_url():
    # Before 4.6 the URL was derived from HUB_CURIOSITY_LEASE_VALIDATION_URL's default
    # http://127.0.0.1:8124/leases/validate; the new default must land on the same endpoint.
    loop = _loop(_CortexBus(), kickoff_via_cortex=True)
    assert loop._outreach_lease_release_url("x") == "http://127.0.0.1:8124/runs/x/release-outreach-lease"


def test_main_wires_door_a_to_the_durable_runs_base_url():
    from pathlib import Path
    main = (Path(ci.__file__).parent / "main.py").read_text()
    assert "durable_runs_url=settings.HUB_CURIOSITY_DURABLE_RUNS_URL" in main


def test_deleted_lease_token_is_refused_on_the_wire():
    wire = request().model_dump(mode="json")
    wire["lease"] = {"lease_id": "old"}
    with pytest.raises(ValueError):
        CuriosityTurnRequestV1.model_validate(wire)
