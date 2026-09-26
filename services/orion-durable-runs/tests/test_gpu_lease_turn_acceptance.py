"""Stage 4.4 acceptance: one curiosity turn under a GPU pool hold, through the real consumers.

Durable-runs does not issue pool holds until 4.5, so this hands Hub the ``CuriosityTurnRequestV1``
4.5 will send (``gpu_lease``, no durable ``lease``) and runs it through the production Hub, Thought,
Governor, Cortex Exec and Gateway adapters (acceptance_turn.py). Evidence it proves:

* Hub fences the ref with the pool's ``status`` verb before any generation;
* every model call -- stance, FCC over HTTP, finalize reflect/repair -- carries the ref, and the
  Gateway places each one with ``attach`` (``hold=``), never a plain lease that would queue behind
  the run's own hold;
* a stale hold generation is refused at Hub, with no model call at all.

No Postgres: the durable broker is not involved in a pool-held turn.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi import FastAPI

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.durable_run import (
    CURIOSITY_TURN_REQUEST_KIND, CuriosityTurnRequestV1, CuriosityTurnResultV1,
)
from orion.schemas.gpu_pool import GpuLeaseRefV1

from .acceptance_bus import TypedBus
from .acceptance_turn import DRAFT, REPAIRED, SOURCE, build_turn_adapter

RUN = "acceptance-held-run"
HOLD = GpuLeaseRefV1(lease_id="hold-acceptance", generation=4, role="agent", holder=f"durable-runs:{RUN}")


def _turn(ref: GpuLeaseRefV1, reply_to: str) -> BaseEnvelope:
    request = CuriosityTurnRequestV1(
        run_id=RUN, correlation_id=str(uuid.uuid4()), prompt="Inspect the isolated fixture ledger.",
        timeout_sec=30, source_tag="curiosity_investigate", gpu_lease=ref)
    return BaseEnvelope(kind=CURIOSITY_TURN_REQUEST_KIND, source=SOURCE, reply_to=reply_to,
                        payload=request.model_dump(mode="json"))


@pytest.mark.parametrize("repair_required", [False, True], ids=["accepted-draft", "conditional-repair"])
def test_held_turn_attaches_every_call_to_the_hold(monkeypatch, repair_required):
    async def scenario():
        bus = TypedBus()
        adapter = build_turn_adapter(monkeypatch, bus, None, repair_required, authority_app=FastAPI())
        adapter.pool_holds[HOLD.lease_id] = HOLD
        try:
            await adapter.handle_turn(_turn(HOLD, "orion:curiosity:turn:reply:held"))
            await bus.drain()
            reply = [env for ch, env in bus.events if ch == "orion:curiosity:turn:reply:held"]
            result = CuriosityTurnResultV1.model_validate(reply[-1].payload)
            assert result.ok, result.error
            assert result.text == (REPAIRED if repair_required else DRAFT)
            expected = ["stance_react", "fcc_primary", "harness_finalize_reflect"]
            if repair_required:
                expected.append("orion_response_repair")
            assert [row["stage"] for row in adapter.stages] == expected
            for row in adapter.stages:
                assert (row["hold_lease_id"], row["hold_generation"]) == (HOLD.lease_id, HOLD.generation)
                assert row["hold_holder"] == HOLD.holder
                assert "lease_id" not in row  # no durable token rode this turn
            # Every pool placement was an attach under the hold: the run never queued behind itself.
            assert len(adapter.pool_grants) == len(expected)
            assert all(grant.get("hold") == HOLD for grant in adapter.pool_grants)
            # Hub's fence was the pool's status verb, over the bus.
            status_reads = [env for ch, env in bus.events
                            if ch == "orion:gpu_pool:lease:request" and env.payload.get("verb") == "status"]
            assert [env.payload["lease_id"] for env in status_reads] == [HOLD.lease_id]
        finally:
            await adapter.close()
            await bus.close()
    asyncio.run(scenario())


def test_stale_hold_generation_is_refused_before_any_model_call(monkeypatch):
    async def scenario():
        bus = TypedBus()
        adapter = build_turn_adapter(monkeypatch, bus, None, False, authority_app=FastAPI())
        adapter.pool_holds[HOLD.lease_id] = HOLD.model_copy(update={"generation": HOLD.generation + 1})
        try:
            await adapter.handle_turn(_turn(HOLD, "orion:curiosity:turn:reply:stale"))
            await bus.drain()
            reply = [env for ch, env in bus.events if ch == "orion:curiosity:turn:reply:stale"]
            result = CuriosityTurnResultV1.model_validate(reply[-1].payload)
            assert not result.ok and "gpu_lease_stale_generation" in (result.error or "")
            assert adapter.stages == [] and adapter.pool_grants == []
        finally:
            await adapter.close()
            await bus.close()
    asyncio.run(scenario())
