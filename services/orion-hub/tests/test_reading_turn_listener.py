import asyncio
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.reading_turn import ReadingTurnRequestV1
from scripts.reading_turn_listener import ReadingTurnListener
from scripts.world_pulse_read_pipeline import WorldPulseReadPipeline
from scripts.world_pulse_read_stage2 import WorldPulseReadStage2Pipeline


def request():
    return ReadingTurnRequestV1(
        run_id="reading-test",
        correlation_id=str(uuid4()),
        brief={
            "seed_id": "reading:seed",
            "stage": 1,
            "prompt": "Read the source",
            "session_id": "reading",
            "timeout_sec": 10,
        },
        gpu_lease={
            "lease_id": "hold",
            "generation": 2,
            "role": "agent",
            "holder": "durable-runs:reading-test",
        },
    )


def test_rpc_validates_hold_and_deduplicates_read_only_turn(monkeypatch):
    validator = AsyncMock()
    monkeypatch.setitem(
        ReadingTurnListener.handle.__globals__, "validate_hold_ref", validator
    )
    turn = AsyncMock(
        return_value=[
            {
                "type": "final",
                "llm_response": "A grounded response",
                "harness_source_fetches": [],
            }
        ]
    )
    monkeypatch.setattr("orion.hub.turn_orchestrator.execute_unified_turn", turn)

    async def run():
        listener = ReadingTurnListener(ServiceRef(name="orion-hub"))
        listener.bus = AsyncMock()
        listener.rpc_bus = listener.bus
        req = request()
        envelope = BaseEnvelope(
            kind="reading.turn.request.v1",
            source=ServiceRef(name="orion-durable-runs"),
            correlation_id=req.correlation_id,
            reply_to=f"orion:reading:turn:reply:{req.correlation_id}",
            payload=req.model_dump(),
        )
        await asyncio.gather(listener.handle(envelope), listener.handle(envelope))
        assert validator.await_count == 2
        assert turn.await_count == 1
        kwargs = turn.call_args.kwargs
        assert kwargs["reading_only"] and kwargs["payload"]["no_write"]
        assert kwargs["payload"]["gpu_lease"] == req.gpu_lease.model_dump()
        assert kwargs["correlation_id"] == req.correlation_id
        assert listener.bus.publish.await_count == 2
        await listener.stop()

    asyncio.run(run())


def test_stale_hold_cannot_execute(monkeypatch):
    from orion.gpu_pool.client import LeaseUnavailable

    monkeypatch.setitem(
        ReadingTurnListener.handle.__globals__,
        "validate_hold_ref",
        AsyncMock(side_effect=LeaseUnavailable("stale_hold")),
    )

    async def run():
        listener = ReadingTurnListener(ServiceRef(name="orion-hub"))
        listener.bus = AsyncMock()
        listener._execute = AsyncMock()
        req = request()
        envelope = BaseEnvelope(
            kind="reading.turn.request.v1",
            source=ServiceRef(name="orion-durable-runs"),
            correlation_id=req.correlation_id,
            reply_to=f"orion:reading:turn:reply:{req.correlation_id}",
            payload=req.model_dump(),
        )
        await listener.handle(envelope)
        listener._execute.assert_not_called()
        response = listener.bus.publish.call_args.args[1].payload
        assert response["error"] == "turn_deferred:reading_admission:stale_hold"

    asyncio.run(run())


@pytest.mark.parametrize(
    "cls,stage", [(WorldPulseReadPipeline, 1), (WorldPulseReadStage2Pipeline, 2)]
)
def test_production_generate_only_binds_and_polls_durable_run(monkeypatch, cls, stage):
    from types import SimpleNamespace
    from orion.schemas.reading_turn import ReadingTurnResultV1
    from orion.world_pulse_read.durable import ReadingPending

    pipe = object.__new__(cls)
    pipe.session_id, pipe.timeout_sec = "reading", 900
    pipe._fcc_model_label, pipe.durable_url = "llamacpp/agent", "http://durable"
    conn = object()

    async def with_conn(callback):
        return await callback(conn)

    pipe._with_conn = with_conn
    bound = SimpleNamespace(run_id="immutable-run", brief=SimpleNamespace(prompt="Read source"))
    bind = AsyncMock(return_value=bound)
    poll = AsyncMock(side_effect=ReadingPending("waiting_resource"))
    monkeypatch.setitem(cls._generate.__globals__, "bind_turn", bind)
    monkeypatch.setitem(cls._generate.__globals__, "poll_turn", poll)

    async def run():
        with pytest.raises(ReadingPending):
            await pipe._generate("Read source", "requested-corr", seed_id="seed")
        assert bind.call_args.args[0] is conn
        assert bind.call_args.args[1].stage == stage
        assert bind.call_args.args[1].seed_id == "seed"
        poll.assert_awaited_once()
        assert poll.call_args.args == (bound, "http://durable")
        if stage == 1:  # stage 1 watches for work start (its `look` act opens then)
            assert callable(poll.call_args.kwargs["on_state"])
        poll.side_effect = None
        poll.return_value = ReadingTurnResultV1(
            run_id="immutable-run",
            correlation_id="actual-held-corr",
            ok=True,
            text="Grounded result",
            source_fetches=[],
        )
        result = await pipe._generate("Read source", "requested-corr", seed_id="seed")
        assert result.trace_id == "actual-held-corr"
        assert pipe._settlement_run_id == "immutable-run"

    asyncio.run(run())


def test_reading_turn_budget_is_the_briefs_wall_not_the_harness_default():
    """Seed c31227dc Stage 2 (2026-10-14): no budget on the payload -> motor ran on
    HARNESS_FCC_TIMEOUT_SEC=7200 inside a 3500 s durable attempt."""
    from scripts.reading_turn_listener import (
        READING_REPLY_RESERVE_SEC, REPLY_DEADLINE_PAYLOAD_KEY, stamp_reading_turn_budget,
    )
    from scripts.curiosity_investigation import REPLY_DEADLINE_PAYLOAD_KEY as CURIOSITY_KEY
    from orion.hub import turn_orchestrator

    assert REPLY_DEADLINE_PAYLOAD_KEY == CURIOSITY_KEY  # the key _held_turn_budgets reads
    payload = {}
    stamp_reading_turn_budget(payload, timeout_sec=3500.0, started=1000.0)
    assert payload["inference_timeout_sec"] == 3500.0 - READING_REPLY_RESERVE_SEC
    assert payload[REPLY_DEADLINE_PAYLOAD_KEY] == 4500.0
    # After 249 s of stance/recall the harness gets what is really left (10 s slack).
    monkey_now = 1249.0
    orig = turn_orchestrator.time.monotonic
    turn_orchestrator.time.monotonic = lambda: monkey_now
    try:
        inference, reply = turn_orchestrator._held_turn_budgets(payload)
    finally:
        turn_orchestrator.time.monotonic = orig
    assert reply == 3500.0 - 249 - 10
    assert reply - inference == READING_REPLY_RESERVE_SEC  # reserve survives slow stance
    # A caller that sets no reserve (curiosity held turns) is unchanged.
    no_reserve = {k: v for k, v in payload.items() if k != "harness_finalize_reserve_sec"}
    turn_orchestrator.time.monotonic = lambda: monkey_now
    try:
        inference2, reply2 = turn_orchestrator._held_turn_budgets(no_reserve)
    finally:
        turn_orchestrator.time.monotonic = orig
    assert inference2 == reply2
    small = {}
    stamp_reading_turn_budget(small, timeout_sec=10.0, started=0.0)
    assert small["inference_timeout_sec"] == 10.0


def test_listener_stamps_budget_on_the_turn_payload(monkeypatch):
    from scripts.reading_turn_listener import REPLY_DEADLINE_PAYLOAD_KEY

    monkeypatch.setitem(ReadingTurnListener.handle.__globals__, "validate_hold_ref", AsyncMock())
    turn = AsyncMock(return_value=[{"type": "final", "llm_response": "ok", "harness_source_fetches": []}])
    monkeypatch.setattr("orion.hub.turn_orchestrator.execute_unified_turn", turn)

    async def run():
        listener = ReadingTurnListener(ServiceRef(name="orion-hub"))
        listener.bus = AsyncMock()
        listener.rpc_bus = listener.bus
        req = request()
        envelope = BaseEnvelope(
            kind="reading.turn.request.v1", source=ServiceRef(name="orion-durable-runs"),
            correlation_id=req.correlation_id,
            reply_to=f"orion:reading:turn:reply:{req.correlation_id}", payload=req.model_dump(),
        )
        await listener.handle(envelope)
        payload = turn.call_args.kwargs["payload"]
        assert payload["inference_timeout_sec"] == 10.0
        assert REPLY_DEADLINE_PAYLOAD_KEY in payload
        await listener.stop()

    asyncio.run(run())
