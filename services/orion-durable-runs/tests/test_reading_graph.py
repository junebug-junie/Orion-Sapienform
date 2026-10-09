"""Reading uses real checkpointed admission, without nested model retries."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from test_admitted_graph import World, CFG, initial
from app.admitted_graph import AdmissionDeps, HoldLost
from app.graph import recorded_turn_correlation_id
from app.reading_graph import build_reading_graph, finish_detail
from orion.schemas.reading_turn import ReadingTurnResultV1


def graph(world, saver, turn):
    return build_reading_graph(
        turn,
        AdmissionDeps(
            world.register, world.lease, world.execute, world.release, world.event
        ),
        saver,
    )


def state():
    value = initial()
    value["brief"].update(seed_id="reading:seed", stage=1)
    return value


def test_wait_restart_preserves_hold_and_checkpoints_result():
    async def run():
        world, saver = World(), InMemorySaver()
        calls = []

        async def turn(request):
            calls.append(request)
            assert request.gpu_lease.model_dump() == world.ref()
            return ReadingTurnResultV1(
                run_id=request.run_id,
                correlation_id=request.correlation_id,
                ok=True,
                text="Grounded result",
                source_fetches=[],
            )

        await graph(world, saver, turn).ainvoke(state(), CFG)
        assert calls == []
        snapshot = await graph(world, saver, turn).aget_state(CFG)
        assert snapshot.next == ("resource_wait",)
        world.grant()
        result = await graph(world, saver, turn).ainvoke(Command(resume={}), CFG)
        assert len(calls) == 1
        assert result["status"] == "completed"
        assert world.releases == ["completed"]
        assert finish_detail(result)["reading_result"]["text"] == "Grounded result"
        assert recorded_turn_correlation_id(result) == calls[0].correlation_id

    asyncio.run(run())


def test_failed_model_turn_releases_and_records_actual_trace_without_retry():
    async def run():
        world = World()
        world.grant()
        calls = []

        async def turn(request):
            calls.append(request)
            return ReadingTurnResultV1(
                run_id=request.run_id,
                correlation_id=request.correlation_id,
                ok=False,
                error="fcc_stream_stalled",
            )

        result = await graph(world, InMemorySaver(), turn).ainvoke(state(), CFG)
        assert result["status"] == "failed" and len(calls) == 1
        assert world.releases == ["failed"]
        assert recorded_turn_correlation_id(result) == calls[0].correlation_id

    asyncio.run(run())


def test_lost_hold_requeues_without_model_call():
    async def run():
        world = World()
        world.grant()

        async def execute(state, node):
            world.pool_status = "queued"
            raise HoldLost("gpu_hold_lost")

        world.execute = execute

        async def turn(request):
            raise AssertionError("no inference before admission")

        result = await graph(world, InMemorySaver(), turn).ainvoke(state(), CFG)
        assert result["status"] == "waiting_resource"
        assert result.get("attempt", 0) == 0

    asyncio.run(run())
