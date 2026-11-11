"""Actual checkpoint waits/restarts around a single frozen resonance judgment."""
import asyncio
import json
import sys
from pathlib import Path
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
sys.path.insert(0, str(Path(__file__).parent))
from test_admitted_graph import World, CFG
from app.admitted_graph import AdmissionDeps
from app.reading_resonance_graph import build_reading_resonance_graph
from orion.world_pulse_read.tests.test_resonance_runtime import packet, answer, NOW
from orion.schemas.reading_resonance import ReadingResonanceRunBriefV1


def initial():
    return dict(run_id="resonance-packet-1", correlation_id="trace-1", workflow="reading.resonance", attempt=0,
                admission={"resource": "llm.route.agent"},
                brief=ReadingResonanceRunBriefV1(packet=packet()).model_dump(mode="json"))


def test_busy_hold_restart_and_persist_retry_never_regenerate():
    async def run():
        world, saver = World(), InMemorySaver()
        world.now = NOW
        calls, writes = [], []
        async def call(prompt, *, brief, run_id, correlation_id, gpu_lease):
            assert gpu_lease.model_dump() == world.ref()
            assert brief.packet == packet()
            calls.append(prompt)
            return {"text": json.dumps(answer()), "model": "fixture", "latency_ms": 17}
        async def persist(result):
            assert world.releases and all(r == "completed" for r in world.releases)
            writes.append(result)
            if len(writes) == 1:
                raise RuntimeError("database temporarily down")
        def graph():
            return build_reading_resonance_graph(call, persist, AdmissionDeps(
                world.register, world.lease, world.execute, world.release, world.event, now=lambda: world.now), saver)
        await graph().ainvoke(initial(), CFG)
        assert (await graph().aget_state(CFG)).next == ("resource_wait",)
        assert not calls
        await graph().ainvoke(Command(resume={}), CFG)
        assert not calls
        world.grant()
        with pytest.raises(RuntimeError, match="database temporarily down"):
            await graph().ainvoke(Command(resume={}), CFG)
        assert len(calls) == 1
        result = await graph().ainvoke(None, CFG)
        assert result["status"] == "completed" and result["persisted"]
        assert len(calls) == 1 and writes[0] == writes[1]
    asyncio.run(run())


@pytest.mark.parametrize("raw", ["not JSON", json.dumps(answer("none"))])
def test_rejected_or_none_answer_is_recorded_once(raw):
    async def run():
        world, saver = World(), InMemorySaver()
        world.now = NOW
        world.grant()
        calls, writes = [], []
        async def call(*args, **kw):
            calls.append(1)
            return {"text": raw}
        async def persist(r):
            writes.append(r)
        graph = build_reading_resonance_graph(call, persist, AdmissionDeps(
            world.register, world.lease, world.execute, world.release, world.event, now=lambda: world.now), saver)
        result = await graph.ainvoke(initial(), CFG)
        assert result["status"] == "completed" and len(calls) == len(writes) == 1
        assert writes[0].check.decision == "none"
    asyncio.run(run())


def test_gateway_uses_background_hold_and_no_recall(monkeypatch):
    from test_journal_compose_graph import _runner_with_reply
    runner, bus = _runner_with_reply({"content": json.dumps(answer()), "raw": {"model": "fixture"}})
    from orion.schemas.gpu_pool import GpuLeaseRefV1
    world = World()
    world.grant()
    result = asyncio.run(runner._call_reading_resonance_llm("Frozen prompt", brief=ReadingResonanceRunBriefV1(packet=packet()),
        run_id="resonance-1", correlation_id="trace-1", gpu_lease=GpuLeaseRefV1.model_validate(world.ref())))
    assert result["text"]
    request = bus.rpc_request.call_args.args[1]
    assert request.kind == "llm.chat.request"
    assert request.payload["options"]["gpu_lease"] == world.ref()
    assert request.payload["options"]["purpose"] == "reading_resonance"
    assert request.payload["options"]["chat_template_kwargs"]["enable_thinking"] is False
