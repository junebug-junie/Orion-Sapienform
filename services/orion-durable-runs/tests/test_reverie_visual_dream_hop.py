"""reverie.visual as a dream.carry image hop: the brief's dream_hop rides on every step request
(prepare/generate/caption/abandon) and the completed detail carries what was seen (``caption``).
A waking painting is unchanged: no dream_hop on the wire, no caption key in its detail."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))

from langgraph.checkpoint.memory import InMemorySaver

from test_admitted_graph import World
from test_reverie_visual_graph import CFG, Thought, graph, initial
from app.reverie_visual_graph import abandon_request, finish_detail
from orion.schemas.reverie_visual import VisualRunRequestV1
from orion.schemas.reverie_visual_run import (
    REVERIE_VISUAL_STEP_RESULT_KIND, DreamHopImageV1, ReverieVisualRunBriefV1, ReverieVisualStepRequestV1,
    ReverieVisualStepResultV1,
)

HOP = DreamHopImageV1(carry_run_id="dream-carry-abc", hop_index=3, prompt="a lighthouse made of moths")


def dream_initial(world):
    state = initial(world)
    brief = ReverieVisualRunBriefV1.model_validate(state["brief"])
    state["brief"] = brief.model_copy(update={"dream_hop": HOP}).model_dump(mode="json")
    return state


def test_dream_hop_rides_on_every_step_and_the_detail_carries_the_caption():
    async def run():
        world, saver = World(), InMemorySaver()
        thought = Thought(world, caption=[("done", {"caption": "Moths circle a white tower at dusk."})])
        world.grant()
        result = await graph(world, saver, thought).ainvoke(dream_initial(world), CFG)
        assert result["status"] == "completed"
        assert thought.steps() == ["prepare", "generate", "caption"]
        assert all(req.dream_hop == HOP for _, req in thought.calls)
        detail = finish_detail(result)
        assert detail["caption"] == "Moths circle a white tower at dusk."
        assert detail["outcome"] == "produced" and detail["artifact_sha256"] == "a" * 64
    asyncio.run(run())


def test_waking_painting_sends_no_dream_hop_and_has_no_caption_key():
    async def run():
        world, saver = World(), InMemorySaver()
        thought = Thought(world)
        world.grant()
        result = await graph(world, saver, thought).ainvoke(initial(world), CFG)
        assert result["status"] == "completed"
        assert all(req.dream_hop is None for _, req in thought.calls)
        assert "caption" not in finish_detail(result) and "caption" not in result
    asyncio.run(run())


def test_abandon_request_carries_the_dream_hop_and_waking_abandon_does_not():
    dream = abandon_request("run-1", {"dispatch_id": "dream-carry:x:3"}, None, HOP.model_dump(mode="json"))
    assert dream.dream_hop == HOP and dream.step == "abandon"
    assert abandon_request("run-1", {"dispatch_id": "d"}, "att-1").dream_hop is None


def _runner(holder):
    from app.runner import DurableRunner
    from app.settings import Settings
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
    from orion.core.bus.codec import OrionCodec

    codec = OrionCodec()

    async def rpc_request(channel, envelope, *, reply_channel, timeout_sec, **_):
        holder.append(envelope.payload)
        p = envelope.payload
        result = ReverieVisualStepResultV1(run_id=p["run_id"], correlation_id=p["correlation_id"], step=p["step"],
                                           status="retry", reason="busy")
        reply = BaseEnvelope(kind=REVERIE_VISUAL_STEP_RESULT_KIND, source=ServiceRef(name="orion-thought"),
                             correlation_id=envelope.correlation_id, payload=result.model_dump(mode="json"))
        return {"data": codec.encode(reply)}

    settings = Settings(_env_file=None, DURABLE_RUNS_GRAPH_HOST="", POSTGRES_URI="postgresql://unused",
                        ORION_BUS_ENABLED=False)
    return DurableRunner(settings, bus=SimpleNamespace(codec=codec, rpc_request=rpc_request), checkpointer=None)


def test_runner_omits_dream_hop_for_waking_steps_and_sends_it_for_dream_steps():
    """Rolling deploy: an orion-thought still on the pre-carry (extra="forbid") request schema must
    keep accepting waking steps, so their payload has no dream_hop key at all."""
    payloads = []
    runner = _runner(payloads)
    waking = ReverieVisualStepRequestV1(run_id="r", correlation_id="c-1", step="prepare",
                                        visual_request=VisualRunRequestV1(dispatch_id="d"))
    dream = waking.model_copy(update={"correlation_id": "c-2", "dream_hop": HOP})
    asyncio.run(runner._run_reverie_visual_step(waking))
    asyncio.run(runner._run_reverie_visual_step(dream))
    assert "dream_hop" not in payloads[0]
    assert payloads[1]["dream_hop"] == HOP.model_dump(mode="json")


def test_runtime_abandon_of_a_dream_child_names_its_dream_hop():
    """A dream child that ends failed is abandoned with its dream_hop, so thought closes the dream
    attempt (never a waking one)."""
    from app.admission_runtime import AdmissionRuntime

    recorded = []

    async def get_run(run_id):
        return {"terminal": None, "control": None}

    async def record_event(run_id, event, detail, event_id=None):
        recorded.append((event, detail))

    rt = object.__new__(AdmissionRuntime)
    rt.store = SimpleNamespace(get_run=get_run, record_event=record_event)
    brief = ReverieVisualRunBriefV1(visual_request=VisualRunRequestV1(dispatch_id="dream-carry:x:3"), dream_hop=HOP)
    entry = asyncio.run(rt._record_reverie_abandon("child-1", {"brief": brief.model_dump(mode="json"),
                                                               "last_error": "retry_window_expired"}))
    assert entry["dream_hop"] == HOP.model_dump(mode="json")
    req = abandon_request("child-1", entry["visual_request"], entry["attempt_id"], entry["dream_hop"])
    assert req.dream_hop == HOP


def test_a_carry_that_ends_failed_cancels_its_in_flight_child():
    from app.admission_runtime import AdmissionRuntime

    cancelled = []
    rt = object.__new__(AdmissionRuntime)

    async def terminal_detail(run_id):
        return None

    async def control(run_id, action):
        cancelled.append((run_id, action))

    rt.store = SimpleNamespace(terminal_detail=terminal_detail)
    rt.control = control
    asyncio.run(rt._cancel_carry_child("carry-1", "child-1"))
    assert cancelled == [("child-1", "cancel")]
