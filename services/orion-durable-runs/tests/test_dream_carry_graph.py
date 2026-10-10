"""dream.carry on real checkpointed admission: text hops under the run's LLM hold, image hops as
child reverie.visual runs (nothing held while waiting), checkpoint-per-hop replay, retries that never
spend attempts, and a deadline that finishes partial instead of failing empty."""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from test_admitted_graph import World
from app.admission_runtime import TIMED_WAIT_NODES, WORK_NODES, AdmissionRuntime
from app.admitted_graph import AdmissionDeps
from app.dream_carry_graph import (
    CHILD_POLL_SEC, DreamCarryDeps, build_dream_carry_graph, finish_detail, terminal_detail,
)
from orion.schemas.dream_carry import (
    DREAM_CARRY_WORKFLOW, IMAGE_PROMPT_MAX_WORDS, DreamCarryBriefV1, DreamCarryStepResultV1,
    dream_hop_dispatch_id,
)
from orion.schemas.reverie_visual_run import REVERIE_VISUAL_WORKFLOW, reverie_visual_run_id

RUN = "dream-carry-001"
CFG = {"configurable": {"thread_id": RUN}}
SHA = "b" * 64


class Crash(BaseException):
    """Process death mid-node: not an Exception, so no node turns it into a retry."""


class CarryWorld(World):
    """The test_admitted_graph pool, but every fresh hold request is granted at once (a text hop
    per even hop index needs a new hold each time)."""

    def __init__(self, auto_grant=True):
        super().__init__()
        self.auto_grant = auto_grant

    async def register(self, state):
        update = await super().register(state)
        if self.auto_grant:
            self.grant()
        return update


class Dream:
    """orion-dream's step executor. ``script[step]`` is a list of (status, extra) answers consumed in
    order; the last one repeats. A BaseException status is raised."""

    def __init__(self, **script):
        self.script = {"text": [("done", {})], "finish": [("done", {})], **script}
        self.calls = []   # requests

    def steps(self, name=None):
        return [(r.step, r.hop_index) for r in self.calls if name is None or r.step == name]

    async def __call__(self, req, budget_sec=None):
        self.calls.append(req)
        answers = self.script[req.step]
        status, extra = answers.pop(0) if len(answers) > 1 else answers[0]
        if isinstance(status, BaseException):
            raise status
        base = {"run_id": req.run_id, "correlation_id": req.correlation_id, "step": req.step, "status": status}
        if status == "done" and req.step == "text":
            seen = req.hops[-1].caption if req.hops else "the sleep"
            base["hop"] = {"index": req.hop_index, "kind": "text", "passage": f"passage {req.hop_index} from {seen}",
                           "image_prompt": f"prompt {req.hop_index} " + "word " * 80, "elapsed_sec": 4.0}
        if status == "done" and req.step == "finish":
            base["dream_id"] = "dream-42" if req.hops else f"story-fallback:{req.brief.trigger_id}"
        return DreamCarryStepResultV1(**{**base, **extra})


class Children:
    """The child reverie.visual runs: submissions recorded; ``outcome[hop]`` decides the terminal
    (None = still running). Default: completed, produced, with a caption."""

    def __init__(self, **outcome):
        self.submitted = []   # DurableRunRequestV1
        self.outcome = outcome   # f"h{hop}" -> terminal, or a list of terminals by attempt (last repeats)
        self.reads = []
        self.crash_reads = 0
        self.crash_submits = 0   # die right after a submit lands (before the carry checkpoints it)
        self.clock = None        # with ready_at: a child reads as still running until clock() >= ready_at
        self.ready_at = None

    def hop_of(self, run_id):
        for req in self.submitted:
            if req.run_id == run_id:
                return req.brief.dream_hop.hop_index
        raise KeyError(run_id)

    def attempt_of(self, run_id):
        for req in self.submitted:
            if req.run_id == run_id:
                tail = req.brief.visual_request.dispatch_id.rsplit(":", 1)[-1]
                return int(tail[1:]) if tail.startswith("r") else 0
        raise KeyError(run_id)

    async def submit(self, request):
        self.submitted.append(request)
        if self.crash_submits:
            self.crash_submits -= 1
            raise Crash()
        return {"run_id": request.run_id}

    async def terminal(self, run_id):
        self.reads.append(run_id)
        if self.crash_reads:
            self.crash_reads -= 1
            raise Crash()
        hop = self.hop_of(run_id)
        if self.ready_at is not None and self.clock() < self.ready_at:
            return None
        default = ("completed", {"outcome": "produced", "artifact_sha256": SHA, "caption": f"seen {hop}",
                                 "visual_elapsed_sec": 50.0})
        got = self.outcome.get(f"h{hop}", default)
        if isinstance(got, list):
            got = got[min(self.attempt_of(run_id), len(got) - 1)]
        return got


def initial(world, hops=6, deadline=timedelta(hours=4)):
    brief = DreamCarryBriefV1(trigger_id="sleep-1", hops=hops, timeout_sec=5.0)
    return {"run_id": RUN, "correlation_id": "trace-dream", "attempt": 0, "workflow": DREAM_CARRY_WORKFLOW,
            "admission": {"resource": "llm.route.metacog_background", "preferred_lane": "metacog_background",
                          "deadline_at": (world.now + deadline).isoformat()},
            "brief": brief.model_dump(mode="json")}


def graph(world, saver, dream, children, grace=1800.0, child_max_attempts=3, child_min_window_sec=900.0):
    return build_dream_carry_graph(
        DreamCarryDeps(run_step=dream, submit_child=children.submit, child_terminal=children.terminal,
                       finish_grace_sec=grace, child_max_attempts=child_max_attempts,
                       child_min_window_sec=child_min_window_sec),
        AdmissionDeps(world.register, world.lease, world.execute, world.release, world.event,
                      now=lambda: world.now, max_attempts=1, retry_base_seconds=30.0, retry_max_seconds=300.0,
                      guard=world.guard), saver)


async def drive(g, world, first=None, limit=600, on_wait=None):
    """The admission driver in miniature: run, and at each interrupt advance the clock to the wake
    time (retry_at) or grant the hold, then resume. Returns the final state."""
    await g.ainvoke(first, CFG) if first is not None else None
    for _ in range(limit):
        snap = await g.aget_state(CFG)
        if not snap.next:
            return dict(snap.values)
        if on_wait is not None:
            on_wait(snap)
        node = snap.next[0]
        if node in TIMED_WAIT_NODES and snap.values.get("retry_at"):
            at = datetime.fromisoformat(snap.values["retry_at"])
            world.now = max(world.now, at)
        elif node == "resource_wait":
            world.grant()
        await g.ainvoke(Command(resume=True), CFG)
    raise AssertionError("carry did not finish")


def test_six_hops_alternate_text_and_image_and_finish_gets_all_six():
    async def run():
        world, saver = CarryWorld(), InMemorySaver()
        dream, children = Dream(), Children()
        g = graph(world, saver, dream, children)
        result = await drive(g, world, initial(world))
        assert result["status"] == "completed"
        hops = result["hops"]
        assert [h["kind"] for h in hops] == ["text", "image"] * 3
        assert [h["index"] for h in hops] == list(range(6))
        # Each text hop ran under the run's own hold, which was handed back right after.
        texts = [r for r in dream.calls if r.step == "text"]
        assert [r.hop_index for r in texts] == [0, 2, 4]
        assert all(r.gpu_lease is not None for r in texts)
        assert world.releases.count("text_done") == 3
        # T2 dreams from what I1's painting turned out to show.
        assert texts[1].hops[-1].caption == "seen 1" and hops[2]["passage"] == "passage 2 from seen 1"
        # The image prompt is clipped to what CLIP reads.
        assert len(hops[0]["image_prompt"].split()) == IMAGE_PROMPT_MAX_WORDS
        # Image hops: one child reverie.visual run each, deterministic per (carry, hop), painting the
        # previous text hop's prompt, on a diffusion hold, never past the carry's deadline.
        assert [c.brief.dream_hop.hop_index for c in children.submitted] == [1, 3, 5]
        for child, hop in zip(children.submitted, (1, 3, 5)):
            dispatch = dream_hop_dispatch_id(RUN, hop)
            assert child.run_id == reverie_visual_run_id(dispatch) and child.workflow == REVERIE_VISUAL_WORKFLOW
            assert child.brief.visual_request.dispatch_id == dispatch
            assert child.brief.dream_hop.carry_run_id == RUN
            assert child.brief.dream_hop.prompt == hops[hop - 1]["image_prompt"]
            assert child.admission.resource == "service.route.diffusion"
            assert child.admission.preferred_lane == "diffusion"
            assert child.admission.deadline_at <= datetime.fromisoformat(initial(world)["admission"]["deadline_at"])
        assert hops[1]["sha256"] == SHA and hops[1]["caption"] == "seen 1"
        assert hops[1]["child_run_id"] == children.submitted[0].run_id and hops[1]["elapsed_sec"] == 50.0
        # finish got every hop, and no stop reason.
        [fin] = [r for r in dream.calls if r.step == "finish"]
        assert [h.kind for h in fin.hops] == ["text", "image"] * 3 and fin.stopped_reason is None
        assert fin.gpu_lease is None
        detail = finish_detail(result)
        assert detail["hops_made"] == 6 and detail["stopped_reason"] is None
        assert detail["dream_id"] == "dream-42"
        assert detail["child_run_ids"] == [c.run_id for c in children.submitted]
    asyncio.run(run())


def test_crash_after_an_image_hop_resumes_without_resubmitting_or_redoing_a_text_hop():
    async def run():
        world, saver = CarryWorld(), InMemorySaver()
        dream = Dream(text=[("done", {}), (Crash(), {}), ("done", {})])
        children = Children()
        g = graph(world, saver, dream, children)
        with pytest.raises(Crash):
            await drive(g, world, initial(world))
        snap = await g.aget_state(CFG)
        # Died inside T2, after I1 was checkpointed, under T2's hold.
        assert snap.next == ("text_hop",) and [h["kind"] for h in snap.values["hops"]] == ["text", "image"]
        assert snap.values["lease"] == world.ref()
        rt = object.__new__(AdmissionRuntime)
        await rt._recover(g, CFG, DREAM_CARRY_WORKFLOW, dict(snap.values), snap)   # the restart fence
        snap = await g.aget_state(CFG)
        assert snap.next == ("resource_wait",)
        restarted = graph(world, saver, dream, children)
        result = await drive(restarted, world)
        assert result["status"] == "completed" and len(result["hops"]) == 6
        assert [r.hop_index for r in dream.calls if r.step == "text"] == [0, 2, 2, 4]   # T0 never redone
        assert [c.brief.dream_hop.hop_index for c in children.submitted] == [1, 3, 5]   # I1 never resubmitted
    asyncio.run(run())


def test_crash_while_waiting_on_a_child_rereads_it_and_never_resubmits():
    async def run():
        world, saver = CarryWorld(), InMemorySaver()
        dream, children = Dream(), Children()
        children.crash_reads = 1
        g = graph(world, saver, dream, children)
        with pytest.raises(Crash):
            await drive(g, world, initial(world))
        snap = await g.aget_state(CFG)
        assert snap.next == ("image_wait",) and snap.values["child_run_id"] == children.submitted[0].run_id
        assert snap.values["lease"] is None and snap.values["hold"] is None   # nothing held while waiting
        result = await drive(graph(world, saver, dream, children), world)
        assert result["status"] == "completed"
        assert [c.brief.dream_hop.hop_index for c in children.submitted] == [1, 3, 5]
    asyncio.run(run())


def test_image_wait_polls_a_running_child_holding_nothing():
    async def run():
        world, saver = CarryWorld(), InMemorySaver()
        dream, children = Dream(), Children(h1=None)
        g = graph(world, saver, dream, children)
        await g.ainvoke(initial(world), CFG)
        snap = await g.aget_state(CFG)
        assert snap.next == ("image_wait",) and snap.tasks[0].interrupts
        assert snap.values["retry_at"] == (world.now + timedelta(seconds=CHILD_POLL_SEC)).isoformat()
        world.now += timedelta(seconds=CHILD_POLL_SEC)
        await g.ainvoke(Command(resume=True), CFG)
        snap = await g.aget_state(CFG)
        assert snap.next == ("image_wait",) and len(children.reads) == 1 and len(snap.values["hops"]) == 1
        del children.outcome["h1"]   # the painting finished
        result = await drive(g, world)
        assert result["status"] == "completed" and len(children.submitted) == 3
    asyncio.run(run())


def test_text_retry_backs_off_and_never_spends_an_attempt():
    async def run():
        world, saver = CarryWorld(), InMemorySaver()
        dream = Dream(text=[("done", {}), ("retry", {"reason": "gateway_refused:thermal"}), ("done", {})])
        children = Children()
        g = graph(world, saver, dream, children)
        seen = []

        def watch(snap):
            if snap.next == ("retry_wait",):
                seen.append(dict(snap.values))

        result = await drive(g, world, initial(world), on_wait=watch)
        assert result["status"] == "completed" and len(result["hops"]) == 6
        [waiting] = seen
        assert waiting["attempt"] == 0 and waiting["retries"] == 1 and waiting["retry_node"] == "resource_request"
        assert waiting["hold"] is None and waiting["reason"] == "gateway_refused:thermal"
        assert "step_retry" in world.releases
        assert result["attempt"] == 0   # max_attempts=1: a spent attempt would have failed the run
        assert [r.hop_index for r in dream.calls if r.step == "text"] == [0, 2, 2, 4]
    asyncio.run(run())


def test_child_failure_finishes_partial_with_the_reason():
    async def run():
        world, saver = CarryWorld(), InMemorySaver()
        dream = Dream()
        children = Children(h3=("failed", {"error": "checkpoint_resume_failed: boom", "reason": "thermal_refused"}))
        g = graph(world, saver, dream, children)
        result = await drive(g, world, initial(world))
        assert result["status"] == "completed"
        assert [h["kind"] for h in result["hops"]] == ["text", "image", "text"]
        [fin] = [r for r in dream.calls if r.step == "finish"]
        reason = "image hop 3: checkpoint_resume_failed: boom"
        assert len(fin.hops) == 3 and fin.stopped_reason == reason   # not retryable: no second child
        detail = finish_detail(result)
        assert detail["hops_made"] == 3 and detail["stopped_reason"] == reason
        assert len(children.submitted) == 2
        assert [r.hop_index for r in dream.calls if r.step == "text"] == [0, 2]   # no hop after the stop
    asyncio.run(run())


def test_a_child_that_completed_without_a_picture_stops_the_carry():
    async def run():
        world, saver = CarryWorld(), InMemorySaver()
        dream = Dream()
        children = Children(h1=("completed", {"outcome": "already_satisfied", "reason": "dispatch_request_mismatch"}))
        result = await drive(graph(world, saver, dream, children), world, initial(world))
        assert result["status"] == "completed" and len(result["hops"]) == 1
        assert result["stopped_reason"] == "image hop 1: outcome:already_satisfied (dispatch_request_mismatch)"
        assert len(children.submitted) == 1   # not a deferral: no fresh child
    asyncio.run(run())


def test_deadline_finishes_partial_naming_the_hop_and_the_last_reason():
    async def run():
        world, saver = CarryWorld(), InMemorySaver()
        dream = Dream(text=[("done", {}), ("retry", {"reason": "gateway_refused:thermal", "retry_after_sec": 3600})])
        children = Children()
        g = graph(world, saver, dream, children)
        deadline = world.now + timedelta(hours=2)
        result = await drive(g, world, initial(world, deadline=timedelta(hours=2)))
        assert result["status"] == "completed"
        assert [h["kind"] for h in result["hops"]] == ["text", "image"]
        assert result["stopped_reason"] == "deadline at hop 2: gateway_refused:thermal"
        [fin] = [r for r in dream.calls if r.step == "finish"]
        assert len(fin.hops) == 2 and fin.stopped_reason.startswith("deadline at hop 2")
        # The retry never slept past the deadline: it woke at it and stopped.
        assert world.now == deadline
    asyncio.run(run())


def test_deadline_while_queued_for_the_hold_finishes_partial():
    async def run():
        world, saver = CarryWorld(auto_grant=False), InMemorySaver()
        dream, children = Dream(), Children()
        g = graph(world, saver, dream, children)
        world.grant()
        start = world.now
        await g.ainvoke(initial(world, deadline=timedelta(hours=1)), CFG)   # T0 under the first grant
        for _ in range(10):   # I1 completes; T2 then queues for a hold that never comes
            snap = await g.aget_state(CFG)
            if snap.next == ("resource_wait",):
                break
            world.now = max(world.now, datetime.fromisoformat(snap.values["retry_at"]))
            await g.ainvoke(Command(resume=True), CFG)
        assert snap.next == ("resource_wait",) and len(snap.values["hops"]) == 2
        world.now = start + timedelta(minutes=70)   # past the deadline, inside the finish grace
        result = await g.ainvoke(Command(resume=True), CFG)
        assert result["status"] == "completed"
        assert len(result["hops"]) == 2 and result["stopped_reason"] == "deadline at hop 2: waiting for the LLM hold"
        assert world.releases[-1] == "workflow_deadline"   # the queued hold is handed back
    asyncio.run(run())


def test_finish_retries_past_the_deadline_within_the_grace():
    async def run():
        world, saver = CarryWorld(), InMemorySaver()
        dream = Dream(finish=[("retry", {"reason": "sql_writer_busy", "retry_after_sec": 600}),
                              ("retry", {"reason": "sql_writer_busy", "retry_after_sec": 600}), ("done", {})])
        children = Children(h3=None)   # I3 never finishes: the deadline stops the carry
        g = graph(world, saver, dream, children)
        start = world.now
        result = await drive(g, world, initial(world, deadline=timedelta(hours=1)))
        assert result["status"] == "completed" and result["dream_id"] == "dream-42"
        assert result["stopped_reason"].startswith("deadline at hop 3")
        assert [r.step for r in dream.calls].count("finish") == 3
        assert world.now > start + timedelta(hours=1)   # finish kept going past the deadline
    asyncio.run(run())


def test_finish_past_the_grace_fails_with_the_last_finish_error():
    async def run():
        world, saver = CarryWorld(), InMemorySaver()
        dream = Dream(finish=[("retry", {"reason": "sql_writer_busy", "retry_after_sec": 600})])
        children = Children(h1=None)
        g = graph(world, saver, dream, children, grace=1800.0)
        result = await drive(g, world, initial(world, deadline=timedelta(hours=1)))
        assert result["status"] == "failed"
        assert result["last_error"] == "finish_grace_expired: finish:sql_writer_busy"
        # 600 s retries; the one due at deadline + grace fails without another call.
        assert [r.step for r in dream.calls].count("finish") == 3   # at 1h, 1h10, 1h20; 1h30 is the bound
        detail = terminal_detail(result, "failed")
        assert detail["hops_made"] == 1 and detail["error"].startswith("finish_grace_expired")
    asyncio.run(run())


def test_a_carry_that_made_no_hop_still_calls_finish_and_gets_the_story_fallback():
    async def run():
        world, saver = CarryWorld(auto_grant=False), InMemorySaver()
        dream, children = Dream(), Children()
        g = graph(world, saver, dream, children)
        await g.ainvoke(initial(world, deadline=timedelta(minutes=5)), CFG)
        world.now += timedelta(minutes=10)
        result = await g.ainvoke(Command(resume=True), CFG)
        assert result["status"] == "completed" and result["dream_id"] == "story-fallback:sleep-1"
        [fin] = dream.calls
        assert fin.step == "finish" and fin.hops == []
        assert fin.stopped_reason == "deadline at hop 0: waiting for the LLM hold"
        detail = finish_detail(result)
        assert detail["hops_made"] == 0 and detail["dream_id"] == "story-fallback:sleep-1"
        assert detail["stopped_reason"] == fin.stopped_reason
    asyncio.run(run())


def test_a_hand_started_carry_with_no_hop_fails_on_finish_terminal():
    async def run():
        world, saver = CarryWorld(auto_grant=False), InMemorySaver()
        dream = Dream(finish=[("terminal", {"reason": "no_sleep_to_fall_back_to"})])
        g = graph(world, saver, dream, Children())
        await g.ainvoke(initial(world, deadline=timedelta(minutes=5)), CFG)
        world.now += timedelta(minutes=10)
        result = await g.ainvoke(Command(resume=True), CFG)
        assert result["status"] == "failed" and result["last_error"] == "finish:no_sleep_to_fall_back_to"
        assert terminal_detail(result, "failed")["stopped_reason"].startswith("deadline at hop 0")
    asyncio.run(run())


def test_text_terminal_stops_and_finishes_with_what_was_made():
    async def run():
        world, saver = CarryWorld(), InMemorySaver()
        dream = Dream(text=[("done", {}), ("terminal", {"reason": "sleep_material_gone"})])
        children = Children()
        result = await drive(graph(world, saver, dream, children), world, initial(world))
        assert result["status"] == "completed" and len(result["hops"]) == 2
        assert result["stopped_reason"] == "text hop 2: sleep_material_gone"
        assert world.releases.count("terminal") == 1
    asyncio.run(run())


# --- runtime registration ----------------------------------------------------------------------

def _runtime():
    class FakeRunner:
        _checkpointer = InMemorySaver()
        _bus = None

        def _curiosity_deps(self):
            from app.graph import Deps
            return Deps(None, None, None, None)

        def _self_sense_deps(self):
            from app.self_sense_graph import Deps
            return Deps(run_turn=None, publish_rows=None)

        def _reflect_deps(self):
            from app.reflect_graph import Deps
            return Deps(call_reflect_llm=None)

    settings = SimpleNamespace(service_name="orion-durable-runs", lease_heartbeat_sec=15, admission_tick_sec=1,
                               retry_max_attempts=1, hold_max_takebacks=12, retry_base_sec=1, retry_max_sec=300,
                               state_channel="orion:durable:state", dream_carry_finish_grace_sec=900.0)
    return AdmissionRuntime(settings, FakeRunner(), pool=None, store=SimpleNamespace(),
                            holds=SimpleNamespace(hold_ttl_sec=90))


def test_runtime_routes_dream_carry_to_its_own_graph_and_details():
    rt = _runtime()
    g = rt._graph_for(DREAM_CARRY_WORKFLOW)
    assert g is rt.graphs[DREAM_CARRY_WORKFLOW] and g is not rt._graph_for("curiosity.investigate")
    assert {"text_hop", "image_wait", "finish_dream"} <= set(g.get_graph().nodes)
    assert WORK_NODES[DREAM_CARRY_WORKFLOW] == {"text_hop"}
    state = {"brief": {"trigger_id": "s", "hops": 6}, "hops": [{}, {}], "dream_id": "d", "stopped_reason": "x",
             "child_run_ids": ["c1"], "last_error": "boom"}
    assert rt._finish_detail_for(DREAM_CARRY_WORKFLOW, state)["hops_made"] == 2
    assert rt._terminal_detail_for(DREAM_CARRY_WORKFLOW, "failed", state)["error"] == "boom"


def test_driver_backstops_a_carry_only_after_deadline_plus_grace():
    rt = _runtime()
    now = datetime.fromisoformat("2026-10-10T00:00:00+00:00")
    rt.now = lambda: now
    state = {"admission": {"deadline_at": "2026-10-09T23:00:00+00:00"}}
    # A reverie/journal run past its deadline is failed by the driver; a carry is not (yet).
    assert rt._driver_deadline(REVERIE_VISUAL_WORKFLOW, state) <= now
    assert rt._driver_deadline(DREAM_CARRY_WORKFLOW, state) == \
        datetime.fromisoformat("2026-10-09T23:00:00+00:00") + timedelta(seconds=900 + 300)
    assert rt._carry_past_deadline(DREAM_CARRY_WORKFLOW, state)
    assert not rt._carry_past_deadline(REVERIE_VISUAL_WORKFLOW, state)


def test_runner_sends_the_carry_step_on_its_channel_with_its_reply_prefix():
    from app.runner import DurableRunner
    from app.settings import Settings
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
    from orion.core.bus.codec import OrionCodec
    from orion.schemas.dream_carry import (
        DREAM_CARRY_STEP_CHANNEL, DREAM_CARRY_STEP_REPLY_PREFIX, DREAM_CARRY_STEP_RESULT_KIND,
        DreamCarryStepRequestV1,
    )
    from orion.schemas.gpu_pool import GpuLeaseRefV1

    codec = OrionCodec()
    req = DreamCarryStepRequestV1(run_id=RUN, correlation_id="c-1", step="text", hop_index=0,
                                  brief=DreamCarryBriefV1(trigger_id="s", timeout_sec=7.0),
                                  gpu_lease=GpuLeaseRefV1(lease_id="hold-1", generation=1, role="metacog",
                                                          holder=f"durable-runs:{RUN}"))
    holder = {}

    async def rpc_request(channel, envelope, *, reply_channel, timeout_sec, **_):
        holder.update(channel=channel, envelope=envelope, reply=reply_channel, timeout=timeout_sec)
        result = DreamCarryStepResultV1(run_id=RUN, correlation_id="c-1", step="text", status="retry", reason="busy")
        reply = BaseEnvelope(kind=DREAM_CARRY_STEP_RESULT_KIND, source=ServiceRef(name="orion-dream"),
                             correlation_id=envelope.correlation_id, payload=result.model_dump(mode="json"))
        return {"data": codec.encode(reply)}

    settings = Settings(_env_file=None, DURABLE_RUNS_GRAPH_HOST="", POSTGRES_URI="postgresql://unused",
                        ORION_BUS_ENABLED=False)
    runner = DurableRunner(settings, bus=SimpleNamespace(codec=codec, rpc_request=rpc_request), checkpointer=None)
    result = asyncio.run(runner._run_dream_carry_step(req, 7.0))
    assert result.status == "retry"
    assert holder["channel"] == DREAM_CARRY_STEP_CHANNEL
    assert holder["reply"] == f"{DREAM_CARRY_STEP_REPLY_PREFIX}:c-1" and holder["timeout"] == 7.0
    assert holder["envelope"].payload["gpu_lease"]["lease_id"] == "hold-1"


# --- review follow-ups ---------------------------------------------------------------------------

def test_submit_refuses_a_carry_without_a_deadline():
    from orion.schemas.durable_run import DurableRunRequestV1

    rt = _runtime()
    req = DurableRunRequestV1(run_id="dream-carry-x", workflow=DREAM_CARRY_WORKFLOW, correlation_id="c",
                              brief=DreamCarryBriefV1(trigger_id="s"),
                              admission={"resource": "llm.route.metacog_background",
                                         "preferred_lane": "metacog_background"})
    with pytest.raises(ValueError, match="deadline_at"):
        asyncio.run(rt.submit(req))


def test_an_unspaced_image_prompt_is_clipped_to_what_the_child_brief_accepts():
    async def run():
        world, saver = CarryWorld(), InMemorySaver()
        dream = Dream(text=[("done", {"hop": {"index": 0, "kind": "text", "passage": "p",
                                              "image_prompt": "x" * 5000}}), ("done", {})])
        children = Children()
        result = await drive(graph(world, saver, dream, children), world, initial(world, hops=2))
        assert result["status"] == "completed" and len(result["hops"]) == 2
        assert len(children.submitted[0].brief.dream_hop.prompt) == 1000
    asyncio.run(run())


def test_a_text_answer_for_the_wrong_hop_is_retried_not_checkpointed():
    async def run():
        world, saver = CarryWorld(), InMemorySaver()
        bad = {"index": 1, "kind": "text", "passage": "p", "image_prompt": "q"}
        dream = Dream(text=[("done", {"hop": bad}), ("done", {})])
        children = Children()
        result = await drive(graph(world, saver, dream, children), world, initial(world, hops=2))
        assert result["status"] == "completed" and [h["index"] for h in result["hops"]] == [0, 1]
        assert result["retries"] == 1 and [r.hop_index for r in dream.calls if r.step == "text"] == [0, 0]
    asyncio.run(run())


class TakeBackWorld(CarryWorld):
    """execute raises ``exc`` once, as the runtime does for a pool take-back or the run deadline."""

    def __init__(self, exc):
        super().__init__()
        self.exc = exc

    async def execute(self, state, node):
        if self.exc is not None:
            exc, self.exc = self.exc, None
            raise exc
        return await super().execute(state, node)


def test_a_pool_take_back_mid_text_hop_replays_it_without_a_retry_or_attempt():
    from app.admitted_graph import HoldLost

    async def run():
        world, saver = TakeBackWorld(HoldLost("gpu_hold_lost:queued")), InMemorySaver()
        dream, children = Dream(), Children()
        result = await drive(graph(world, saver, dream, children), world, initial(world, hops=2))
        assert result["status"] == "completed" and len(result["hops"]) == 2
        assert result["attempt"] == 0 and int(result.get("retries") or 0) == 0
        assert result["hold_takebacks"] == 1 and "hold_lost" in world.releases
        assert result["lease"] is None and result["hold"] is None
    asyncio.run(run())


def test_the_deadline_inside_a_text_hop_releases_the_hold_and_finishes_partial_or_fails_empty():
    from app.admitted_graph import WorkflowDeadline

    async def run():
        world, saver = TakeBackWorld(WorkflowDeadline("workflow_deadline")), InMemorySaver()
        dream, children = Dream(), Children()
        result = await drive(graph(world, saver, dream, children), world, initial(world))
        # Nothing made yet: the hold is handed back and finish asks for the story fallback.
        assert result["status"] == "completed" and result["dream_id"] == "story-fallback:sleep-1"
        assert result["stopped_reason"].startswith("deadline at hop 0")
        assert world.releases == ["workflow_deadline"] and [r.step for r in dream.calls] == ["finish"]
    asyncio.run(run())


def test_a_carry_cancelled_mid_submit_still_cancels_the_deterministic_child():
    """The child submit landed but the driver was cancelled before child_run_id was checkpointed."""
    rt = _runtime()
    cancelled = []

    async def finish_projection(run_id, status, detail, **_):
        return status

    async def cancel_child(run_id, child):
        cancelled.append(child)

    rt.store = SimpleNamespace(finish_projection=finish_projection)
    rt._cancel_carry_child = cancel_child
    state = {"workflow": DREAM_CARRY_WORKFLOW, "hops": [{"index": 0}], "brief": {"trigger_id": "s", "hops": 6}}
    asyncio.run(rt._terminal(RUN, "cancelled", state, workflow=DREAM_CARRY_WORKFLOW))
    assert cancelled == [reverie_visual_run_id(dream_hop_dispatch_id(RUN, 1))]
    cancelled.clear()
    # A completed-partial carry (stopped with a picture still being painted) cancels it too.
    asyncio.run(rt._terminal(RUN, "completed", {**state, "child_run_id": "child-x"}, workflow=DREAM_CARRY_WORKFLOW))
    assert cancelled == ["child-x"]
    cancelled.clear()
    full = {**state, "hops": [{}] * 6}   # every hop made: no child to name
    asyncio.run(rt._terminal(RUN, "completed", full, workflow=DREAM_CARRY_WORKFLOW))
    assert cancelled == []


# --- fresh child after a retryable miss (heat / busy) ---------------------------------------------

HOT = ("failed", {"error": "retry_window_expired", "last_error": "retry_window_expired", "reason": "thermal_refused"})


def test_dispatch_id_attempt_zero_is_unchanged_and_retries_are_suffixed():
    assert dream_hop_dispatch_id("run", 3) == dream_hop_dispatch_id("run", 3, 0) == "dream-carry:run:3"
    assert dream_hop_dispatch_id("run", 3, 2) == "dream-carry:run:3:r2"


def test_a_child_that_ran_out_of_window_on_heat_is_replaced_and_the_carry_reaches_six_hops():
    async def run():
        world, saver = CarryWorld(), InMemorySaver()
        dream, children = Dream(), Children(h3=[HOT, ("completed", {"outcome": "produced", "artifact_sha256": SHA,
                                                                     "caption": "seen 3 again",
                                                                     "visual_elapsed_sec": 50.0})])
        waits = []

        def watch(snap):
            if snap.next == ("retry_wait",) and snap.values.get("retry_node") == "image_submit":
                waits.append((dict(snap.values), world.now))

        result = await drive(graph(world, saver, dream, children), world, initial(world), on_wait=watch)
        assert result["status"] == "completed" and len(result["hops"]) == 6 and result["stopped_reason"] is None
        hop3 = [c for c in children.submitted if c.brief.dream_hop.hop_index == 3]
        assert [c.brief.visual_request.dispatch_id for c in hop3] == [dream_hop_dispatch_id(RUN, 3),
                                                                      dream_hop_dispatch_id(RUN, 3, 1)]
        assert hop3[1].run_id == reverie_visual_run_id(dream_hop_dispatch_id(RUN, 3, 1))
        assert hop3[1].brief.dream_hop.prompt == hop3[0].brief.dream_hop.prompt
        assert result["hops"][3]["child_run_id"] == hop3[1].run_id
        assert result["hops"][3]["caption"] == "seen 3 again"
        assert result["child_run_ids"] == [c.run_id for c in children.submitted] and len(children.submitted) == 4
        # Backed off before the fresh child (the admission backoff, never immediate).
        [(state, at)] = waits
        assert state["child_attempt"] == 1 and state["reason"] == "child_retry:thermal_refused"
        # max(admission backoff 30 s, DREAM_CARRY_CHILD_RETRY_GAP_SEC 900 s): a waking painting can
        # claim thought's slot in between.
        assert datetime.fromisoformat(state["retry_at"]) - at == timedelta(seconds=900)
        assert result["child_attempt"] == 0   # reset for the next hop
        assert finish_detail(result)["child_run_ids"] == result["child_run_ids"]
    asyncio.run(run())


def test_retryable_misses_are_bounded_per_hop_and_then_finish_partial():
    async def run():
        world, saver = CarryWorld(), InMemorySaver()
        dream = Dream()
        children = Children(h1=("completed", {"outcome": "deferred_thermal", "reason": "thermal_refused"}))
        result = await drive(graph(world, saver, dream, children), world, initial(world))
        assert result["status"] == "completed" and len(result["hops"]) == 1
        assert result["stopped_reason"] == "image hop 1: thermal_refused x3"
        assert [c.brief.visual_request.dispatch_id for c in children.submitted] == [
            dream_hop_dispatch_id(RUN, 1, n) for n in range(3)]
    asyncio.run(run())


def test_no_fresh_child_without_enough_carry_left():
    async def run():
        world, saver = CarryWorld(), InMemorySaver()
        dream, children = Dream(), Children(h1=HOT)
        start = world.now
        children.clock, children.ready_at = (lambda: world.now), start + timedelta(minutes=50)
        result = await drive(graph(world, saver, dream, children), world, initial(world, deadline=timedelta(hours=1)))
        assert result["status"] == "completed" and len(result["hops"]) == 1 and len(children.submitted) == 1
        assert result["stopped_reason"].startswith("image hop 1: thermal_refused (only ")
    asyncio.run(run())


def test_no_first_child_without_enough_carry_left():
    async def run():
        world, saver = CarryWorld(), InMemorySaver()
        dream, children = Dream(), Children()
        g = graph(world, saver, dream, children, child_min_window_sec=5 * 3600)   # more than the 4 h carry
        result = await drive(g, world, initial(world))
        assert result["status"] == "completed" and len(result["hops"]) == 1 and children.submitted == []
        assert result["stopped_reason"].startswith("image hop 1: only ") and result["stopped_reason"].endswith("s left")
    asyncio.run(run())


def test_a_dream_child_window_is_short_so_it_cannot_hog_the_painting_slot():
    async def run():
        world, saver = CarryWorld(), InMemorySaver()
        dream, children = Dream(), Children()
        await drive(graph(world, saver, dream, children), world, initial(world))
        for child in children.submitted:
            assert child.admission.deadline_at - child.requested_at == timedelta(seconds=2400)
        world2, children2 = CarryWorld(), Children()   # a carry with less than 2400 s left: its deadline wins
        carry_deadline = world2.now + timedelta(minutes=20)
        await drive(graph(world2, InMemorySaver(), Dream(), children2), world2,
                    initial(world2, deadline=timedelta(minutes=20)))
        assert children2.submitted[0].admission.deadline_at == carry_deadline
    asyncio.run(run())


def test_a_carry_stopped_by_its_deadline_keeps_the_running_child_for_the_terminal_cancel():
    async def run():
        world, saver = CarryWorld(), InMemorySaver()
        children = Children(h1=None)
        result = await drive(graph(world, saver, Dream(), children), world, initial(world, deadline=timedelta(hours=1)))
        assert result["status"] == "completed" and result["stopped_reason"].startswith("deadline at hop 1")
        assert result["child_run_id"] == children.submitted[0].run_id   # _terminal cancels it (L2)
    asyncio.run(run())


def test_a_restart_mid_resubmit_replays_the_same_child_never_a_new_one():
    async def run():
        world, saver = CarryWorld(), InMemorySaver()
        dream, children = Dream(), Children(h1=[HOT, None])
        g = graph(world, saver, dream, children)

        def arm(snap):
            if snap.next == ("retry_wait",) and snap.values.get("child_attempt") == 1:
                children.crash_submits = 1   # the r1 submit lands, then the process dies

        with pytest.raises(Crash):
            await drive(g, world, initial(world), on_wait=arm)
        snap = await g.aget_state(CFG)
        assert snap.next == ("image_submit",) and snap.values["child_attempt"] == 1
        await graph(world, saver, dream, children).ainvoke(None, CFG)   # restarted driver replays the node
        ids = [c.run_id for c in children.submitted]
        r1 = reverie_visual_run_id(dream_hop_dispatch_id(RUN, 1, 1))
        assert ids == [reverie_visual_run_id(dream_hop_dispatch_id(RUN, 1)), r1, r1]   # same id: store dedupes
        snap = await g.aget_state(CFG)
        assert snap.next == ("image_wait",) and snap.values["child_run_id"] == r1
        assert snap.values["child_run_ids"] == ids[:2]
    asyncio.run(run())


def test_cancel_cancels_the_current_attempts_child():
    rt = _runtime()
    cancelled = []

    async def finish_projection(run_id, status, detail, **_):
        return status

    async def cancel_child(run_id, child):
        cancelled.append(child)

    rt.store = SimpleNamespace(finish_projection=finish_projection)
    rt._cancel_carry_child = cancel_child
    state = {"workflow": DREAM_CARRY_WORKFLOW, "hops": [{}, {}, {}], "child_attempt": 2,
             "brief": {"trigger_id": "s", "hops": 6}}
    asyncio.run(rt._terminal(RUN, "cancelled", state, workflow=DREAM_CARRY_WORKFLOW))
    assert cancelled == [reverie_visual_run_id(dream_hop_dispatch_id(RUN, 3, 2))]
