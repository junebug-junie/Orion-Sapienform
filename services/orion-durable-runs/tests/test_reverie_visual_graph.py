"""reverie.visual on real checkpointed admission: per-stage checkpoints, retries that never spend
attempts, the diffusion hold scoped to generate only, and the deadline as the only failure."""
from __future__ import annotations

import asyncio
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from test_admitted_graph import World
from app.admission_runtime import AdmissionRuntime, TERMINAL_STATE_NODE
from app.admitted_graph import AdmissionDeps, HoldLost
from app.reverie_visual_graph import (
    RETRY_WINDOW_EXPIRED, build_reverie_visual_graph, finish_detail, terminal_detail,
)
from orion.schemas.durable_run import DURABLE_RUN_STATE_KIND, DurableRunStateV1
from orion.schemas.reverie_visual import VisualRunRequestV1
from orion.schemas.reverie_visual_run import (
    NEEDS_GENERATE, REVERIE_VISUAL_WORKFLOW, ReverieVisualRunBriefV1, ReverieVisualStepResultV1,
)

RUN = "reverie-001"
CFG = {"configurable": {"thread_id": RUN}}
SHA = "a" * 64


class Crash(BaseException):
    """Process death mid-node: not an Exception, so no node turns it into a retry."""


class Thought:
    """orion-thought's stage executor. ``script[step]`` is a list of (status, extra) answers,
    consumed in order; the last one repeats."""

    def __init__(self, world, **script):
        self.world = world
        self.script = {"prepare": [("done", {})], "generate": [("done", {})], "caption": [("done", {})],
                       "abandon": [("done", {})], **script}
        self.calls = []            # (step, request)
        self.pool_at_caption = []  # the pool's view of the run's hold when caption ran

    def steps(self, name=None):
        return [s for s, _ in self.calls if name is None or s == name]

    async def __call__(self, req, budget_sec=None):
        self.calls.append((req.step, req))
        if req.step == "caption":
            self.pool_at_caption.append(self.world.pool_status)
        answers = self.script[req.step]
        status, extra = answers.pop(0) if len(answers) > 1 else answers[0]
        if isinstance(status, BaseException):
            raise status
        base = {"run_id": req.run_id, "correlation_id": req.correlation_id, "step": req.step, "status": status}
        if status == "done":
            base.update({
                "prepare": {"attempt_id": "att-1", "elapsed_sec": 2.0},
                "generate": {"attempt_id": "att-1", "artifact_sha256": SHA, "elapsed_sec": 10.0},
                "caption": {"attempt_id": "att-1", "chain_id": "att-1", "outcome": "produced", "elapsed_sec": 3.0,
                            "execution_receipt": {"outcome": "produced"}},
                "abandon": {"attempt_id": req.attempt_id},
            }[req.step])
        return ReverieVisualStepResultV1(**{**base, **extra})


def initial(world, deadline=timedelta(minutes=90)):
    brief = ReverieVisualRunBriefV1(visual_request=VisualRunRequestV1(
        dispatch_id="dispatch-1", proposal_id="proposal-1", decision_id="decision-1"), timeout_sec=5.0)
    return {"run_id": RUN, "correlation_id": "trace-001", "attempt": 0, "workflow": REVERIE_VISUAL_WORKFLOW,
            "admission": {"resource": "service.route.diffusion", "preferred_lane": "diffusion",
                          "deadline_at": (world.now + deadline).isoformat()},
            "brief": brief.model_dump(mode="json")}


def graph(world, saver, thought):
    return build_reverie_visual_graph(thought, AdmissionDeps(
        world.register, world.lease, world.execute, world.release, world.event, now=lambda: world.now,
        max_attempts=1, retry_base_seconds=30.0, retry_max_seconds=300.0, guard=world.guard), saver)


def test_happy_path_holds_diffusion_for_generate_only_and_reports_real_work_seconds():
    async def run():
        world, saver = World(), InMemorySaver()
        thought = Thought(world)
        world.grant()
        result = await graph(world, saver, thought).ainvoke(initial(world), CFG)
        assert result["status"] == "completed"
        assert thought.steps() == ["prepare", "generate", "caption"]
        [(_, gen)] = [c for c in thought.calls if c[0] == "generate"]
        assert gen.gpu_lease.model_dump() == world.ref() and gen.attempt_id == "att-1"
        # Released right after generate; caption ran with the pool no longer granting the run.
        assert world.releases == ["generated"] and thought.pool_at_caption == ["released"]
        assert len({req.correlation_id for _, req in thought.calls}) == 3
        detail = finish_detail(result)
        assert detail["dispatch_id"] == "dispatch-1" and detail["proposal_id"] == "proposal-1"
        assert detail["decision_id"] == "decision-1" and detail["attempt_id"] == "att-1"
        assert detail["chain_id"] == "att-1" and detail["outcome"] == "produced"
        assert detail["artifact_sha256"] == SHA and detail["execution_receipt"] == {"outcome": "produced"}
        assert detail["retries"] == 0 and detail["generate_elapsed_sec"] == 10.0
        assert detail["visual_elapsed_sec"] == 15.0   # thought's work, never queue/hold wait
        assert detail["started_at"] and detail["finished_at"]
    asyncio.run(run())


def test_wait_restart_grant_resumes_at_the_hold_without_repeating_prepare():
    async def run():
        world, saver = World(), InMemorySaver()
        thought = Thought(world)
        await graph(world, saver, thought).ainvoke(initial(world), CFG)
        snap = await graph(world, saver, thought).aget_state(CFG)
        assert snap.next == ("resource_wait",) and snap.tasks[0].interrupts
        assert thought.steps() == ["prepare"] and snap.values["attempt_id"] == "att-1"
        world.grant()
        result = await graph(world, saver, thought).ainvoke(Command(resume={"lease_id": "forged"}), CFG)
        assert result["status"] == "completed"
        assert thought.steps() == ["prepare", "generate", "caption"]
    asyncio.run(run())


def test_crash_after_generate_resumes_at_caption_with_one_generate_call():
    async def run():
        world, saver = World(), InMemorySaver()
        thought = Thought(world, caption=[(Crash(), {}), ("done", {})])
        world.grant()
        with pytest.raises(Crash):
            await graph(world, saver, thought).ainvoke(initial(world), CFG)
        snap = await graph(world, saver, thought).aget_state(CFG)
        # generate's result is checkpointed and the hold already gone: nothing for _recover to fence.
        assert snap.next == ("caption",) and snap.values["lease"] is None
        assert snap.values["artifact_sha256"] == SHA
        result = await graph(world, saver, thought).ainvoke(None, CFG)
        assert result["status"] == "completed"
        assert thought.steps("generate") == ["generate"]
        assert thought.steps("caption") == ["caption", "caption"]
    asyncio.run(run())


def test_restart_mid_generate_fences_and_replays_generate_under_the_same_hold():
    async def run():
        world, saver = World(), InMemorySaver()
        thought = Thought(world, generate=[(Crash(), {}), ("done", {})])
        world.grant()
        g = graph(world, saver, thought)
        with pytest.raises(Crash):
            await g.ainvoke(initial(world), CFG)
        snap = await g.aget_state(CFG)
        assert snap.next == ("generate",) and snap.values["lease"] == world.ref()
        rt = object.__new__(AdmissionRuntime)
        await rt._recover(g, CFG, REVERIE_VISUAL_WORKFLOW, dict(snap.values), snap)
        snap = await g.aget_state(CFG)
        assert snap.next == ("resource_wait",) and snap.values["turn_fence"] == 1
        assert snap.values["hold"] is not None   # same hold, re-confirmed with the pool
        result = await g.ainvoke(Command(resume=True), CFG)
        assert result["status"] == "completed"
        first, replay = [req for step, req in thought.calls if step == "generate"]
        assert first.correlation_id != replay.correlation_id   # a late reply cannot cross
        assert world.requests == ["study-001:1"]   # the fence re-reads the pool; no new request
    asyncio.run(run())


def test_generate_deferral_releases_the_hold_and_requeues_without_spending_an_attempt():
    async def run():
        world, saver = World(), InMemorySaver()
        thought = Thought(world, generate=[("retry", {"reason": "thermal_refused"}), ("done", {})])
        world.grant()
        g = graph(world, saver, thought)
        await g.ainvoke(initial(world), CFG)
        snap = await g.aget_state(CFG)
        assert snap.next == ("retry_wait",) and snap.tasks[0].interrupts
        assert world.releases == ["step_retry"] and snap.values["hold"] is None
        assert snap.values["attempt"] == 0 and snap.values["retries"] == 1
        assert snap.values["retry_at"] == (world.now + timedelta(seconds=30)).isoformat()
        world.now += timedelta(seconds=31)
        await g.ainvoke(Command(resume=True), CFG)
        assert (await g.aget_state(CFG)).next == ("resource_wait",)
        world.grant()
        result = await g.ainvoke(Command(resume=True), CFG)
        assert result["status"] == "completed" and result["attempt"] == 0
        assert thought.steps() == ["prepare", "generate", "generate", "caption"]
        assert world.requests == ["study-001:1", "study-001:2"]   # a released hold is never re-asked
        assert finish_detail(result)["retries"] == 1
    asyncio.run(run())


def test_retry_after_sec_from_thought_overrides_the_backoff_but_never_goes_to_zero():
    async def run():
        world, saver = World(), InMemorySaver()
        thought = Thought(world, prepare=[("retry", {"reason": "context_busy", "retry_after_sec": 0.0}),
                                          ("done", {})])
        g = graph(world, saver, thought)
        await g.ainvoke(initial(world), CFG)
        snap = await g.aget_state(CFG)
        assert snap.next == ("retry_wait",) and snap.values["retry_node"] == "prepare"
        assert snap.values["retry_at"] == (world.now + timedelta(seconds=1)).isoformat()
        assert world.requests == []   # prepare never takes a hold
    asyncio.run(run())


def test_generate_transport_error_is_a_retry_not_a_failure():
    async def run():
        world, saver = World(), InMemorySaver()
        thought = Thought(world, generate=[(TimeoutError("rpc"), {}), ("done", {})])
        world.grant()
        g = graph(world, saver, thought)
        await g.ainvoke(initial(world), CFG)
        snap = await g.aget_state(CFG)
        assert snap.values["status"] == "retrying" and snap.values["retry_node"] == "resource_request"
        assert snap.values["last_error"].startswith("transport:TimeoutError")
        assert world.releases == ["step_retry"] and snap.values["attempt"] == 0
    asyncio.run(run())


def test_lost_hold_requeues_without_a_generate_call():
    async def run():
        world, saver = World(), InMemorySaver()
        thought = Thought(world)
        world.grant()

        async def execute(state, node):
            world.pool_status = "queued"
            raise HoldLost("gpu_hold_lost")

        world.execute = execute
        result = await graph(world, saver, thought).ainvoke(initial(world), CFG)
        assert result["status"] == "waiting_resource" and result.get("retries", 0) == 0
        assert thought.steps() == ["prepare"]
    asyncio.run(run())


def test_caption_needs_generate_goes_back_through_the_hold_after_a_backoff():
    async def run():
        world, saver = World(), InMemorySaver()
        thought = Thought(world, caption=[("retry", {"reason": NEEDS_GENERATE}), ("done", {})])
        world.grant()
        g = graph(world, saver, thought)
        await g.ainvoke(initial(world), CFG)
        snap = await g.aget_state(CFG)
        assert snap.next == ("retry_wait",) and snap.values["retry_node"] == "resource_request"
        assert snap.values["regenerations"] == 1 and snap.values["attempt"] == 0
        assert snap.values["retry_at"] == (world.now + timedelta(seconds=30)).isoformat()
        world.now += timedelta(seconds=31)
        await g.ainvoke(Command(resume=True), CFG)
        assert (await g.aget_state(CFG)).next == ("resource_wait",)
        world.grant()
        result = await g.ainvoke(Command(resume=True), CFG)
        assert result["status"] == "completed"
        assert thought.steps() == ["prepare", "generate", "caption", "generate", "caption"]
        assert world.releases == ["generated", "generated"]
        assert finish_detail(result)["generate_elapsed_sec"] == 20.0
    asyncio.run(run())


def test_repeated_needs_generate_backs_off_exponentially_on_the_regeneration_count():
    """A storage fault that keeps eating the image must not re-render back to back."""
    async def run():
        world, saver = World(), InMemorySaver()
        thought = Thought(world, caption=[("retry", {"reason": NEEDS_GENERATE})])
        g = graph(world, saver, thought)
        world.grant()
        await g.ainvoke(initial(world), CFG)
        delays = []
        for _ in range(4):
            snap = await g.aget_state(CFG)
            assert snap.next == ("retry_wait",) and snap.values["retry_node"] == "resource_request"
            retry_at = datetime.fromisoformat(snap.values["retry_at"])
            delays.append((retry_at - world.now).total_seconds())
            world.now = retry_at + timedelta(seconds=1)
            await g.ainvoke(Command(resume=True), CFG)   # retry_wait -> resource_request -> wait
            world.grant()
            await g.ainvoke(Command(resume=True), CFG)   # granted -> generate -> caption (needs_generate)
        assert delays == [30.0, 60.0, 120.0, 240.0]
        snap = await g.aget_state(CFG)
        assert snap.values["regenerations"] == 5 and snap.values["attempt"] == 0
        assert (datetime.fromisoformat(snap.values["retry_at"]) - world.now).total_seconds() == 300.0   # capped
    asyncio.run(run())


@pytest.mark.parametrize("reason", ["not_prepared", "plan_not_frozen"])
def test_generate_retry_without_a_prepared_attempt_releases_the_hold_and_reprepares(reason):
    async def run():
        world, saver = World(), InMemorySaver()
        thought = Thought(world, generate=[("retry", {"reason": reason}), ("done", {})])
        world.grant()
        g = graph(world, saver, thought)
        await g.ainvoke(initial(world), CFG)
        snap = await g.aget_state(CFG)
        assert snap.next == ("retry_wait",) and snap.values["retry_node"] == "prepare"
        assert world.releases == ["step_retry"] and snap.values["hold"] is None
        world.now += timedelta(seconds=31)
        await g.ainvoke(Command(resume=True), CFG)
        assert (await g.aget_state(CFG)).next == ("resource_wait",)
        world.grant()
        result = await g.ainvoke(Command(resume=True), CFG)
        assert result["status"] == "completed"
        assert thought.steps() == ["prepare", "generate", "prepare", "generate", "caption"]
    asyncio.run(run())


@pytest.mark.parametrize("reason", ["not_prepared", "plan_not_frozen"])
def test_caption_retry_without_a_prepared_attempt_goes_back_to_prepare(reason):
    async def run():
        world, saver = World(), InMemorySaver()
        thought = Thought(world, caption=[("retry", {"reason": reason}), ("done", {})])
        world.grant()
        g = graph(world, saver, thought)
        await g.ainvoke(initial(world), CFG)
        snap = await g.aget_state(CFG)
        assert snap.next == ("retry_wait",) and snap.values["retry_node"] == "prepare"
        assert world.releases == ["generated"]   # caption never held anything to release
        world.now += timedelta(seconds=31)
        await g.ainvoke(Command(resume=True), CFG)
        assert (await g.aget_state(CFG)).next == ("resource_wait",)
        assert thought.steps() == ["prepare", "generate", "caption", "prepare"]
    asyncio.run(run())


def test_generate_terminal_keeps_attempt_and_chain_ids_in_the_finish_detail():
    async def run():
        world, saver = World(), InMemorySaver()
        thought = Thought(world, generate=[("terminal", {"outcome": "failed", "reason": "attempt_mismatch",
                                                         "attempt_id": "att-1", "chain_id": "att-1"})])
        world.grant()
        result = await graph(world, saver, thought).ainvoke(initial(world), CFG)
        assert result["status"] == "completed" and world.releases == ["terminal"]
        detail = finish_detail(result)
        assert detail["attempt_id"] == "att-1" and detail["chain_id"] == "att-1"
        assert detail["outcome"] == "failed" and detail["reason"] == "attempt_mismatch"
    asyncio.run(run())


def test_deadline_fails_retry_window_expired_with_the_last_stage_reason():
    async def run():
        world, saver = World(), InMemorySaver()
        thought = Thought(world, generate=[("retry", {"reason": "thermal_refused"})])
        world.grant()
        g = graph(world, saver, thought)
        await g.ainvoke(initial(world, deadline=timedelta(seconds=10)), CFG)
        assert (await g.aget_state(CFG)).next == ("retry_wait",)
        world.now += timedelta(seconds=60)
        result = await g.ainvoke(Command(resume=True), CFG)
        assert result["status"] == "failed" and result["last_error"] == RETRY_WINDOW_EXPIRED
        # Abandon belongs to the runtime's terminal (durable + retried), never the graph node.
        assert thought.steps("abandon") == [] and thought.steps("generate") == ["generate"]
        detail = terminal_detail(result, "failed")
        assert detail["dispatch_id"] == "dispatch-1" and detail["attempt_id"] == "att-1"
        assert detail["retries"] == 1 and detail["last_error"] == RETRY_WINDOW_EXPIRED
        assert detail["reason"] == "thermal_refused"
    asyncio.run(run())


def test_terminal_from_prepare_finishes_without_ever_requesting_a_hold():
    async def run():
        world, saver = World(), InMemorySaver()
        thought = Thought(world, prepare=[("terminal", {"outcome": "already_satisfied",
                                                        "reason": "baseline_satisfied",
                                                        "attempt_id": "att-0", "chain_id": "att-0"})])
        result = await graph(world, saver, thought).ainvoke(initial(world), CFG)
        assert result["status"] == "completed"
        assert world.requests == [] and world.releases == [] and thought.steps() == ["prepare"]
        detail = finish_detail(result)
        assert detail["outcome"] == "already_satisfied" and detail["reason"] == "baseline_satisfied"
        assert detail["attempt_id"] == "att-0" and detail["chain_id"] == "att-0"
        assert detail["visual_elapsed_sec"] == 0.0
    asyncio.run(run())


def test_legacy_resume_sweep_ignores_admitted_reverie_threads_silently(caplog):
    from app.runner import DurableRunner
    from app.settings import Settings

    async def run():
        world, saver = World(), InMemorySaver()
        await graph(world, saver, Thought(world)).ainvoke(initial(world), CFG)   # waiting at the hold
        runner = DurableRunner(Settings(_env_file=None, POSTGRES_URI="postgresql://unused"),
                               bus=None, checkpointer=saver)
        assert await runner.unfinished_threads() == []
    caplog.set_level("WARNING")
    asyncio.run(run())
    assert "durable_run_resume_unknown_workflow" not in caplog.text


def test_reverie_visual_timeouts_nest_and_the_hold_heartbeat_fits_its_ttl():
    """Each waiter outlasts the one inside it, so a reply always lands before its caller gives up:
    durable-runs' stage RPC wait > the brief's generate budget > thought's own generate deadline."""
    from app.settings import Settings
    from orion.gpu_pool.config import load_pool_config

    settings = Settings(_env_file=None, POSTGRES_URI="postgresql://unused")
    brief_budget = ReverieVisualRunBriefV1.model_fields["timeout_sec"].default
    # services/orion-thought/app/settings.py: visual_chain_step_generate_deadline_sec
    # (ORION_VISUAL_CHAIN_STEP_GENERATE_DEADLINE_SEC), read from source: thought is another service.
    thought_settings = Path(__file__).resolve().parents[3] / "services/orion-thought/app/settings.py"
    match = re.search(r"visual_chain_step_generate_deadline_sec: float = Field\(\s*([0-9.]+)",
                      thought_settings.read_text())
    assert match, "thought's generate deadline setting moved; update this test"
    thought_generate_deadline = float(match.group(1))
    assert settings.reverie_visual_step_timeout_sec > brief_budget > thought_generate_deadline
    # AdmissionRuntime refuses to start otherwise: two missed beats must land inside the TTL.
    assert settings.lease_heartbeat_sec * 2 <= load_pool_config().defaults.hold_lease_ttl_sec


# --- terminal projection: every admitted terminal reaches orion:durable:run:state ---------------

class FakeStore:
    """The admission store's run row, terminal outbox and run.abandon_* records (same semantics as
    DurableRunRegistryStore: event ids are ON CONFLICT DO NOTHING; pending = terminal failed/cancelled
    with a pending record and no acked one). Shared across runtimes to model a restart."""

    def __init__(self, workflow):
        self.workflow, self.events, self.acked, self.control = workflow, [], [], None
        self.cancelled_details = []
        self.terminal = None
        self.recorded = {}   # event_id -> (event, detail)

    async def get_run(self, run_id):
        return {"run_id": run_id, "terminal": self.terminal, "control": self.control,
                "request": {"workflow": self.workflow, "admission": {}, "brief": initial(World())["brief"]}}

    async def finish_projection(self, run_id, status, detail, *, cancelled_detail=None):
        self.cancelled_details.append(cancelled_detail)
        if self.control == "cancelled":
            status, detail = "cancelled", dict(cancelled_detail or {})
        self.events.append({"schema_version": "durable.resource.event.v1", "entry_id": f"{run_id}:terminal:{status}",
                            "event": "run." + status, "run_id": run_id, "thread_id": run_id,
                            "correlation_id": "trace-001", "generated_at": "2026-09-28T00:00:00+00:00",
                            "detail": detail})
        self.terminal = status
        return status

    async def record_event(self, run_id, event, detail, event_id=None):
        self.recorded.setdefault(event_id, (event, detail))

    def recorded_events(self):
        return [event for event, _ in self.recorded.values()]

    async def abandons_pending(self, limit=100):
        acked = "run.abandon_acked" in self.recorded_events()
        if acked or self.terminal not in ("failed", "cancelled"):
            return []
        return [{"run_id": RUN, "detail": detail, "request": (await self.get_run(RUN))["request"]}
                for event, detail in self.recorded.values() if event == "run.abandon_pending"]

    async def list_pending(self, limit=100):
        return []

    async def outreach_holds_pending(self, max_age_seconds):
        return []

    async def pending_outbox(self):
        return [e for e in self.events if e["entry_id"] not in self.acked]

    async def ack_outbox(self, entry_id):
        self.acked.append(entry_id)


def runtime(workflow, thought=None, store=None):
    rt = object.__new__(AdmissionRuntime)
    rt.store = store or FakeStore(workflow)
    published = []

    async def publish(channel, kind, model, corr):
        published.append((channel, kind, model))
        return True

    rt.runner = SimpleNamespace(_publish=publish, _corr_for_admission=lambda c: c,
                                _run_reverie_visual_step=thought)
    rt.settings = SimpleNamespace(state_channel="orion:durable:run:state", retry_base_sec=30.0,
                                  retry_max_sec=300.0, hold_status_poll_sec=60.0, outreach_hold_max_sec=1800.0)
    rt._wake = asyncio.Event()
    rt.outreach, rt._hints, rt._checked = {}, set(), {}
    rt._outreach_loaded_at, rt._pending_release, rt.active = None, {}, {}
    rt._abandons, rt._abandons_loaded_at, rt._abandoning = {}, None, {}
    return rt, published


async def settle(rt):
    """Let every in-flight abandon try finish (they run off the reconcile loop)."""
    while rt._abandoning:
        await asyncio.gather(*list(rt._abandoning.values()))


def _failed_state():
    world = World()
    return {**initial(world), "status": "failed", "last_error": RETRY_WINDOW_EXPIRED, "attempt_id": "att-1",
            "retries": 3, "reason": "thermal_refused"}


def test_failed_reverie_run_publishes_a_durable_run_state_with_the_dispatch_detail():
    async def run():
        rt, published = runtime(REVERIE_VISUAL_WORKFLOW)
        await rt._terminal(RUN, "failed", _failed_state(), workflow=REVERIE_VISUAL_WORKFLOW)
        await settle(rt)
        await rt._publish_outbox()
        [state] = [m for ch, kind, m in published if kind == DURABLE_RUN_STATE_KIND]
        assert isinstance(state, DurableRunStateV1)
        assert state.status == "failed" and state.node == "failed" and state.workflow == REVERIE_VISUAL_WORKFLOW
        assert state.entry_id == f"{RUN}:terminal:failed:state"
        for key in ("dispatch_id", "attempt_id", "retries", "last_error", "reason", "error"):
            assert key in state.detail
        assert state.detail["last_error"] == RETRY_WINDOW_EXPIRED and state.detail["retries"] == 3
        assert rt.store.acked == [f"{RUN}:terminal:failed"]
    asyncio.run(run())


def test_driver_side_terminal_abandons_the_attempt_and_a_cancel_keeps_the_dispatch_detail():
    async def run():
        world = World()
        thought = Thought(world)
        rt, published = runtime(REVERIE_VISUAL_WORKFLOW, thought)
        rt.store.control = "cancelled"
        await rt._terminal(RUN, "cancelled", _failed_state(), workflow=REVERIE_VISUAL_WORKFLOW)
        await settle(rt)
        [(_, abandon)] = thought.calls
        assert abandon.step == "abandon" and abandon.attempt_id == "att-1"
        assert rt.store.recorded_events() == ["run.abandon_pending", "run.abandon_acked"]
        assert await rt.store.abandons_pending() == [] and rt._abandons == {}
        await rt._publish_outbox()
        [event] = [m for ch, kind, m in published if kind == DURABLE_RUN_STATE_KIND]
        assert event.status == "cancelled" and event.detail["dispatch_id"] == "dispatch-1"
        assert event.detail["attempt_id"] == "att-1" and event.detail["last_error"] == "cancelled"
    asyncio.run(run())


def test_abandon_thought_misses_goes_terminal_anyway_and_reconcile_retries_until_acked():
    """Thought down (then deferring) at terminal time: the run still goes terminal, the pending
    record survives a durable-runs restart, and reconcile retries with backoff until thought answers."""
    async def run():
        world = World()
        thought = Thought(world, abandon=[(ConnectionError("thought down"), {}),
                                          ("retry", {"reason": "store_busy"}), ("done", {})])
        rt, published = runtime(REVERIE_VISUAL_WORKFLOW, thought)
        await rt._terminal(RUN, "failed", _failed_state(), workflow=REVERIE_VISUAL_WORKFLOW)
        await settle(rt)
        assert thought.steps() == ["abandon"]                     # attempted
        assert rt.store.terminal == "failed"                      # terminal anyway
        [pending] = await rt.store.abandons_pending()
        assert pending["detail"]["attempt_id"] == "att-1"
        # Backed off: a reconcile before the retry is due sends nothing.
        assert rt._abandons[RUN]["failures"] == 1
        await rt.reconcile()
        await settle(rt)
        assert thought.steps() == ["abandon"]

        # durable-runs restarts: the new process has no memory, only the store.
        rt2, _ = runtime(REVERIE_VISUAL_WORKFLOW, thought, store=rt.store)
        await rt2.reconcile()
        await settle(rt2)
        assert thought.steps() == ["abandon", "abandon"]          # retried; thought deferred
        assert len(await rt2.store.abandons_pending()) == 1 and rt2._abandons[RUN]["failures"] == 1
        rt2._abandons[RUN]["due"] = 0.0                           # backoff elapsed
        await rt2.reconcile()
        await settle(rt2)
        assert thought.steps() == ["abandon", "abandon", "abandon"]
        assert await rt2.store.abandons_pending() == [] and rt2._abandons == {}
        first, second, third = [req for _, req in thought.calls]
        assert len({first.correlation_id, second.correlation_id, third.correlation_id}) == 3
        assert {req.attempt_id for _, req in thought.calls} == {"att-1"}
        await rt2.reconcile()
        await settle(rt2)
        assert len(thought.calls) == 3                            # cleared: never re-sent
    asyncio.run(run())


def test_run_that_never_learned_its_attempt_abandons_by_dispatch_id():
    """Deadline before prepare answered (or prepare's reply lost): no attempt_id, but thought may
    still have claimed one for the dispatch -- abandon names the dispatch."""
    async def run():
        world, saver = World(), InMemorySaver()
        thought = Thought(world)
        result = await graph(world, saver, thought).ainvoke(initial(world, deadline=timedelta(seconds=-1)), CFG)
        assert result["status"] == "failed" and result["last_error"] == RETRY_WINDOW_EXPIRED
        assert result.get("attempt_id") is None and thought.steps() == []
        rt, _ = runtime(REVERIE_VISUAL_WORKFLOW, thought)
        await rt._terminal(RUN, "failed", result, workflow=REVERIE_VISUAL_WORKFLOW)
        await settle(rt)
        [(_, abandon)] = thought.calls
        assert abandon.step == "abandon" and abandon.attempt_id is None
        assert abandon.visual_request.dispatch_id == "dispatch-1"
        assert rt.store.recorded_events() == ["run.abandon_pending", "run.abandon_acked"]
    asyncio.run(run())


def test_terminal_abandon_answer_that_closed_nothing_stays_pending():
    """An older thought rejects an abandon without attempt_id as invalid (terminal): nothing was
    closed, so it is not an ack. attempt_mismatch (dispatch claimed by another request) is."""
    async def run():
        world = World()
        thought = Thought(world, abandon=[("terminal", {"outcome": "failed", "reason": "invalid_step_request"}),
                                          ("terminal", {"outcome": "failed", "reason": "attempt_mismatch"})])
        rt, _ = runtime(REVERIE_VISUAL_WORKFLOW, thought)
        await rt._terminal(RUN, "failed", _failed_state(), workflow=REVERIE_VISUAL_WORKFLOW)
        await settle(rt)
        assert rt.store.recorded_events() == ["run.abandon_pending"] and rt._abandons[RUN]["failures"] == 1
        rt._abandons[RUN]["due"] = 0.0
        await rt.reconcile()
        await settle(rt)
        assert rt.store.recorded_events() == ["run.abandon_pending", "run.abandon_acked"]
        assert rt._abandons == {}
    asyncio.run(run())


def test_unconfirmed_abandon_is_given_up_after_thoughts_own_sweep_would_have_released_it():
    from orion.durable_runs.registry_store import ABANDON_GIVE_UP_SEC

    async def run():
        world = World()
        thought = Thought(world, abandon=[(ConnectionError("thought down"), {})] * 3)
        rt, _ = runtime(REVERIE_VISUAL_WORKFLOW, thought)
        await rt._terminal(RUN, "failed", _failed_state(), workflow=REVERIE_VISUAL_WORKFLOW)
        await settle(rt)
        assert rt._abandons[RUN]["failures"] == 1
        rt._abandons[RUN].update(due=0.0, pending_since=time.time() - ABANDON_GIVE_UP_SEC - 1)
        await rt.reconcile()
        await settle(rt)
        assert rt._abandons == {} and len(thought.calls) == 2
        await rt.reconcile()
        await settle(rt)
        assert len(thought.calls) == 2                            # never re-sent
    asyncio.run(run())


def test_completed_or_paused_reverie_run_never_abandons():
    async def run():
        thought = Thought(World())
        rt, _ = runtime(REVERIE_VISUAL_WORKFLOW, thought)
        await rt._terminal(RUN, "completed", {**_failed_state(), "status": "completed"},
                           workflow=REVERIE_VISUAL_WORKFLOW)
        rt2, _ = runtime(REVERIE_VISUAL_WORKFLOW, thought)
        rt2.store.control = "paused"
        await rt2._terminal(RUN, "failed", _failed_state(), workflow=REVERIE_VISUAL_WORKFLOW)
        await settle(rt2)
        assert thought.calls == [] and rt.store.recorded == {} and rt2.store.recorded == {}
    asyncio.run(run())


@pytest.mark.parametrize("status", sorted(TERMINAL_STATE_NODE))
def test_every_admitted_terminal_of_every_workflow_reaches_the_state_channel(status):
    async def run():
        rt, published = runtime("self_study.reflect")
        state = {"run_id": RUN, "correlation_id": "trace-001", "status": status, "last_error": "boom",
                 "llm_call_ok": False, "findings": []}
        if status == "completed":
            rt.store.events.append({"schema_version": "durable.resource.event.v1",
                                    "entry_id": f"{RUN}:terminal:completed", "event": "run.completed",
                                    "run_id": RUN, "thread_id": RUN, "correlation_id": "trace-001",
                                    "generated_at": "2026-09-28T00:00:00+00:00", "detail": {}})
        else:
            await rt._terminal(RUN, status, state, workflow="self_study.reflect")
        await rt._publish_outbox()
        [event] = [m for ch, kind, m in published if kind == DURABLE_RUN_STATE_KIND]
        assert event.status == status and event.node == TERMINAL_STATE_NODE[status]
        if status == "failed":
            assert event.detail == {"error": "boom"}
    asyncio.run(run())
