"""Urgent preemption on the durable-runs side (Plan 2 Task 4).

An urgent run took this run's GPU slot mid-node: the pool re-queued the hold in its original place
(queued, reason ``urgent_preempt``). The node replays on the next grant under the same lease_id and
it is never a failed attempt. And reconcile drives urgent runs first, past MAX_CONCURRENT_DRIVERS.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

sys.path.insert(0, str(Path(__file__).parent))

from test_admitted_graph import CFG, World, initial
from app.admission_runtime import MAX_CONCURRENT_DRIVERS, AdmissionRuntime
from app.admitted_graph import GONE, GRANTED, WAITING, AdmissionDeps, HoldLost, HoldPreempted, HoldRecalled
from app.pool_hold import URGENT_PREEMPT, PoolHolds
from app.reading_graph import build_reading_graph
from orion.schemas.gpu_pool import GpuLeaseGrantV1, GpuLeaseReplyV1
from orion.schemas.reading_turn import ReadingTurnResultV1
from pool_fixture import CFG as POOL_CFG, InProcessPool, PoolBus

NOW = datetime(2026, 9, 28, tzinfo=timezone.utc)
HOLDER = "durable-runs:study-001"


# --- graph: HoldPreempted / HoldLost replay the node; a real turn failure spends an attempt --------

class PreemptWorld(World):
    def __init__(self, raise_once: BaseException):
        super().__init__()
        self.raise_once: BaseException | None = raise_once
        self.release_calls: list[tuple[str, bool]] = []

    async def execute(self, state, node):
        if self.raise_once is not None:
            exc, self.raise_once = self.raise_once, None
            self.pool_status = "queued"          # the pool put the hold back in line, same lease_id
            raise exc
        return await super().execute(state, node)

    async def release(self, state, reason, keep_requeued=False):
        self.release_calls.append((reason, keep_requeued))
        return await super().release(state, reason, keep_requeued)


async def _turn_update(graph, value):
    deltas = [update["harness_turn"] async for update in graph.astream(value, CFG, stream_mode="updates")
              if "harness_turn" in update]
    assert deltas, "harness_turn never ran"
    return deltas[0]


def test_preempted_turn_retries_now_under_the_same_hold_without_spending_an_attempt():
    async def scenario():
        world, saver = PreemptWorld(HoldPreempted("gpu_hold_preempted:hold-1")), InMemorySaver()
        world.grant()
        graph = world.graph(saver)
        delta = await _turn_update(graph, initial())
        assert delta["status"] == "retrying" and "attempt" not in delta
        assert delta["retry_at"] == world.now.isoformat() and delta["retry_node"] is None
        assert delta["hold"] == {"request_id": "study-001:1", "lease_id": "hold-1"}   # kept: same place
        assert world.release_calls == [("urgent_preempt", True)] and world.releases == []
        snap = await graph.aget_state(CFG)
        # retry_at is now: straight back to resource_request, which re-asks with the SAME request id.
        assert snap.next == ("resource_wait",) and snap.values["attempt"] == 0
        assert world.requests == ["study-001:1", "study-001:1"]
        world.grant()
        result = await graph.ainvoke(Command(resume=True), CFG)
        assert result["status"] == "completed" and world.calls == [1] and result["attempt"] == 1
    asyncio.run(scenario())


def test_a_lost_hold_waits_for_the_same_hold_without_spending_an_attempt():
    """Live 2026-09-26..28: a recall past its grace (gpu2 max_hold, an owner reclaim) came back as
    HoldLost and spent one of three attempts; the third recall failed the run. The pool took the seat
    back -- the run did nothing wrong."""
    async def scenario():
        world, saver = PreemptWorld(HoldLost("gpu_hold_lost:queued:recall_grace_exceeded")), InMemorySaver()
        world.grant()
        delta = await _turn_update(world.graph(saver), initial())
        assert delta["status"] == "retrying" and "attempt" not in delta
        assert delta["retry_at"] == world.now.isoformat()
        assert delta["hold"] == {"request_id": "study-001:1", "lease_id": "hold-1"}
        assert world.release_calls == [("hold_lost", True)]
    asyncio.run(scenario())


def test_a_real_turn_failure_still_spends_an_attempt_with_backoff():
    async def scenario():
        world, saver = PreemptWorld(RuntimeError("harness_boom")), InMemorySaver()
        world.grant()
        delta = await _turn_update(world.graph(saver), initial())
        assert delta["status"] == "retrying" and delta["attempt"] == 1
        assert delta["retry_at"] == (world.now + timedelta(seconds=30)).isoformat()
        assert world.release_calls == [("attempt_failed", True)]
    asyncio.run(scenario())


# --- reading: a failed turn comes back as a RESULT, not an exception ---------------------------------

def _reading_state():
    value = initial()
    value["brief"].update(seed_id="reading:seed", stage=1)
    return value


def _reading_graph(world, saver, turn, preempted):
    return build_reading_graph(turn, AdmissionDeps(world.register, world.lease, world.execute, world.release,
                                                   world.event, requeued=preempted), saver)


@pytest.mark.parametrize("preempted", [True, False])
def test_a_reading_turn_that_failed_because_its_hold_was_preempted_waits_instead_of_failing(preempted):
    async def scenario():
        world = World()
        world.grant()
        asked = []

        async def turn(request):
            world.pool_status = "queued" if preempted else "granted"
            return ReadingTurnResultV1(run_id=request.run_id, correlation_id=request.correlation_id,
                                       ok=False, error="gpu_lease_attach_refused")

        async def was_preempted(state):
            asked.append(state["lease"]["lease_id"])
            return URGENT_PREEMPT if preempted else None

        result = await _reading_graph(world, InMemorySaver(), turn, was_preempted).ainvoke(_reading_state(), CFG)
        assert asked == ["hold-1"]
        if preempted:
            assert result["status"] == "waiting_resource" and result.get("attempt", 0) == 0
            assert result["hold"] == {"request_id": "study-001:1", "lease_id": "hold-1"}
            assert world.releases == []
        else:
            assert result["status"] == "failed" and world.releases == ["failed"]
    asyncio.run(scenario())


def test_a_reading_turn_that_succeeded_never_asks_the_pool():
    async def scenario():
        world = World()
        world.grant()

        async def turn(request):
            return ReadingTurnResultV1(run_id=request.run_id, correlation_id=request.correlation_id,
                                       ok=True, text="Grounded result")

        was_preempted = AsyncMock(return_value=True)
        result = await _reading_graph(world, InMemorySaver(), turn, was_preempted).ainvoke(_reading_state(), CFG)
        assert result["status"] == "completed" and was_preempted.await_count == 0
    asyncio.run(scenario())


# --- reflect / self-sense: preemption waits for the hold instead of failing or finishing empty ------

class HeldAdmission:
    """One granted hold; ``execute`` raises ``raise_exc`` or runs the node; ``release`` keeps a
    re-queued hold the way the runtime does when the pool says queued."""

    def __init__(self, raise_exc=None, preempted=False):
        self.raise_exc, self.is_preempted = raise_exc, preempted
        self.release_calls: list[tuple[str, bool]] = []
        self.granted = True

    def deps(self):
        return AdmissionDeps(self.register, self.lease, self.execute, self.release, self.event,
                             now=lambda: NOW, max_attempts=3, requeued=self.preempted)

    async def register(self, state):
        return {"status": "waiting_resource", "hold": {"request_id": "r-1:1", "lease_id": "hold-1"}, "hold_seq": 1}

    async def lease(self, state):
        if self.granted:
            return GRANTED, {"status": "admitted", "lease": dict(LEASE), "hold": state.get("hold")}
        return WAITING, {"status": "waiting_resource", "lease": None}

    async def execute(self, state, node):
        if self.raise_exc is not None:
            self.granted = False
            raise self.raise_exc
        result = await node(state)
        if self.is_preempted:
            self.granted = False
        return result

    async def release(self, state, reason, keep_requeued=False):
        self.release_calls.append((reason, keep_requeued))
        if keep_requeued and not self.granted:
            return {"lease": None, "hold": state.get("hold")}
        return {"lease": None, "hold": None}

    async def preempted(self, state):
        return URGENT_PREEMPT if self.is_preempted else None

    async def event(self, *args):
        pass


def _stop_at_wait(graph, value, thread):
    async def run():
        await graph.ainvoke(value, {"configurable": {"thread_id": thread}})
        return await graph.aget_state({"configurable": {"thread_id": thread}})
    return run()


@pytest.mark.parametrize("how", ["raised", "returned"])
def test_a_preempted_reflect_call_waits_for_its_hold_instead_of_finishing_empty(how):
    from app.admitted_reflect_graph import build_admitted_reflect_graph
    from app.reflect_graph import Deps as ReflectDeps

    async def scenario():
        admission = HeldAdmission(HoldPreempted("gpu_hold_preempted:hold-1") if how == "raised" else None,
                                  preempted=how == "returned")

        async def call(reflect_input, llm_route, gpu_lease=None):
            return None                                        # the non-ok result a lost attach gives

        graph = build_admitted_reflect_graph(ReflectDeps(call_reflect_llm=call), admission.deps(), InMemorySaver())
        snap = await _stop_at_wait(graph, {"run_id": "r-1", "correlation_id": "c", "workflow": "self_study.reflect",
                                           "attempt": 0, "admission": {}, "brief": {"timeout_sec": 5}}, "r-1")
        assert snap.next == ("resource_wait",) and snap.values["status"] == "waiting_resource"
        assert snap.values["attempt"] == 0 and snap.values["hold"]["lease_id"] == "hold-1"
        assert admission.release_calls == [(URGENT_PREEMPT, True)]
    asyncio.run(scenario())


def test_a_preempted_self_sense_node_waits_for_its_hold_instead_of_failing_the_run():
    from app.admitted_self_sense_graph import build_admitted_self_sense_graph
    from app.self_sense_graph import Deps as SelfSenseDeps

    async def scenario():
        admission = HeldAdmission(HoldPreempted("gpu_hold_preempted:hold-1"))
        graph = build_admitted_self_sense_graph(SelfSenseDeps(run_turn=AsyncMock(), publish_rows=AsyncMock()),
                                                admission.deps(), InMemorySaver())
        snap = await _stop_at_wait(graph, {"run_id": "r-1", "correlation_id": "c", "workflow": "self_sense_eval",
                                           "attempt": 0, "admission": {}, "brief": {"timeout_sec": 5}}, "r-1")
        assert snap.next == ("resource_wait",) and snap.values["status"] == "waiting_resource"
        assert snap.values["hold"]["lease_id"] == "hold-1"
        assert admission.release_calls == [(URGENT_PREEMPT, True)]
    asyncio.run(scenario())


@pytest.mark.parametrize("preempted", [True, False])
def test_a_self_sense_question_that_came_back_empty(preempted):
    """ask_questions records a failed question as an empty answer (a result, not an exception)."""
    from app.admitted_self_sense_graph import build_admitted_self_sense_graph
    from app.self_sense_graph import Deps as SelfSenseDeps
    from orion.schemas.durable_run import CuriosityTurnResultV1
    from test_admitted_self_sense_graph import _brief

    async def scenario():
        admission = HeldAdmission(preempted=preempted)
        asked, published = [], []

        async def turn(req):
            asked.append(req.prompt)
            text = "" if len(asked) == 2 else "an answer"                  # the second question fails
            return CuriosityTurnResultV1(run_id=req.run_id, correlation_id=req.correlation_id, text=text,
                                         ok=bool(text))

        async def publish_rows(rows):
            published.extend(rows)
            return len(rows), 0

        graph = build_admitted_self_sense_graph(SelfSenseDeps(run_turn=turn, publish_rows=publish_rows),
                                                admission.deps(), InMemorySaver())
        snap = await _stop_at_wait(graph, {"run_id": "r-1", "correlation_id": "c", "workflow": "self_sense_eval",
                                           "attempt": 0, "admission": {}, "brief": _brief()}, "r-1")
        if preempted:
            # The node replays under the kept hold; nothing from the failed pass is kept or published.
            assert snap.next == ("resource_wait",) and snap.values["status"] == "waiting_resource"
            assert snap.values["hold"]["lease_id"] == "hold-1" and not snap.values.get("answers")
            assert admission.release_calls == [(URGENT_PREEMPT, True)] and published == []
        else:
            # A real empty answer: published as before, the run completes.
            assert snap.next == () and snap.values["status"] == "completed"
            assert len(published) == len(asked) == 4
            assert admission.release_calls == [("completed", False)]
    asyncio.run(scenario())


# --- runtime: _beat / execute / release against scripted pool replies --------------------------------

def _reply(status, *, generation=1, reason=None, recall_in=5.0):
    grant = None
    if status in ("granted", "recall"):
        grant = GpuLeaseGrantV1(lease_id="hold-1", generation=generation, role="agent", cards=["gpu1"],
                                url="http://agent", served_by="circe")
    recall_by = NOW + timedelta(seconds=recall_in) if status == "recall" else None
    return GpuLeaseReplyV1(status=status, lease_id="hold-1", grant=grant, reason=reason, recall_by=recall_by)


class Holds:
    def __init__(self, heartbeats=(), status=None):
        self.heartbeats = list(heartbeats)
        self.statuses = list(status) if isinstance(status, (list, tuple)) else [status]
        self.status_calls = 0
        self.released: list[str] = []
        self.cfg = POOL_CFG

    async def heartbeat(self, lease_id):
        return self.heartbeats.pop(0) if len(self.heartbeats) > 1 else self.heartbeats[0]

    async def status(self, lease_id):
        self.status_calls += 1
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]

    async def release(self, lease_id, *, outcome="ok", detail=None):
        self.released.append(detail)
        return GpuLeaseReplyV1(status="ok", lease_id=lease_id)


class Store:
    def __init__(self, rows=()):
        self.events: list[tuple[str, str, dict, str | None]] = []
        self.rows = list(rows)

    async def get_run(self, run_id):
        return {"run_id": run_id, "control": None}

    async def record_event(self, run_id, event, detail, event_id=None):
        self.events.append((run_id, event, detail, event_id))

    def names(self):
        return [e[1] for e in self.events]

    async def list_pending(self, limit=100):
        return list(self.rows)

    async def outreach_holds_pending(self, max_age_seconds):
        return []

    async def abandons_pending(self, limit=100):
        return []

    async def pending_outbox(self):
        return []


def bare_runtime(holds, store=None):
    rt = object.__new__(AdmissionRuntime)
    rt.holds, rt.store = holds, store or Store()
    rt.now = lambda: NOW
    rt.settings = SimpleNamespace(lease_heartbeat_sec=0.01, hold_status_poll_sec=60.0, outreach_hold_max_sec=1800.0,
                                  retry_base_sec=30.0, retry_max_sec=300.0)
    rt.runner = SimpleNamespace(_publish=AsyncMock(return_value=True), _corr_for_admission=lambda c: c)
    rt.outreach, rt._pending_release, rt.active = {}, {}, {}
    rt._hints, rt._checked, rt._wake = set(), {}, asyncio.Event()
    rt._outreach_loaded_at = None
    rt._abandons, rt._abandons_loaded_at, rt._abandoning = {}, None, {}
    rt._urgent_drivers = set()
    rt._system_drivers = set()
    return rt


LEASE = {"lease_id": "hold-1", "generation": 1, "role": "agent", "holder": HOLDER}


def _held_state(workflow="self_study.reflect"):
    return {"run_id": "study-001", "correlation_id": "trace-001", "workflow": workflow, "attempt": 0,
            "lease": dict(LEASE), "hold": {"request_id": "study-001:1", "lease_id": "hold-1"},
            "brief": {"timeout_sec": 5}, "admission": {}}


def test_beat_on_a_hold_requeued_for_urgent_work_raises_preempted_and_a_plain_requeue_is_lost():
    """Both are HoldLost (neither spends an attempt); only the urgent one is HoldPreempted."""
    async def scenario():
        rt = bare_runtime(Holds([_reply("queued", reason=URGENT_PREEMPT)]))
        with pytest.raises(HoldPreempted):
            await rt._beat(LEASE)
        rt = bare_runtime(Holds([_reply("queued")]))
        with pytest.raises(HoldLost) as lost:
            await rt._beat(LEASE)
        assert not isinstance(lost.value, HoldPreempted)
        rt = bare_runtime(Holds([_reply("recall", reason=URGENT_PREEMPT)]))
        assert (await rt._beat(LEASE)).status == "recall"   # the 5 s grace: keep working
    asyncio.run(scenario())


@pytest.mark.parametrize("pool_says", ["preempted", "granted"])
def test_a_work_failure_while_the_pool_says_preempted_becomes_hold_preempted(pool_says):
    async def scenario():
        status = _reply("queued", reason=URGENT_PREEMPT) if pool_says == "preempted" else _reply("granted")
        rt = bare_runtime(Holds([_reply("granted")], status=status))

        async def node(state):
            raise RuntimeError("gpu_lease_attach_refused")

        with pytest.raises(RuntimeError) as raised:
            await rt.execute(_held_state(), node)
        if pool_says == "preempted":
            assert isinstance(raised.value, HoldPreempted)
            assert isinstance(raised.value.__cause__, RuntimeError)
            assert "attach_refused" in str(raised.value.__cause__)
        else:
            assert type(raised.value) is RuntimeError and "attach_refused" in str(raised.value)
    asyncio.run(scenario())


def test_a_preempt_seen_by_the_heartbeat_stops_the_turn_and_cancels_it_in_hub():
    async def scenario():
        rt = bare_runtime(Holds([_reply("granted"), _reply("recall"), _reply("queued", reason=URGENT_PREEMPT)]))
        started, cancelled = asyncio.Event(), []

        async def node(state):
            started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        with pytest.raises(HoldPreempted):
            await asyncio.wait_for(rt.execute(_held_state("curiosity.investigate"), node), 2)
        assert started.is_set() and cancelled == [True]
        channel = rt.runner._publish.await_args.args[0]
        assert channel == "orion:harness:run:cancel"
    asyncio.run(scenario())


def test_the_pool_and_durable_runs_share_one_preempt_reason():
    from orion.schemas.gpu_pool import URGENT_PREEMPT as SCHEMA_PREEMPT
    assert URGENT_PREEMPT is SCHEMA_PREEMPT


def test_after_an_urgent_recall_the_heartbeat_polls_fast_so_the_requeue_frees_the_slot_within_a_second(
        monkeypatch):
    """The paused run's in-flight call keeps the slot the urgent run is waiting for until the
    harness is cancelled: past the grace the next beat must come within PREEMPT_POLL_SEC, not a
    whole heartbeat interval later."""
    import app.admission_runtime as runtime_module
    monkeypatch.setattr(runtime_module, "PREEMPT_POLL_SEC", 0.01)

    async def scenario():
        holds = Holds([_reply("granted"), _reply("recall", reason=URGENT_PREEMPT),
                       _reply("queued", reason=URGENT_PREEMPT)])
        rt = bare_runtime(holds)
        rt.settings.lease_heartbeat_sec = 0.5
        cancelled = []

        async def node(state):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(HoldPreempted):
            await asyncio.wait_for(rt.execute(_held_state("curiosity.investigate"), node), 5)
        elapsed = loop.time() - started
        assert cancelled == [True]
        assert elapsed < 0.5 + 0.3, elapsed          # one heartbeat, then a fast poll -- not two heartbeats
    asyncio.run(scenario())


@pytest.mark.parametrize("cap, expected", [(3, 0.05), (0, 0.2)])
def test_a_held_run_beats_within_the_urgent_grace_so_a_pause_is_seen_before_the_abort(cap, expected):
    """A heartbeat longer than the grace would usually first see the pause after the abort, leaving
    the paused call on the slot for up to a whole heartbeat. Rollback (cap 0) keeps the heartbeat."""
    async def scenario():
        holds = Holds([_reply("granted")])
        holds.cfg = POOL_CFG.model_copy(update={"defaults": POOL_CFG.defaults.model_copy(
            update={"urgent_preempt_grace_sec": 0.05, "urgent_max_concurrent": cap})})
        rt = bare_runtime(holds)
        rt.settings.lease_heartbeat_sec = 0.2
        loop = asyncio.get_running_loop()
        beats = []
        real_beat = rt._beat

        async def timed(lease, admission=None):
            beats.append(loop.time())
            return await real_beat(lease, admission)

        rt._beat = timed

        async def node(state):
            await asyncio.sleep(0.3)
            return "done"

        assert await asyncio.wait_for(rt.execute(_held_state("curiosity.investigate"), node), 5) == "done"
        gap = beats[1] - beats[0]                  # beats[0] is the pre-step check
        assert abs(gap - expected) < 0.04, (gap, expected)
    asyncio.run(scenario())


def test_an_other_recall_keeps_the_normal_heartbeat(monkeypatch):
    import app.admission_runtime as runtime_module
    monkeypatch.setattr(runtime_module, "PREEMPT_POLL_SEC", 0.001)

    async def scenario():
        holds = Holds([_reply("granted"), _reply("recall", reason="owner_waiting")])
        rt = bare_runtime(holds)
        rt.settings.lease_heartbeat_sec = 0.05
        beats = []
        real_beat = rt._beat

        async def counted(lease, admission=None):
            beats.append(1)
            return await real_beat(lease, admission)

        rt._beat = counted

        async def node(state):
            await asyncio.sleep(0.3)
            return "done"

        assert await asyncio.wait_for(rt.execute(_held_state("curiosity.investigate"), node), 5) == "done"
        assert len(beats) <= 1 + 0.3 / 0.05 + 1, len(beats)
    asyncio.run(scenario())


def test_release_keeps_a_preempted_hold_and_records_the_preemption_not_an_expiry():
    async def scenario():
        holds = Holds(status=_reply("queued", reason=URGENT_PREEMPT))
        rt = bare_runtime(holds)
        update = await rt.release(_held_state(), URGENT_PREEMPT, keep_requeued=True)
        assert update == {"lease": None, "hold": {"request_id": "study-001:1", "lease_id": "hold-1"}}
        assert holds.released == []                                   # never handed back to the pool
        [(run_id, name, detail, event_id)] = rt.store.events
        assert name == "run.preempted" and detail["lease_id"] == "hold-1" and detail["generation"] == 1
        assert detail["reason"] == URGENT_PREEMPT and event_id == "preempted:hold-1:1"
    asyncio.run(scenario())


# --- paused before the step started: wait for the in-place re-queue, never release ----------------

@pytest.fixture
def fast_poll(monkeypatch):
    import app.admission_runtime as runtime_module
    monkeypatch.setattr(runtime_module, "PREEMPT_POLL_SEC", 0.001)
    return runtime_module


def _never_runs(state):
    raise AssertionError("the step must not start on a recalled hold")


def test_an_urgent_recall_before_the_step_waits_for_the_requeue_and_keeps_the_hold(fast_poll):
    async def scenario():
        holds = Holds([_reply("recall", reason=URGENT_PREEMPT)],
                      status=[_reply("recall", reason=URGENT_PREEMPT), _reply("recall", reason=URGENT_PREEMPT),
                              _reply("queued", reason=URGENT_PREEMPT)])
        rt = bare_runtime(holds)
        with pytest.raises(HoldPreempted):
            await asyncio.wait_for(rt.execute(_held_state(), _never_runs), 2)
        assert holds.released == [] and holds.status_calls == 3
        update = await rt.release(_held_state(), URGENT_PREEMPT, keep_requeued=True)
        assert update["hold"] == {"request_id": "study-001:1", "lease_id": "hold-1"}   # same place, same id
        assert holds.released == [] and "run.preempted" in rt.store.names()
    asyncio.run(scenario())


def test_any_other_recall_before_the_step_still_hands_the_hold_back_at_once(fast_poll):
    async def scenario():
        holds = Holds([_reply("recall", reason="owner_waiting")], status=_reply("queued", reason=URGENT_PREEMPT))
        rt = bare_runtime(holds)
        with pytest.raises(HoldRecalled):
            await rt.execute(_held_state(), _never_runs)
        assert holds.released == ["recalled_before_start"] and holds.status_calls == 0
    asyncio.run(scenario())


def test_an_urgent_recall_the_pool_never_aborts_falls_back_to_releasing_it(fast_poll, monkeypatch):
    monkeypatch.setattr(fast_poll, "PREEMPT_REQUEUE_MARGIN_SEC", 0.05)

    async def scenario():
        recall = _reply("recall", reason=URGENT_PREEMPT, recall_in=0.02)    # grace nearly over
        holds = Holds([recall], status=recall)
        rt = bare_runtime(holds)
        with pytest.raises(HoldRecalled):
            await asyncio.wait_for(rt.execute(_held_state(), _never_runs), 2)
        assert holds.status_calls >= 1 and holds.released == ["recalled_before_start"]
    asyncio.run(scenario())


@pytest.mark.parametrize("recall_in,expected", [(600, "grace"), (-30, "grace"), (2.0, 2.0)])
def test_the_requeue_wait_trusts_the_pools_recall_by_only_to_shorten_it(fast_poll, recall_in, expected):
    """recall_by is the pool's clock: far off or already past (clock skew) means the local grace;
    only a sane, sooner recall_by shortens the wait."""
    rt = bare_runtime(Holds())
    grace = POOL_CFG.defaults.urgent_preempt_grace_sec
    wait = rt._requeue_wait_sec(_reply("recall", reason=URGENT_PREEMPT, recall_in=recall_in))
    assert wait == (grace if expected == "grace" else expected) + fast_poll.PREEMPT_REQUEUE_MARGIN_SEC


def test_a_requeue_for_another_reason_is_not_an_urgent_pause(fast_poll):
    async def scenario():
        holds = Holds([_reply("recall", reason=URGENT_PREEMPT)],
                      status=[_reply("recall", reason=URGENT_PREEMPT), _reply("queued", reason="expired")])
        rt = bare_runtime(holds)
        with pytest.raises(HoldRecalled):
            await asyncio.wait_for(rt.execute(_held_state(), _never_runs), 2)
        assert holds.released == ["recalled_before_start"] and "run.preempted" not in rt.store.names()
    asyncio.run(scenario())


def test_a_hanging_status_call_cannot_stretch_the_requeue_wait(fast_poll, monkeypatch):
    monkeypatch.setattr(fast_poll, "PREEMPT_REQUEUE_MARGIN_SEC", 0.2)

    class HangingHolds(Holds):
        async def status(self, lease_id):
            self.status_calls += 1
            await asyncio.sleep(60)

    async def scenario():
        holds = HangingHolds([_reply("recall", reason=URGENT_PREEMPT, recall_in=0.05)])
        rt = bare_runtime(holds)
        started = asyncio.get_running_loop().time()
        with pytest.raises(HoldRecalled):
            await asyncio.wait_for(rt.execute(_held_state(), _never_runs), 5)
        elapsed = asyncio.get_running_loop().time() - started
        assert elapsed < 0.05 + 0.2 + 0.1 + 0.2              # bound + one min status slice + slack
        assert holds.status_calls >= 1 and holds.released == ["recalled_before_start"]
    asyncio.run(scenario())


def test_a_real_work_error_near_the_budget_end_is_not_replaced_by_a_timeout():
    """The preempt check after a work failure reads the pool outside the turn's time budget: a slow
    read must not turn the turn's own error into a TimeoutError."""
    class SlowStatus(Holds):
        async def status(self, lease_id):
            await asyncio.sleep(0.3)
            return _reply("granted")

    async def scenario():
        rt = bare_runtime(SlowStatus([_reply("granted")]))

        async def node(state):
            await asyncio.sleep(0.1)
            raise RuntimeError("real_turn_error")

        state = {**_held_state(), "brief": {"timeout_sec": 0.2}}
        with pytest.raises(RuntimeError) as raised:
            await asyncio.wait_for(rt.execute(state, node), 3)
        assert type(raised.value) is RuntimeError and "real_turn_error" in str(raised.value)
    asyncio.run(scenario())


@pytest.mark.parametrize("reason", [URGENT_PREEMPT, "owner_waiting"])
def test_a_granted_then_recalled_hold_seen_in_resource_wait(fast_poll, reason):
    async def scenario():
        holds = Holds(status=[_reply("recall", reason=reason), _reply("recall", reason=reason),
                              _reply("queued", reason=URGENT_PREEMPT)])
        rt = bare_runtime(holds)
        state = {"run_id": "study-001", "admission": {}, "hold": {"request_id": "study-001:1", "lease_id": "hold-1"}}
        outcome, update = await asyncio.wait_for(rt.lease(state), 2)
        if reason == URGENT_PREEMPT:
            assert outcome == WAITING and "hold" not in update       # checkpointed hold stays: same place
            assert holds.released == [] and "run.preempted" in rt.store.names()
        else:
            assert outcome == GONE and update["hold"] is None
            assert holds.released == ["recalled_before_start"]
    asyncio.run(scenario())


# --- the REAL pool: pause, keep the place, pick the same lease back up -------------------------------

def test_preempted_hold_is_kept_through_release_and_regranted_under_the_same_lease_id():
    async def scenario():
        pool = await InProcessPool().boot()
        holds = PoolHolds(PoolBus(pool), source="durable-runs-test", cfg=POOL_CFG)
        rt = bare_runtime(holds)
        admission = {"resource": "llm.route.agent", "preferred_lane": "agent", "priority": "background"}
        first = await holds.acquire("study-001", "study-001:1", admission, correlation_id="trace-001")
        assert first.status == "granted"
        state = {"run_id": "study-001", "admission": admission,
                 "hold": {"request_id": "study-001:1", "lease_id": first.lease_id}}
        outcome, update = await rt.lease(state)
        assert outcome == GRANTED
        lease = update["lease"]

        urgent = await holds.acquire("urgent-001", "urgent-001:1", {**admission, "priority": "urgent"},
                                     correlation_id="trace-u")
        assert urgent.status == "queued"
        beat = await rt._beat(lease)
        assert beat.status == "recall" and beat.reason == URGENT_PREEMPT   # grace: the turn keeps going
        await pool.later(POOL_CFG.defaults.urgent_preempt_grace_sec, beat=[urgent.lease_id])
        with pytest.raises(HoldPreempted):
            await rt._beat(lease)

        kept = await rt.release({**state, "lease": lease}, URGENT_PREEMPT, keep_requeued=True)
        assert kept["hold"] == state["hold"] and kept["lease"] is None
        assert "run.preempted" in rt.store.names() and "resource.lease_expired" not in rt.store.names()
        assert (await pool.lease(first.lease_id))["status"] == "queued"
        again = await holds.acquire("study-001", "study-001:1", admission, correlation_id="trace-001")
        assert again.status == "queued" and again.lease_id == first.lease_id   # the re-ask is idempotent
        kept_state = {**state, "hold": kept["hold"]}
        assert (await rt.lease(kept_state))[0] == WAITING

        await pool.later(1, beat=[urgent.lease_id])
        await holds.release(urgent.lease_id, detail="done")
        await pool.later(1)
        outcome, update = await rt.lease(kept_state)
        assert outcome == GRANTED and update["lease"]["lease_id"] == first.lease_id
        assert update["lease"]["generation"] > lease["generation"]
    asyncio.run(scenario())


def test_a_hold_recalled_before_its_step_is_requeued_in_place_by_the_real_pool(fast_poll):
    async def scenario():
        pool = await InProcessPool().boot()
        holds = PoolHolds(PoolBus(pool), source="durable-runs-test", cfg=POOL_CFG)
        rt = bare_runtime(holds)
        rt.now = pool.clock
        admission = {"resource": "llm.route.agent", "preferred_lane": "agent", "priority": "background"}
        first = await holds.acquire("study-001", "study-001:1", admission, correlation_id="trace-001")
        state = {**_held_state(), "admission": admission,
                 "hold": {"request_id": "study-001:1", "lease_id": first.lease_id}}
        state["lease"] = (await rt.lease(state))[1]["lease"]
        created_at = (await pool.lease(first.lease_id))["created_at"]
        urgent = await holds.acquire("urgent-001", "urgent-001:1", {**admission, "priority": "urgent"},
                                     correlation_id="trace-u")

        async def pool_aborts_after_grace():
            await asyncio.sleep(0.02)
            await pool.later(POOL_CFG.defaults.urgent_preempt_grace_sec, beat=[urgent.lease_id])

        aborting = asyncio.create_task(pool_aborts_after_grace())
        with pytest.raises(HoldPreempted):
            await asyncio.wait_for(rt.execute(state, _never_runs), 5)
        await aborting
        kept = await rt.release(state, URGENT_PREEMPT, keep_requeued=True)
        assert kept["hold"]["lease_id"] == first.lease_id
        row = await pool.lease(first.lease_id)
        assert row["status"] == "queued" and row["created_at"] == created_at      # its original place
        assert not [r for r in pool.requests if r.verb == "release" and r.lease_id == first.lease_id]
    asyncio.run(scenario())


# --- reconcile: urgent first, past MAX_CONCURRENT_DRIVERS, capped -----------------------------------

def _row(run_id, priority="background"):
    return {"run_id": run_id, "control": None, "request": {"admission": {"priority": priority}}}


def _driving(rt):
    order, gate = [], asyncio.Event()

    async def drive(row):
        order.append(row["run_id"])
        await gate.wait()

    rt._drive = drive
    return order, gate


async def _settle():
    for _ in range(3):
        await asyncio.sleep(0)


def test_an_urgent_run_is_driven_even_when_every_background_driver_slot_is_busy():
    async def scenario():
        rt = bare_runtime(Holds(), Store([_row(f"bg-{i}") for i in range(6)] + [_row("u-1", "urgent")]))
        order, gate = _driving(rt)
        await rt.reconcile()
        await _settle()
        assert order[0] == "u-1"                                       # urgent first
        assert order[1:] == [f"bg-{i}" for i in range(MAX_CONCURRENT_DRIVERS)]
        assert set(rt.active) == {"u-1", *(f"bg-{i}" for i in range(MAX_CONCURRENT_DRIVERS))}
        gate.set()
        await asyncio.gather(*rt.active.values())
        assert rt.active == {} and rt._urgent_drivers == set()
    asyncio.run(scenario())


def test_urgent_drivers_are_capped_at_the_pools_urgent_max_concurrent():
    async def scenario():
        cap = POOL_CFG.defaults.urgent_max_concurrent
        rows = [_row(f"u-{i}", "urgent") for i in range(cap + 2)] + [_row("bg-0")]
        rt = bare_runtime(Holds(), Store(rows))
        order, gate = _driving(rt)
        await rt.reconcile()
        await _settle()
        assert order == [f"u-{i}" for i in range(cap)] + ["bg-0"]
        await rt.reconcile()                                           # still capped on the next tick
        await _settle()
        assert len(order) == cap + 1
        gate.set()
        await asyncio.gather(*rt.active.values())
    asyncio.run(scenario())


def test_urgent_max_concurrent_zero_drives_urgent_like_background():
    async def scenario():
        holds = Holds()
        holds.cfg = POOL_CFG.model_copy(update={"defaults": POOL_CFG.defaults.model_copy(
            update={"urgent_max_concurrent": 0})})
        rows = [_row(f"bg-{i}") for i in range(MAX_CONCURRENT_DRIVERS)] + [_row("u-1", "urgent")]
        rt = bare_runtime(holds, Store(rows))
        order, gate = _driving(rt)
        await rt.reconcile()
        await _settle()
        assert order == [f"bg-{i}" for i in range(MAX_CONCURRENT_DRIVERS)]   # list order, no bypass
        gate.set()
        await asyncio.gather(*rt.active.values())
    asyncio.run(scenario())


# --- any pool take-back, not only urgent (2026-09-29) -----------------------------------------------

def test_taken_back_classifies_every_mid_node_pool_answer():
    """A re-queue for any reason or a newer-generation grant is HoldLost (same hold, no attempt); an
    ended hold is HoldLost unless the reason is the run's own (deadline -> WorkflowDeadline, a class
    nothing serves -> a plain error for the attempt path)."""
    from app.admitted_graph import WorkflowDeadline
    rt = bare_runtime(Holds([_reply("granted")]))
    cases = {
        ("queued", "recall_grace_exceeded", 1): HoldLost,
        ("queued", None, 1): HoldLost,
        ("backlogged", "no_serviceable_role", 1): HoldLost,
        ("granted", None, 2): HoldLost,
        ("recall", "max_hold", 2): HoldLost,
        ("unavailable", "recall_grace_exceeded", 1): HoldLost,
        ("ok", None, 1): HoldLost,
        ("unavailable", "deadline", 1): WorkflowDeadline,
        ("unavailable", "unknown_class:nothing-serves-this", 1): RuntimeError,
    }
    for (status, reason, generation), expected in cases.items():
        exc = rt._taken_back(LEASE, _reply(status, reason=reason, generation=generation), {})
        assert type(exc) is expected, (status, reason, exc)
        assert not isinstance(exc, HoldPreempted)
    assert type(rt._taken_back(LEASE, _reply("queued", reason=URGENT_PREEMPT), {})) is HoldPreempted
    assert HoldLost("x").release_reason == "hold_lost" and HoldPreempted("x").release_reason == URGENT_PREEMPT


def test_a_work_failure_after_a_non_urgent_requeue_is_a_take_back_not_an_attempt():
    async def scenario():
        rt = bare_runtime(Holds([_reply("granted")], status=_reply("queued", reason="recall_grace_exceeded")))

        async def node(state):
            raise RuntimeError("gpu_lease_attach_refused")

        with pytest.raises(HoldLost) as raised:
            await rt.execute(_held_state(), node)
        assert type(raised.value) is HoldLost and raised.value.release_reason == "hold_lost"
        assert await rt.requeued(_held_state()) == "hold_lost"
        rt = bare_runtime(Holds([_reply("granted")], status=_reply("granted")))
        assert await rt.requeued(_held_state()) is None                   # still ours: a real failure
    asyncio.run(scenario())


@pytest.mark.parametrize("reason,kept", [("hold_lost", True), (URGENT_PREEMPT, True), ("attempt_failed", False)])
def test_release_keeps_a_regranted_hold_only_on_a_take_back(reason, kept):
    """A take-back goes straight back to resource_wait, which reads the new grant; a failed attempt
    backs off with nothing heartbeating the hold, so the seat is handed back instead of idling."""
    async def scenario():
        holds = Holds([_reply("granted")], status=_reply("granted", generation=2))
        rt = bare_runtime(holds)
        update = await rt.release(_held_state(), reason, keep_requeued=True)
        if kept:
            assert update["hold"]["lease_id"] == "hold-1" and holds.released == []
            assert "resource.lease_expired" in rt.store.names()
        else:
            assert update["hold"] is None and holds.released == [reason]
    asyncio.run(scenario())


def test_the_take_back_limit_fails_the_run_instead_of_replaying_forever():
    async def scenario():
        world, saver = PreemptWorld(HoldLost("gpu_hold_lost:queued:recall_grace_exceeded")), InMemorySaver()
        world.grant()
        graph = world.graph(saver, max_takebacks=1)
        delta = await _turn_update(graph, {**initial(), "hold_takebacks": 1})
        assert delta["status"] == "failed" and delta["hold_takebacks"] == 2
        assert delta["last_error"].startswith("hold_takeback_limit:1: HoldLost")
        assert world.release_calls[0] == ("hold_takeback_limit", False)
        world, saver = PreemptWorld(HoldLost("gpu_hold_lost:queued:recall_grace_exceeded")), InMemorySaver()
        world.grant()
        delta = await _turn_update(world.graph(saver, max_takebacks=2), {**initial(), "hold_takebacks": 1})
        assert delta["status"] == "retrying" and delta["hold_takebacks"] == 2 and "attempt" not in delta
    asyncio.run(scenario())


def test_a_system_run_is_driven_ahead_of_background_even_when_every_background_slot_is_busy():
    """memory.episode_distill (2026-10-02) runs at "system" priority: it must reach the pool
    while four background turns are already driving, and rank behind urgent."""
    from app.admission_runtime import MAX_CONCURRENT_SYSTEM_DRIVERS

    async def scenario():
        rows = [_row(f"bg-{i}") for i in range(6)] + [_row("s-1", "system"), _row("s-2", "system"),
                                                     _row("u-1", "urgent")]
        rt = bare_runtime(Holds(), Store(rows))
        order, gate = _driving(rt)
        await rt.reconcile()
        await _settle()
        assert order[0] == "u-1"
        assert order[1:1 + MAX_CONCURRENT_SYSTEM_DRIVERS] == ["s-1"]          # capped at one system driver
        assert order[1 + MAX_CONCURRENT_SYSTEM_DRIVERS:] == [f"bg-{i}" for i in range(MAX_CONCURRENT_DRIVERS)]
        assert "s-2" not in rt.active
        gate.set()
        await asyncio.gather(*rt.active.values())
        assert rt.active == {} and rt._system_drivers == set()
    asyncio.run(scenario())
