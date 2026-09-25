"""Actual LangGraph interrupts around the curiosity graph, with a fake pool and a fake clock.

The admission deps here model the GPU pool's answers (queued / granted / recalled / refused); the
real pool runtime is exercised end to end in test_pool_hold_runtime_postgres.py and
test_durable_acceptance.py.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parents[1])]

from app.admitted_graph import GONE, GRANTED, REFUSED, WAITING, AdmissionDeps, build_admitted_graph
from app.graph import Deps, finish_detail
from orion.schemas.durable_run import CuriosityTurnResultV1

HOLDER = "durable-runs:study-001"


class World:
    """A pool with one hold: ``queued`` until ``grant()``; ``recall()``/``refuse()`` change it."""

    def __init__(self):
        self.now = datetime(2026, 9, 12, tzinfo=timezone.utc)
        self.pool_status = "queued"
        self.generation = 0
        self.requests = []      # request ids asked for
        self.calls = []         # turn attempts
        self.turn_requests = []
        self.releases = []
        self.guard_status = "granted"
        self.fail = False
        self.events = []

    def grant(self):
        self.pool_status, self.generation = "granted", self.generation + 1

    def ref(self):
        return {"lease_id": "hold-1", "generation": self.generation, "role": "agent-gpu2", "holder": HOLDER}

    async def turn(self, req):
        self.calls.append(req.attempt)
        self.turn_requests.append(req)
        return CuriosityTurnResultV1(run_id=req.run_id, correlation_id=req.correlation_id,
                                    text="A grounded study finding." if not self.fail else "", ok=not self.fail)

    async def read(self, run_id):
        return {"graph_readable": True, "hops": [[1, "inspected evidence"]]}

    async def row(self, facts):
        return True

    async def journal(self, entry):
        return entry.entry_id

    async def register(self, state):
        hold = state.get("hold") or {}
        seq = int(state.get("hold_seq") or 0) + (0 if hold else 1)
        request_id = hold.get("request_id") or f"study-001:{seq}"
        self.requests.append(request_id)
        if self.pool_status == "released":
            self.pool_status = "queued"
        return {"status": "waiting_resource", "hold": {"request_id": request_id, "lease_id": "hold-1"}, "hold_seq": seq}

    async def lease(self, state):
        if self.pool_status == "granted":
            return GRANTED, {"status": "admitted", "lease": self.ref(), "hold": state.get("hold")}
        if self.pool_status == "queued":
            return WAITING, {"status": "waiting_resource", "lease": None}
        if self.pool_status == "refused":
            return REFUSED, {"status": "failed", "last_error": "gpu_pool_unavailable:backlog_max_age",
                             "lease": None, "hold": None}
        return GONE, {"status": "waiting_resource", "lease": None, "hold": None}

    async def execute(self, state, node):
        if self.pool_status != "granted" or state["lease"] != self.ref():
            raise RuntimeError("gpu_hold_lost")
        return await node(state)

    async def release(self, state, reason, keep_requeued=False):
        if not state.get("hold") and not state.get("lease"):
            return {"lease": None, "hold": None}   # nothing held: nothing to release (as the runtime)
        if keep_requeued and self.pool_status == "queued":
            return {"lease": None, "hold": state.get("hold")}
        self.releases.append(reason)
        self.pool_status = "released"
        return {"lease": None, "hold": None}

    async def guard(self, state):
        if not state.get("lease"):
            return None
        if self.guard_status == "recall":
            await self.release(state, "recalled")
            return None
        return state["lease"]

    async def event(self, state, name, detail):
        self.events.append((name, detail))

    async def keep(self, state):
        self.events.append(("kept_for_outreach", state["lease"]))

    def graph(self, saver):
        return build_admitted_graph(Deps(self.turn, self.read, self.row, self.journal),
            AdmissionDeps(self.register, self.lease, self.execute, self.release, self.event,
                          now=lambda: self.now, max_attempts=2, guard=self.guard, keep_for_outreach=self.keep), saver)


def initial():
    return {"run_id": "study-001", "correlation_id": "trace-001", "attempt": 0,
            "admission": {"resource": "llm.route.agent"},
            "brief": {"prompt": "Study evidence.", "session_id": "curiosity", "timeout_sec": 0.05}}


CFG = {"configurable": {"thread_id": "study-001"}}


def test_wait_is_checkpointed_without_turn_or_timeout_and_restart_resumes_once():
    async def scenario():
        world, saver = World(), InMemorySaver()
        graph = world.graph(saver)
        await asyncio.wait_for(graph.ainvoke(initial(), CFG), 1)
        snap = await graph.aget_state(CFG)
        assert snap.next == ("resource_wait",) and snap.tasks[0].interrupts
        assert snap.values["hold"] == {"request_id": "study-001:1", "lease_id": "hold-1"}
        assert world.calls == [] and world.requests == ["study-001:1"]
        world.now += timedelta(days=2)  # queue is independent of legacy 24h age / 50ms turn timeout
        restarted = world.graph(saver)
        world.grant()
        result = await restarted.ainvoke(Command(resume={"untrusted_grant": "ignored"}), CFG)
        assert result["status"] == "completed" and world.calls == [1]
        assert world.releases == ["completed"]
        assert (await restarted.aget_state(CFG)).next == ()
    asyncio.run(scenario())


def test_turn_carries_the_hold_ref_and_never_the_pool_role_as_a_route():
    async def scenario():
        world, saver = World(), InMemorySaver()
        world.grant()
        await world.graph(saver).ainvoke(initial(), CFG)
        [req] = world.turn_requests
        assert req.gpu_lease is not None and req.gpu_lease.model_dump() == world.ref()
        # The hold landed on agent-gpu2: that is a pool role, never a route label.
        assert req.assigned_lane is None and req.lease is None and req.fcc_model_label is None
        assert "agent-gpu2" not in req.model_dump_json(exclude={"gpu_lease"})
    asyncio.run(scenario())


def test_door_a_reach_out_keeps_the_hold_until_hub_finishes():
    """When Orion asked to share, finish keeps the hold (durable-runs heartbeats it) and hands its
    ref to Hub in the finish detail; Hub releases it via release-outreach-lease."""
    async def scenario():
        world, saver = World(), InMemorySaver()

        async def read_reach(run_id):
            return {"graph_readable": True, "hops": [[1, "note"]],
                    "outcome": {"reach_out": True, "reach_out_why": "she should know", "continue_line": False}}

        world.read = read_reach  # type: ignore[method-assign]
        graph = world.graph(saver)
        world.grant()
        result = await graph.ainvoke(initial(), CFG)
        assert result["status"] == "completed"
        assert world.releases == [], f"hold must stay held for Door-A, got {world.releases}"
        assert ("kept_for_outreach", world.ref()) in world.events
        assert any(name == "run.outreach_pending" and detail["lease_id"] == "hold-1" for name, detail in world.events)
        detail = finish_detail(result)
        assert detail["reach_out"] is True and "resource_lease" not in detail
        assert detail["gpu_lease"] == world.ref()
    asyncio.run(scenario())


def test_recall_at_a_node_boundary_releases_the_hold_and_the_tail_continues_without_it():
    async def scenario():
        world, saver = World(), InMemorySaver()
        world.grant()

        async def read_then_recall(run_id):
            world.guard_status = "recall"   # the pool wants the seat back while the tail runs
            return {"graph_readable": True, "hops": [], "outcome": {"reach_out": True}}

        world.read = read_then_recall  # type: ignore[method-assign]
        result = await world.graph(saver).ainvoke(initial(), CFG)
        assert result["status"] == "completed" and result["journal_entry_id"]
        assert world.releases == ["recalled"]            # let go at the first boundary after recall
        assert result["lease"] is None and "gpu_lease" not in finish_detail(result)
    asyncio.run(scenario())


def test_duplicate_wakeup_cannot_fake_a_grant_or_duplicate_the_hold_request():
    async def scenario():
        world, saver = World(), InMemorySaver()
        graph = world.graph(saver)
        await graph.ainvoke(initial(), CFG)
        await graph.ainvoke(Command(resume={"lease_id": "forged"}), CFG)
        assert world.calls == []
        # The re-ask after a wakeup reuses the same request id (idempotent at the pool).
        assert set(world.requests) == {"study-001:1"}
        assert (await graph.aget_state(CFG)).tasks[0].interrupts
    asyncio.run(scenario())


def test_failure_releases_then_checkpoints_bounded_backoff_before_retry_under_a_new_request():
    async def scenario():
        world, saver = World(), InMemorySaver()
        world.grant()
        world.fail = True
        graph = world.graph(saver)
        await graph.ainvoke(initial(), CFG)
        snap = await graph.aget_state(CFG)
        assert snap.values["status"] == "retrying" and snap.next == ("retry_wait",)
        assert snap.tasks[0].interrupts and world.calls == [1]
        assert world.releases == ["attempt_failed"] and snap.values["hold"] is None
        world.now += timedelta(seconds=31)
        world.grant()
        result = await graph.ainvoke(Command(resume=True), CFG)
        assert result["status"] == "failed" and world.calls == [1, 2]
        assert world.requests == ["study-001:1", "study-001:2"]   # a released hold is never re-asked
        assert (await graph.aget_state(CFG)).next == ()
        assert "attempt_failed" in world.releases
    asyncio.run(scenario())


def test_a_hold_the_pool_refuses_fails_the_run_with_the_pool_reason():
    async def scenario():
        world, saver = World(), InMemorySaver()
        graph = world.graph(saver)
        await graph.ainvoke(initial(), CFG)
        world.pool_status = "refused"
        result = await graph.ainvoke(Command(resume=True), CFG)
        assert result["status"] == "failed" and world.calls == []
        assert result["last_error"] == "gpu_pool_unavailable:backlog_max_age"
    asyncio.run(scenario())
