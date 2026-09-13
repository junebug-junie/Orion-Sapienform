"""Actual LangGraph interrupts with deterministic resources and a fake clock."""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parents[1])]

from app.admitted_graph import AdmissionDeps, build_admitted_graph
from app.graph import Deps
from orion.schemas.durable_run import CuriosityTurnResultV1


class World:
    def __init__(self):
        self.now = datetime(2026, 9, 12, tzinfo=timezone.utc)
        self.current_lease = None
        self.demands = set()
        self.calls = []
        self.releases = []
        self.fail = False

    def grant(self):
        self.current_lease = dict(run_id="study-001", demand_id="study-001:harness_turn:llm.route.agent",
            lease_id="lease-001", resource_key="llm.route.agent", lane="agent", backend_key="http://worker",
            generation=1, granted_at=self.now.isoformat(), expires_at=(self.now+timedelta(seconds=90)).isoformat(),
            heartbeat_at=self.now.isoformat(), status="active")

    async def turn(self, req):
        self.calls.append(req.attempt)
        assert req.lease and req.assigned_lane == "agent"
        return CuriosityTurnResultV1(run_id=req.run_id, correlation_id=req.correlation_id,
                                    text="A grounded study finding." if not self.fail else "", ok=not self.fail)

    async def read(self, run_id):
        return {"graph_readable": True, "hops": [[1, "inspected evidence"]]}

    async def row(self, facts):
        return True

    async def journal(self, entry):
        return entry.entry_id

    async def register(self, state):
        self.demands.add(state["run_id"])

    async def lease(self, run_id):
        return self.current_lease

    async def execute(self, state, node):
        if not self.current_lease or state["lease"] != self.current_lease:
            raise RuntimeError("stale_lease")
        return await node(state)

    async def release(self, run_id, reason):
        self.releases.append(reason)
        self.current_lease = None

    async def event(self, *args):
        pass

    def graph(self, saver):
        return build_admitted_graph(Deps(self.turn, self.read, self.row, self.journal),
            AdmissionDeps(self.register, self.lease, self.execute, self.release, self.event,
                          now=lambda: self.now, max_attempts=2), saver)


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
        assert world.calls == [] and world.demands == {"study-001"}
        world.now += timedelta(days=2)  # queue is independent of legacy 24h age / 50ms turn timeout
        restarted = world.graph(saver)
        world.grant()
        result = await restarted.ainvoke(Command(resume={"untrusted_grant": "ignored"}), CFG)
        assert result["status"] == "completed" and world.calls == [1]
        assert world.releases == ["completed"]
        assert (await restarted.aget_state(CFG)).next == ()
    asyncio.run(scenario())


def test_duplicate_wakeup_cannot_fake_a_grant_or_duplicate_demand():
    async def scenario():
        world, saver = World(), InMemorySaver()
        graph = world.graph(saver)
        await graph.ainvoke(initial(), CFG)
        await graph.ainvoke(Command(resume={"lease_id": "forged"}), CFG)
        assert world.calls == [] and len(world.demands) == 1
        assert (await graph.aget_state(CFG)).tasks[0].interrupts
    asyncio.run(scenario())


def test_failure_releases_then_checkpoints_bounded_backoff_before_retry():
    async def scenario():
        world, saver = World(), InMemorySaver()
        world.grant()
        world.fail = True
        graph = world.graph(saver)
        await graph.ainvoke(initial(), CFG)
        snap = await graph.aget_state(CFG)
        assert snap.values["status"] == "retrying" and snap.next == ("retry_wait",)
        assert snap.tasks[0].interrupts and world.calls == [1]
        assert world.current_lease is None
        world.now += timedelta(seconds=31)
        world.grant()
        result = await graph.ainvoke(Command(resume=True), CFG)
        assert result["status"] == "failed" and world.calls == [1, 2]
        assert (await graph.aget_state(CFG)).next == ()
        assert "failed" in world.releases
    asyncio.run(scenario())
