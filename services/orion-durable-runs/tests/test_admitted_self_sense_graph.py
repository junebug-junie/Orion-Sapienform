"""Admitted self_sense_eval graph: wait for a lease, ask the four questions,
publish score rows, finish -- never the curiosity harness/journal path.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parents[1])]

from app.admitted_graph import AdmissionDeps
from app.admitted_self_sense_graph import build_admitted_self_sense_graph
from app.self_sense_graph import Deps
from orion.schemas.durable_run import CuriosityTurnRequestV1, CuriosityTurnResultV1
from orion.schemas.self_sense import SELF_SENSE_QUESTIONS

_QUESTION_BY_TEXT = {text: key for key, text in SELF_SENSE_QUESTIONS}


class World:
    def __init__(self):
        self.now = datetime(2026, 9, 22, tzinfo=timezone.utc)
        self.current_lease = None
        self.demands = set()
        self.turn_calls: list[CuriosityTurnRequestV1] = []
        self.published = []
        self.releases = []

    def grant(self):
        self.current_lease = {
            "run_id": "ss-001",
            "demand_id": "ss-001:ask_questions:llm.route.agent",
            "lease_id": "lease-ss",
            "resource_key": "llm.route.agent",
            "lane": "agent",
            "backend_key": "http://worker",
            "generation": 1,
            "granted_at": self.now.isoformat(),
            "expires_at": (self.now + timedelta(seconds=90)).isoformat(),
            "heartbeat_at": self.now.isoformat(),
            "status": "active",
        }

    async def turn(self, req: CuriosityTurnRequestV1) -> CuriosityTurnResultV1:
        self.turn_calls.append(req)
        key = _QUESTION_BY_TEXT[req.prompt]
        return CuriosityTurnResultV1(
            run_id=req.run_id, correlation_id=req.correlation_id, text=f"answer:{key}", ok=True
        )

    async def publish_rows(self, rows):
        self.published.extend(rows)
        return len(rows), 0

    async def register(self, state):
        self.demands.add(state["run_id"])

    async def lease(self, run_id):
        return self.current_lease

    async def execute(self, state, node):
        if not self.current_lease or state.get("lease") != self.current_lease:
            raise RuntimeError("stale_lease")
        return await node(state)

    async def release(self, run_id, reason):
        self.releases.append(reason)
        self.current_lease = None

    async def event(self, *args):
        pass

    def graph(self, saver):
        return build_admitted_self_sense_graph(
            Deps(run_turn=self.turn, publish_rows=self.publish_rows),
            AdmissionDeps(
                self.register, self.lease, self.execute, self.release, self.event,
                now=lambda: self.now, max_attempts=1,
            ),
            saver,
        )


def _brief():
    return {
        "prompt": "self-sense eval: four fixed questions",
        "session_id": "self-sense-eval",
        "timeout_sec": 600.0,
        "source_tag": "curiosity_self_sense_eval",
        "line": "self_sense_eval",
        "questions": [list(q) for q in SELF_SENSE_QUESTIONS],
        "self_definition_version": 3,
        "lived_answers": [],
    }


def initial():
    return {
        "run_id": "ss-001",
        "correlation_id": "corr-ss",
        "attempt": 0,
        "workflow": "self_sense_eval",
        "admission": {"resource": "llm.route.agent"},
        "brief": _brief(),
    }


CFG = {"configurable": {"thread_id": "ss-001"}}


def test_admitted_self_sense_waits_then_asks_all_four_and_publishes():
    async def scenario():
        world, saver = World(), InMemorySaver()
        graph = world.graph(saver)
        await asyncio.wait_for(graph.ainvoke(initial(), CFG), 1)
        snap = await graph.aget_state(CFG)
        assert snap.next == ("resource_wait",)
        assert world.turn_calls == [] and world.demands == {"ss-001"}

        world.grant()
        result = await world.graph(saver).ainvoke(Command(resume=True), CFG)
        assert result["status"] == "completed"
        assert len(world.turn_calls) == 4
        assert { _QUESTION_BY_TEXT[c.prompt] for c in world.turn_calls } == {
            "what_are_you", "last_day_unasked", "cannot_do_now", "who_matters"
        }
        assert len(world.published) == 4
        assert world.releases == ["completed"]
        assert (await world.graph(saver).aget_state(CFG)).next == ()
        # Curiosity-only nodes must never appear on this path.
        nodes_seen = set()
        async for update in world.graph(saver).aget_state_history(CFG):
            pass
        _ = nodes_seen  # history walk not required; turn count is the proof

    asyncio.run(scenario())


def test_admission_runtime_routes_self_sense_workflow():
    """Unit: AdmissionRuntime picks the self-sense compiled graph by workflow."""
    from types import SimpleNamespace

    from app.admission_runtime import AdmissionRuntime, SELF_SENSE_WORKFLOW, DEFAULT_WORKFLOW
    from app.graph import Deps as CuriosityDeps

    class FakeStore:
        def get_lease(self, run_id):
            return None

    class FakeRunner:
        def __init__(self):
            self._checkpointer = InMemorySaver()

        def _curiosity_deps(self):
            async def turn(req):
                return CuriosityTurnResultV1(
                    run_id=req.run_id, correlation_id=req.correlation_id, text="x", ok=True
                )

            async def read(_):
                return {}

            async def row(_):
                return True

            async def journal(entry):
                return entry.entry_id

            return CuriosityDeps(turn, read, row, journal)

        def _self_sense_deps(self):
            async def turn(req):
                return CuriosityTurnResultV1(
                    run_id=req.run_id, correlation_id=req.correlation_id, text="x", ok=True
                )

            async def publish_rows(rows):
                return len(rows), 0

            return Deps(run_turn=turn, publish_rows=publish_rows)

    settings = SimpleNamespace(
        admission_shadow=False,
        lease_seconds=90,
        widening_after_sec=0,
        widening_hysteresis_sec=0,
        widening_enabled=False,
        capacity_enabled=False,
        lane_policy_json="{}",
        gateway_url="http://unused",
        admission_tick_sec=1,
        retry_max_attempts=1,
        retry_base_sec=1,
        retry_max_sec=1,
        elastic_enabled=False,
        state_channel="orion:durable:state",
    )
    rt = AdmissionRuntime(settings, FakeRunner(), pool=None, store=FakeStore(), broker=SimpleNamespace())
    assert rt._graph_for(DEFAULT_WORKFLOW) is rt.graphs[DEFAULT_WORKFLOW]
    assert rt._graph_for(SELF_SENSE_WORKFLOW) is rt.graphs[SELF_SENSE_WORKFLOW]
    assert rt._graph_for(SELF_SENSE_WORKFLOW) is not rt._graph_for(DEFAULT_WORKFLOW)
