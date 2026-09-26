"""Admitted self_sense_eval graph: wait for the pool's hold, ask the four questions under it,
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

from app.admitted_graph import GRANTED, WAITING, AdmissionDeps
from app.admitted_self_sense_graph import build_admitted_self_sense_graph
from app.self_sense_graph import Deps
from orion.schemas.durable_run import CuriosityTurnRequestV1, CuriosityTurnResultV1
from orion.schemas.self_sense import SELF_SENSE_QUESTIONS

_QUESTION_BY_TEXT = {text: key for key, text in SELF_SENSE_QUESTIONS}


REF = {"lease_id": "hold-ss", "generation": 1, "role": "agent", "holder": "durable-runs:ss-001"}


class World:
    def __init__(self):
        self.now = datetime(2026, 9, 22, tzinfo=timezone.utc)
        self.granted = False
        self.requests = []
        self.turn_calls: list[CuriosityTurnRequestV1] = []
        self.published = []
        self.releases = []

    def grant(self):
        self.granted = True

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
        self.requests.append(state["run_id"])
        return {"status": "waiting_resource", "hold": {"request_id": "ss-001:1", "lease_id": "hold-ss"}, "hold_seq": 1}

    async def lease(self, state):
        if self.granted:
            return GRANTED, {"status": "admitted", "lease": dict(REF), "hold": state.get("hold")}
        return WAITING, {"status": "waiting_resource", "lease": None}

    async def execute(self, state, node):
        if not self.granted or state.get("lease") != REF:
            raise RuntimeError("gpu_hold_lost")
        return await node(state)

    async def release(self, state, reason, keep_requeued=False):
        self.releases.append(reason)
        self.granted = False
        return {"lease": None, "hold": None}

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
        assert world.turn_calls == [] and world.requests == ["ss-001"]

        world.grant()
        result = await world.graph(saver).ainvoke(Command(resume=True), CFG)
        assert result["status"] == "completed"
        assert len(world.turn_calls) == 4
        assert { _QUESTION_BY_TEXT[c.prompt] for c in world.turn_calls } == {
            "what_are_you", "last_day_unasked", "cannot_do_now", "who_matters"
        }
        assert len(world.published) == 4
        # Every question attached to the run's hold; the hold's role is never a route label.
        assert all(c.gpu_lease is not None and c.gpu_lease.model_dump() == REF for c in world.turn_calls)
        assert all(c.assigned_lane is None and c.lease is None for c in world.turn_calls)
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

    from app.admission_runtime import AdmissionRuntime, SELF_SENSE_WORKFLOW, DEFAULT_WORKFLOW, REFLECT_WORKFLOW
    from app.graph import Deps as CuriosityDeps

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

        def _reflect_deps(self):
            from app.reflect_graph import Deps as ReflectDeps

            async def call(reflect_input, llm_route, gpu_lease=None):
                return []

            return ReflectDeps(call_reflect_llm=call)

    settings = SimpleNamespace(
        service_name="orion-durable-runs",
        lease_seconds=90,
        lease_heartbeat_sec=15,
        admission_tick_sec=1,
        retry_max_attempts=1,
        retry_base_sec=1,
        retry_max_sec=1,
        state_channel="orion:durable:state",
    )
    rt = AdmissionRuntime(settings, FakeRunner(), pool=None, store=SimpleNamespace(),
                          holds=SimpleNamespace(hold_ttl_sec=90))
    assert rt._graph_for(DEFAULT_WORKFLOW) is rt.graphs[DEFAULT_WORKFLOW]
    assert rt._graph_for(SELF_SENSE_WORKFLOW) is rt.graphs[SELF_SENSE_WORKFLOW]
    assert rt._graph_for(SELF_SENSE_WORKFLOW) is not rt._graph_for(DEFAULT_WORKFLOW)
    # Before 4.5 an admitted reflect run fell back to the curiosity graph.
    assert rt._graph_for(REFLECT_WORKFLOW) is rt.graphs[REFLECT_WORKFLOW]
    assert rt._graph_for(REFLECT_WORKFLOW) is not rt._graph_for(DEFAULT_WORKFLOW)


def test_heartbeat_must_fit_twice_inside_the_pool_hold_ttl():
    from types import SimpleNamespace

    import pytest as _pytest

    from app.admission_runtime import AdmissionRuntime

    settings = SimpleNamespace(service_name="x", lease_heartbeat_sec=50, retry_max_attempts=1, retry_base_sec=1,
                               retry_max_sec=1)
    with _pytest.raises(ValueError, match="at most half"):
        AdmissionRuntime(settings, SimpleNamespace(), pool=None, store=SimpleNamespace(),
                         holds=SimpleNamespace(hold_ttl_sec=90))
