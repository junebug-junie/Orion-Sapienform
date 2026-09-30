"""journal.compose: the world-news journal composed under a GPU pool hold.

Real LangGraph interrupts + checkpointer around the graph, with the fake pool from
test_admitted_graph (queued until ``grant()``). Proves: a busy pool is a checkpointed wait, not an
attempt; a restart resumes; compose failures are bounded attempts; a replayed publish re-sends the
identical write (same entry_id / created_at), never a second entry.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from test_admitted_graph import CFG, World
from app.admitted_graph import AdmissionDeps, HoldLost
from app.journal_compose_graph import build_journal_compose_graph, build_write, finish_detail
from orion.journaler.schemas import JournalEntryDraftV1
from orion.schemas.durable_run import DurableRunRequestV1
from orion.schemas.journal_compose_run import JOURNAL_COMPOSE_WORKFLOW, JournalComposeRunBriefV1
from orion.schemas.resource_admission import ResourceRequirementV1

BRIEF = {
    "trigger": {"trigger_kind": "world_pulse_digest", "source_kind": "world_pulse", "source_ref": "wp-1",
                "summary": "Daily World Pulse"},
    "entry_id": "entry-wp-1",
    "author": "orion",
    "session_id": "orion_journal",
    "recall_profile": "journal.world_pulse.grounded.v1",
    "llm_route": "quick_background",
    "timeout_sec": 5.0,
}


def initial():
    return {"run_id": "study-001", "correlation_id": "trace-001", "workflow": JOURNAL_COMPOSE_WORKFLOW,
            "attempt": 0, "admission": {"resource": "llm.route.quick_background"}, "brief": dict(BRIEF)}


class Journal:
    def __init__(self, world, fail_times=0, publish_ok=True):
        self.world, self.fail_times, self.publish_ok = world, fail_times, publish_ok
        self.composes, self.writes = [], []

    async def compose(self, brief, *, run_id, correlation_id, gpu_lease):
        assert isinstance(brief, JournalComposeRunBriefV1)
        assert gpu_lease.model_dump() == self.world.ref()   # the call attaches to the run's hold
        self.composes.append(correlation_id)
        if len(self.composes) <= self.fail_times:
            raise RuntimeError("journal_compose_failed:{'message': 'cortex_orch_missing_final_text'}")
        return JournalEntryDraftV1(mode="digest", title="Pulse", body="World news today.")

    async def publish(self, write):
        self.writes.append(write)
        return self.publish_ok


def graph(world, saver, journal, max_attempts=3):
    return build_journal_compose_graph(
        journal.compose, journal.publish,
        AdmissionDeps(world.register, world.lease, world.execute, world.release, world.event,
                      now=lambda: world.now, max_attempts=max_attempts),
        saver)


def test_busy_pool_waits_checkpointed_and_restart_resumes_without_spending_an_attempt():
    async def run():
        world, saver = World(), InMemorySaver()
        journal = Journal(world)
        await asyncio.wait_for(graph(world, saver, journal).ainvoke(initial(), CFG), 1)
        snap = await graph(world, saver, journal).aget_state(CFG)
        assert snap.next == ("resource_wait",)
        assert journal.composes == [] and snap.values.get("attempt", 0) == 0
        # Still busy after a "restart": asking again is a wait, never an attempt.
        await graph(world, saver, journal).ainvoke(Command(resume={}), CFG)
        assert (await graph(world, saver, journal).aget_state(CFG)).values.get("attempt", 0) == 0
        world.grant()
        result = await graph(world, saver, journal).ainvoke(Command(resume={}), CFG)
        assert result["status"] == "completed" and result["published"] is True
        assert len(journal.composes) == 1 and len(journal.writes) == 1
        write = journal.writes[0]
        assert write.entry_id == "entry-wp-1" and write.trigger_kind == "world_pulse_digest"
        assert write.source_ref == "wp-1" and write.correlation_id == "trace-001"
        assert world.releases == ["completed"]   # released once, before publish
        assert finish_detail(result) == {"line": "journal", "entry_id": "entry-wp-1",
                                         "trigger_kind": "world_pulse_digest", "published": True, "attempts": 1}

    asyncio.run(run())


def test_compose_failure_is_a_bounded_attempt_then_succeeds():
    async def run():
        world = World()
        world.grant()
        journal = Journal(world, fail_times=1)

        async def release(state, reason, keep_requeued=False):   # the pool re-grants at once
            world.releases.append(reason)
            return {"lease": None, "hold": None}

        world.release = release
        result = await graph(world, InMemorySaver(), journal).ainvoke(initial(), CFG)
        assert result["status"] == "completed" and result["attempt"] == 2
        assert len(journal.composes) == 2 and len(journal.writes) == 1

    asyncio.run(run())


def test_compose_fails_after_max_attempts_without_publishing():
    async def run():
        world = World()
        world.grant()
        journal = Journal(world, fail_times=99)

        async def release(state, reason, keep_requeued=False):
            world.releases.append(reason)
            return {"lease": None, "hold": None}

        world.release = release
        result = await graph(world, InMemorySaver(), journal, max_attempts=2).ainvoke(initial(), CFG)
        assert result["status"] == "failed" and result["attempt"] == 2
        assert "cortex_orch_missing_final_text" in result["last_error"]
        assert journal.writes == []

    asyncio.run(run())


def test_lost_hold_is_not_an_attempt():
    async def run():
        world = World()
        world.grant()
        journal = Journal(world)

        async def execute(state, node):
            world.pool_status = "queued"
            raise HoldLost("gpu_hold_lost")

        world.execute = execute
        result = await graph(world, InMemorySaver(), journal).ainvoke(initial(), CFG)
        assert result["status"] == "waiting_resource" and result.get("attempt", 0) == 0
        assert journal.composes == []

    asyncio.run(run())


def test_publish_failure_resumes_at_publish_with_identical_write_and_no_recompose():
    async def run():
        world, saver = World(), InMemorySaver()
        world.grant()
        journal = Journal(world, publish_ok=False)
        try:
            await graph(world, saver, journal).ainvoke(initial(), CFG)
        except RuntimeError as exc:
            assert "journal_publish_failed" in str(exc)
        snap = await graph(world, saver, journal).aget_state(CFG)
        assert snap.next == ("publish",)
        journal.publish_ok = True
        result = await graph(world, saver, journal).ainvoke(None, CFG)   # the driver's resume
        assert result["status"] == "completed"
        assert len(journal.composes) == 1                                  # never recomposed
        assert len(journal.writes) == 2
        first, second = (w.model_dump(mode="json") for w in journal.writes)
        assert first == second                                             # same entry_id + created_at

    asyncio.run(run())


def test_build_write_is_a_pure_function_of_state():
    state = {**initial(), "draft": {"mode": "digest", "title": "t", "body": "b"},
             "created_at": "2026-09-30T12:00:00+00:00"}
    assert build_write(state).model_dump(mode="json") == build_write(dict(state)).model_dump(mode="json")


def test_request_contract_requires_admission_and_matching_brief():
    import pytest

    ok = DurableRunRequestV1(run_id="world-pulse-journal:wp-1", workflow=JOURNAL_COMPOSE_WORKFLOW,
                             correlation_id="c", brief=BRIEF,
                             admission=ResourceRequirementV1(resource="llm.route.quick_background",
                                                             preferred_lane="quick_background"))
    assert isinstance(ok.brief, JournalComposeRunBriefV1)
    with pytest.raises(ValueError):
        DurableRunRequestV1(run_id="world-pulse-journal:wp-1", workflow=JOURNAL_COMPOSE_WORKFLOW,
                            correlation_id="c", brief=BRIEF)
    with pytest.raises(ValueError):
        DurableRunRequestV1(run_id="world-pulse-journal:wp-1", workflow="self_study.reflect",
                            correlation_id="c", brief=BRIEF,
                            admission=ResourceRequirementV1())


def test_admission_runtime_registers_journal_compose():
    import inspect

    from app import admission_runtime

    assert admission_runtime.WORK_NODES[JOURNAL_COMPOSE_WORKFLOW] == {"compose"}
    src = inspect.getsource(admission_runtime.AdmissionRuntime.__init__)
    assert "JOURNAL_COMPOSE_WORKFLOW: build_journal_compose_graph" in src


def _runner_with_reply(payload):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from uuid import uuid4

    from app.runner import DurableRunner
    from app.settings import Settings
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
    from orion.core.bus.codec import OrionCodec

    codec = OrionCodec()
    reply = BaseEnvelope(kind="cortex.orch.result", source=ServiceRef(name="orion-cortex-orch"),
                         correlation_id=str(uuid4()), payload=payload)
    bus = SimpleNamespace(codec=codec, rpc_request=AsyncMock(return_value={"data": codec.encode(reply)}))
    settings = Settings(_env_file=None, DURABLE_RUNS_GRAPH_HOST="", POSTGRES_URI="postgresql://unused",
                        ORION_BUS_ENABLED=False)
    return DurableRunner(settings, bus=bus, checkpointer=None), bus


def _hold():
    from orion.schemas.gpu_pool import GpuLeaseRefV1

    return GpuLeaseRefV1(lease_id="hold-1", generation=1, role="fast", holder="durable-runs:world-pulse-journal:wp-1")


def test_runner_compose_sends_journal_verb_attached_to_the_hold():
    runner, bus = _runner_with_reply(
        {"ok": True, "status": "success", "final_text": '{"mode":"digest","title":"Pulse","body":"World news."}'})
    draft = asyncio.run(runner._compose_journal(JournalComposeRunBriefV1.model_validate(BRIEF),
                                                run_id="r", correlation_id="trace-1", gpu_lease=_hold()))
    assert draft.body == "World news."
    env = bus.rpc_request.await_args.args[1]
    options = env.payload["options"]
    assert env.payload["verb"] == "journal.compose"
    assert options["gpu_lease"]["lease_id"] == "hold-1"
    assert options["llm_route"] == "quick_background"                 # the route, never the hold's role
    assert env.payload["context"]["metadata"]["journal_trigger"]["trigger_kind"] == "world_pulse_digest"
    assert env.payload["recall"]["profile"] == "journal.world_pulse.grounded.v1"
    assert bus.rpc_request.await_args.kwargs["timeout_sec"] == BRIEF["timeout_sec"]


def test_runner_compose_raises_on_not_ok_and_on_empty_text():
    import pytest

    for payload in ({"ok": False, "error": {"message": "gpu_pool_unavailable:deadline"}},
                    {"ok": True, "status": "success", "final_text": ""}):
        runner, _ = _runner_with_reply(payload)
        with pytest.raises(Exception):
            asyncio.run(runner._compose_journal(JournalComposeRunBriefV1.model_validate(BRIEF),
                                                run_id="r", correlation_id="t", gpu_lease=_hold()))
