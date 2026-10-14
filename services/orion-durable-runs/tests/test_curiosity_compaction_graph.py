"""Real LangGraph checkpoints around Curiosity topic/overall consolidation."""
import asyncio
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

sys.path.insert(0, str(Path(__file__).parent))
from test_admitted_graph import CFG, World
from app.admitted_graph import AdmissionDeps, HoldLost
from app.curiosity_compaction_graph import build_curiosity_compaction_graph, finish_detail
from orion.curiosity.compaction import batches
from orion.schemas.curiosity_compaction import CURIOSITY_COMPACTION_WORKFLOW, CuriosityCompactionBriefV1
from orion.schemas.durable_run import DurableRunRequestV1

BODY = "On October 5 Orion investigated missing deliveries; on October 8 the new records weakened the timeout explanation. The delivery cause remains unresolved and needs a fresh observation."
NOW = datetime(2026, 10, 14, tzinfo=timezone.utc)


class Crash(BaseException):
    pass


class Store:
    def __init__(self, *, empty=False, crash_after_save=False):
        self.row = dict(compaction_id="curiosity-compaction:weekly:2026-10-05", run_id="study-001",
                        cadence="weekly", window_start=NOW, window_end=NOW, cycle_id="dc-test", status="pending",
                        updated_at=NOW, state={"phase": "topics", "sources": [] if empty else [
                            {"entry_id": "j1", "run_ref": "curiosity:r1", "created_at": NOW.isoformat()}],
                            "batches": [] if empty else batches([{"ref": "j1", "body": BODY, "sources": ["j1"]}]),
                            "partials": []})
        self.crash_after_save = crash_after_save
    def load(self, key, run_id):
        assert key == self.row["compaction_id"] and run_id == self.row["run_id"]
        return copy.deepcopy(self.row)
    def persist(self, row):
        self.row = copy.deepcopy(row)
        if self.crash_after_save:
            self.crash_after_save = False
            raise Crash()


class Model:
    def __init__(self, world, *, fail=False, preempt=False):
        self.world, self.calls, self.entries = world, [], []
        self.fail, self.preempt = fail, preempt
        self.publish_ok = True
    async def call(self, verb, metadata, route, *, gpu_lease, timeout_sec, user_text):
        assert gpu_lease.model_dump() == self.world.ref() and route == "agent"
        assert verb == "curiosity_compactor_digest_v1"
        prompt = metadata["curiosity_compaction_prompt"]
        phase = "overall" if 'overall chronological synthesis' in prompt else "topics"
        self.calls.append(phase)
        if self.preempt:
            self.preempt = False
            raise HoldLost("hold recalled")
        if self.fail:
            return ""
        if phase == "overall":
            return json.dumps({"body": BODY})
        records = json.loads(prompt.split("SOURCE RECORDS:\n")[1])
        return json.dumps({"topics": [{"topic": "Delivery reliability", "body": BODY,
                                       "refs": [r["ref"] for r in records]}]})
    async def publish(self, entry):
        self.entries.append(entry)
        return entry.entry_id if self.publish_ok else None


def initial():
    return {"run_id": "study-001", "workflow": CURIOSITY_COMPACTION_WORKFLOW, "correlation_id": "trace-001",
            "brief": CuriosityCompactionBriefV1(compaction_id="curiosity-compaction:weekly:2026-10-05").model_dump(),
            "admission": {"resource": "llm.route.agent"}, "attempt": 0}


def graph(world, saver, store, model):
    return build_curiosity_compaction_graph(store, model.call, model.publish,
        AdmissionDeps(world.register, world.lease, world.execute, world.release, world.event,
                      now=lambda: world.now, max_attempts=2), saver)


def test_busy_pool_waits_without_attempt_then_topics_precede_overall():
    async def run():
        world, saver, store = World(), InMemorySaver(), Store()
        model = Model(world)
        await graph(world, saver, store, model).ainvoke(initial(), CFG)
        snap = await graph(world, saver, store, model).aget_state(CFG)
        assert snap.next == ("resource_wait",) and not model.calls and snap.values["attempt"] == 0
        world.grant()
        done = await graph(world, saver, store, model).ainvoke(Command(resume={}), CFG)
        assert done["status"] == "completed" and model.calls == ["topics", "overall"]
        assert len(model.entries) == 1 and finish_detail(done)["topics"] == 1
        assert store.row["status"] == "completed"
    asyncio.run(run())


def test_crash_after_artifact_commit_before_graph_checkpoint_never_repeats_topic_pass():
    async def run():
        world, saver, store = World(), InMemorySaver(), Store(crash_after_save=True)
        world.grant()
        model = Model(world)
        with pytest.raises(Crash):
            await graph(world, saver, store, model).ainvoke(initial(), CFG)
        assert store.row["state"]["phase"] == "overall"
        done = await graph(world, saver, store, model).ainvoke(None, CFG)
        assert done["status"] == "completed" and model.calls == ["topics", "overall"]
    asyncio.run(run())


def test_failed_finalize_resumes_without_regenerating():
    async def run():
        world, saver, store = World(), InMemorySaver(), Store()
        world.grant()
        model = Model(world)
        model.publish_ok = False
        with pytest.raises(RuntimeError, match="publish failed"):
            await graph(world, saver, store, model).ainvoke(initial(), CFG)
        model.publish_ok = True
        done = await graph(world, saver, store, model).ainvoke(None, CFG)
        assert done["status"] == "completed" and model.calls == ["topics", "overall"]
        assert model.entries[0].entry_id == model.entries[1].entry_id
    asyncio.run(run())


def test_empty_period_finishes_without_gpu_or_fake_journal():
    async def run():
        world, saver, store = World(), InMemorySaver(), Store(empty=True)
        model = Model(world)
        done = await graph(world, saver, store, model).ainvoke(initial(), CFG)
        assert done["status"] == "completed" and done["result"]["artifact_status"] == "empty"
        assert world.requests == model.calls == model.entries == []
    asyncio.run(run())


def test_preemption_costs_no_attempt_and_empty_generation_is_bounded():
    async def run():
        world, saver, store = World(), InMemorySaver(), Store()
        world.grant()
        model = Model(world, preempt=True)
        await graph(world, saver, store, model).ainvoke(initial(), CFG)
        snap = await graph(world, saver, store, model).aget_state(CFG)
        assert snap.values["attempt"] == 0 and snap.next == ("resource_wait",)
        model.fail = True
        world.grant()
        await graph(world, saver, store, model).ainvoke(Command(resume={}), CFG)
        assert (await graph(world, saver, store, model).aget_state(CFG)).values["attempt"] == 1
        world.grant()
        done = await graph(world, saver, store, model).ainvoke(Command(resume={}), CFG)
        assert done["status"] == "failed" and store.row["status"] == "pending" and model.entries == []
    asyncio.run(run())


def test_contract_requires_matching_brief_and_admission():
    req = initial()
    req.pop("attempt")
    assert DurableRunRequestV1.model_validate(req).workflow == CURIOSITY_COMPACTION_WORKFLOW
    with pytest.raises(ValueError):
        DurableRunRequestV1.model_validate({**req, "admission": None})
    with pytest.raises(ValueError):
        DurableRunRequestV1.model_validate({**req, "workflow": "dream.carry"})


def test_large_window_exceeds_default_25_step_limit_without_dropping_material():
    async def run():
        world, saver, store = World(), InMemorySaver(), Store()
        store.row["state"]["batches"] *= 40  # forty segments from one long source journal
        world.grant()
        model = Model(world)
        done = await graph(world, saver, store, model).ainvoke(initial(), CFG)
        assert done["status"] == "completed" and len(model.calls) > 40
        assert model.calls[-1] == "overall"
        assert store.row["state"]["topics"][0]["sources"] == ["j1"]
    asyncio.run(run())
