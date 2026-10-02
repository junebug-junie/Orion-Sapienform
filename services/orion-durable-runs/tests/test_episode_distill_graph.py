"""memory.episode_distill: one closed episode distilled under a GPU pool hold, persisted in shadow.

Real LangGraph interrupts + checkpointer, with the fake pool from test_admitted_graph (queued until
``grant()``). Proves: the turns are read before any hold is asked for; a busy pool is a checkpointed
wait, not an attempt; the LLM call attaches to the run's hold; an unparseable answer is a bounded,
backed-off attempt; the hold is released before persisting; persist receives validated memories
(an unverifiable quote never reaches it).
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from test_admitted_graph import CFG, World
from app.admitted_graph import AdmissionDeps
from app.episode_distill_graph import build_episode_distill_graph, finish_detail, request_from_closed_event
from orion.memory.episode.distill import turns_from_rows, turns_to_state
from orion.schemas.durable_run import DurableRunRequestV1
from orion.schemas.memory_episode import MEMORY_EPISODE_DISTILL_WORKFLOW, EpisodeDistillBriefV1

T0 = datetime(2026, 9, 28, 12, 26, tzinfo=timezone.utc)
BRIEF = {"episode_id": "ep-1", "turn_ids": ["c-1", "c-2"], "started_at": T0.isoformat(),
         "ended_at": (T0 + timedelta(hours=3)).isoformat(), "close_reason": "v2:phase_next_day",
         "llm_route": "memory_distill", "timeout_sec": 30.0}
ROWS = [
    {"correlation_id": "c-1", "prompt": "Headed to Austin and will fly back on Wednesday.",
     "response": "Austin! Safe travels.", "created_at": T0},
    {"correlation_id": "c-2", "prompt": "Run github compactor.", "response": "Workflow: GitHub Compactor",
     "created_at": T0 + timedelta(hours=3)},
]
GOOD = ('{"memories": ['
        '{"purpose": "happened", "voice": "juniper_said", "statement": "Juniper flew to Austin and returns Wednesday.",'
        ' "referents": [{"key": "event:austin-offsite-2026-09", "aliases": ["austin"]}],'
        ' "evidence": [{"turn": "t1", "field": "prompt", "quote": "Headed to Austin"}]},'
        '{"purpose": "about_juniper", "voice": "juniper_said", "statement": "Juniper said she loves flying to Austin.",'
        ' "evidence": [{"turn": "t1", "field": "prompt", "quote": "I love flying"}]}'
        '], "questions": []}')


def initial():
    return {"run_id": "memdistill-ep-1", "correlation_id": "ep-1", "workflow": MEMORY_EPISODE_DISTILL_WORKFLOW,
            "attempt": 0, "admission": {"resource": "llm.route.memory_distill"}, "brief": dict(BRIEF)}


class Distiller:
    def __init__(self, world, answers):
        self.world, self.answers = world, list(answers)
        self.loads, self.calls, self.persisted = 0, [], []

    async def load(self, brief):
        assert isinstance(brief, EpisodeDistillBriefV1)
        self.loads += 1
        return {"turns": turns_to_state(turns_from_rows(ROWS)), "candidate_referents": ["person:juniper"]}

    async def call_llm(self, prompt, *, brief, run_id, correlation_id, gpu_lease):
        assert gpu_lease.model_dump() == self.world.ref()        # attaches to the run's hold
        assert "Headed to Austin" in prompt and "COMMAND" in prompt
        self.calls.append(correlation_id)
        return {"text": self.answers.pop(0), "usage": {"prompt_tokens": 900, "completion_tokens": 120},
                "model": "qwen-27b", "latency_ms": 4200}

    async def persist(self, **kw):
        assert self.world.releases[-1] == "completed"            # hold let go before the write
        self.persisted.append(kw)
        r = kw["result"]
        return {"memories": len(r.memories), "rejections": len(r.rejections), "questions": len(r.questions),
                "downgrades": r.downgrades}


def graph(world, saver, d, max_attempts=3):
    return build_episode_distill_graph(
        d.load, d.call_llm, d.persist,
        AdmissionDeps(world.register, world.lease, world.execute, world.release, world.event,
                      now=lambda: world.now, max_attempts=max_attempts),
        saver)


def test_loads_first_waits_for_the_hold_then_distills_and_persists_validated_memories():
    async def run():
        world, saver = World(), InMemorySaver()
        d = Distiller(world, [GOOD])
        await asyncio.wait_for(graph(world, saver, d).ainvoke(initial(), CFG), 1)
        snap = await graph(world, saver, d).aget_state(CFG)
        assert snap.next == ("resource_wait",)
        assert d.loads == 1 and d.calls == [] and snap.values.get("attempt", 0) == 0
        world.grant()
        result = await graph(world, saver, d).ainvoke(Command(resume={}), CFG)
        assert result["status"] == "completed"
        assert len(d.calls) == 1 and d.loads == 1                 # resumed: not re-read
        kw = d.persisted[0]
        assert kw["episode_id"] == "ep-1" and kw["model_route"] == "memory_distill"
        assert kw["usage"] == {"prompt_tokens": 900, "completion_tokens": 120}
        kept = kw["result"].memories
        assert [m.statement for m in kept] == ["Juniper flew to Austin and returns Wednesday."]
        assert [r.reason for r in kw["result"].rejections] == ["no_verified_quote"]
        assert kw["coverage"] == 1.0                              # the only content turn is cited
        assert finish_detail(result) == {"line": "memory", "episode_id": "ep-1", "turns": 2, "memories": 1,
                                         "rejections": 1, "questions": 0, "downgrades": 0, "attempts": 1}

    asyncio.run(run())


def _regranting_release(world):
    async def release(state, reason, keep_requeued=False):
        world.releases.append(reason)
        return {"lease": None, "hold": None}
    return release


def test_unparseable_answer_is_a_backed_off_attempt():
    async def run():
        world, saver = World(), InMemorySaver()
        world.grant()
        world.release = _regranting_release(world)
        d = Distiller(world, ["Sorry, I can't produce JSON right now.", GOOD])
        await graph(world, saver, d).ainvoke(initial(), CFG)
        snap = await graph(world, saver, d).aget_state(CFG)
        assert snap.next == ("retry_wait",)
        assert snap.values["attempt"] == 1 and "distill_no_json_object" in snap.values["last_error"]
        assert d.persisted == []
        world.now = datetime.fromisoformat(snap.values["retry_at"])
        result = await graph(world, saver, d).ainvoke(Command(resume=True), CFG)
        assert result["status"] == "completed" and result["attempt"] == 2 and len(d.persisted) == 1

    asyncio.run(run())


def test_missing_turns_fail_without_asking_for_a_hold():
    async def run():
        world, saver = World(), InMemorySaver()
        d = Distiller(world, [GOOD])

        async def empty(brief):
            return {"turns": [], "candidate_referents": []}

        d.load = empty
        result = await graph(world, saver, d).ainvoke(initial(), CFG)
        assert result["status"] == "failed" and result["last_error"] == "episode_turns_missing"
        assert d.calls == []

    asyncio.run(run())


SETTINGS = SimpleNamespace(memory_episode_distill_route="memory_distill", memory_episode_distill_timeout_sec=600.0,
                           memory_episode_distill_max_tokens=4096, memory_episode_distill_deadline_hours=20.0)
CLOSED = {"episode_id": "ep-9", "source_platform": None, "started_at": T0.isoformat(),
          "ended_at": T0.isoformat(), "closed_at": (T0 + timedelta(hours=9)).isoformat(),
          "turn_ids": ["c-1"], "juniper_turn_count": 1, "close_reason": "v2:phase_long_gap",
          "episode_status": "closed"}


def test_closed_event_becomes_a_system_priority_run_on_the_distill_route():
    req = request_from_closed_event(CLOSED, settings=SETTINGS, now=T0)
    assert isinstance(req, DurableRunRequestV1)
    assert req.run_id == "memdistill-ep-9" and req.workflow == MEMORY_EPISODE_DISTILL_WORKFLOW
    assert (req.admission.priority, req.admission.preferred_lane, req.admission.resource) == (
        "system", "memory_distill", "llm.route.memory_distill")
    assert req.admission.deadline_at == T0 + timedelta(hours=20)
    assert req.brief.turn_ids == ["c-1"]


def test_skipped_and_external_episodes_get_no_run():
    assert request_from_closed_event({**CLOSED, "episode_status": "skipped", "skip_reason": "command_only"},
                                     settings=SETTINGS) is None
    assert request_from_closed_event({**CLOSED, "source_platform": "aitown"}, settings=SETTINGS) is None


def test_the_hold_route_is_agent_class_at_system_priority():
    from app.pool_hold import hold_placement
    from orion.gpu_pool.config import load_pool_config

    req = request_from_closed_event(CLOSED, settings=SETTINGS, now=T0)
    work_class, priority, _ = hold_placement(load_pool_config(), req.admission.model_dump(mode="json"))
    assert (work_class, priority) == ("agent", "system")
