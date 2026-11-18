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
from app.episode_distill_graph import (
    build_episode_distill_graph,
    finish_detail,
    rendered_prompt_version,
    request_from_closed_event,
)
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
        answer = self.answers.pop(0)
        text, finish = answer if isinstance(answer, tuple) else (answer, "stop")
        return {"text": text, "finish_reason": finish,
                "usage": {"prompt_tokens": 900, "completion_tokens": 120},
                "model": "qwen-27b", "latency_ms": 4200}

    async def persist(self, **kw):
        assert self.world.releases[-1] == "completed"            # hold let go before the write
        self.persisted.append(kw)
        r = kw["result"]
        return {"memories": len(r.memories), "rejections": len(r.rejections), "questions": len(r.questions),
                "downgrades": r.downgrades}


def graph(world, saver, d, max_attempts=3, link=None):
    return build_episode_distill_graph(
        d.load, d.call_llm, d.persist,
        AdmissionDeps(world.register, world.lease, world.execute, world.release, world.event,
                      now=lambda: world.now, max_attempts=max_attempts),
        saver, link=link)


class CheckingDistiller(Distiller):
    """The distill answer first, then supersede-check answers on the same hold (issue #3587)."""

    def __init__(self, world, answers):
        super().__init__(world, answers)
        self.check_briefs = []

    async def call_llm(self, prompt, *, brief, run_id, correlation_id, gpu_lease, **kw):
        if prompt.startswith("CHECK:"):
            assert gpu_lease.model_dump() == self.world.ref()    # the check rides the same hold
            assert kw == {"purpose": "memory_supersede_check", "temperature": 0.0}
            self.check_briefs.append(brief)
            return {"text": "yes", "finish_reason": "stop"}
        return await super().call_llm(prompt, brief=brief, run_id=run_id, correlation_id=correlation_id,
                                      gpu_lease=gpu_lease)


def test_supersede_check_runs_on_the_hold_and_its_links_reach_persist():
    async def link(result, ask, *, episode_id):
        assert episode_id == "ep-1"
        assert [m.statement for m in result.memories] == ["Juniper flew to Austin and returns Wednesday."]
        assert await ask("CHECK: does B correct A?") == "yes"
        from orion.memory.episode.supersede import CHECK_VERSION

        return {"links": [{"old_id": "old-1", "new_id": result.memories[0].memory_id, "reason": "r",
                           "check_version": CHECK_VERSION, "relation": "correction"}],
                "asked": 1, "apply": "on", "refused_because": []}

    async def run():
        world, saver = World(), InMemorySaver()
        d = CheckingDistiller(world, [GOOD])
        await asyncio.wait_for(graph(world, saver, d, link=link).ainvoke(initial(), CFG), 1)
        world.grant()
        result = await graph(world, saver, d, link=link).ainvoke(Command(resume={}), CFG)
        assert result["status"] == "completed"
        assert [b.max_tokens for b in d.check_briefs] == [200] and d.check_briefs[0].timeout_sec <= 120
        assert d.persisted[0]["supersede_links"][0]["old_id"] == "old-1"
        assert d.persisted[0]["supersede_plan"]["apply"] == "on"   # the whole plan reaches the audit row
        assert result["supersede"]["asked"] == 1

    asyncio.run(run())


def test_a_plan_that_does_not_say_apply_on_reaches_persist_with_no_links():
    """Issue #3659: a degraded plan (or one checkpointed before the rule) applies nothing."""
    async def link(result, ask, *, episode_id):
        mine = result.memories[0].memory_id
        return {"links": [{"old_id": "old-1", "new_id": mine, "reason": "r"}], "asked": 1, "dense": "unavailable"}

    async def run():
        world, saver = World(), InMemorySaver()
        d = Distiller(world, [GOOD])
        await asyncio.wait_for(graph(world, saver, d, link=link).ainvoke(initial(), CFG), 1)
        world.grant()
        result = await graph(world, saver, d, link=link).ainvoke(Command(resume={}), CFG)
        assert result["status"] == "completed" and "supersede_links" not in d.persisted[0]

    asyncio.run(run())


def test_a_failing_supersede_check_links_nothing_and_never_fails_the_run():
    async def link(result, ask, **_kw):
        raise RuntimeError("index down")

    async def run():
        world, saver = World(), InMemorySaver()
        d = Distiller(world, [GOOD])
        await asyncio.wait_for(graph(world, saver, d, link=link).ainvoke(initial(), CFG), 1)
        world.grant()
        result = await graph(world, saver, d, link=link).ainvoke(Command(resume={}), CFG)
        assert result["status"] == "completed" and len(d.calls) == 1
        assert "supersede_links" not in d.persisted[0]
        assert result["supersede"] == {"links": [], "error": "RuntimeError"}

    asyncio.run(run())


def test_a_slow_supersede_check_times_out_without_costing_an_attempt(monkeypatch):
    import app.episode_distill_graph as g

    monkeypatch.setattr(g, "SUPERSEDE_NODE_BUDGET_SEC", 0.05)

    async def link(result, ask, **_kw):
        await asyncio.sleep(5)

    async def run():
        world, saver = World(), InMemorySaver()
        d = Distiller(world, [GOOD])
        await asyncio.wait_for(graph(world, saver, d, link=link).ainvoke(initial(), CFG), 1)
        world.grant()
        result = await asyncio.wait_for(graph(world, saver, d, link=link).ainvoke(Command(resume={}), CFG), 3)
        assert result["status"] == "completed" and result["attempt"] == 1 and len(d.calls) == 1
        assert result["supersede"]["links"] == [] and "supersede_links" not in d.persisted[0]

    asyncio.run(run())


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
    assert (work_class, priority) == ("memory_distill", "system")


def test_stored_prompt_version_is_the_rendered_template_not_the_brief():
    """Mixed deploy: memory-consolidation still on the old image sends a v2 brief, durable-runs renders
    v3. The run must record v3 and judge stakes as a v3 answer."""
    async def run():
        world, saver = World(), InMemorySaver()
        d = Distiller(world, [GOOD])
        state = initial()
        state["brief"] = {**BRIEF, "prompt_version": "memory_episode_distill.v2"}
        await asyncio.wait_for(graph(world, saver, d).ainvoke(state, CFG), 1)
        world.grant()
        result = await graph(world, saver, d).ainvoke(Command(resume={}), CFG)
        assert result["status"] == "completed"
        kw = d.persisted[0]
        assert kw["prompt_version"] == "memory_episode_distill.v6"
        assert result["rendered_prompt_version"] == "memory_episode_distill.v6"
        # GOOD names no stakes category: under v3 that is escalated and labelled, not left NULL.
        assert [(m.stakes, m.stakes_reason) for m in kw["result"].memories] == [("high", "unjudged")]

    asyncio.run(run())


def test_checkpoint_without_a_stamp_was_rendered_from_v2():
    assert rendered_prompt_version({}) == "memory_episode_distill.v2"
    assert rendered_prompt_version({"rendered_prompt_version": "memory_episode_distill.v3"}) == "memory_episode_distill.v3"


def test_candidate_referents_reach_validation_so_an_imported_name_is_asked_about():
    """The load node's candidate keys are the names the validator checks. A statement naming a
    known place Juniper never said in the episode is kept but escalated (situation-graph spec §7)."""
    imported = ('{"memories": [{"purpose": "happened", "voice": "juniper_said", "stakes": "low", "stakes_reason": "none",'
                ' "statement": "Juniper flew from Seattle to Austin and returns Wednesday.",'
                ' "evidence": [{"turn": "t1", "field": "prompt", "quote": "Headed to Austin"}]}], "questions": []}')

    class WithCandidates(Distiller):
        async def load(self, brief):
            self.loads += 1
            return {"turns": turns_to_state(turns_from_rows(ROWS)), "candidate_referents": ["place:seattle"]}

    async def run():
        world, saver = World(), InMemorySaver()
        d = WithCandidates(world, [imported])
        await asyncio.wait_for(graph(world, saver, d).ainvoke(initial(), CFG), 1)
        world.grant()
        await graph(world, saver, d).ainvoke(Command(resume={}), CFG)
        (m,) = d.persisted[0]["result"].memories
        assert (m.stakes, m.stakes_reason, m.confirmation_state) == ("high", "ungrounded_name", "pending_confirmation")

    asyncio.run(run())


def test_answer_cut_at_max_tokens_is_named_truncation_not_invalid_json():
    """Live 2026-10-31: 12 of 12 calls on one episode stopped at 4096 tokens (finish_reason=length).
    Each failed as ``distill_json_invalid:Expecting ',' delimiter``, which reads as a model that writes
    broken JSON. The attempt must say what happened: the answer was cut at the budget."""
    async def run():
        world, saver = World(), InMemorySaver()
        world.grant()
        world.release = _regranting_release(world)
        d = Distiller(world, [(GOOD[: len(GOOD) // 2], "length"), GOOD])
        await graph(world, saver, d).ainvoke(initial(), CFG)
        snap = await graph(world, saver, d).aget_state(CFG)
        assert snap.next == ("retry_wait",)
        assert "distill_truncated_at_max_tokens:" in snap.values["last_error"]
        assert "distill_json_invalid" not in snap.values["last_error"]
        assert d.persisted == []

    asyncio.run(run())
