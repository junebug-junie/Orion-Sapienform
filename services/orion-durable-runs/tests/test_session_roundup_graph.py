"""session_roundup.orion_turn: Orion's hourly read of the messages coding sessions sent, as one admitted
durable run (the single-step harness graph consult.orion_turn uses). What must hold: the turn runs under
the hold in the round's own session, lookup NOT required (the specs are in the prompt), a per-round source
tag, the full text checkpointed; a failed turn is retried by the shared policy; a cut-short turn is NOT
retried (Hub holds the threads for the next hour); its own act family, never the consult family's."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver

from app import run_acts
from app.admitted_graph import AdmissionDeps
from app.harness_smoke_graph import build_session_roundup_graph, session_roundup_finish_detail
from app.run_bids import BINDING_FAMILIES
from orion.schemas.durable_run import DurableRunRequestV1
from orion.schemas.resource_admission import ResourceRequirementV1
from orion.schemas.session_roundup_run import (
    SESSION_ROUNDUP_STEP_KEY, SESSION_ROUNDUP_TAG, SESSION_ROUNDUP_WORKFLOW, SessionRoundupBriefV1,
    session_roundup_run_id, session_roundup_session_id,
)
from tests.test_consult_turn_graph import CUT_TEXT, CutWorld
from tests.test_harness_smoke_graph import REF, World

RUN_ID = session_roundup_run_id(datetime(2026, 11, 15, 11, 0, tzinfo=timezone.utc))
CFG = {"configurable": {"thread_id": RUN_ID}}
LONG = "### thread:abc123\n1. Why hourly?\n" * 150   # longer than the 2000-char state-row cap
BRIEF = {"roundup": "session_roundup.orion_turn", "roundup_id": RUN_ID, "thread_ids": ["abc123"],
         "prompt": "Hourly round-up.", "session_id": session_roundup_session_id(RUN_ID),
         "retrieval_query": "session threads design", "timeout_sec": 900}


def _graph(world, saver):
    return build_session_roundup_graph(
        world.turn,
        AdmissionDeps(world.register, world.lease, world.execute, world.release, world.event,
                      now=lambda: world.now, max_attempts=world.max_attempts, retry_base_seconds=10.0),
        saver)


def _initial():
    return {"run_id": RUN_ID, "correlation_id": "trace-roundup", "workflow": SESSION_ROUNDUP_WORKFLOW,
            "attempt": 0, "admission": {"resource": "llm.route.agent"}, "brief": dict(BRIEF)}


def test_run_id_is_one_per_hour():
    assert RUN_ID == "session-roundup-20261115T1100"
    assert session_roundup_session_id(RUN_ID) == "orion_session-roundup-20261115T1100"


def test_the_request_validates_and_needs_admission_and_its_own_brief():
    ok = DurableRunRequestV1(run_id=RUN_ID, workflow=SESSION_ROUNDUP_WORKFLOW, correlation_id="c",
                             brief=SessionRoundupBriefV1(**BRIEF), admission=ResourceRequirementV1())
    assert isinstance(DurableRunRequestV1.model_validate(ok.model_dump(mode="json")).brief, SessionRoundupBriefV1)
    with pytest.raises(ValueError):
        DurableRunRequestV1(run_id=RUN_ID, workflow=SESSION_ROUNDUP_WORKFLOW, correlation_id="c",
                            brief=SessionRoundupBriefV1(**BRIEF))
    with pytest.raises(ValueError):
        DurableRunRequestV1(run_id=RUN_ID, workflow="consult.orion_turn", correlation_id="c",
                            brief=SessionRoundupBriefV1(**BRIEF), admission=ResourceRequirementV1())
    with pytest.raises(ValueError):
        SessionRoundupBriefV1(**{**BRIEF, "prompt": ""})


def test_roundup_turn_runs_in_its_own_session_and_keeps_the_full_text():
    async def scenario():
        world, saver = World(replies=[LONG]), InMemorySaver()
        world.grant()
        result = await asyncio.wait_for(_graph(world, saver).ainvoke(_initial(), CFG), 2)
        assert result["status"] == "completed"
        [req] = world.turns
        assert req.require_lookup is False and req.session_id == f"orion_{RUN_ID}"
        assert req.source_tag == f"{SESSION_ROUNDUP_TAG}:{RUN_ID}" and req.retrieval_query == "session threads design"
        assert req.gpu_lease.model_dump() == REF
        assert result["steps"][SESSION_ROUNDUP_STEP_KEY]["text"] == LONG
        detail = session_roundup_finish_detail(result)
        assert detail["line"] == "session_roundup" and detail["thread_ids"] == ["abc123"]
        assert detail["steps"]["turn"]["text_truncated"] is True
    asyncio.run(scenario())


def test_a_failed_turn_is_retried_and_a_cut_short_turn_is_not():
    async def scenario():
        world, saver = World(replies=[("fail", "timeout"), LONG]), InMemorySaver()
        world.grant()
        await asyncio.wait_for(_graph(world, saver).ainvoke(_initial(), CFG), 2)
        assert (await _graph(world, saver).aget_state(CFG)).next == ("retry_wait",)

        world, saver = CutWorld(replies=[("cut", CUT_TEXT)]), InMemorySaver()
        world.grant()
        result = await asyncio.wait_for(_graph(world, saver).ainvoke(_initial(), CFG), 2)
        # One attempt, completed with the honest cut-short text and its code: Hub reads the code and
        # holds the threads for the next hour, instead of two more hour-long attempts.
        assert result["status"] == "completed" and len(world.turns) == 1
        step = result["steps"][SESSION_ROUNDUP_STEP_KEY]
        assert step["text"] == CUT_TEXT and step["debug"]["harness_grounding_status"] == "fcc_timeout"
    asyncio.run(scenario())


def test_admission_runtime_registers_the_roundup():
    from app.admission_runtime import HARNESS_STEP_WORKFLOWS, WORK_NODES, AdmissionRuntime

    assert WORK_NODES[SESSION_ROUNDUP_WORKFLOW] == {"harness_step"}
    assert HARNESS_STEP_WORKFLOWS[SESSION_ROUNDUP_WORKFLOW] == SESSION_ROUNDUP_STEP_KEY
    runtime = object.__new__(AdmissionRuntime)
    runtime.runner = SimpleNamespace(_publish=AsyncMock(return_value=True), _corr_for_admission=lambda v: v)
    detail = runtime._finish_detail_for(SESSION_ROUNDUP_WORKFLOW, {
        "brief": dict(BRIEF), "steps": {SESSION_ROUNDUP_STEP_KEY: {"text": "ok", "turn_correlation_id": "c-1"}}})
    assert detail["line"] == "session_roundup" and detail["turn_correlation_id"] == "c-1"


def test_its_own_act_family_tier_2_and_shadow_bids():
    family = run_acts.run_act_family({"workflow": SESSION_ROUNDUP_WORKFLOW, "brief": dict(BRIEF)})
    assert (family.actor, family.verb) == ("durable.session_roundup", "ask")
    assert family.actor != run_acts.FAMILIES["consult_orion_turn"].actor
    assert "durable.session_roundup" not in BINDING_FAMILIES   # the hourly cadence is Juniper's rule

    from pathlib import Path

    import yaml

    doc = yaml.safe_load((Path(__file__).resolve().parents[3] / "config/physiology/families.yaml").read_text())
    assert doc["families"]["durable.session_roundup"]["tier"] == 2
