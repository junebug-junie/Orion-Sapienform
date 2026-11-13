"""intention.carry (choice-as-EFE Step -1): the carry run of a concern Prime took on, as an admitted durable run.

What must hold: the request contract (brief and workflow agree, admission required); the run's act family is
``durable.intention`` (make); its bid BINDS at TIER 1 (Juniper 2026-11-13); a held run is admitted at its hold end,
never dropped (S-1.3), and the hold end is capped by the concern's horizon; the one turn runs in session
``orion_intention`` with lookup not required and its prompt citing ``prime_concern:<id>``."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.run_acts import run_act_family
from app.run_bids import BINDING_PRODUCER, RunBidder, binding_family, hold_end, want_for_run
from orion.physiology.families import load_families
from orion.schemas.durable_run import DurableRunRequestV1
from orion.schemas.intention import (
    INTENTION_CARRY_STEP_KEY, INTENTION_CARRY_TAG, INTENTION_CARRY_WORKFLOW, INTENTION_FAMILY, INTENTION_SESSION_ID,
    intention_run_id,
)

ROOT = Path(__file__).resolve().parents[3]
FAMS, _ = load_families(ROOT / "config/physiology/families.yaml")
T0 = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)
CID = "d8db9d7b1bba4cf7"
HORIZON = T0 + timedelta(hours=60)


def _brief(**kw):
    return {"intention": "intention.carry", "concern_id": CID, "source": "turn_promise", "prime_act_id": "act_1",
            "prompt": f"prime_concern:{CID}\n\nI am carrying out something I told Juniper I would do.",
            "route": "chat_thread", "route_target": "9c91d646-cbd0-4977-8cdd-d2c82045c7f9",
            "quote_ref": "harness_turn_trace:x#promises_made[0]", "horizon_at": HORIZON.isoformat(),
            "retrieval_query": "a lazy browse", "timeout_sec": 1800, **kw}


def _req(requested_at=T0, deadline=HORIZON):
    return {"run_id": intention_run_id(CID), "correlation_id": "c-1", "workflow": INTENTION_CARRY_WORKFLOW,
            "requested_at": requested_at.isoformat(), "brief": _brief(),
            "admission": {"resource": "llm.route.agent", "preferred_lane": "agent",
                          "deadline_at": deadline.isoformat() if deadline else None}}


def _dec(outcome="deferred", reason="price", net=-0.2, retry=900.0):
    return SimpleNamespace(decision_id=f"d@{reason}", outcome=outcome, reason=reason, net=net, charge=0.3,
                           retry_after_sec=retry, binding_constraint="lane:agent", lane="agent",
                           cost_frac={"lane:agent": 1.0}, why=f"why {reason}", state_tick_id="tick")


class _Market:
    def __init__(self, decision):
        self.families, self.decision, self.bids, self.released = FAMS, decision, [], []

    async def bid(self, want, now=None, *, reserve_at_open=False, object_text=None):
        self.bids.append(want)
        return self.decision

    def release(self, want_id):
        self.released.append(want_id)
        return 1


# --- contract --------------------------------------------------------------------------------------------------

def test_the_request_validates_and_the_brief_and_workflow_must_agree():
    req = DurableRunRequestV1.model_validate(_req())
    assert req.brief.concern_id == CID and req.admission.deadline_at == HORIZON
    with pytest.raises(ValueError):
        DurableRunRequestV1.model_validate({**_req(), "workflow": "consult.orion_turn"})
    with pytest.raises(ValueError):
        DurableRunRequestV1.model_validate({**_req(), "admission": None})
    with pytest.raises(ValueError):
        DurableRunRequestV1.model_validate({**_req(), "brief": _brief(concern_id="not-hex")})


# --- the family: tier 1, binding (Juniper 2026-11-13) ------------------------------------------------------------

def test_the_run_is_a_make_act_of_durable_intention_at_tier_1_and_binds():
    fam = run_act_family(_req())
    assert fam.actor == INTENTION_FAMILY and fam.verb == "make"
    cfg = FAMS.get(INTENTION_FAMILY)
    assert cfg.tier == 1 and cfg.deferrable_window_s == 86400 and "agent" in cfg.lanes and cfg.placement == "one_of"
    assert binding_family(_req(), True) == INTENTION_FAMILY and binding_family(_req(), False) is None
    want = want_for_run(_req(), cfg.tier)
    assert want.tier == 1 and want.family == INTENTION_FAMILY and want.window.latest == HORIZON


def test_hold_end_is_the_window_capped_by_the_concerns_horizon():
    cfg = FAMS.get(INTENTION_FAMILY)
    far = want_for_run(_req(deadline=T0 + timedelta(hours=60)), 1)
    assert hold_end(far, cfg.deferrable_window_s) == T0 + timedelta(seconds=86400)
    near = want_for_run(_req(deadline=T0 + timedelta(hours=5)), 1)
    assert hold_end(near, cfg.deferrable_window_s) == T0 + timedelta(hours=5) - timedelta(seconds=1800)


def test_a_held_tier_1_carry_run_rebids_and_is_admitted_at_its_hold_end_never_dropped():
    async def go():
        m = _Market(_dec())
        b = RunBidder(enabled=True, binding_enabled=True, market_getter=lambda: m)
        held = await b.bind(_req(), T0)
        assert not held.admit and held.why == "deferred:price" and held.retry_at == T0 + timedelta(seconds=900)
        assert m.bids[0].tier == 1 and m.bids[0].producer == BINDING_PRODUCER and m.released == [intention_run_id(CID)]
        end = T0 + timedelta(seconds=86400)
        last = await b.bind(_req(), end - timedelta(seconds=60))
        assert not last.admit and last.retry_at == end
        released = await b.bind(_req(), end)
        assert released.admit and released.why == "window_end"
        funded = await RunBidder(enabled=True, binding_enabled=True,
                                 market_getter=lambda: _Market(_dec("funded", "net_positive", 0.4))).bind(_req(), T0)
        assert funded.admit and funded.why == "funded"
    asyncio.run(go())


# --- the turn --------------------------------------------------------------------------------------------------

def test_the_carry_turn_runs_in_the_intention_session_writing_allowed_and_cites_the_concern():
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import InMemorySaver

    from app.admitted_graph import AdmissionDeps
    from app.harness_smoke_graph import build_intention_carry_graph, intention_carry_finish_detail
    from tests.test_harness_smoke_graph import World

    run_id = intention_run_id(CID)
    cfg = {"configurable": {"thread_id": run_id}}
    initial = {"run_id": run_id, "correlation_id": "c-1", "workflow": INTENTION_CARRY_WORKFLOW, "attempt": 0,
               "admission": {"resource": "llm.route.agent"}, "brief": _brief()}

    async def scenario():
        world, saver = World(replies=["I browsed the PRs. Two stand out."]), InMemorySaver()
        world.grant()
        graph = build_intention_carry_graph(
            world.turn, AdmissionDeps(world.register, world.lease, world.execute, world.release, world.event,
                                      now=lambda: world.now, max_attempts=world.max_attempts,
                                      retry_base_seconds=10.0), saver)
        result = await asyncio.wait_for(graph.ainvoke(initial, cfg), 2)
        assert result["status"] == "completed"
        [req] = world.turns
        assert req.session_id == INTENTION_SESSION_ID and req.require_lookup is False
        assert req.source_tag == f"{INTENTION_CARRY_TAG}:{CID}" and req.retrieval_query == "a lazy browse"
        assert req.prompt.startswith(f"prime_concern:{CID}\n")
        assert result["steps"][INTENTION_CARRY_STEP_KEY]["text"] == "I browsed the PRs. Two stand out."
        detail = intention_carry_finish_detail(result)
        assert detail["line"] == "intention_carry" and detail["concern_id"] == CID and detail["route"] == "chat_thread"
    asyncio.run(scenario())


def test_admission_runtime_registers_the_carry_run():
    from app.admission_runtime import HARNESS_STEP_WORKFLOWS, WORK_NODES, AdmissionRuntime

    assert WORK_NODES[INTENTION_CARRY_WORKFLOW] == {"harness_step"}
    assert HARNESS_STEP_WORKFLOWS[INTENTION_CARRY_WORKFLOW] == INTENTION_CARRY_STEP_KEY
    runtime = object.__new__(AdmissionRuntime)
    runtime.runner = SimpleNamespace(_publish=AsyncMock(return_value=True), _corr_for_admission=lambda v: v)
    detail = runtime._finish_detail_for(INTENTION_CARRY_WORKFLOW, {
        "brief": _brief(), "steps": {INTENTION_CARRY_STEP_KEY: {"text": "ok", "turn_correlation_id": "t-1"}}})
    assert detail["line"] == "intention_carry" and detail["turn_correlation_id"] == "t-1"
