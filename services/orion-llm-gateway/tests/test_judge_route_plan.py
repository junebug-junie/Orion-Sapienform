"""The finalize 5b judge's wire shape off a gpu2-seat hold, planned by the gateway against the repo's
real config/gpu_pool.yaml with live lane-routing settings (2026-11-12).

orion/harness/tests/test_finalize_judge_placement.py pins the other half: the judge context is
route `judge`, lane `agent`, and no `gpu_lease`. This file does not import orion.harness (the
gateway's CI job installs only the gateway's requirements)."""
from __future__ import annotations

from pathlib import Path

import pytest

from orion.gpu_pool.config import load_pool_config
from orion.llm.routes import CORTEX_OVERRIDE_SYSTEM_ROUTES, normalize_llm_route

from app import llm_backend as lb
from app import pool_placement
from app.llm_backend import plan_llm_chat
from app.models import ChatBody, ChatMessage

CFG = load_pool_config(Path(__file__).resolve().parents[3] / "config" / "gpu_pool.yaml")
HOLD = {"lease_id": "h", "generation": 1, "role": "agent", "holder": "durable-runs:r"}


@pytest.fixture(autouse=True)
def _live_gateway(monkeypatch):
    monkeypatch.setattr(pool_placement, "pool_routes", lambda: dict(CFG.routes))
    monkeypatch.setattr(lb.settings, "llm_lane_routing_enabled", True)   # live .env
    monkeypatch.setattr(lb.settings, "llm_lane_default", "chat")
    monkeypatch.setattr(lb.settings, "llm_route_default", "quick")


def _plan(route: str, **options):
    accepted = normalize_llm_route(route, allow_system=CORTEX_OVERRIDE_SYSTEM_ROUTES)  # cortex-exec's override
    return plan_llm_chat(ChatBody(route=accepted, messages=[ChatMessage(role="user", content="x")],
                                  options=options, trace_id="c"))


def test_judge_off_the_hold_plans_on_class_judge_never_the_gpu2_seat():
    # Before the lane_routes fix this planned ('agent', 'agent'): a class that lists agent-gpu2.
    plan = _plan("judge", llm_lane="agent")
    assert plan.error is None
    assert (plan.route, plan.work_class) == ("judge", "judge")
    assert "agent-gpu2" not in CFG.classes[plan.work_class].roles


def test_judge_on_an_agent_hold_stays_attached_on_route_agent():
    plan = _plan("agent", llm_lane="agent", gpu_lease=HOLD)
    assert plan.route == "agent" and plan.body.options["gpu_lease"] == HOLD
