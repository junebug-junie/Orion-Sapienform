"""The 5b judge leaves a hold that is not on a judge role (2026-11-12).

Live 11-11 05:29Z..11-12: 22 of 90 judge calls ran on the gpu2 seat's model because the
run's hold was on pool role ``agent-gpu2`` and a hold's calls run on the hold's role.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from orion.gpu_pool.config import load_pool_config
from orion.harness.finalize import (
    FINALIZE_JUDGE_HOLD_ROLES,
    FINALIZE_JUDGE_ROUTE,
    build_finalize_reflect_context,
    build_response_repair_context,
    resolve_finalize_judge_placement,
)
from orion.harness.tests.fixtures import make_appraisal, make_reflection, make_repair_overlay, make_thought
from orion.llm.routes import CORTEX_OVERRIDE_SYSTEM_ROUTES, normalize_llm_route
from orion.llm.routes import AGENT_ROUTE_FCC_MODEL_LABEL
from orion.schemas.gpu_pool import GpuLeaseRefV1

ROOT = Path(__file__).resolve().parents[3]
CFG = load_pool_config(ROOT / "config" / "gpu_pool.yaml")


def _hold(role: str) -> GpuLeaseRefV1:
    return GpuLeaseRefV1(lease_id="hold-1", generation=3, role=role, holder="durable-runs:run-1")


def _judge_ctx(gpu_lease=None, fcc_model_label=None) -> dict:
    return build_finalize_reflect_context(
        correlation_id="c", draft_text="draft", thought=make_thought(), substrate_appraisal=make_appraisal(),
        repair_overlay=make_repair_overlay(), user_message="m", gpu_lease=gpu_lease,
        fcc_model_label=fcc_model_label,
    )


def test_gpu2_seat_hold_sends_the_judge_off_the_hold():
    ctx = _judge_ctx(gpu_lease=_hold("agent-gpu2"))
    assert ctx["llm_route"] == FINALIZE_JUDGE_ROUTE == "judge"
    assert ctx["llm_lane"] == "agent"          # cortex-exec lane class; "judge" is not one
    assert "gpu_lease" not in ctx              # with the ref the gateway would attach to agent-gpu2


@pytest.mark.parametrize("role", ["agent", "chat"])
def test_hold_on_a_judge_role_keeps_the_judge_on_the_hold(role):
    hold = _hold(role)
    ctx = _judge_ctx(gpu_lease=hold, fcc_model_label="MODEL_SONNET")
    assert ctx["llm_route"] == ctx["llm_lane"] == "agent"
    assert ctx["gpu_lease"] == hold.model_dump(mode="json")


@pytest.mark.parametrize(("label", "lane"), [(None, "chat"), ("MODEL_SONNET", "chat"),
                                             (AGENT_ROUTE_FCC_MODEL_LABEL, "agent")])
def test_no_hold_keeps_the_owner_rule(label, lane):
    # corr 60f0e051: a chat turn's judge stays on chat, never pushed to agent work.
    ctx = _judge_ctx(fcc_model_label=label)
    assert ctx["llm_route"] == ctx["llm_lane"] == lane
    assert "gpu_lease" not in ctx


def test_repair_keeps_the_hold_whatever_its_role():
    hold = _hold("agent-gpu2")
    ctx = build_response_repair_context(correlation_id="c", draft_text="d", reflection=make_reflection(),
                                        user_message="m", gpu_lease=hold)
    assert ctx["llm_route"] == "agent" and ctx["gpu_lease"] == hold.model_dump(mode="json")


def test_any_non_judge_role_leaves_the_hold():
    for role in CFG.roles:
        route, _lane, lease = resolve_finalize_judge_placement(gpu_lease=_hold(role))
        assert (lease is None) == (role not in FINALIZE_JUDGE_HOLD_ROLES), role
        assert (route == FINALIZE_JUDGE_ROUTE) == (role not in FINALIZE_JUDGE_HOLD_ROLES), role


def test_judge_roles_match_the_pool_config():
    assert CFG.routes[FINALIZE_JUDGE_ROUTE].work_class == "judge"
    assert set(CFG.classes["judge"].roles) == set(FINALIZE_JUDGE_HOLD_ROLES)
    # The seat the pool loads Bonsai on (launch.profiles[0]) is never a judge role.
    seat = CFG.roles["agent-gpu2"]
    assert seat.cards == ["gpu2"] and seat.launch is not None
    assert "agent-gpu2" not in CFG.classes["judge"].roles


def test_judge_owns_gpu1_and_only_borrows_chat():
    # As a borrower on gpu1, rule 5 would hold every judge call behind any queued agent lease.
    assert CFG.owns("judge", "agent")
    assert not CFG.owns("judge", "chat")       # chat's card only while Juniper lends it (rule 9)


def test_cortex_exec_accepts_the_judge_route_as_an_override():
    assert normalize_llm_route("judge", allow_system=CORTEX_OVERRIDE_SYSTEM_ROUTES) == "judge"
    assert normalize_llm_route("judge") is None  # never a human Compute choice
