"""Stage 4.4: a durable run's GPU pool hold ref (GpuLeaseRefV1) rides every LLM call of the turn.

FCC gets it as ``X-Orion-Gpu-Lease`` in ANTHROPIC_CUSTOM_HEADERS (straight to the gateway, like the
old durable token); every finalize hop carries it as ``gpu_lease`` so cortex-exec forwards it to
the gateway as ``options.gpu_lease``. Both coexist with the old ``resource_lease`` until 4.6.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from orion.harness.fcc_motor import _build_subprocess_env
from orion.harness.finalize import resolve_finalize_llm_lane, run_harness_finalize_chain
from orion.harness.runner import HarnessRunner, build_coalition_snapshot, build_draft_molecule
from orion.harness.tests.fixtures import make_appraisal, make_reflection, make_repair_overlay, make_thought
from orion.llm.resource_lease import (
    GPU_LEASE_HEADER, GPU_LEASE_ROUTE, LEASE_HEADER, decode_gpu_lease_header, decode_lease_header,
)
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.context_exec import ContextExecPermissionV1
from orion.schemas.gpu_pool import GpuLeaseRefV1
from orion.schemas.harness_finalize import HarnessRunRequestV1
from orion.schemas.resource_admission import ResourceLeaseV1

REF = GpuLeaseRefV1(lease_id="hold-1", generation=4, role="agent-gpu2", holder="durable-runs:run-1")


def _token() -> ResourceLeaseV1:
    now = datetime.now(timezone.utc)
    return ResourceLeaseV1(run_id="run-1", demand_id="run-1:turn", lease_id="legacy", resource_key="llm.route.agent",
                           lane="agent", backend_key="http://worker:8000", generation=2,
                           granted_at=now, heartbeat_at=now, expires_at=now + timedelta(seconds=60))


def test_fcc_gets_the_ref_header_and_goes_straight_to_the_gateway(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", f"X-Trace: t\n{GPU_LEASE_HEADER.lower()}: another-runs-stale-ref")
    monkeypatch.setenv("HARNESS_LLM_GATEWAY_URL", "http://llm-gateway:8210/")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "external-proxy-secret")
    env = _build_subprocess_env(fcc_server_url="http://fcc-proxy:8082", auth_token="fcc-secret",
                                gpu_lease=REF.model_dump(mode="json"))
    headers = env["ANTHROPIC_CUSTOM_HEADERS"].splitlines()
    assert headers[0] == "X-Trace: t" and len(headers) == 2  # the inherited ref is gone
    name, value = headers[1].split(": ", 1)
    assert name == GPU_LEASE_HEADER and decode_gpu_lease_header(value) == REF
    # Same direct-to-gateway rule as the durable token: the FCC proxy forwards no custom headers.
    assert env["ANTHROPIC_BASE_URL"] == "http://llm-gateway:8210"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "orion-resource-lease"
    assert "ANTHROPIC_API_KEY" not in env


def test_fcc_carries_both_old_token_and_new_ref_until_4_6(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)
    token = _token().model_dump(mode="json")
    env = _build_subprocess_env(fcc_server_url="http://fcc-proxy:8082", auth_token="tok",
                                resource_lease=token, gpu_lease=REF.model_dump(mode="json"))
    headers = dict(line.split(": ", 1) for line in env["ANTHROPIC_CUSTOM_HEADERS"].splitlines())
    assert decode_lease_header(headers[LEASE_HEADER]) == token
    assert decode_gpu_lease_header(headers[GPU_LEASE_HEADER]) == REF


def test_a_turn_without_a_ref_sends_no_ref_header(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", f"{GPU_LEASE_HEADER}: stale")
    env = _build_subprocess_env(fcc_server_url="http://fcc-proxy:8082", auth_token="tok")
    assert "ANTHROPIC_CUSTOM_HEADERS" not in env
    assert env["ANTHROPIC_BASE_URL"] == "http://fcc-proxy:8082"


@pytest.mark.asyncio
async def test_motor_receives_the_ref():
    captured = {}

    async def motor(**kwargs):
        captured.update(kwargs)
        yield {"type": "error", "error": "synthetic stop", "error_code": "test"}
    request = HarnessRunRequestV1(correlation_id="run-1", thought_event=make_thought(), user_message="study",
                                  permissions=ContextExecPermissionV1(), answer_contract=AnswerContract(),
                                  gpu_lease=REF)
    await HarnessRunner(AsyncMock(), fcc_runner=motor, fcc_timeout_sec=999).run(request)
    assert captured["gpu_lease"] == REF.model_dump(mode="json")
    assert "resource_lease" not in captured


def test_request_without_a_ref_keeps_its_old_wire_shape():
    request = HarnessRunRequestV1(correlation_id="c", thought_event=make_thought(), user_message="m",
                                  permissions=ContextExecPermissionV1(), answer_contract=AnswerContract())
    assert request.gpu_lease is None
    assert HarnessRunRequestV1.model_validate(request.model_dump(mode="json")).gpu_lease is None


def test_finalize_lane_under_a_hold_is_the_agent_route_not_the_role():
    # "agent-gpu2" is a pool role, not a route: naming it would 404 in the gateway.
    assert resolve_finalize_llm_lane(gpu_lease=REF) == GPU_LEASE_ROUTE == "agent"
    assert resolve_finalize_llm_lane(resource_lease=_token().model_copy(update={"lane": "chat"}), gpu_lease=REF) == "chat"


@pytest.mark.asyncio
@pytest.mark.parametrize("repair_required", [False, True])
async def test_every_finalize_hop_carries_the_ref(monkeypatch, repair_required):
    monkeypatch.setenv("HARNESS_FINALIZE_TOOL_LOOP_ENABLED", "true")
    thought, overlay, draft = make_thought(), make_repair_overlay(), "A grounded internal draft."
    molecule = build_draft_molecule(correlation_id="t", thought=thought, draft_text=draft, grammar_receipts=[],
                                    coalition_snapshot=build_coalition_snapshot(thought), repair_overlay=overlay)
    seen = []
    reflections = 0

    async def cortex_client(request):
        nonlocal reflections
        verb = request.plan.verb_name
        if verb == "look_at_camera":
            return {"final_text": "The camera shows a table."}
        seen.append(verb)
        assert request.context["gpu_lease"] == REF.model_dump(mode="json")
        assert "resource_lease" not in request.context
        assert request.context["llm_route"] == request.context["llm_lane"] == "agent"
        if verb == "harness_finalize_reflect":
            reflections += 1
            reflection = make_reflection(
                alignment_verdict="misaligned" if reflections == 1 or repair_required else "aligned",
                recommended_tool="look_at_camera" if reflections == 1 else None)
            return {"final_text": json.dumps(reflection.model_dump(mode="json"))}
        return {"final_text": "The grounded final answer."}

    async def substrate_client(_molecule):
        return make_appraisal(surprise_level=0.5)

    await run_harness_finalize_chain(
        correlation_id="t", draft_text=draft, draft_molecule=molecule, thought=thought, grammar_receipts=[],
        repair_overlay=overlay, user_message="What does the camera show?", voice_contract=None,
        cortex_client=cortex_client, substrate_client=substrate_client, gpu_lease=REF,
    )
    assert seen[:2] == ["harness_finalize_reflect", "harness_finalize_reflect"]
    assert ("orion_response_repair" in seen) is repair_required
