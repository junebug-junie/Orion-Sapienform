"""Explicit admission tokens survive the existing Cortex Exec -> Gateway seam."""
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.executor import call_step_services
from orion.core.bus.bus_schemas import ChatResponsePayload, ServiceRef
from orion.schemas.cortex.schemas import ExecutionStep
from orion.schemas.resource_admission import ResourceLeaseV1


@pytest.mark.parametrize("location", ["options", "context"])
@pytest.mark.parametrize("lease_lane,override,expected_route", [
    ("metacog", "metacog", "metacog"),
    ("harness", "harness", "harness"),
    ("catalog-alias", None, "catalog-alias"),
    ("metacog", "agent", "agent"),
])
def test_admitted_token_reaches_gateway_with_existing_route_override(location, lease_lane, override, expected_route):
    now = datetime.now(timezone.utc)
    token = ResourceLeaseV1(
        run_id="study-one", demand_id="study-one:harness", lease_id="lease-one",
        resource_key=f"llm.route.{lease_lane}", lane=lease_lane, backend_key="http://worker:8000", generation=2,
        granted_at=now, heartbeat_at=now, expires_at=now + timedelta(seconds=60),
    ).model_dump(mode="json")
    ctx = {"mode": "brain", "llm_route": override, "session_id": "study-session",
           "raw_user_text": "Evaluate this study result.",
           "messages": [{"role": "user", "content": "Evaluate this study result."}]}
    if location == "options":
        ctx["options"] = {"resource_lease": token}
    else:
        ctx["resource_lease"] = token
    step = ExecutionStep(step_name="llm_harness_finalize_reflect", verb_name="harness_finalize_reflect",
                         services=["LLMGatewayService"], order=0, prompt_template="{{ raw_user_text }}")
    with patch("app.executor.LLMGatewayClient.chat", new=AsyncMock(
        return_value=ChatResponsePayload(content='{"verdict":"aligned"}')
    )) as chat:
        result = asyncio.run(call_step_services(bus=MagicMock(), source=ServiceRef(name="test"),
                            step=step, ctx=ctx, correlation_id=str(uuid4())))
    assert result.status == "success"
    req = chat.await_args.kwargs["req"]
    assert req.route == expected_route
    assert req.options["resource_lease"] == token


def _run_finalize_step(ctx):
    step = ExecutionStep(step_name="llm_harness_finalize_reflect", verb_name="harness_finalize_reflect",
                         services=["LLMGatewayService"], order=0, prompt_template="{{ raw_user_text }}")
    base = {"mode": "brain", "session_id": "study-session", "raw_user_text": "Evaluate this study result.",
            "messages": [{"role": "user", "content": "Evaluate this study result."}]}
    with patch("app.executor.LLMGatewayClient.chat", new=AsyncMock(
        return_value=ChatResponsePayload(content='{"verdict":"aligned"}')
    )) as chat:
        result = asyncio.run(call_step_services(bus=MagicMock(), source=ServiceRef(name="test"),
                            step=step, ctx={**base, **ctx}, correlation_id=str(uuid4())))
    assert result.status == "success"
    return chat.await_args.kwargs["req"]


@pytest.mark.parametrize("location", ["options", "context"])
@pytest.mark.parametrize("override,expected_route", [(None, "agent"), ("agent", "agent"), ("chat", "chat"),
                                                     ("AGENT", "agent"), ("not-a-route", "agent")])
def test_gpu_pool_hold_ref_reaches_gateway_as_options_gpu_lease(location, override, expected_route):
    """Stage 4.4: the run's hold ref rides to the gateway (which attaches the call to the hold). With
    no explicit route the call names the hold's work-class route, never the role ("agent-gpu2")."""
    ref = {"lease_id": "hold-1", "generation": 3, "role": "agent-gpu2", "holder": "durable-runs:run-1"}
    ctx = {"llm_route": override}
    if location == "options":
        ctx["options"] = {"gpu_lease": ref}
    else:
        ctx["gpu_lease"] = ref
    req = _run_finalize_step(ctx)
    assert req.route == expected_route
    assert req.options["gpu_lease"] == ref
    assert "resource_lease" not in req.options


def test_old_token_and_new_ref_both_reach_the_gateway():
    now = datetime.now(timezone.utc)
    token = ResourceLeaseV1(
        run_id="run-1", demand_id="run-1:harness", lease_id="legacy", resource_key="llm.route.agent",
        lane="agent", backend_key="http://worker:8000", generation=2,
        granted_at=now, heartbeat_at=now, expires_at=now + timedelta(seconds=60),
    ).model_dump(mode="json")
    ref = {"lease_id": "hold-1", "generation": 3, "role": "agent", "holder": "durable-runs:run-1"}
    req = _run_finalize_step({"resource_lease": token, "gpu_lease": ref})
    assert req.route == "agent"
    assert req.options["resource_lease"] == token and req.options["gpu_lease"] == ref
