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
def test_admitted_token_reaches_gateway_with_existing_route_override(location):
    now = datetime.now(timezone.utc)
    token = ResourceLeaseV1(
        run_id="study-one", demand_id="study-one:harness", lease_id="lease-one",
        resource_key="llm.route.metacog", lane="metacog", backend_key="http://worker:8000", generation=2,
        granted_at=now, heartbeat_at=now, expires_at=now + timedelta(seconds=60),
    ).model_dump(mode="json")
    ctx = {"mode": "brain", "llm_route": "metacog", "session_id": "study-session",
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
    assert req.route == "metacog"
    assert req.options["resource_lease"] == token
