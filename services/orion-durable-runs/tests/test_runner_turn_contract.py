"""An admitted inference budget and identity survive the real Hub RPC seam."""
import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.runner import DurableRunner
from app.settings import Settings
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.schemas.durable_run import CuriosityTurnRequestV1, CuriosityTurnResultV1, CURIOSITY_TURN_RESULT_KIND
from orion.schemas.resource_admission import ResourceLeaseV1


def turn_request(*, admitted=True):
    now = datetime.now(timezone.utc)
    lease = ResourceLeaseV1(run_id="study-budget", demand_id="study-budget:turn", lease_id="lease-one",
        generation=1, resource_key="llm.route.agent", lane="agent", backend_key="http://fixture",
        granted_at=now, heartbeat_at=now, expires_at=now + timedelta(seconds=90))
    return CuriosityTurnRequestV1(run_id="study-budget", correlation_id=str(uuid4()),
        prompt="Inspect the supplied study fixture", timeout_sec=7200,
        lease=lease if admitted else None, assigned_lane="agent" if admitted else None)


def runner_for(request, *, mutate=None):
    reply = BaseEnvelope(kind=CURIOSITY_TURN_RESULT_KIND, source=ServiceRef(name="orion-hub"),
        correlation_id=request.correlation_id, payload=CuriosityTurnResultV1(
            run_id=request.run_id, correlation_id=request.correlation_id, text="An inspected fixture finding").model_dump(mode="json"))
    if mutate:
        reply = reply.model_dump(mode="json", by_alias=True)
        mutate(reply)
    codec = OrionCodec()
    bus = SimpleNamespace(codec=codec, rpc_request=AsyncMock(return_value={"data": codec.encode(reply)}))
    settings = Settings(_env_file=None, DURABLE_RUNS_GRAPH_HOST="", POSTGRES_URI="postgresql://unused",
                        ORION_BUS_ENABLED=False, DURABLE_RUNS_TURN_RPC_TIMEOUT_SEC=0.05)
    return DurableRunner(settings, bus=bus, checkpointer=None), bus


@pytest.mark.parametrize("admitted", [False, True])
def test_declared_inference_budget_outlives_legacy_rpc_timeout(admitted):
    request = turn_request(admitted=admitted)
    runner, bus = runner_for(request)
    result = asyncio.run(runner._run_turn(request))
    assert result.ok
    timeout = bus.rpc_request.await_args.kwargs["timeout_sec"]
    assert timeout >= request.timeout_sec if admitted else timeout == 0.05


@pytest.mark.parametrize("field", ["kind", "envelope_correlation", "run_id", "correlation_id"])
def test_admitted_turn_rejects_misrouted_or_stale_reply(field):
    request = turn_request()
    def mutate(reply):
        if field == "kind":
            reply["kind"] = "cortex.exec.result"
        elif field == "envelope_correlation":
            reply["correlation_id"] = str(uuid4())
        else:
            reply["payload"][field] = str(uuid4())
    runner, _bus = runner_for(request, mutate=mutate)
    result = asyncio.run(runner._run_turn(request))
    assert not result.ok
    assert result.error.startswith("bad_reply:")
