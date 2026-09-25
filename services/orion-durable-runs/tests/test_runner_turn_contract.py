"""An admitted inference budget and identity survive the real Hub RPC seam."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.runner import DurableRunner
from app.settings import Settings
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.schemas.durable_run import CuriosityTurnRequestV1, CuriosityTurnResultV1, CURIOSITY_TURN_RESULT_KIND
from orion.schemas.gpu_pool import GpuLeaseRefV1


HOLD = GpuLeaseRefV1(lease_id="hold-one", generation=1, role="agent-gpu2", holder="durable-runs:study-budget")


def turn_request(*, admitted=True):
    # Stage 4.5: an admitted turn carries the run's GPU pool hold, never a pool role as a route.
    return CuriosityTurnRequestV1(run_id="study-budget", correlation_id=str(uuid4()),
        prompt="Inspect the supplied study fixture", timeout_sec=7200,
        gpu_lease=HOLD if admitted else None)


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


def test_admitted_turn_on_the_wire_carries_the_hold_and_no_route_label():
    request = turn_request()
    runner, bus = runner_for(request)
    asyncio.run(runner._run_turn(request))
    envelope = bus.rpc_request.await_args.args[1]
    assert envelope.payload["gpu_lease"] == HOLD.model_dump(mode="json")
    assert "assigned_lane" not in envelope.payload and "lease" not in envelope.payload
    assert "fcc_model_label" not in envelope.payload


def test_reflect_llm_call_sends_the_hold_ref_in_options_and_keeps_its_route():
    """Stage 4.4 hazard: the reflect call carried neither lease. Under a hold it must carry the ref
    (cortex-exec forwards options.gpu_lease, the gateway attaches), with llm_route unchanged."""
    from orion.core.bus.bus_schemas import BaseEnvelope as Env
    from orion.schemas.cortex.contracts import CortexClientRequest

    codec = OrionCodec()
    reply = Env(kind="cortex.orch.result", source=ServiceRef(name="cortex-orch"),
                payload={"ok": True, "final_text": '{"findings": [{"kind": "k"}]}'})
    bus = SimpleNamespace(codec=codec, rpc_request=AsyncMock(return_value={"data": codec.encode(reply)}))
    settings = Settings(_env_file=None, DURABLE_RUNS_GRAPH_HOST="", POSTGRES_URI="postgresql://unused",
                        ORION_BUS_ENABLED=False)
    runner = DurableRunner(settings, bus=bus, checkpointer=None)
    for ref in (HOLD, None):
        asyncio.run(runner._call_reflect_llm({"snapshot_id": "s"}, "agent", **({"gpu_lease": ref} if ref else {})))
        sent = CortexClientRequest.model_validate(bus.rpc_request.await_args.args[1].payload)
        assert sent.options["llm_route"] == "agent"
        if ref:
            assert sent.options["gpu_lease"] == HOLD.model_dump(mode="json")
        else:
            assert "gpu_lease" not in sent.options
