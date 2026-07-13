from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orion.execution_dispatch.cortex_client import ExecutionDispatchCortexClient


def _bus_returning(payload: dict) -> AsyncMock:
    bus = AsyncMock()
    decode_result = MagicMock(ok=True, envelope=MagicMock(payload=payload))
    bus.codec = MagicMock()
    bus.codec.decode = MagicMock(return_value=decode_result)
    bus.rpc_request = AsyncMock(return_value={"data": b"payload"})
    return bus


@pytest.mark.asyncio
async def test_dispatch_sends_real_verb_plan_and_returns_payload() -> None:
    bus = _bus_returning({"result": {"final_text": '{"observation": "steady", "confidence": 0.8}'}})
    client = ExecutionDispatchCortexClient(
        bus,
        request_channel="orion:cortex:exec:request:background",
        result_prefix="orion:exec:result",
        timeout_sec=120.0,
    )

    payload = await client.dispatch(
        verb="substrate.inspect",
        mode="brain",
        context={"target_id": "capability:orchestration", "target_kind": "capability"},
        dispatch_id="dispatch:test:1",
    )

    assert payload["result"]["final_text"]
    bus.rpc_request.assert_awaited_once()
    assert bus.rpc_request.await_args.kwargs["timeout_sec"] == 120.0
    call_args = bus.rpc_request.await_args
    assert call_args.args[0] == "orion:cortex:exec:request:background"
    sent_envelope = call_args.args[1]
    # correlation_id must be a real UUID (BaseEnvelope requirement); the
    # human-readable dispatch_id travels in args.extra instead.
    assert sent_envelope.payload["args"]["extra"]["dispatch_id"] == "dispatch:test:1"
    assert sent_envelope.payload["plan"]["verb_name"] == "substrate.inspect"
    assert sent_envelope.payload["context"]["target_id"] == "capability:orchestration"


@pytest.mark.asyncio
async def test_dispatch_custom_timeout_overrides_default() -> None:
    bus = _bus_returning({"result": {"final_text": "{}"}})
    client = ExecutionDispatchCortexClient(
        bus,
        request_channel="orion:cortex:exec:request:background",
        result_prefix="orion:exec:result",
        timeout_sec=120.0,
    )

    await client.dispatch(
        verb="substrate.observe",
        mode="brain",
        context={},
        dispatch_id="dispatch:test:2",
        timeout_sec=30.0,
    )

    assert bus.rpc_request.await_args.kwargs["timeout_sec"] == 30.0


@pytest.mark.asyncio
async def test_dispatch_raises_on_rpc_not_ok() -> None:
    bus = AsyncMock()
    decode_result = MagicMock(ok=False, error="timeout")
    bus.codec = MagicMock()
    bus.codec.decode = MagicMock(return_value=decode_result)
    bus.rpc_request = AsyncMock(return_value={"data": b"payload"})
    client = ExecutionDispatchCortexClient(
        bus,
        request_channel="orion:cortex:exec:request:background",
        result_prefix="orion:exec:result",
    )

    with pytest.raises(RuntimeError, match="execution_dispatch cortex RPC failed"):
        await client.dispatch(
            verb="substrate.inspect", mode="brain", context={}, dispatch_id="dispatch:test:3"
        )


@pytest.mark.asyncio
async def test_dispatch_raises_on_non_dict_payload() -> None:
    bus = _bus_returning({})
    bus.codec.decode.return_value = MagicMock(ok=True, envelope=MagicMock(payload="not-a-dict"))
    client = ExecutionDispatchCortexClient(
        bus,
        request_channel="orion:cortex:exec:request:background",
        result_prefix="orion:exec:result",
    )

    with pytest.raises(RuntimeError, match="non-dict payload"):
        await client.dispatch(
            verb="substrate.inspect", mode="brain", context={}, dispatch_id="dispatch:test:4"
        )
