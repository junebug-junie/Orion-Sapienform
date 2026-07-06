from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1


@pytest.mark.asyncio
async def test_stance_react_error_publishes_defer_thought_event() -> None:
    from app import bus_listener

    corr_id = uuid4()
    corr = str(corr_id)
    reply_to = f"orion:thought:result:{corr}"
    req = StanceReactRequestV1(
        correlation_id=corr,
        session_id="sess-1",
        user_message="hello",
        association=HubAssociationBundleV1(
            correlation_id=corr,
            broadcast=None,
            broadcast_stale=True,
            read_source="felt_state_reader",
        ),
        repair_bundle=None,
        stance_inputs={"user_message": "hello"},
    )
    request_envelope = BaseEnvelope(
        kind="stance.react.request.v1",
        source=ServiceRef(name="orion-hub"),
        correlation_id=corr_id,
        reply_to=reply_to,
        payload=req.model_dump(mode="json"),
    )

    class _Codec:
        @staticmethod
        def decode(data):
            class _Decoded:
                ok = True
                envelope = request_envelope
                error = None

            return _Decoded()

    bus = AsyncMock()
    bus.codec = _Codec()

    with patch.object(bus_listener, "run_stance_react", AsyncMock(side_effect=TimeoutError("cortex timeout"))):
        await bus_listener._handle_bus_message(bus, {"data": b"ignored"})

    bus.publish.assert_awaited_once()
    channel, envelope = bus.publish.await_args.args
    assert channel == reply_to
    assert envelope.kind == "thought.event.v1"
    assert envelope.payload["disposition"] == "defer"
    assert envelope.payload["correlation_id"] == corr
    assert any("stance_react_failed" in r for r in envelope.payload["disposition_reasons"])
