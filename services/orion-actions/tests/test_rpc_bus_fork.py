from __future__ import annotations

from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock

import pytest

from app import main as actions_main


@pytest.mark.asyncio
async def test_rpc_request_with_retry_uses_supplied_bus_not_listener() -> None:
    rpc_bus = AsyncMock()
    rpc_bus.rpc_request = AsyncMock(return_value={"data": b"ok"})
    rpc_bus.codec = MagicMock()

    def _factory(reply_channel: str, attempt: int):
        from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

        return BaseEnvelope(
            kind="test.request",
            source=ServiceRef(name="test", version="1"),
            correlation_id=str(uuid4()),
            reply_to=reply_channel,
            payload={"attempt": attempt},
        )

    msg = await actions_main._rpc_request_with_retry(
        bus=rpc_bus,
        request_channel="orion:test:request",
        reply_prefix="orion:test:result:",
        timeout_sec=1.0,
        envelope_factory=_factory,
        operation_name="test op",
        max_attempts=1,
    )
    assert msg == {"data": b"ok"}
    rpc_bus.rpc_request.assert_awaited_once()
    args, kwargs = rpc_bus.rpc_request.await_args
    assert args[0] == "orion:test:request"
    assert kwargs["reply_channel"].startswith("orion:test:result:")
