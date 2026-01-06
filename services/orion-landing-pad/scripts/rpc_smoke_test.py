#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
from uuid import uuid4

from loguru import logger

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.pad import KIND_PAD_RPC_REQUEST_V1, PadRpcRequestV1


BUS_URL = os.getenv("ORION_BUS_URL", "redis://localhost:6379/0")
REQUEST_CHANNEL = os.getenv("PAD_RPC_REQUEST_CHANNEL", "orion:pad:rpc:request")
REPLY_PREFIX = os.getenv("PAD_RPC_REPLY_PREFIX", "orion:pad:rpc:reply:")


async def main() -> None:
    bus = OrionBusAsync(BUS_URL)
    await bus.connect()
    source = ServiceRef(name="pad-rpc-smoke", node="local", version="0.0.0")

    correlation_id = uuid4()
    reply_channel = f"{REPLY_PREFIX}{correlation_id}"

    req = PadRpcRequestV1(
        request_id=str(uuid4()),
        reply_channel=reply_channel,
        method="get_latest_frame",
        args={},
    )
    env = BaseEnvelope(
        kind=KIND_PAD_RPC_REQUEST_V1,
        source=source,
        correlation_id=correlation_id,
        payload=req.model_dump(mode="json"),
        reply_to=reply_channel,
    )

    logger.info(f"Publishing RPC request to {REQUEST_CHANNEL}, awaiting reply on {reply_channel}")
    msg = await bus.rpc_request(REQUEST_CHANNEL, env, reply_channel=reply_channel, timeout_sec=10.0)
    decoded = bus.codec.decode(msg.get("data"))
    logger.info(f"RPC reply ok={decoded.ok} envelope={decoded.envelope}")
    await bus.close()


if __name__ == "__main__":
    asyncio.run(main())
