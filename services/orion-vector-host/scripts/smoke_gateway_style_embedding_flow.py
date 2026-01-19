#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
import sys
import uuid

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vector.schemas import EmbeddingGenerateV1, VectorUpsertV1

CHANNEL_EMBEDDING_GENERATE = "orion:embedding:generate"
CHANNEL_SEMANTIC_UPSERT = "orion:vector:semantic:upsert"
BUS_URL = os.getenv("ORION_BUS_URL", "redis://orion-redis:6379/0")
TIMEOUT_SECONDS = 10.0
DOC_ID = "pf-smoke-1"


def _source() -> ServiceRef:
    return ServiceRef(name="vector-host-smoke", version="0.0.0", node="local")


async def _wait_for_semantic_upsert(bus: OrionBusAsync) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + TIMEOUT_SECONDS

    async with bus.subscribe(CHANNEL_SEMANTIC_UPSERT) as pubsub:
        messages = bus.iter_messages(pubsub)
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            try:
                msg = await asyncio.wait_for(messages.__anext__(), timeout=remaining)
            except asyncio.TimeoutError:
                return False

            decoded = bus.codec.decode(msg.get("data"))
            if not decoded:
                continue
            try:
                env = BaseEnvelope.model_validate(decoded)
            except Exception:
                continue

            payload_obj = env.payload.model_dump(mode="json") if hasattr(env.payload, "model_dump") else env.payload
            if not isinstance(payload_obj, dict):
                continue

            try:
                upsert = VectorUpsertV1.model_validate(payload_obj)
            except Exception:
                continue

            if upsert.doc_id == DOC_ID and upsert.embedding_kind == "semantic" and upsert.embedding:
                return True


async def main() -> int:
    bus = OrionBusAsync(url=BUS_URL)
    try:
        await bus.connect()
    except Exception as exc:
        print(f"Redis bus unreachable at {BUS_URL}: {exc}")
        return 2

    envelope = BaseEnvelope(
        kind="embedding.generate.v1",
        source=_source(),
        correlation_id=uuid.uuid4(),
        payload=EmbeddingGenerateV1(
            doc_id=DOC_ID,
            text="hello world",
            embedding_profile="default",
            include_latent=False,
        ).model_dump(mode="json"),
    )

    try:
        await bus.publish(CHANNEL_EMBEDDING_GENERATE, envelope)
        ok = await _wait_for_semantic_upsert(bus)
    finally:
        await bus.close()

    if ok:
        print("OK: semantic upsert received")
        return 0

    print("FAIL: semantic upsert not observed within timeout")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
