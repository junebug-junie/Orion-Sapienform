#!/usr/bin/env python3
"""
Smoke test: publish memory.vector.upsert.v1 messages (with and without
embeddings) to the vector-writer intake channel.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vector.schemas import VectorDocumentUpsertV1


BUS_URL = os.getenv("ORION_BUS_URL", "redis://localhost:6379/0")
UPSERT_CHANNEL = os.getenv("VECTOR_UPSERT_CHANNEL", "orion:memory:vector:upsert")
ERROR_CHANNEL = os.getenv("ORION_ERROR_CHANNEL", "system.error")
TIMEOUT_SEC = float(os.getenv("VECTOR_SMOKE_TIMEOUT_SEC", "15"))


def _build_envelope(doc_id: str, *, with_embedding: bool) -> BaseEnvelope:
    payload = VectorDocumentUpsertV1(
        doc_id=doc_id,
        kind="smoke.test",
        text="vector smoke payload",
        metadata={"source": "smoke-vector-upsert"},
        collection="smoke",
        embedding=[0.1, 0.2, 0.3] if with_embedding else None,
        embedding_dim=3 if with_embedding else None,
        embedding_model="smoke-embedder" if with_embedding else None,
    )
    return BaseEnvelope(
        kind="memory.vector.upsert.v1",
        source=ServiceRef(name="smoke-vector-writer", node="local", version="0.0.0"),
        correlation_id=uuid4(),
        payload=payload.model_dump(mode="json"),
    )


async def _collect_errors(bus: OrionBusAsync, window_sec: float) -> list[str]:
    errors: list[str] = []
    async with bus.subscribe(ERROR_CHANNEL) as pubsub:
        start = time.time()
        while time.time() - start < window_sec:
            try:
                msg = await asyncio.wait_for(bus.iter_messages(pubsub).__anext__(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            decoded = bus.codec.decode(msg.get("data"))
            if decoded.ok and decoded.envelope:
                errors.append(decoded.envelope.kind)
    return errors


async def main() -> int:
    bus = OrionBusAsync(BUS_URL)
    await bus.connect()

    with_embedding = _build_envelope("smoke-with-embed", with_embedding=True)
    without_embedding = _build_envelope("smoke-no-embed", with_embedding=False)

    async with bus.subscribe(UPSERT_CHANNEL) as _:
        await bus.publish(UPSERT_CHANNEL, with_embedding)
        await bus.publish(UPSERT_CHANNEL, without_embedding)

    errors = await _collect_errors(bus, window_sec=TIMEOUT_SEC)
    await bus.close()

    print("[ok] published vector upserts (with + without embedding)")
    if errors:
        print(f"[warn] observed error kinds on bus: {errors}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130)
