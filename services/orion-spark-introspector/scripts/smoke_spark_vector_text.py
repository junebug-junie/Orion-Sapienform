#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
import sys
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.schemas.vector.schemas import EmbeddingGenerateV1


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    raise SystemExit(1)


async def _request_embedding(bus: OrionBusAsync, *, doc_id: str, text: str, collection: str) -> None:
    reply_channel = f"orion:embedding:result:smoke:{doc_id}:{uuid4()}"
    env = BaseEnvelope(
        kind="embedding.generate.v1",
        source=ServiceRef(name="spark-introspector", version="smoke", node="smoke"),
        correlation_id=uuid4(),
        reply_to=reply_channel,
        payload=EmbeddingGenerateV1(
            doc_id=doc_id,
            text=text,
            collection=collection,
            include_latent=False,
        ).model_dump(mode="json"),
    )
    msg = await bus.rpc_request(
        os.getenv("CHANNEL_EMBEDDING_GENERATE", "orion:embedding:generate"),
        env,
        reply_channel=reply_channel,
        timeout_sec=float(os.getenv("VALENCE_ANCHOR_TIMEOUT_SEC", "15")),
    )
    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok or decoded.envelope is None:
        _fail(f"embedding rpc decode failed for {doc_id}")
    payload = decoded.envelope.payload
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    if not isinstance(payload, dict):
        _fail(f"embedding rpc payload invalid for {doc_id}: {type(payload)}")
    if payload.get("error"):
        _fail(f"embedding rpc returned error for {doc_id}: {payload['error']}")


async def main() -> None:
    try:
        import chromadb
    except Exception as exc:  # pragma: no cover
        _fail(f"chromadb import failed (install dependency first): {exc}")

    bus_url = os.getenv("ORION_BUS_URL", "redis://localhost:6379/0")
    target_collection = os.getenv("SPARK_VECTOR_COLLECTION", "orion_spark_store")
    chroma_host = os.getenv("CHROMA_HOST", "localhost")
    chroma_port = int(os.getenv("CHROMA_PORT", "8000"))

    anchors = {
        "valence-anchor-pos": os.getenv("VALENCE_ANCHOR_POS_TEXT", "I feel hopeful and grateful."),
        "valence-anchor-neg": os.getenv("VALENCE_ANCHOR_NEG_TEXT", "I feel hopeless and afraid."),
    }

    bus = OrionBusAsync(bus_url, enabled=True, codec=OrionCodec())
    await bus.connect()
    try:
        for doc_id, text in anchors.items():
            await _request_embedding(bus, doc_id=doc_id, text=text, collection=target_collection)
    finally:
        await bus.close()

    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    coll = client.get_or_create_collection(name=target_collection)

    for doc_id in anchors:
        found = None
        for _ in range(10):
            res = coll.get(ids=[doc_id], include=["documents", "metadatas", "embeddings"])
            ids = res.get("ids") or []
            if ids:
                found = res
                break
            await asyncio.sleep(0.5)
        if not found:
            _fail(f"id not found in collection={target_collection}: {doc_id}")

        docs = found.get("documents") or []
        metas = found.get("metadatas") or []
        embeds = found.get("embeddings") or []

        doc_val = docs[0] if docs else None
        if not doc_val or not str(doc_val).strip():
            _fail(f"empty documents field for id={doc_id} in collection={target_collection}")
        if not embeds or not embeds[0]:
            _fail(f"missing embedding for id={doc_id}")

        meta = metas[0] if metas else {}
        if (meta or {}).get("requester_service") != "spark-introspector":
            _fail(f"requester_service mismatch for id={doc_id}: {meta}")

        print(f"PASS id={doc_id} collection={target_collection} doc_preview={str(doc_val)[:120]!r} meta={meta}")

    print("PASS: spark vector smoke test complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
