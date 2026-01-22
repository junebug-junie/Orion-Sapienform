#!/usr/bin/env python3
"""
Smoke test: publish chat.history.message.v1 via the bus and verify retrieval.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

import requests

from chromadb import HttpClient  # type: ignore
from chromadb.config import Settings as ChromaSettings  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.chat_history import ChatHistoryMessageV1

BUS_URL = os.getenv("ORION_BUS_URL", "redis://localhost:6379/0")
CHAT_HISTORY_CHANNEL = os.getenv("CHAT_HISTORY_CHANNEL", "orion:chat:history:log")
ERROR_CHANNEL = os.getenv("ORION_ERROR_CHANNEL", "system.error")
TIMEOUT_SEC = float(os.getenv("VECTOR_SMOKE_TIMEOUT_SEC", "15"))

RECALL_URL = os.getenv("RECALL_URL", "http://localhost:8260/recall")
VECTOR_DB_HOST = os.getenv("VECTOR_DB_HOST", "orion-athena-vector-db")
VECTOR_DB_PORT = int(os.getenv("VECTOR_DB_PORT", "8000"))
VECTOR_DB_COLLECTION = os.getenv("VECTOR_DB_COLLECTION", "orion_chat")


def _build_envelope(doc_id: str, *, session_id: str) -> BaseEnvelope:
    payload = ChatHistoryMessageV1(
        message_id=doc_id,
        session_id=session_id,
        role="assistant",
        speaker="smoke-tester",
        content="Synthetic chat turn about the cpu card for recall testing.",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        tags=["brain", "chat_general"],
    )
    return BaseEnvelope(
        kind="chat.history.message.v1",
        source=ServiceRef(name="vector-smoke-test", node="local", version="0.0.0"),
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


def _query_recall(session_id: str) -> None:
    payload = {"query_text": "cpu card", "session_id": session_id, "diagnostic": True}
    try:
        resp = requests.post(RECALL_URL, json=payload, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[warn] recall query failed: {exc}")
        return
    data = resp.json()
    counts = data.get("debug", {}).get("backend_counts")
    print(f"[ok] recall debug backend_counts={counts}")


def _query_chroma(session_id: str) -> None:
    client = HttpClient(
        host=VECTOR_DB_HOST,
        port=VECTOR_DB_PORT,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    col = client.get_or_create_collection(name=VECTOR_DB_COLLECTION)
    res = col.get(where={"session_id": session_id}, include=["metadatas", "documents"], limit=5)
    metas = (res.get("metadatas") or [])
    print(f"[ok] chroma hits for session_id={session_id}: {len(res.get('ids', []))}")
    if metas:
        meta = metas[0] or {}
        missing = [key for key in ("role", "speaker", "created_at", "original_channel") if key not in meta]
        if missing:
            print(f"[warn] missing metadata keys: {missing}")
        else:
            print("[ok] metadata includes role/speaker/created_at/original_channel")


async def main() -> int:
    bus = OrionBusAsync(BUS_URL)
    await bus.connect()

    session_id = f"smoke-session-{uuid4()}"
    doc_id = f"smoke-doc-{uuid4()}"

    envelope = _build_envelope(doc_id, session_id=session_id)
    async with bus.subscribe(CHAT_HISTORY_CHANNEL) as _:
        await bus.publish(CHAT_HISTORY_CHANNEL, envelope)

    errors = await _collect_errors(bus, window_sec=TIMEOUT_SEC)
    await bus.close()

    print("[ok] published chat.history.message.v1")
    if errors:
        print(f"[warn] observed error kinds on bus: {errors}")

    try:
        _query_chroma(session_id)
    except Exception as exc:
        print(f"[warn] chroma query failed: {exc}")

    _query_recall(session_id)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130)
