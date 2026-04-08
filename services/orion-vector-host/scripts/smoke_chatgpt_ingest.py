#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chromadb
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef


async def publish_chatgpt_samples(bus_url: str, message_channel: str, turn_channel: str, msg_id: str, turn_id: str) -> None:
    bus = OrionBusAsync(bus_url)
    await bus.connect()
    try:
        source = ServiceRef(name="chatgpt-import", version="0.1.0", node="smoke")
        message_env = BaseEnvelope(
            kind="chat.gpt.message.v1",
            source=source,
            correlation_id=uuid.uuid4(),
            payload={
                "message_id": msg_id,
                "session_id": "chatgpt:smoke",
                "role": "assistant",
                "speaker": "ChatGPT",
                "content": f"smoke gpt message {msg_id}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "tags": ["source:chatgpt_export", "smoke:test"],
            },
        )
        await bus.publish(message_channel, message_env)

        turn_env = BaseEnvelope(
            kind="chat.gpt.log.v1",
            source=source,
            correlation_id=uuid.UUID(turn_id),
            payload={
                "id": turn_id,
                "correlation_id": turn_id,
                "source": "chatgpt_import",
                "prompt": "smoke prompt",
                "response": "smoke response",
                "session_id": "chatgpt:smoke",
                "spark_meta": {
                    "source_conversation_id": "smoke-conv",
                    "conversation_title": "Smoke GPT",
                },
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        await bus.publish(turn_channel, turn_env)
    finally:
        await bus.close()


def has_doc(client: chromadb.HttpClient, collection: str, doc_id: str) -> bool:
    col = client.get_or_create_collection(name=collection)
    result = col.get(ids=[doc_id], include=["metadatas", "documents"])
    docs = result.get("documents") or []
    return bool(docs and docs[0] and str(docs[0]).strip())


async def main() -> int:
    bus_url = os.getenv("ORION_BUS_URL", "redis://localhost:6379/0")
    chroma_host = os.getenv("VECTOR_DB_HOST", "localhost")
    chroma_port = int(os.getenv("VECTOR_DB_PORT", "8000"))
    message_channel = os.getenv("VECTOR_HOST_GPT_MESSAGE_CHANNEL", "orion:chat:gpt:log")
    turn_channel = os.getenv("VECTOR_HOST_GPT_TURN_CHANNEL", "orion:chat:gpt:turn")
    message_collection = os.getenv("VECTOR_HOST_GPT_MESSAGE_COLLECTION", "orion_chat_gpt")
    turn_collection = os.getenv("VECTOR_HOST_GPT_TURN_COLLECTION", "orion_chat_gpt_turns")

    msg_id = f"smoke-gpt-msg-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    turn_uuid = uuid.uuid4()
    turn_id = str(turn_uuid)

    await publish_chatgpt_samples(bus_url, message_channel, turn_channel, msg_id, turn_id)
    await asyncio.sleep(3)

    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

    message_ok = has_doc(client, message_collection, msg_id)
    turn_ok = has_doc(client, turn_collection, turn_id)

    print({
        "message_collection": message_collection,
        "message_doc_id": msg_id,
        "message_found": message_ok,
        "turn_collection": turn_collection,
        "turn_doc_id": turn_id,
        "turn_found": turn_ok,
    })

    return 0 if (message_ok and turn_ok) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
