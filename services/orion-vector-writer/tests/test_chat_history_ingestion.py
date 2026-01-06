import sys
from pathlib import Path
from uuid import uuid4

# Ensure service-local imports resolve
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.chat_history import (
    CHAT_HISTORY_COLLECTION,
    chat_history_envelope_to_request,
)
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.chat_history import (
    CHAT_HISTORY_MESSAGE_KIND,
    ChatHistoryMessageEnvelope,
    ChatHistoryMessageV1,
)


def test_chat_history_payload_to_document_includes_lineage():
    payload = ChatHistoryMessageV1(
        message_id="msg-123",
        session_id="sess-42",
        role="assistant",
        speaker="orion",
        content="Hello, world!",
        tags=["brain"],
        model="gpt-4",
        provider="openai",
    )
    corr_id = uuid4()
    envelope = ChatHistoryMessageEnvelope(
        correlation_id=corr_id,
        source=ServiceRef(name="hub", node="athena", version="0.4.0"),
        payload=payload,
    )

    doc = payload.to_document(
        kind=CHAT_HISTORY_MESSAGE_KIND,
        source=envelope.source,
        correlation_id=envelope.correlation_id,
        channel="orion:chat:history:log",
    )

    assert doc["id"] == payload.message_id
    assert doc["text"] == payload.content
    metadata = doc["metadata"]
    assert metadata["role"] == "assistant"
    assert metadata["session_id"] == "sess-42"
    assert metadata["source_channel"] == "orion:chat:history:log"
    assert metadata["correlation_id"] == str(corr_id)
    assert metadata["kind"] == CHAT_HISTORY_MESSAGE_KIND
    assert metadata["model"] == "gpt-4"
    assert "tags" in metadata and metadata["tags"] == ["brain"]


def test_chat_history_envelope_normalizes_to_vector_request():
    payload = ChatHistoryMessageV1(
        message_id="msg-abc",
        session_id="sess-1",
        role="user",
        speaker="test-user",
        content="Hi there",
    )
    envelope = ChatHistoryMessageEnvelope(
        correlation_id=uuid4(),
        source=ServiceRef(name="hub", node="athena"),
        payload=payload,
    )

    req = chat_history_envelope_to_request(
        envelope,
        channel="orion:chat:history:log",
        collection_name=CHAT_HISTORY_COLLECTION,
    )

    assert req is not None
    assert req.id == payload.message_id
    assert req.kind == CHAT_HISTORY_MESSAGE_KIND
    assert req.collection_name == CHAT_HISTORY_COLLECTION
    assert req.metadata["role"] == "user"
    assert req.metadata["speaker"] == "test-user"
    assert req.metadata["source_channel"] == "orion:chat:history:log"
