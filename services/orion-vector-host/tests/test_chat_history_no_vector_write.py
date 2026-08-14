from __future__ import annotations

import asyncio
from typing import Any, List
from unittest.mock import AsyncMock

import pytest

from app import main as vector_host_main
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.chat_history import (
    CHAT_HISTORY_MESSAGE_KIND,
    CHAT_HISTORY_TURN_KIND,
)


class _FakeEmbedder:
    """Deterministic stand-in for app.embedder.Embedder -- returns a fixed
    small vector so tests don't need a real embedding backend."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    async def embed(self, text: str):
        self.calls.append(text)
        return [0.1, 0.2, 0.3, 0.4], "fake-model", 4


def _source() -> ServiceRef:
    return ServiceRef(name="orion-hub", node="athena", version="0.1.0")


def _chat_message_env(content: str = "hello world", role: str = "user") -> BaseEnvelope:
    return BaseEnvelope(
        kind=CHAT_HISTORY_MESSAGE_KIND,
        source=_source(),
        payload={
            "message_id": "msg-1",
            "session_id": "sess-1",
            "role": role,
            "speaker": "june",
            "content": content,
        },
    )


def _chat_turn_env(prompt: str = "hi", response: str = "hello") -> BaseEnvelope:
    return BaseEnvelope(
        kind=CHAT_HISTORY_TURN_KIND,
        source=_source(),
        payload={
            "id": "turn-1",
            "source": "hub_ws",
            "prompt": prompt,
            "response": response,
            "session_id": "sess-1",
        },
    )


@pytest.fixture(autouse=True)
def _fake_embedder(monkeypatch):
    embedder = _FakeEmbedder()
    monkeypatch.setattr(vector_host_main, "embedder", embedder)
    # hunter stays None -- these handlers must not need the bus at all
    # post-kill, so leaving it unset also proves no publish call is attempted.
    monkeypatch.setattr(vector_host_main, "hunter", None)
    yield embedder


def test_handle_chat_history_does_not_publish_semantic_upsert(monkeypatch):
    """Core kill assertion: _handle_chat_history must never call
    _publish_semantic_upsert (the function that publishes vector.upsert.v1,
    which orion-vector-writer turns into a Chroma write for orion_chat)."""
    publish_mock = AsyncMock()
    monkeypatch.setattr(vector_host_main, "_publish_semantic_upsert", publish_mock)

    tissue_mock = AsyncMock(return_value={"polarity_diff": 0.0})
    monkeypatch.setattr(vector_host_main, "feed_tissue", tissue_mock)

    asyncio.run(vector_host_main._handle_chat_history(_chat_message_env()))

    publish_mock.assert_not_called()
    tissue_mock.assert_called_once()


def test_handle_chat_turn_does_not_publish_semantic_upsert(monkeypatch):
    publish_mock = AsyncMock()
    monkeypatch.setattr(vector_host_main, "_publish_semantic_upsert", publish_mock)

    tissue_mock = AsyncMock(return_value={"polarity_diff": 0.0})
    monkeypatch.setattr(vector_host_main, "feed_tissue", tissue_mock)

    asyncio.run(vector_host_main._handle_chat_turn(_chat_turn_env()))

    publish_mock.assert_not_called()
    tissue_mock.assert_called_once()


def test_handle_chat_history_still_embeds_and_feeds_tissue(monkeypatch, _fake_embedder):
    """Confirms the still-live half of this kill: the embedding is computed
    and handed to the real feed_tissue() (not a stub) so OrionTissue keeps
    getting a real stimulus from chat traffic."""
    calls: List[Any] = []
    real_feed_tissue = vector_host_main.feed_tissue

    async def _spy_feed_tissue(embedding, **kwargs):
        calls.append((embedding, kwargs))
        return await real_feed_tissue(embedding, broadcast=False, **kwargs)

    monkeypatch.setattr(vector_host_main, "feed_tissue", _spy_feed_tissue)

    asyncio.run(vector_host_main._handle_chat_history(_chat_message_env(content="a real message")))

    assert _fake_embedder.calls == ["a real message"]
    assert len(calls) == 1
    embedding_arg, kwargs = calls[0]
    assert embedding_arg == [0.1, 0.2, 0.3, 0.4]
    assert kwargs["doc_id"] == "msg-1"


def test_handle_chat_history_skips_non_embed_role(monkeypatch, _fake_embedder):
    tissue_mock = AsyncMock()
    monkeypatch.setattr(vector_host_main, "feed_tissue", tissue_mock)

    env = _chat_message_env(role="system")
    asyncio.run(vector_host_main._handle_chat_history(env))

    tissue_mock.assert_not_called()
    assert _fake_embedder.calls == []


def test_handle_chat_turn_skips_blank_turn(monkeypatch, _fake_embedder):
    tissue_mock = AsyncMock()
    monkeypatch.setattr(vector_host_main, "feed_tissue", tissue_mock)

    env = _chat_turn_env(prompt="", response="")
    asyncio.run(vector_host_main._handle_chat_turn(env))

    tissue_mock.assert_not_called()
    assert _fake_embedder.calls == []
