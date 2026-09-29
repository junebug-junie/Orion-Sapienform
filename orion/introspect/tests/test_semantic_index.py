"""Shared search plumbing: embed, nearest, stored hashes, upsert publish."""
import asyncio
import json

import httpx
import pytest

from orion.core.bus.bus_schemas import ServiceRef
from orion.introspect.semantic_index import (
    UPSERT_CHANNEL,
    UPSERT_KIND,
    SearchConfig,
    SearchUnavailableError,
    embed,
    nearest,
    publish_upsert,
    stored_hashes,
)
from orion.schemas.vector.schemas import VectorUpsertV1

CFG = SearchConfig(
    chroma_url="http://chroma.test", embed_url="http://embed.test/embedding",
    collection="orion_things", min_similarity=0.5,
)


def _client(*, missing=False, query=(), stored=None):
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.host == "embed.test":
            body = json.loads(request.content)
            return httpx.Response(200, json={"doc_id": body["doc_id"], "embedding": [1.0, 0.0], "embedding_model": "bge"})
        if request.url.path == "/api/v1/collections/orion_things":
            if missing:
                return httpx.Response(500, json={"error": "ValueError('Collection orion_things does not exist.')"})
            return httpx.Response(200, json={"id": "cid", "metadata": {"hnsw:space": "cosine"}})
        if request.url.path == "/api/v1/collections/cid/query":
            ids, dists = zip(*query) if query else ((), ())
            return httpx.Response(200, json={"ids": [list(ids)], "distances": [list(dists)]})
        if request.url.path == "/api/v1/collections/cid/get":
            wanted = json.loads(request.content)["ids"]
            have = [i for i in wanted if i in (stored or {})]
            return httpx.Response(200, json={"ids": have, "metadatas": [{"content_hash": stored[i]} for i in have]})
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), seen


def _run(coro_fn, client):
    async def go():
        async with client:
            return await coro_fn(client)
    return asyncio.run(go())


def test_embed_uses_the_given_doc_prefix():
    client, seen = _client()
    vector, model = _run(lambda c: embed(c, CFG, "hello", doc_prefix="dream-search"), client)
    assert vector == [1.0, 0.0] and model == "bge"
    assert json.loads(seen[0].content)["doc_id"].startswith("dream-search-")


def test_nearest_sorts_by_similarity_and_names_the_collection_when_unbuilt():
    client, _ = _client(query=[("a", 0.4), ("b", 0.1)])
    assert _run(lambda c: nearest(c, CFG, [1.0, 0.0], 20), client) == [("b", 0.9), ("a", 0.6)]
    client, _ = _client(missing=True)
    with pytest.raises(SearchUnavailableError, match="orion_things"):
        _run(lambda c: nearest(c, CFG, [1.0, 0.0], 20), client)


def test_stored_hashes_of_missing_collection_is_empty():
    client, _ = _client(missing=True)
    assert _run(lambda c: stored_hashes(c, CFG, ["a"]), client) == {}
    client, _ = _client(stored={"a": "h1"})
    assert _run(lambda c: stored_hashes(c, CFG, ["a", "b"]), client) == {"a": "h1"}


def test_publish_upsert_sends_a_semantic_vector_upsert():
    published = []

    class Bus:
        async def publish(self, channel, envelope):
            published.append((channel, envelope))

    asyncio.run(publish_upsert(
        Bus(), ServiceRef(name="orion-dream"), CFG,
        doc_id="dream:1", text="t", vector=[1.0, 0.0], model="bge", meta={"kind": "narrative"},
    ))
    [(channel, envelope)] = published
    assert channel == UPSERT_CHANNEL and envelope.kind == UPSERT_KIND
    payload = VectorUpsertV1.model_validate(envelope.payload)
    assert payload.collection == "orion_things" and payload.embedding_dim == 2
    assert payload.embedding_kind == "semantic" and payload.meta == {"kind": "narrative"}
