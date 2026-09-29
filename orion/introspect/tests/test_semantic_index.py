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
    content_hash,
    embed,
    index_docs,
    nearest,
    publish_upsert,
    stored_hashes,
)
from orion.schemas.vector.schemas import VectorUpsertV1

CFG = SearchConfig(
    chroma_url="http://chroma.test", embed_url="http://embed.test/embedding",
    collection="orion_things", min_similarity=0.5,
)


def _client(*, missing=False, query=(), stored=None, get_status=200, count=0):
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
        if request.url.path == "/api/v1/collections/cid/count":
            return httpx.Response(200, json=count)
        if request.url.path == "/api/v1/collections/cid/get":
            if get_status != 200:
                return httpx.Response(get_status, json={"error": "boom"})
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


def _queries(seen):
    return [json.loads(r.content) for r in seen if r.url.path.endswith("/query")]


def test_nearest_without_where_sends_no_filter_and_empty_is_unknown():
    client, seen = _client(query=[("a", 0.4)])
    _run(lambda c: nearest(c, CFG, [1.0, 0.0], 20), client)
    assert "where" not in _queries(seen)[0]
    client, seen = _client(query=(), count=5)
    with pytest.raises(SearchUnavailableError, match="empty"):
        _run(lambda c: nearest(c, CFG, [1.0, 0.0], 20), client)
    assert not any(r.url.path.endswith("/count") for r in seen)


def test_nearest_sends_where_to_chroma():
    where = {"$and": [{"kind": "narrative"}, {"occurred_ts": {"$gte": 1.5}}]}
    client, seen = _client(query=[("a", 0.4)])
    assert _run(lambda c: nearest(c, CFG, [1.0, 0.0], 20, where=where), client) == [("a", 0.6)]
    assert _queries(seen)[0]["where"] == where


def test_filtered_zero_hits_in_non_empty_collection_is_no_match():
    client, seen = _client(query=(), count=7)
    assert _run(lambda c: nearest(c, CFG, [1.0, 0.0], 20, where={"kind": "narrative"}), client) == []
    assert [r.method for r in seen if r.url.path.endswith("/count")] == ["GET"]


def test_filtered_zero_hits_in_empty_or_unreadable_collection_is_unknown():
    client, _ = _client(query=(), count=0)
    with pytest.raises(SearchUnavailableError, match="orion_things is empty"):
        _run(lambda c: nearest(c, CFG, [1.0, 0.0], 20, where={"kind": "narrative"}), client)
    client, _ = _client(query=(), count="nope")
    with pytest.raises(SearchUnavailableError, match="count"):
        _run(lambda c: nearest(c, CFG, [1.0, 0.0], 20, where={"kind": "narrative"}), client)
    client, _ = _client(missing=True)
    with pytest.raises(SearchUnavailableError, match="not built"):
        _run(lambda c: nearest(c, CFG, [1.0, 0.0], 20, where={"kind": "narrative"}), client)


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


class _Bus:
    def __init__(self):
        self.published = []

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))


def _index(docs, client, batch=None):
    bus = _Bus()
    result = _run(lambda c: index_docs(
        docs, CFG, client=c, bus=bus, source=ServiceRef(name="orion-dream"),
        doc_prefix="dream-search", batch=batch,
    ), client)
    return result, bus


DOCS = [("a", "alpha", {"kind": "narrative"}), ("b", "beta", {"kind": "hypothesis"}), ("c", "gamma", {})]


def test_index_docs_publishes_only_new_and_changed_with_content_hash():
    client, seen = _client(stored={"a": content_hash("alpha"), "b": "stale"})
    result, bus = _index(DOCS, client)
    assert (result.indexed, result.pending) == (2, 0)
    payloads = [VectorUpsertV1.model_validate(e.payload) for _, e in bus.published]
    assert [p.doc_id for p in payloads] == ["b", "c"]
    assert payloads[0].meta == {"kind": "hypothesis", "content_hash": content_hash("beta")}
    assert payloads[1].meta == {"content_hash": content_hash("gamma")} and payloads[1].text == "gamma"
    embeds = [json.loads(r.content) for r in seen if r.url.host == "embed.test"]
    assert all(e["doc_id"].startswith("dream-search-") for e in embeds) and len(embeds) == 2


def test_index_docs_hash_keys_fold_meta_into_the_hash():
    docs = [("a", "alpha", {"kind": "narrative", "occurred_ts": 2.0})]
    client, _ = _client(stored={"a": content_hash("alpha")})
    bus = _Bus()
    result = _run(lambda c: index_docs(
        docs, CFG, client=c, bus=bus, source=ServiceRef(name="orion-dream"),
        doc_prefix="x", hash_keys=("occurred_ts",),
    ), client)
    assert (result.indexed, result.pending) == (1, 0)
    [(_, envelope)] = bus.published
    folded = envelope.payload["meta"]["content_hash"]
    assert folded != content_hash("alpha")
    client, _ = _client(stored={"a": folded})
    again = _run(lambda c: index_docs(
        docs, CFG, client=c, bus=_Bus(), source=ServiceRef(name="orion-dream"),
        doc_prefix="x", hash_keys=("occurred_ts",),
    ), client)
    assert (again.indexed, again.pending) == (0, 0)


def test_index_docs_batch_leaves_the_rest_pending():
    client, _ = _client(missing=True)
    result, bus = _index(DOCS, client, batch=1)
    assert (result.indexed, result.pending) == (1, 2) and len(bus.published) == 1
    client, _ = _client(stored={})
    result, _ = _index(DOCS, client)
    assert (result.indexed, result.pending) == (3, 0)


def test_index_docs_empty_makes_no_http_call():
    client, seen = _client()
    result, bus = _index([], client)
    assert (result.indexed, result.pending) == (0, 0) and seen == [] and bus.published == []


def test_index_docs_stored_hashes_failure_raises_and_publishes_nothing():
    client, seen = _client(get_status=500)
    bus = _Bus()
    with pytest.raises(SearchUnavailableError):
        _run(lambda c: index_docs(
            DOCS, CFG, client=c, bus=bus, source=ServiceRef(name="orion-dream"), doc_prefix="x",
        ), client)
    assert bus.published == [] and not any(r.url.host == "embed.test" for r in seen)
