"""Dream search: index text per kind, hash-aware index pass, floor-gated ranking."""
import asyncio
import json
from datetime import date, datetime, timezone

import httpx
import pytest

from app import dream_search as ds
from orion.core.bus.bus_schemas import ServiceRef
from orion.introspect.semantic_index import SearchConfig, SearchUnavailableError, content_hash
from orion.schemas.vector.schemas import VectorUpsertV1

NOW = datetime(2026, 9, 29, 6, 0, tzinfo=timezone.utc)
CFG = SearchConfig(
    chroma_url="http://chroma.test", embed_url="http://embed.test/embedding",
    collection="orion_dreams", min_similarity=0.6,
)
SOURCE = ServiceRef(name="orion-dream")
NARR = {"id": 19, "dream_date": date(2026, 9, 28), "tldr": "Infrastructure dream.", "narrative": "Cables hum.",
        "themes": ["infrastructure", "connection"], "occurred_at": NOW}
HYP = {"hypothesis_id": "dh-aaa111", "cycle_id": "dc-1", "claim": "RPC timeouts link.", "why": "Both fail.",
       "occurred_at": NOW, "expires_at": NOW}


def _client(*, query=(), stored=None, embed_status=200):
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.host == "embed.test":
            if embed_status != 200:
                return httpx.Response(embed_status, json={})
            body = json.loads(request.content)
            return httpx.Response(200, json={"doc_id": body["doc_id"], "embedding": [1.0, 0.0], "embedding_model": "bge"})
        if request.url.path == "/api/v1/collections/orion_dreams":
            return httpx.Response(200, json={"id": "cid", "metadata": None})
        if request.url.path == "/api/v1/collections/cid/query":
            ids, dists = zip(*query) if query else ((), ())
            return httpx.Response(200, json={"ids": [list(ids)], "distances": [list(dists)]})
        if request.url.path == "/api/v1/collections/cid/get":
            wanted = json.loads(request.content)["ids"]
            have = [i for i in wanted if i in (stored or {})]
            return httpx.Response(200, json={"ids": have, "metadatas": [{"content_hash": stored[i]} for i in have]})
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), seen


class Bus:
    def __init__(self):
        self.published = []

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))


def _index(pairs, client, batch=None):
    bus = Bus()

    async def run():
        async with client:
            return await ds.index_missing(pairs, CFG, client=client, bus=bus, source=SOURCE, batch=batch)
    return asyncio.run(run()), bus


def _rank(client, query="vision"):
    async def run():
        async with client:
            return await ds.rank(client, CFG, query)
    return asyncio.run(run())


def test_document_text_per_kind():
    assert ds.document_text("narrative", NARR) == (
        "Infrastructure dream.\n\nThemes: infrastructure, connection\n\nCables hum."
    )
    assert ds.document_text("hypothesis", HYP) == "RPC timeouts link.\nWhy: Both fail."
    assert ds.document_text("hypothesis", {**HYP, "claim": " "}) is None
    assert len(ds.document_text("narrative", {**NARR, "narrative": "n" * 5000})) == ds.INDEX_TEXT_CHARS
    assert ds.document_text("narrative", {**NARR, "themes": ["", " "]}) == "Infrastructure dream.\n\nCables hum."
    assert ds.document_text("narrative", {**NARR, "tldr": None, "narrative": "", "themes": None}) is None


def test_index_upserts_new_and_changed_only_with_kind_meta():
    fresh_hash = content_hash(ds.document_text("narrative", NARR))
    client, _ = _client(stored={"dream:19": fresh_hash, "dh-aaa111": "stale"})
    result, bus = _index([("narrative", NARR), ("hypothesis", HYP)], client)
    assert (result.indexed, result.pending) == (1, 0)
    [(_, envelope)] = bus.published
    payload = VectorUpsertV1.model_validate(envelope.payload)
    assert payload.doc_id == "dh-aaa111" and payload.collection == "orion_dreams"
    assert payload.meta["kind"] == "hypothesis" and payload.meta["content_hash"] == content_hash(payload.text)
    assert payload.meta["occurred_at"] == NOW.isoformat()


def test_index_never_publishes_blind_experiment_fields():
    leaky = {**HYP, "arm": "treatment", "ref_a": "secret-a", "ref_b": "secret-b"}
    client, _ = _client(stored={})
    _, bus = _index([("hypothesis", leaky)], client)
    [(_, envelope)] = bus.published
    blob = json.dumps(envelope.payload)
    assert not {"arm", "ref_a", "ref_b"} & set(envelope.payload["meta"])
    assert "treatment" not in blob and "secret-a" not in blob and "secret-b" not in blob


def test_index_honors_batch_and_embed_failure_is_unknown():
    client, _ = _client(stored={})
    result, bus = _index([("narrative", NARR), ("hypothesis", HYP)], client, batch=1)
    assert (result.indexed, result.pending) == (1, 1) and len(bus.published) == 1
    client, _ = _client(stored={}, embed_status=503)
    with pytest.raises(SearchUnavailableError):
        _index([("narrative", NARR)], client)


def test_rank_keeps_hits_at_or_above_floor_best_first():
    client, seen = _client(query=[("dream:19", 0.5), ("dh-aaa111", 0.9)])
    assert _rank(client) == [("dream:19", 0.75)]
    embeds = [r for r in seen if r.url.host == "embed.test"]
    assert len(embeds) == 1 and json.loads(embeds[0].content)["text"] == "vision"


def test_rank_orders_multiple_hits_best_first_and_keeps_exact_floor():
    # metadata None -> l2 space: similarity = 1 - d/2; floor 0.6 == d 0.8.
    client, _ = _client(query=[("dream:19", 0.8), ("dh-aaa111", 0.2), ("dream:20", 0.4), ("dh-bbb222", 1.0)])
    assert _rank(client) == [("dh-aaa111", 0.9), ("dream:20", 0.8), ("dream:19", 0.6)]


def test_rank_on_existing_but_empty_collection_is_unknown_not_empty():
    client, _ = _client(query=())
    with pytest.raises(SearchUnavailableError):
        _rank(client)
