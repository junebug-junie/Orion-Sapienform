"""Semantic reading search over a mocked Chroma REST API and embedder."""
import asyncio
import json
from datetime import datetime, timezone

import httpx
import pytest

from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.vector.schemas import VectorUpsertV1
from orion.world_pulse_read.search import (
    UPSERT_CHANNEL,
    ReadingSearchConfig,
    SearchUnavailableError,
    content_hash,
    document_text,
    index_missing_readings,
    search_readings,
    similarity,
)

NOW = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)
CFG = ReadingSearchConfig(
    chroma_url="http://chroma.test", embed_url="http://embed.test/embedding",
    collection="orion_reading_results", min_similarity=0.60,
)
SOURCE = ServiceRef(name="orion-hub")


def _row(seed_id, url, summary, *, read=True, title="T"):
    evidence = [{"tool_name": "WebFetch", "url": url, "content_chars": 500}] if read else []
    return {
        "seed_id": seed_id, "request_id": None, "url": url, "title": title,
        "status": "done", "stage2_status": "done", "created_at": NOW, "handoff_at": NOW,
        "stage2_completed_at": NOW, "landing_at": NOW,
        "handoff_json": {"what_i_learned": "stage one", "read_evidence": evidence},
        "stage2_result_json": {"summary": summary}, "trace_id": None, "stage2_trace_id": None,
        "why_now": None,
    }


class FakeConn:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    async def fetch(self, sql, *args):
        self.calls.append((sql, args))
        if "ANY($1::text[])" in sql:
            wanted = set(args[0])
            return [r for r in self.rows if r["seed_id"] in wanted]
        return list(self.rows)


class RecordingBus:
    def __init__(self):
        self.published = []

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))


def _chroma(*, space=None, query=None, stored=None, missing=False, embed_status=200, query_status=200):
    """Mock transport: embedder + Chroma 0.4.24 REST."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        path = request.url.path
        if request.url.host == "embed.test":
            if embed_status != 200:
                return httpx.Response(embed_status, json={"detail": "down"})
            body = json.loads(request.content)
            return httpx.Response(200, json={
                "doc_id": body["doc_id"], "embedding": [1.0, 0.0], "embedding_model": "bge", "embedding_dim": 2,
            })
        if path == "/api/v1/collections/orion_reading_results":
            if missing:
                return httpx.Response(500, json={"error": "ValueError('Collection orion_reading_results does not exist.')"})
            meta = {"hnsw:space": space} if space else None
            return httpx.Response(200, json={"name": "orion_reading_results", "id": "cid", "metadata": meta})
        if path == "/api/v1/collections/cid/query":
            if query_status != 200:
                return httpx.Response(query_status, json={"error": "boom"})
            ids, distances = zip(*query) if query else ((), ())
            return httpx.Response(200, json={"ids": [list(ids)], "distances": [list(distances)]})
        if path == "/api/v1/collections/cid/get":
            wanted = json.loads(request.content)["ids"]
            have = [i for i in wanted if i in (stored or {})]
            return httpx.Response(200, json={"ids": have, "metadatas": [{"content_hash": stored[i]} for i in have]})
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), seen


def _search(conn, client, query="graphics cards", limit=5, since=None):
    async def run():
        async with client:
            return await search_readings(conn, CFG, client=client, query=query, limit=limit, since=since)
    return asyncio.run(run())


def test_similarity_converts_l2_and_cosine_distances():
    assert similarity(0.5, "l2") == pytest.approx(0.75)
    assert similarity(0.25, "cosine") == pytest.approx(0.75)
    assert similarity(0.25, "ip") == pytest.approx(0.75)


def test_search_keeps_only_hits_above_floor_in_similarity_order():
    rows = [_row("a", "https://x.org/a", "gpu story"), _row("b", "https://x.org/b", "chip exports")]
    # l2: sim = 1 - d/2 -> a=0.70, b=0.80, c=0.50 (below floor)
    client, _ = _chroma(query=[("a", 0.6), ("b", 0.4), ("c", 1.0)])
    result = _search(FakeConn(rows), client)
    assert result.ok and result.total_available == 2
    assert [i.id for i in result.items] == ["b", "a"]
    assert result.items[0].extra["similarity"] == pytest.approx(0.8)
    assert result.items[0].text == "chip exports"


def test_search_regates_hits_against_postgres():
    rows = [_row("hollow", "https://x.org/h", "guess", read=False)]
    # "gone" is in the index but no longer in Postgres; "hollow" never read its source.
    client, _ = _chroma(space="cosine", query=[("hollow", 0.1), ("gone", 0.1)])
    result = _search(FakeConn(rows), client)
    assert result.ok and result.items == [] and result.total_available == 0


def test_search_nothing_above_floor_is_empty_not_unknown():
    client, _ = _chroma(query=[("a", 1.2)])
    conn = FakeConn([_row("a", "https://x.org/a", "gpu")])
    result = _search(conn, client)
    assert result.ok and result.items == [] and result.total_available == 0
    assert conn.calls == []


def test_search_respects_limit_but_counts_all_hits():
    rows = [_row(s, f"https://x.org/{s}", "gpu") for s in "abc"]
    client, _ = _chroma(query=[("a", 0.1), ("b", 0.2), ("c", 0.3)])
    result = _search(FakeConn(rows), client, limit=2)
    assert len(result.items) == 2 and result.total_available == 3


def test_search_passes_since_to_postgres():
    conn = FakeConn([_row("a", "https://x.org/a", "gpu")])
    client, _ = _chroma(query=[("a", 0.1)])
    _search(conn, client, since=NOW)
    [(_, args)] = conn.calls
    assert args == (["a"], NOW)


@pytest.mark.parametrize(
    "kwargs",
    [{"missing": True}, {"embed_status": 503}, {"query_status": 500}],
)
def test_search_failures_are_unknown(kwargs):
    client, _ = _chroma(query=[("a", 0.1)], **kwargs)
    with pytest.raises(SearchUnavailableError):
        _search(FakeConn([]), client)


def test_search_embeds_only_the_query():
    client, seen = _chroma(query=[("a", 0.1)])
    _search(FakeConn([_row("a", "https://x.org/a", "gpu")]), client)
    assert [r.url.host for r in seen].count("embed.test") == 1


def test_document_text_is_title_plus_learned_and_none_when_hollow():
    assert document_text(_row("a", "https://x.org/a", "gpu story", title="Nvidia")) == "Nvidia\n\ngpu story"
    assert document_text(_row("h", "https://x.org/h", "guess", read=False)) is None


def _index(rows, client, batch=10):
    bus = RecordingBus()

    async def run():
        async with client:
            return await index_missing_readings(rows, CFG, client=client, bus=bus, source=SOURCE, batch=batch)
    return asyncio.run(run()), bus


def test_index_upserts_missing_and_changed_verified_readings_only():
    fresh = _row("fresh", "https://x.org/f", "same")
    changed = _row("changed", "https://x.org/c", "new summary")
    missing = _row("missing", "https://x.org/m", "never indexed")
    hollow = _row("hollow", "https://x.org/h", "guess", read=False)
    stored = {"fresh": content_hash(document_text(fresh)), "changed": "stale-hash"}
    client, _ = _chroma(stored=stored)
    result, bus = _index([fresh, changed, missing, hollow], client)
    assert (result.indexed, result.pending) == (2, 0)
    ids = sorted(e.payload["doc_id"] for _, e in bus.published)
    assert ids == ["changed", "missing"]
    channel, envelope = bus.published[0]
    assert channel == UPSERT_CHANNEL and envelope.kind == "vector.upsert.v1"
    payload = VectorUpsertV1.model_validate(envelope.payload)
    assert payload.collection == "orion_reading_results" and payload.embedding_kind == "semantic"
    assert payload.meta["content_hash"] == content_hash(payload.text)


def test_index_treats_missing_collection_as_everything_missing_and_honors_batch():
    rows = [_row(s, f"https://x.org/{s}", "gpu") for s in "abc"]
    client, _ = _chroma(missing=True)
    result, bus = _index(rows, client, batch=2)
    assert (result.indexed, result.pending) == (2, 1) and len(bus.published) == 2


def test_index_embed_failure_is_unknown():
    client, _ = _chroma(stored={}, embed_status=503)
    with pytest.raises(SearchUnavailableError):
        _index([_row("a", "https://x.org/a", "gpu")], client)


def test_config_is_off_without_urls():
    assert CFG.enabled
    assert not ReadingSearchConfig(chroma_url="", embed_url="x", collection="c", min_similarity=0.6).enabled


def _raw_chroma(routes):
    """Embedder always answers; Chroma answers from {path: (status, json)}."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "embed.test":
            return httpx.Response(200, json={"doc_id": "q", "embedding": [1.0, 0.0]})
        status, body = routes.get(request.url.path, (404, {}))
        return httpx.Response(status, json=body)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


_COLL = ("/api/v1/collections/orion_reading_results", (200, {"id": "cid", "metadata": None}))


@pytest.mark.parametrize(
    "reply",
    [
        {"ids": [["a", "b"]], "distances": [[0.1]]},
        {"ids": [[None]], "distances": [[0.1]]},
        {"ids": [], "distances": []},
        # Chroma 0.4.24's real reply for a collection with no vectors yet.
        {"ids": [[]], "distances": [[]]},
    ],
)
def test_malformed_query_reply_is_unknown_not_empty(reply):
    client = _raw_chroma(dict([_COLL, ("/api/v1/collections/cid/query", (200, reply))]))
    with pytest.raises(SearchUnavailableError):
        _search(FakeConn([]), client)


def test_malformed_get_reply_is_unknown_not_everything_missing():
    client = _raw_chroma(dict([_COLL, ("/api/v1/collections/cid/get", (200, {"ids": ["a"]}))]))
    with pytest.raises(SearchUnavailableError):
        _index([_row("a", "https://x.org/a", "gpu")], client)


def test_unrelated_does_not_exist_error_is_unknown_not_missing_collection():
    body = {"error": "ValueError('Tenant default does not exist.')"}
    client = _raw_chroma({"/api/v1/collections/orion_reading_results": (500, body)})
    with pytest.raises(SearchUnavailableError):
        _index([_row("a", "https://x.org/a", "gpu")], client)
