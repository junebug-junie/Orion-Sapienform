"""A fake Chroma 0.4 + vector-host embedder for introspect responder tests.

Holds real documents, honours `where` filters ($and, equality, $gte), and
scores each document by a fixed per-id similarity. Upserts published on the
bus are NOT applied until `apply(bus)` is called -- exactly the gap between
"the index loop published" and "orion-vector-writer stored it".
"""
from __future__ import annotations

import json
from typing import Any

import httpx

from orion.introspect.semantic_index import UPSERT_CHANNEL


def _matches(meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    if "$and" in where:
        return all(_matches(meta, w) for w in where["$and"])
    for key, cond in where.items():
        value = meta.get(key)
        if isinstance(cond, dict):
            if "$gte" in cond and not (value is not None and value >= cond["$gte"]):
                return False
        elif value != cond:
            return False
    return True


class FakeChroma:
    def __init__(self, collection: str, *, scores: dict[str, float] | None = None):
        self.collection = collection
        self.docs: dict[str, dict[str, Any]] = {}  # id -> metadata (holds content_hash)
        self.scores = dict(scores or {})
        self.queries: list[dict[str, Any]] = []
        self.created = False

    def put(self, doc_id: str, meta: dict[str, Any], score: float | None = None) -> None:
        self.created = True
        self.docs[doc_id] = dict(meta)
        if score is not None:
            self.scores[doc_id] = score

    def apply(self, bus: Any) -> int:
        """Store every upsert the bus saw (what orion-vector-writer would do)."""
        n = 0
        for channel, envelope in bus.published:
            if channel != UPSERT_CHANNEL:
                continue
            payload = envelope.payload
            if payload["collection"] == self.collection:
                self.put(payload["doc_id"], payload["meta"])
                n += 1
        return n

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.url.host == "embed.test":
            body = json.loads(request.content)
            return httpx.Response(200, json={"doc_id": body["doc_id"], "embedding": [1.0, 0.0], "embedding_model": "bge"})
        if path == f"/api/v1/collections/{self.collection}":
            if not self.created:
                return httpx.Response(500, json={"error": f"ValueError('Collection {self.collection} does not exist.')"})
            return httpx.Response(200, json={"id": "cid", "metadata": {"hnsw:space": "cosine"}})
        if path == "/api/v1/collections/cid/get":
            wanted = json.loads(request.content)["ids"]
            have = [i for i in wanted if i in self.docs]
            return httpx.Response(200, json={"ids": have, "metadatas": [self.docs[i] for i in have]})
        if path == "/api/v1/collections/cid/count":
            return httpx.Response(200, json=len(self.docs))
        if path == "/api/v1/collections/cid/query":
            body = json.loads(request.content)
            self.queries.append(body)
            hits = [i for i, m in self.docs.items() if _matches(m, body.get("where"))]
            hits.sort(key=lambda i: -self.scores.get(i, 0.0))
            hits = hits[: body["n_results"]]
            return httpx.Response(200, json={
                "ids": [hits], "distances": [[1.0 - self.scores.get(i, 0.0) for i in hits]],
            })
        return httpx.Response(404)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))
