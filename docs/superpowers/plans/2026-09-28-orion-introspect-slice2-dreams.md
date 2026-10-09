# Orion Introspect — Slice 2 (dreams, with search by meaning) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Orion can call `dreams` to read back their own narrative dreams and the sleep-cycle hypotheses they have already been offered: recent, one by id, or by meaning.

**Architecture:**
- **Transport.** The harness-side MCP tool sends `IntrospectRequestV1` over bus RPC on `orion:introspect:dream:request`.
- **Responder.** orion-dream answers with a read-only Postgres lookup, plus Chroma for `query=`, and replies `IntrospectResultV1` on `orion:introspect:result:<correlation_id>`.
- **Shared plumbing.** Embed, nearest-match and index-upsert move from reading search into `orion/introspect/semantic_index.py`. Reading search and dream search both use it.
- **Index.** orion-dream keeps the `orion_dreams` collection current with a hash-aware index loop.

**Tech Stack:** Python 3.12, pydantic v2, SQLAlchemy 2 (sync, psycopg2) in orion-dream, httpx 0.27, Chroma 0.4.24 REST, vector-host HTTP `/embedding` (bge-large, 1024-dim), pytest.

**Spec:** `docs/superpowers/specs/2026-09-28-orion-introspect-slice2-dreams-design.md` (includes the metric-gate record for the similarity floor).

## Global Constraints

- **Workspace.**
  - Worktree `/mnt/scripts/Orion-Sapienform-introspect-dreams`, branch `feat/introspect-dreams`. Never commit from `/mnt/scripts/Orion-Sapienform`.
  - Test interpreter `PY=/tmp/introspect-venv/bin/python`, run from the worktree root with `PYTHONPATH=.`.
  - orion-dream tests run as their own pytest invocation (the service has its own `app` package; its `tests/conftest.py` puts it on `sys.path`).
- **Blind experiment.**
  - Every SQL statement touching `dream_hypothesis` contains `h.offered_at IS NOT NULL`.
  - No statement selects `arm`, `ref_a` or `ref_b`.
  - Never-offered hypotheses are never indexed, returned or counted.
- **Epistemics.** Every dream item is `epistemic_status="unsettled"`.
- **Empty ≠ unknown.**
  - Nothing matched: `ok=True, items=[], total_available=0`.
  - Postgres error, embedder/Chroma failure, search not configured, missing or empty collection: responder `ok=False` with `dreams_unavailable; answer unknown` or `dream_search_unavailable; answer unknown`. The tool raises `IntrospectUnknownError` containing `answer unknown`.
  - Invalid request: `ok=False` with `invalid dreams request: …`.
- **Read-only.** Every orion-dream connection runs `SET TRANSACTION READ ONLY` first. Only the query is embedded at ask time.
- **Bounds.**
  - Output: `MAX_ITEMS=5`, `DEFAULT_TEXT_CAP=900`, `FULL_TEXT_CAP=4000` (one-record fetch only), `THEME_CAP=8`, theme text ≤80 chars.
  - Search: `CANDIDATES=20`, `INDEX_TEXT_CHARS=1800`, `INDEX_SCAN_LIMIT=1000`, `HTTP_TIMEOUT_SEC=5.0`.
- **Contract names.**
  - `DREAM_REQUEST_CHANNEL="orion:introspect:dream:request"`, `RESULT_PREFIX="orion:introspect:result:"`
  - `REQUEST_KIND="introspect.tool.request.v1"`, `RESULT_KIND="introspect.tool.result.v1"`
- **Env keys (orion-dream).**
  - `DREAM_INTROSPECT_ENABLED` (`true`)
  - `DREAM_SEARCH_CHROMA_URL` (`http://${PROJECT}-vector-db:8000`)
  - `DREAM_SEARCH_EMBED_URL` (`http://${PROJECT}-vector-host:8320/embedding`)
  - `DREAM_SEARCH_COLLECTION` (`orion_dreams`)
  - `DREAM_SEARCH_MIN_SIMILARITY` (single-source owner `services/orion-dream/.env_example`; value set in Task 8)
  - `DREAM_SEARCH_INDEX_INTERVAL_SEC` (`300`)
  - `DREAM_SEARCH_INDEX_BATCH` (`10`)
  - Settings defaults for both URLs are empty, so search is off unless configured.
- **Git and deploy.** Never `git commit --no-verify`. Never stage `.env`. Do not deploy. The controller deploys, only from a worktree 0 commits behind `origin/main`.

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `orion/introspect/semantic_index.py` | Create | Shared config, embed, Chroma nearest/get, upsert publish |
| `orion/introspect/tests/test_semantic_index.py` | Create | Unit tests for the shared module |
| `orion/world_pulse_read/search.py` | Modify | Keep reading SQL/gating; import plumbing; keep public names |
| `orion/introspect/transport.py` | Create | Channel and kind constants shared by tool and responder |
| `orion/schemas/introspect.py` | Modify | `IntrospectBusOperation`, `"dreams"` operation, `IntrospectRequestV1`, `DreamsArguments`, caps |
| `orion/schemas/registry.py` | Modify | Register `IntrospectRequestV1`, `DreamsArguments` |
| `orion/bus/channels.yaml` | Modify | Request + result channels; orion-dream produces `orion:vector:semantic:upsert` |
| `orion/introspect/tests/test_introspect_schemas.py` | Modify | Contract tests |
| `services/orion-dream/app/introspect_dreams.py` | Create | SQL + item builders (recent / by ids / one / index rows) |
| `services/orion-dream/tests/test_introspect_dreams.py` | Create | Query-module tests incl. blind-experiment pins |
| `services/orion-dream/app/dream_search.py` | Create | Document text, index pass, rank |
| `services/orion-dream/tests/test_dream_search.py` | Create | Search tests over mocked Chroma/embedder |
| `services/orion-dream/app/introspect_listener.py` | Create | Bus responder + index loop |
| `services/orion-dream/tests/test_introspect_listener.py` | Create | Responder tests over fake bus/engine |
| `services/orion-dream/app/settings.py`, `app/main.py` | Modify | Seven keys; start/stop listener |
| `services/orion-dream/.env_example`, `docker-compose.yml` | Modify | Seven keys |
| `scripts/check_env_key_single_source.py` | Modify | Owner for `DREAM_SEARCH_MIN_SIMILARITY` |
| `orion/introspect/tools.py`, `orion/introspect/brief.py` | Modify | `dreams` tool + brief line |
| `orion/introspect/tests/test_introspect_tools.py`, `test_introspect_harness_wiring.py` | Modify | Tool/brief coverage |
| `scripts/smoke_introspect.py` | Modify | `--tool dreams`, `--dream-id` |
| `services/orion-dream/evals/run_dream_search_calibration.py` | Create | Read-only floor calibration |
| `services/orion-dream/README.md` | Modify | "### Introspect responder: `dreams`" |
| `services/orion-harness-governor/README.md` | Modify | `dreams` row → Live; search paragraph points at shared module |
| `.github/workflows/orion-reading-tests.yml` | Modify | Trigger on orion-dream; run its introspect tests |

---

### Task 1: Shared search module (extract from reading search, no behavior change)

**Files:**
- Create: `orion/introspect/semantic_index.py`, `orion/introspect/tests/test_semantic_index.py`
- Modify: `orion/world_pulse_read/search.py`
- Test: `orion/world_pulse_read/tests/test_reading_search.py` (must pass unchanged), `orion/introspect/tests/test_semantic_index.py`

**Interfaces:**
- Produces:
  - Classes: `SearchConfig`, `SearchUnavailableError`, `IndexPass`
  - Constants: `UPSERT_CHANNEL`, `UPSERT_KIND`, `HTTP_TIMEOUT_SEC`
  - `content_hash(text) -> str`
  - `similarity(distance, space) -> float`
  - `async embed(client, cfg, text, *, doc_prefix) -> tuple[list[float], str | None]`
  - `async nearest(client, cfg, vector, n) -> list[tuple[str, float]]`
  - `async stored_hashes(client, cfg, ids) -> dict[str, str]`
  - `async publish_upsert(bus, source, cfg, *, doc_id, text, vector, model, meta) -> None`
- `orion.world_pulse_read.search` keeps exporting `ReadingSearchConfig` (alias of `SearchConfig`), `SearchUnavailableError`, `IndexPass`, `UPSERT_CHANNEL`, `HTTP_TIMEOUT_SEC`, `content_hash`, `similarity`, `embed(client, cfg, text)`, `nearest(client, cfg, vector, n=20)`, `document_text`, `rank_readings`, `gated_results`, `search_readings`, `verified_rows`, `index_missing_readings`.

- [ ] **Step 1: Baseline.** Record that reading search passes before the change.

Run: `PYTHONPATH=. $PY -m pytest orion/world_pulse_read/tests/test_reading_search.py -q`
Expected: all pass (note the count).

- [ ] **Step 2: Write the failing test** `orion/introspect/tests/test_semantic_index.py`:

```python
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
```

- [ ] **Step 3: Run to verify it fails**

Run: `PYTHONPATH=. $PY -m pytest orion/introspect/tests/test_semantic_index.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'orion.introspect.semantic_index'`.

- [ ] **Step 4: Create** `orion/introspect/semantic_index.py`:

```python
"""Shared search-by-meaning plumbing for orion-introspect responders.

A record is embedded once (vector-host HTTP /embedding, which persists nothing)
and upserted through orion-vector-writer into a per-domain Chroma collection;
a question embeds only the query. Chroma is an index, never the record: every
caller re-reads each hit from its own tables and re-gates it.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.vector.schemas import EmbeddingGenerateV1, EmbeddingResultV1, VectorUpsertV1

UPSERT_CHANNEL = "orion:vector:semantic:upsert"
UPSERT_KIND = "vector.upsert.v1"
HTTP_TIMEOUT_SEC = 5.0


class SearchUnavailableError(RuntimeError):
    """The embedder, index, or configuration could not answer; the result is unknown."""


@dataclass(frozen=True)
class SearchConfig:
    chroma_url: str
    embed_url: str
    collection: str
    min_similarity: float
    index_interval_sec: float = 300.0
    index_batch: int = 10

    @property
    def enabled(self) -> bool:
        return bool(self.chroma_url.strip() and self.embed_url.strip() and self.collection.strip())


@dataclass(frozen=True)
class IndexPass:
    indexed: int
    pending: int


@dataclass(frozen=True)
class _Collection:
    id: str
    space: str


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def similarity(distance: float, space: str) -> float:
    # Chroma's l2 is squared euclidean; for the unit vectors bge emits,
    # ||a-b||^2 = 2 - 2cos, so cos = 1 - d/2. cosine and ip report 1 - cos.
    if space == "l2":
        return 1.0 - distance / 2.0
    return 1.0 - distance


def _base(cfg: SearchConfig) -> str:
    return f"{cfg.chroma_url.rstrip('/')}/api/v1/collections"


async def _json(client: httpx.AsyncClient, method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
    try:
        resp = await client.request(method, url, **kwargs)
        return resp.status_code, resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise SearchUnavailableError(f"chroma unavailable: {type(exc).__name__}") from exc


async def embed(
    client: httpx.AsyncClient, cfg: SearchConfig, text: str, *, doc_prefix: str,
) -> tuple[list[float], str | None]:
    req = EmbeddingGenerateV1(doc_id=f"{doc_prefix}-{uuid4()}", text=text)
    try:
        resp = await client.post(cfg.embed_url, json=req.model_dump(mode="json"))
        resp.raise_for_status()
        result = EmbeddingResultV1.model_validate(resp.json())
    except (httpx.HTTPError, ValueError) as exc:
        raise SearchUnavailableError(f"embedder unavailable: {type(exc).__name__}") from exc
    if not result.embedding:
        raise SearchUnavailableError("embedder returned no vector")
    return result.embedding, result.embedding_model


async def _collection(client: httpx.AsyncClient, cfg: SearchConfig) -> _Collection | None:
    status, body = await _json(client, "GET", f"{_base(cfg)}/{cfg.collection}")
    if status == 200 and isinstance(body, dict) and body.get("id"):
        space = str((body.get("metadata") or {}).get("hnsw:space") or "l2")
        return _Collection(id=str(body["id"]), space=space)
    # Chroma 0.4.24 answers a missing collection with HTTP 500 + ValueError text.
    error = str(body.get("error") or "") if isinstance(body, dict) else ""
    if status == 500 and f"Collection {cfg.collection} does not exist" in error:
        return None
    raise SearchUnavailableError(f"chroma collection lookup failed: HTTP {status}")


async def nearest(
    client: httpx.AsyncClient, cfg: SearchConfig, vector: list[float], n: int,
) -> list[tuple[str, float]]:
    """(doc_id, similarity) for the n nearest indexed records, best first."""
    coll = await _collection(client, cfg)
    if coll is None:
        raise SearchUnavailableError(f"search index {cfg.collection} not built yet")
    status, body = await _json(
        client, "POST", f"{_base(cfg)}/{coll.id}/query",
        json={"query_embeddings": [vector], "n_results": n, "include": ["distances"]},
    )
    if status != 200:
        raise SearchUnavailableError(f"chroma query failed: HTTP {status}")
    try:
        ids, distances = body["ids"][0], body["distances"][0]
        if len(ids) != len(distances) or any(not isinstance(i, str) for i in ids):
            raise ValueError("ids/distances mismatch")
        if not ids:
            # Chroma clamps n_results to the index size, so an existing collection
            # answering nothing holds no vectors: unbuilt, not "no match".
            raise ValueError(f"search index {cfg.collection} is empty")
        scored = [(i, similarity(float(d), coll.space)) for i, d in zip(ids, distances)]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise SearchUnavailableError("malformed chroma query reply") from exc
    return sorted(scored, key=lambda s: -s[1])


async def stored_hashes(client: httpx.AsyncClient, cfg: SearchConfig, ids: list[str]) -> dict[str, str]:
    """content_hash per already-indexed id; a missing collection holds nothing."""
    coll = await _collection(client, cfg)
    if coll is None:
        return {}
    status, body = await _json(
        client, "POST", f"{_base(cfg)}/{coll.id}/get", json={"ids": ids, "include": ["metadatas"]},
    )
    got = body.get("ids") if isinstance(body, dict) else None
    metas = body.get("metadatas") if isinstance(body, dict) else None
    if status != 200 or not isinstance(got, list) or not isinstance(metas, list) or len(got) != len(metas):
        raise SearchUnavailableError(f"chroma get failed: HTTP {status}")
    return {str(i): str((m or {}).get("content_hash") or "") for i, m in zip(got, metas)}


async def publish_upsert(
    bus: Any, source: Any, cfg: SearchConfig, *,
    doc_id: str, text: str, vector: list[float], model: str | None, meta: dict[str, Any],
) -> None:
    payload = VectorUpsertV1(
        doc_id=doc_id, collection=cfg.collection, embedding=vector, embedding_kind="semantic",
        embedding_model=model, embedding_dim=len(vector), text=text, meta=meta,
    )
    await bus.publish(UPSERT_CHANNEL, BaseEnvelope(
        kind=UPSERT_KIND, source=source, payload=payload.model_dump(mode="json"),
    ))
```

- [ ] **Step 5: Rewrite** `orion/world_pulse_read/search.py`. Keep the reading SQL and gating and use the shared plumbing. Replace the whole file with:

```python
"""Semantic search over verified readings for the reading_results introspect tool.

Each verified reading is embedded once, when it lands, and upserted through
orion-vector-writer; a question embeds only the query. Chroma is an index, not
the record: every hit is re-read from world_pulse_read_seed and re-gated.
Shared plumbing: orion/introspect/semantic_index.py.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from orion.introspect.semantic_index import (
    HTTP_TIMEOUT_SEC,
    UPSERT_CHANNEL,
    UPSERT_KIND,
    IndexPass,
    SearchConfig,
    SearchUnavailableError,
    content_hash,
    publish_upsert,
    similarity,
    stored_hashes,
)
from orion.introspect.semantic_index import embed as _embed
from orion.introspect.semantic_index import nearest as _nearest
from orion.schemas.introspect import URL_CAP, IntrospectResultV1, clip_text
from orion.world_pulse_read.introspect import (
    _COLUMNS,
    _OCCURRED,
    _VERIFIED_WHERE,
    _item,
    _learned,
    _source_read,
)

__all__ = [
    "CANDIDATES", "HTTP_TIMEOUT_SEC", "INDEX_SCAN_LIMIT", "INDEX_TEXT_CHARS", "UPSERT_CHANNEL",
    "UPSERT_KIND", "IndexPass", "ReadingSearchConfig", "SearchUnavailableError", "content_hash",
    "document_text", "embed", "gated_results", "index_missing_readings", "nearest",
    "rank_readings", "search_readings", "similarity", "verified_rows",
]

CANDIDATES = 20
INDEX_TEXT_CHARS = 1800
INDEX_SCAN_LIMIT = 1000

ReadingSearchConfig = SearchConfig

_VERIFIED_ROWS_SQL = f"""
SELECT {_COLUMNS} FROM world_pulse_read_seed s
WHERE {_VERIFIED_WHERE}
ORDER BY {_OCCURRED} DESC, s.seed_id DESC
LIMIT $1
"""

_ROWS_BY_ID_SQL = f"""
SELECT {_COLUMNS} FROM world_pulse_read_seed s
WHERE s.seed_id = ANY($1::text[])
  AND {_VERIFIED_WHERE}
  AND ($2::timestamptz IS NULL OR {_OCCURRED} >= $2)
"""


def document_text(row: Any) -> str | None:
    learned = _learned(row, _source_read(row))
    if not learned:
        return None
    title = str(row["title"] or "").strip()
    return clip_text(f"{title}\n\n{learned}" if title else learned, INDEX_TEXT_CHARS)[0]


async def embed(client: httpx.AsyncClient, cfg: SearchConfig, text: str) -> tuple[list[float], str | None]:
    return await _embed(client, cfg, text, doc_prefix="reading-search")


async def nearest(
    client: httpx.AsyncClient, cfg: SearchConfig, vector: list[float], n: int = CANDIDATES,
) -> list[tuple[str, float]]:
    """(seed_id, similarity) for the n nearest indexed readings, best first."""
    return await _nearest(client, cfg, vector, n)


async def rank_readings(
    client: httpx.AsyncClient, cfg: SearchConfig, query: str,
) -> list[tuple[str, float]]:
    """Embed the query once; (seed_id, similarity) hits at or above the floor, best first."""
    vector, _ = await embed(client, cfg, query)
    return [s for s in await nearest(client, cfg, vector) if s[1] >= cfg.min_similarity]


async def gated_results(
    conn: Any,
    scored: list[tuple[str, float]],
    *,
    limit: int,
    since: datetime | None = None,
) -> IntrospectResultV1:
    """Re-read ranked hits from Postgres; only verified readings survive."""
    as_of = datetime.now(timezone.utc)
    if not scored:
        return IntrospectResultV1(ok=True, operation="reading_result", as_of=as_of, total_available=0)
    rows = {str(r["seed_id"]): r for r in await conn.fetch(_ROWS_BY_ID_SQL, [sid for sid, _ in scored], since)}
    hits = [
        (rows[sid], sim) for sid, sim in scored
        if sid in rows and _learned(rows[sid], _source_read(rows[sid]))
    ]
    items = []
    for row, sim in hits[:limit]:
        item = _item(row, request_id=str(row["request_id"]) if row["request_id"] else None)
        items.append(item.model_copy(update={"extra": {**item.extra, "similarity": round(sim, 3)}}))
    return IntrospectResultV1(
        ok=True, operation="reading_result", as_of=as_of, total_available=len(hits), items=items,
    )


async def search_readings(
    conn: Any,
    cfg: SearchConfig,
    *,
    client: httpx.AsyncClient,
    query: str,
    limit: int,
    since: datetime | None = None,
) -> IntrospectResultV1:
    return await gated_results(conn, await rank_readings(client, cfg, query), limit=limit, since=since)


async def verified_rows(conn: Any, scan_limit: int = INDEX_SCAN_LIMIT) -> list[Any]:
    return list(await conn.fetch(_VERIFIED_ROWS_SQL, scan_limit))


async def index_missing_readings(
    rows: list[Any],
    cfg: SearchConfig,
    *,
    client: httpx.AsyncClient,
    bus: Any,
    source: Any,
    batch: int | None = None,
) -> IndexPass:
    """Embed and upsert verified readings the index lacks or holds stale text for."""
    docs = []
    for row in rows:
        text = document_text(row)
        if text:
            docs.append((row, text, content_hash(text)))
    if not docs:
        return IndexPass(indexed=0, pending=0)
    stored = await stored_hashes(client, cfg, [str(row["seed_id"]) for row, _, _ in docs])
    stale = [d for d in docs if stored.get(str(d[0]["seed_id"])) != d[2]]
    batch = cfg.index_batch if batch is None else batch
    for row, text, digest in stale[:batch]:
        vector, model = await embed(client, cfg, text)
        occurred = row["landing_at"] or row["stage2_completed_at"] or row["handoff_at"] or row["created_at"]
        await publish_upsert(
            bus, source, cfg, doc_id=str(row["seed_id"]), text=text, vector=vector, model=model,
            meta={
                "content_hash": digest,
                "url": clip_text(row["url"], URL_CAP)[0],
                "request_id": str(row["request_id"] or ""),
                "occurred_at": occurred.isoformat() if occurred else "",
            },
        )
    done = min(len(stale), batch)
    return IndexPass(indexed=done, pending=len(stale) - done)
```

- [ ] **Step 6: Run the shared tests plus every reading-search consumer.**

Run:

```bash
PYTHONPATH=. $PY -m pytest orion/introspect/tests/test_semantic_index.py orion/world_pulse_read/tests/test_reading_search.py -q
PYTHONPATH=.:services/orion-hub $PY -m pytest services/orion-hub/tests/test_reading_ingress.py -q
```

Expected: all pass. `test_reading_search.py` shows the same count as in Step 1, unchanged.

- [ ] **Step 7: Commit**

```bash
git add orion/introspect/semantic_index.py orion/introspect/tests/test_semantic_index.py orion/world_pulse_read/search.py
git commit -m "refactor(introspect): shared semantic_index; reading search uses it unchanged"
```

---

### Task 2: Contracts — `dreams` operation, request model, channels

**Files:**
- Create: `orion/introspect/transport.py`
- Modify: `orion/schemas/introspect.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`
- Test: `orion/introspect/tests/test_introspect_schemas.py`

**Interfaces:**
- Produces, in `orion.introspect.transport`:
  - `DREAM_REQUEST_CHANNEL`, `RESULT_PREFIX`, `REQUEST_KIND`, `RESULT_KIND`
- Produces, in `orion.schemas.introspect`:
  - Constants: `FULL_TEXT_CAP=4000`, `THEME_CAP=8`, `DREAM_ID_PATTERN`
  - Types: `IntrospectBusOperation = Literal["dreams"]`, `IntrospectOperation = Literal["reading_result", "dreams"]`
  - `IntrospectRequestV1(operation, binding, args)`
  - `DreamsArguments(query, dream_id, kind, limit, since)`

- [ ] **Step 1: Write failing tests.** Append to `orion/introspect/tests/test_introspect_schemas.py`. It already imports `pytest`, `ValidationError` and `NOW`; check the file header and add any missing imports.

```python
from orion.introspect.transport import DREAM_REQUEST_CHANNEL, REQUEST_KIND, RESULT_KIND, RESULT_PREFIX  # noqa: E402
from orion.schemas.introspect import (  # noqa: E402
    FULL_TEXT_CAP,
    DreamsArguments,
    IntrospectRequestV1,
    IntrospectResultV1,
    IntrospectToolBindingV1,
)

_BINDING = IntrospectToolBindingV1(
    invocation_context="unified_chat", parent_run_id="r", parent_trace_id="t", memory_allowed=True,
)


def test_transport_constants():
    assert DREAM_REQUEST_CHANNEL == "orion:introspect:dream:request"
    assert RESULT_PREFIX == "orion:introspect:result:"
    assert (REQUEST_KIND, RESULT_KIND) == ("introspect.tool.request.v1", "introspect.tool.result.v1")


def test_dreams_arguments_defaults_and_modes():
    assert DreamsArguments().limit == 5
    assert DreamsArguments(query="  vision  ").query == "vision"
    assert DreamsArguments(dream_id="dream:19").dream_id == "dream:19"
    assert DreamsArguments(dream_id="dh-33f002b0db4c").dream_id == "dh-33f002b0db4c"
    assert DreamsArguments(kind="hypothesis", since=NOW).kind == "hypothesis"


@pytest.mark.parametrize(
    "fields",
    [
        {"query": "   "},
        {"query": "x" * 501},
        {"dream_id": "19"},
        {"dream_id": "dh-XYZ"},
        {"dream_id": "dream:19", "query": "vision"},
        {"dream_id": "dream:19", "kind": "narrative"},
        {"dream_id": "dream:19", "since": NOW},
        {"kind": "control"},
        {"limit": 6},
        {"since": NOW.replace(tzinfo=None)},
        {"arm": "dream"},
    ],
)
def test_dreams_arguments_reject_bad_input(fields):
    with pytest.raises(ValidationError):
        DreamsArguments(**fields)


def test_request_carries_only_bus_operations():
    req = IntrospectRequestV1(operation="dreams", binding=_BINDING, args={"limit": 2})
    assert req.model_dump(mode="json")["operation"] == "dreams"
    with pytest.raises(ValidationError):
        IntrospectRequestV1(operation="reading_result", binding=_BINDING, args={})


def test_result_accepts_dreams_operation():
    result = IntrospectResultV1(ok=True, operation="dreams", as_of=NOW, total_available=0)
    assert result.operation == "dreams"
    assert FULL_TEXT_CAP == 4000


def test_registry_and_channels_know_the_dream_contract():
    import yaml
    from pathlib import Path

    from orion.schemas.registry import _REGISTRY

    assert {"IntrospectRequestV1", "DreamsArguments", "IntrospectResultV1"} <= set(_REGISTRY)
    channels = yaml.safe_load((Path(__file__).resolve().parents[3] / "orion/bus/channels.yaml").read_text())["channels"]
    by_name = {c["name"]: c for c in channels}
    req = by_name["orion:introspect:dream:request"]
    assert (req["schema_id"], req["message_kind"], req["consumer_services"]) == (
        "IntrospectRequestV1", "introspect.tool.request.v1", ["orion-dream"],
    )
    res = by_name["orion:introspect:result:*"]
    assert (res["schema_id"], res["message_kind"]) == ("IntrospectResultV1", "introspect.tool.result.v1")
    assert "orion-dream" in by_name["orion:vector:semantic:upsert"]["producer_services"]
```

`_REGISTRY` is the registry dict's real name (`orion/schemas/registry.py:803`).

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. $PY -m pytest orion/introspect/tests/test_introspect_schemas.py -q`
Expected: FAIL with ImportError (`orion.introspect.transport` / `DreamsArguments`).

- [ ] **Step 3: Create** `orion/introspect/transport.py`:

```python
"""Bus names shared by the orion-introspect tool and its owning-service responders."""
from __future__ import annotations

DREAM_REQUEST_CHANNEL = "orion:introspect:dream:request"
RESULT_PREFIX = "orion:introspect:result:"
REQUEST_KIND = "introspect.tool.request.v1"
RESULT_KIND = "introspect.tool.result.v1"
```

- [ ] **Step 4: Edit** `orion/schemas/introspect.py`.

Change the module docstring's second paragraph to:

```python
"""Introspect tool contracts: Orion reading back their own recorded activity.

Design: docs/superpowers/specs/2026-09-28-orion-introspect-mcp-design.md.
``reading_result`` travels on the reading contract; every other operation is an
``IntrospectBusOperation`` answered by its owning service over
``orion:introspect:<domain>:request``.
"""
```

Replace `IntrospectOperation = Literal["reading_result"]` with:

```python
FULL_TEXT_CAP = 4000
THEME_CAP = 8
DREAM_ID_PATTERN = r"^(dream:[0-9]{1,12}|dh-[0-9a-f]{6,32})$"

IntrospectBusOperation = Literal["dreams"]
IntrospectOperation = Literal["reading_result", "dreams"]
```

Append at the end of the file:

```python
class IntrospectRequestV1(BaseModel):
    """Harness -> owning service. ``args`` is validated by the owner per operation."""

    model_config = ConfigDict(extra="forbid")
    operation: IntrospectBusOperation
    binding: IntrospectToolBindingV1
    args: dict[str, Any] = Field(default_factory=dict)


class DreamsArguments(BaseModel):
    """Model-supplied arguments for the ``dreams`` tool."""

    model_config = ConfigDict(extra="forbid")
    query: str | None = Field(default=None, min_length=1, max_length=QUERY_CAP)
    dream_id: str | None = Field(default=None, pattern=DREAM_ID_PATTERN)
    kind: Literal["narrative", "hypothesis"] | None = None
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=MAX_ITEMS)
    since: datetime | None = None

    @field_validator("query", mode="before")
    @classmethod
    def _strip_query(cls, value: Any) -> Any:
        return normalize_query(value)

    @model_validator(mode="after")
    def _selectors(self):
        _require_tz(self.since, "since")
        if self.dream_id is not None and (
            self.query is not None or self.kind is not None or self.since is not None
        ):
            raise ValueError("dream_id fetches one dream; it cannot be combined with query, kind or since")
        return self
```

- [ ] **Step 5: Register.** In `orion/schemas/registry.py`, extend the introspect import:

```python
from orion.schemas.introspect import (
    DreamsArguments, IntrospectItemV1, IntrospectRequestV1, IntrospectResultV1, IntrospectToolBindingV1,
    ReadingResultArguments,
)
```

Then add the two entries next to `"ReadingResultArguments": ReadingResultArguments,`:

```python
    "IntrospectRequestV1": IntrospectRequestV1,
    "DreamsArguments": DreamsArguments,
```

- [ ] **Step 6: Channels.** In `orion/bus/channels.yaml`:
  - Change the `orion:vector:semantic:upsert` producers to `producer_services: ["orion-vector-host", "orion-hub", "orion-dream"]`.
  - Add these two entries directly after the `orion:reading:tool:result:*` entry:

```yaml
  - name: "orion:introspect:dream:request"
    kind: "request"
    schema_id: "IntrospectRequestV1"
    message_kind: "introspect.tool.request.v1"
    producer_services: ["orion-harness-governor"]
    consumer_services: ["orion-dream"]
    stability: "experimental"
    since: "2026-09-28"
    description: "orion-introspect dreams tool RPC. Read-only: recent, one by dream_id, or query by meaning over narrative dreams and already-offered sleep-cycle hypotheses (both arms, arm never exposed). Reply on orion:introspect:result:<correlation_id>."

  - name: "orion:introspect:result:*"
    kind: "result"
    schema_id: "IntrospectResultV1"
    message_kind: "introspect.tool.result.v1"
    producer_services: ["orion-dream"]
    consumer_services: ["orion-harness-governor"]
    stability: "experimental"
    since: "2026-09-28"
    description: "Owning-service answer to an introspect request. ok=true with items=[] means nothing matched; ok=false or a timeout means the answer is unknown."
```

- [ ] **Step 7: Run the tests and the contract gates**

```bash
PYTHONPATH=. $PY -m pytest orion/introspect/tests -q
PYTHONPATH=. $PY scripts/check_schema_registry.py
PYTHONPATH=. $PY scripts/check_bus_channels.py
```

Expected:
- The pytest command passes, except `test_readme_coverage.py::test_every_introspect_request_channel_is_a_live_row`. That failure is **intended**: it stays red until Task 7 documents the responder and flips the row to Live, and it proves the gate works. Note it and continue.
- Both gate scripts exit 0.

- [ ] **Step 8: Commit**

```bash
git add orion/introspect/transport.py orion/schemas/introspect.py orion/schemas/registry.py orion/bus/channels.yaml orion/introspect/tests/test_introspect_schemas.py
git commit -m "feat(introspect): dreams contract (request model, arguments, channels)"
```

---

### Task 3: orion-dream query module

**Files:**
- Create: `services/orion-dream/app/introspect_dreams.py`, `services/orion-dream/tests/test_introspect_dreams.py`

**Interfaces:**
- Consumes: `orion.schemas.introspect` (`DEFAULT_TEXT_CAP`, `FULL_TEXT_CAP`, `SHORT_FIELD_CAP`, `THEME_CAP`, `IntrospectItemV1`, `IntrospectResultV1`, `clip_text`).
- Produces, all taking a SQLAlchemy `Connection`-like `conn` with `.execute(clause, params).mappings().all()`:
  - `recent(conn, *, kind, since, limit, now) -> IntrospectResultV1`
  - `by_ids(conn, scored: list[tuple[str, float]], *, kind, since, limit, now) -> IntrospectResultV1`
  - `one(conn, dream_id, *, now) -> IntrospectResultV1`
  - `index_rows(conn) -> list[tuple[str, dict]]` (pairs of `("narrative" | "hypothesis", row)`)
  - `doc_id(kind, row) -> str`
  - `split_ids(ids) -> tuple[list[int], list[str]]`
  - `HYPOTHESIS_SQL: tuple[str, ...]` (every hypothesis statement, for pins)

- [ ] **Step 1: Write the failing test** `services/orion-dream/tests/test_introspect_dreams.py`:

```python
"""Dream lookups: both kinds, blind-experiment rules, empty vs counts, caps."""
import re
from datetime import date, datetime, timedelta, timezone

from app import introspect_dreams as dq

NOW = datetime(2026, 9, 29, 6, 0, tzinfo=timezone.utc)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class FakeConn:
    """Returns given rows (newest first) by table; honors ids and limit like the SQL does."""

    def __init__(self, narratives=(), hypotheses=()):
        self.tables = {"dreams": list(narratives), "dream_hypothesis": list(hypotheses)}
        self.calls = []

    def execute(self, clause, params=None):
        sql, params = str(clause), dict(params or {})
        self.calls.append((sql, params))
        table = "dream_hypothesis" if "FROM dream_hypothesis" in sql else "dreams" if "FROM dreams" in sql else None
        if table is None:
            return _Result([])
        rows = self.tables[table]
        if "ids" in params:
            key = "hypothesis_id" if table == "dream_hypothesis" else "id"
            rows = [r for r in rows if r[key] in params["ids"]]
        total = len(rows)
        return _Result([{**r, "total": total} for r in rows[: params.get("limit", len(rows))]])


def narrative(i, hours_ago, *, tldr="A dream.", story="It went on.", themes=("vision",)):
    return {
        "id": i, "dream_date": date(2026, 9, 28), "tldr": tldr, "narrative": story,
        "themes": list(themes), "occurred_at": NOW - timedelta(hours=hours_ago),
    }


def hypothesis(hid, hours_ago, *, expires_in_hours=24, why="Both mention RPC."):
    return {
        "hypothesis_id": hid, "cycle_id": "dc-1", "claim": f"claim {hid}", "why": why,
        "occurred_at": NOW - timedelta(hours=hours_ago), "expires_at": NOW + timedelta(hours=expires_in_hours),
    }


def test_recent_merges_both_kinds_newest_first_and_counts_the_window():
    conn = FakeConn([narrative(19, 5), narrative(18, 50)], [hypothesis("dh-aaa111", 1), hypothesis("dh-bbb222", 20)])
    result = dq.recent(conn, kind=None, since=None, limit=3, now=NOW)
    assert result.ok and result.operation == "dreams" and result.total_available == 4
    assert [i.id for i in result.items] == ["dh-aaa111", "dream:19", "dh-bbb222"]
    assert {i.kind for i in result.items} == {"dream_narrative", "dream_hypothesis"}
    assert all(i.epistemic_status == "unsettled" for i in result.items)


def test_recent_kind_filter_queries_one_table():
    conn = FakeConn([narrative(19, 5)], [hypothesis("dh-aaa111", 1)])
    result = dq.recent(conn, kind="narrative", since=None, limit=5, now=NOW)
    assert [i.id for i in result.items] == ["dream:19"]
    assert not any("dream_hypothesis" in sql for sql, _ in conn.calls)


def test_empty_window_is_ok_and_zero():
    result = dq.recent(FakeConn(), kind=None, since=NOW, limit=5, now=NOW)
    assert result.ok and result.items == [] and result.total_available == 0


def test_narrative_item_shape_and_utc():
    row = narrative(19, 5, themes=["x" * 200] + [f"t{i}" for i in range(12)])
    row["occurred_at"] = row["occurred_at"].replace(tzinfo=None)
    [item] = dq.recent(FakeConn([row]), kind=None, since=None, limit=5, now=NOW).items
    assert item.text == "A dream.\n\nIt went on." and item.occurred_at.tzinfo is not None
    assert item.extra["dream_date"] == "2026-09-28"
    assert len(item.extra["themes"]) == 8 and len(item.extra["themes"][0]) == 80


def test_hypothesis_item_has_no_arm_or_refs_and_flags_expiry():
    conn = FakeConn([], [hypothesis("dh-aaa111", 1, expires_in_hours=-1)])
    [item] = dq.recent(conn, kind=None, since=None, limit=5, now=NOW).items
    assert item.text == "claim dh-aaa111\nWhy: Both mention RPC."
    assert set(item.extra) == {"cycle_id", "expired"} and item.extra["expired"] is True


def test_every_hypothesis_statement_is_offered_only_and_never_selects_arm_or_refs():
    assert dq.HYPOTHESIS_SQL
    for sql in dq.HYPOTHESIS_SQL:
        assert "h.offered_at IS NOT NULL" in sql
        selected = re.search(r"SELECT(.*?)FROM", sql, re.S).group(1)
        for column in ("arm", "ref_a", "ref_b"):
            assert not re.search(rf"\b{column}\b", selected), (column, sql)


def test_by_ids_keeps_rank_order_attaches_similarity_and_drops_missing():
    conn = FakeConn([narrative(19, 5)], [hypothesis("dh-aaa111", 1)])
    scored = [("dh-aaa111", 0.81), ("dream:7", 0.8), ("dream:19", 0.7)]
    result = dq.by_ids(conn, scored, kind=None, since=None, limit=5, now=NOW)
    assert [(i.id, i.extra["similarity"]) for i in result.items] == [("dh-aaa111", 0.81), ("dream:19", 0.7)]
    assert result.total_available == 2


def test_by_ids_respects_kind_and_limit():
    conn = FakeConn([narrative(19, 5), narrative(18, 6)], [hypothesis("dh-aaa111", 1)])
    scored = [("dh-aaa111", 0.9), ("dream:19", 0.8), ("dream:18", 0.7)]
    result = dq.by_ids(conn, scored, kind="narrative", since=None, limit=1, now=NOW)
    assert [i.id for i in result.items] == ["dream:19"] and result.total_available == 2


def test_by_ids_skips_empty_id_lists():
    conn = FakeConn([narrative(19, 5)])
    dq.by_ids(conn, [("dream:19", 0.9)], kind=None, since=None, limit=5, now=NOW)
    assert not any("dream_hypothesis" in sql for sql, _ in conn.calls)


def test_one_returns_full_text_or_empty():
    long_story = "s" * 5000
    conn = FakeConn([narrative(19, 5, story=long_story)], [hypothesis("dh-aaa111", 1)])
    [item] = dq.one(conn, "dream:19", now=NOW).items
    assert len(item.text) == 4000 and item.truncated
    assert dq.one(conn, "dh-aaa111", now=NOW).items[0].id == "dh-aaa111"
    missing = dq.one(conn, "dh-fffffff", now=NOW)
    assert missing.ok and missing.items == [] and missing.total_available == 0


def test_index_rows_pairs_kinds_and_scans_offered_hypotheses_only():
    conn = FakeConn([narrative(19, 5)], [hypothesis("dh-aaa111", 1)])
    pairs = dq.index_rows(conn)
    assert [(k, dq.doc_id(k, r)) for k, r in pairs] == [("narrative", "dream:19"), ("hypothesis", "dh-aaa111")]
    assert dq.split_ids(["dream:3", "dh-abc123", "dream:x"]) == ([3], ["dh-abc123"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest services/orion-dream/tests/test_introspect_dreams.py -q`
Expected: FAIL with `ImportError: cannot import name 'introspect_dreams'`.

- [ ] **Step 3: Create** `services/orion-dream/app/introspect_dreams.py`:

```python
"""Read-only dream lookups for the orion-introspect ``dreams`` tool.

Two record kinds: narrative dreams (``dreams``, written by orion-sql-writer)
and sleep-cycle hypotheses (``dream_hypothesis``, written by the v2 cycle).
Hypotheses are a blind experiment (orion/dream/hypotheses.py): curiosity shows
each one once with the arm hidden. So only hypotheses already offered to Orion
are ever returned, from both arms, and ``arm`` / ``ref_a`` / ``ref_b`` are
never selected. Every statement here is a SELECT.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Literal

from sqlalchemy import text

from orion.schemas.introspect import (
    DEFAULT_TEXT_CAP,
    FULL_TEXT_CAP,
    SHORT_FIELD_CAP,
    THEME_CAP,
    IntrospectItemV1,
    IntrospectResultV1,
    clip_text,
)

Kind = Literal["narrative", "hypothesis"]
NARRATIVE_PREFIX = "dream:"
THEME_CHARS = 80
INDEX_SCAN_LIMIT = 1000

# dreams.created_at is timestamp without time zone written by now() on a UTC
# server; AT TIME ZONE 'UTC' turns it into the timestamptz it always meant.
_N_OCCURRED = "(d.created_at AT TIME ZONE 'UTC')"
_N_COLS = f"d.id, d.dream_date, d.tldr, d.themes, d.narrative, {_N_OCCURRED} AS occurred_at"
_H_COLS = "h.hypothesis_id, h.cycle_id, h.claim, h.why, h.offered_at AS occurred_at, h.expires_at"
_SINCE = "(CAST(:since AS timestamptz) IS NULL OR {col} >= CAST(:since AS timestamptz))"

NARRATIVE_RECENT_SQL = f"""
SELECT {_N_COLS}, count(*) OVER () AS total FROM dreams d
WHERE d.created_at IS NOT NULL AND {_SINCE.format(col=_N_OCCURRED)}
ORDER BY occurred_at DESC, d.id DESC
LIMIT :limit
"""

NARRATIVE_BY_IDS_SQL = f"""
SELECT {_N_COLS} FROM dreams d
WHERE d.id = ANY(:ids) AND d.created_at IS NOT NULL AND {_SINCE.format(col=_N_OCCURRED)}
"""

HYPOTHESIS_RECENT_SQL = f"""
SELECT {_H_COLS}, count(*) OVER () AS total FROM dream_hypothesis h
WHERE h.offered_at IS NOT NULL AND {_SINCE.format(col="h.offered_at")}
ORDER BY h.offered_at DESC, h.hypothesis_id DESC
LIMIT :limit
"""

HYPOTHESIS_BY_IDS_SQL = f"""
SELECT {_H_COLS} FROM dream_hypothesis h
WHERE h.hypothesis_id = ANY(:ids) AND h.offered_at IS NOT NULL AND {_SINCE.format(col="h.offered_at")}
"""

HYPOTHESIS_SQL = (HYPOTHESIS_RECENT_SQL, HYPOTHESIS_BY_IDS_SQL)


def _rows(conn: Any, sql: str, **params: Any) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(text(sql), params).mappings().all()]


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def doc_id(kind: Kind, row: dict[str, Any]) -> str:
    return f"{NARRATIVE_PREFIX}{row['id']}" if kind == "narrative" else str(row["hypothesis_id"])


def split_ids(ids: Iterable[str]) -> tuple[list[int], list[str]]:
    narratives, hypotheses = [], []
    for item_id in ids:
        if item_id.startswith(NARRATIVE_PREFIX):
            tail = item_id[len(NARRATIVE_PREFIX):]
            if tail.isdigit():
                narratives.append(int(tail))
        elif item_id.startswith("dh-"):
            hypotheses.append(item_id)
    return narratives, hypotheses


def narrative_item(row: dict[str, Any], *, text_cap: int = DEFAULT_TEXT_CAP) -> IntrospectItemV1:
    body = "\n\n".join(p for p in (str(row["tldr"] or "").strip(), str(row["narrative"] or "").strip()) if p)
    body_text, truncated = clip_text(body, text_cap)
    themes = row["themes"] if isinstance(row["themes"], list) else []
    return IntrospectItemV1(
        id=doc_id("narrative", row), occurred_at=_aware(row["occurred_at"]), kind="dream_narrative",
        epistemic_status="unsettled", text=body_text, truncated=truncated,
        extra={
            "dream_date": row["dream_date"].isoformat() if row["dream_date"] else None,
            "themes": [clip_text(str(t), THEME_CHARS)[0] for t in themes if str(t).strip()][:THEME_CAP],
        },
    )


def hypothesis_item(
    row: dict[str, Any], *, now: datetime, text_cap: int = DEFAULT_TEXT_CAP,
) -> IntrospectItemV1:
    claim, why = str(row["claim"] or "").strip(), str(row["why"] or "").strip()
    body_text, truncated = clip_text(f"{claim}\nWhy: {why}" if why else claim, text_cap)
    expires = row["expires_at"]
    return IntrospectItemV1(
        id=doc_id("hypothesis", row), occurred_at=_aware(row["occurred_at"]), kind="dream_hypothesis",
        epistemic_status="unsettled", text=body_text, truncated=truncated,
        extra={
            "cycle_id": clip_text(str(row["cycle_id"] or ""), SHORT_FIELD_CAP)[0],
            "expired": bool(expires is not None and _aware(expires) <= now),
        },
    )


def _result(items: list[IntrospectItemV1], total: int, now: datetime) -> IntrospectResultV1:
    return IntrospectResultV1(ok=True, operation="dreams", as_of=now, total_available=total, items=items)


def recent(
    conn: Any, *, kind: Kind | None, since: datetime | None, limit: int, now: datetime,
) -> IntrospectResultV1:
    items: list[IntrospectItemV1] = []
    total = 0
    if kind in (None, "narrative"):
        rows = _rows(conn, NARRATIVE_RECENT_SQL, since=since, limit=limit)
        total += int(rows[0]["total"]) if rows else 0
        items += [narrative_item(r) for r in rows]
    if kind in (None, "hypothesis"):
        rows = _rows(conn, HYPOTHESIS_RECENT_SQL, since=since, limit=limit)
        total += int(rows[0]["total"]) if rows else 0
        items += [hypothesis_item(r, now=now) for r in rows]
    items.sort(key=lambda i: (i.occurred_at, i.id), reverse=True)
    return _result(items[:limit], total, now)


def by_ids(
    conn: Any, scored: list[tuple[str, float]], *,
    kind: Kind | None, since: datetime | None, limit: int, now: datetime,
) -> IntrospectResultV1:
    """Re-read ranked hits from Postgres (the index is not the record); rank order kept."""
    narrative_ids, hypothesis_ids = split_ids(sid for sid, _ in scored)
    found: dict[str, IntrospectItemV1] = {}
    if narrative_ids and kind in (None, "narrative"):
        for r in _rows(conn, NARRATIVE_BY_IDS_SQL, ids=narrative_ids, since=since):
            found[doc_id("narrative", r)] = narrative_item(r)
    if hypothesis_ids and kind in (None, "hypothesis"):
        for r in _rows(conn, HYPOTHESIS_BY_IDS_SQL, ids=hypothesis_ids, since=since):
            found[doc_id("hypothesis", r)] = hypothesis_item(r, now=now)
    hits = [
        found[sid].model_copy(update={"extra": {**found[sid].extra, "similarity": round(sim, 3)}})
        for sid, sim in scored if sid in found
    ]
    return _result(hits[:limit], len(hits), now)


def one(conn: Any, dream_id: str, *, now: datetime) -> IntrospectResultV1:
    narrative_ids, hypothesis_ids = split_ids([dream_id])
    items: list[IntrospectItemV1] = []
    if narrative_ids:
        items = [narrative_item(r, text_cap=FULL_TEXT_CAP)
                 for r in _rows(conn, NARRATIVE_BY_IDS_SQL, ids=narrative_ids, since=None)]
    elif hypothesis_ids:
        items = [hypothesis_item(r, now=now, text_cap=FULL_TEXT_CAP)
                 for r in _rows(conn, HYPOTHESIS_BY_IDS_SQL, ids=hypothesis_ids, since=None)]
    return _result(items[:1], min(len(items), 1), now)


def index_rows(conn: Any) -> list[tuple[Kind, dict[str, Any]]]:
    pairs: list[tuple[Kind, dict[str, Any]]] = []
    pairs += [("narrative", r) for r in _rows(conn, NARRATIVE_RECENT_SQL, since=None, limit=INDEX_SCAN_LIMIT)]
    pairs += [("hypothesis", r) for r in _rows(conn, HYPOTHESIS_RECENT_SQL, since=None, limit=INDEX_SCAN_LIMIT)]
    return pairs
```

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest services/orion-dream/tests/test_introspect_dreams.py -q`
Expected: all pass.

- [ ] **Step 5: Live read-only SQL check.** This runs the real statements against the real database, as SELECTs inside a read-only transaction. Run from the worktree root:

```bash
PYTHONPATH=.:services/orion-dream $PY - <<'EOF'
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from app import introspect_dreams as dq
eng = create_engine("postgresql://postgres:postgres@127.0.0.1:55432/conjourney")
now = datetime.now(timezone.utc)
with eng.connect() as c:
    c.execute(text("SET TRANSACTION READ ONLY"))
    r = dq.recent(c, kind=None, since=None, limit=5, now=now)
    print("recent", r.total_available, [(i.id, i.kind, i.occurred_at.isoformat()) for i in r.items])
    print("one", [(i.id, len(i.text), i.truncated) for i in dq.one(c, "dream:19", now=now).items])
    pairs = dq.index_rows(c)
    print("index", sum(k == "narrative" for k, _ in pairs), sum(k == "hypothesis" for k, _ in pairs))
    offered = c.execute(text("SELECT count(*) FROM dream_hypothesis WHERE offered_at IS NOT NULL")).scalar()
    print("offered in db", offered)
EOF
```

Expected:
- `recent` prints a total about 19 + offered, and 5 items with tz-aware timestamps.
- `one` prints `dream:19` with text ≤4000.
- The `index` hypothesis count equals `offered in db`: 34 as of 2026-09-29, and never the full 49.

Paste the output into the task report.

- [ ] **Step 6: Commit**

```bash
git add services/orion-dream/app/introspect_dreams.py services/orion-dream/tests/test_introspect_dreams.py
git commit -m "feat(orion-dream): read-only dream lookups for introspect (offered hypotheses only)"
```

---

### Task 4: orion-dream search module

**Files:**
- Create: `services/orion-dream/app/dream_search.py`, `services/orion-dream/tests/test_dream_search.py`

**Interfaces:**
- Consumes:
  - `orion.introspect.semantic_index`: `SearchConfig`, `IndexPass`, `content_hash`, `embed`, `nearest`, `stored_hashes`, `publish_upsert`
  - `app.introspect_dreams`: `doc_id`
- Produces:
  - Constants: `CANDIDATES=20`, `INDEX_TEXT_CHARS=1800`
  - `document_text(kind, row) -> str | None`
  - `async index_missing(pairs, cfg, *, client, bus, source, batch=None) -> IndexPass`
  - `async rank(client, cfg, query) -> list[tuple[str, float]]`

- [ ] **Step 1: Write the failing test** `services/orion-dream/tests/test_dream_search.py`:

```python
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
```

The distances are l2, since the collection metadata is None: 0.5 → 0.75 passes, 0.9 → 0.55 fails.

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest services/orion-dream/tests/test_dream_search.py -q`
Expected: FAIL with `ImportError: cannot import name 'dream_search'`.

- [ ] **Step 3: Create** `services/orion-dream/app/dream_search.py`:

```python
"""Search dreams by meaning. Index text per kind; Chroma is never the record.

Only rows from app.introspect_dreams.index_rows reach the index, so a
never-offered hypothesis is never embedded (blind experiment).
"""
from __future__ import annotations

from typing import Any

import httpx

from app.introspect_dreams import doc_id
from orion.introspect.semantic_index import (
    IndexPass,
    SearchConfig,
    content_hash,
    embed,
    nearest,
    publish_upsert,
    stored_hashes,
)
from orion.schemas.introspect import clip_text

CANDIDATES = 20
INDEX_TEXT_CHARS = 1800
_DOC_PREFIX = "dream-search"


def document_text(kind: str, row: dict[str, Any]) -> str | None:
    if kind == "narrative":
        themes = row.get("themes") if isinstance(row.get("themes"), list) else []
        parts = [
            str(row.get("tldr") or "").strip(),
            f"Themes: {', '.join(str(t) for t in themes)}" if themes else "",
            str(row.get("narrative") or "").strip(),
        ]
    else:
        claim, why = str(row.get("claim") or "").strip(), str(row.get("why") or "").strip()
        if not claim:
            return None
        return clip_text(f"{claim}\nWhy: {why}" if why else claim, INDEX_TEXT_CHARS)[0]
    body = "\n\n".join(p for p in parts if p)
    return clip_text(body, INDEX_TEXT_CHARS)[0] if body else None


async def index_missing(
    pairs: list[tuple[str, dict[str, Any]]],
    cfg: SearchConfig,
    *,
    client: httpx.AsyncClient,
    bus: Any,
    source: Any,
    batch: int | None = None,
) -> IndexPass:
    """Embed and upsert dreams the index lacks or holds stale text for."""
    docs = []
    for kind, row in pairs:
        text = document_text(kind, row)
        if text:
            docs.append((kind, row, doc_id(kind, row), text, content_hash(text)))
    if not docs:
        return IndexPass(indexed=0, pending=0)
    stored = await stored_hashes(client, cfg, [d[2] for d in docs])
    stale = [d for d in docs if stored.get(d[2]) != d[4]]
    batch = cfg.index_batch if batch is None else batch
    for kind, row, did, text, digest in stale[:batch]:
        vector, model = await embed(client, cfg, text, doc_prefix=_DOC_PREFIX)
        occurred = row.get("occurred_at")
        await publish_upsert(
            bus, source, cfg, doc_id=did, text=text, vector=vector, model=model,
            meta={"kind": kind, "content_hash": digest, "occurred_at": occurred.isoformat() if occurred else ""},
        )
    done = min(len(stale), batch)
    return IndexPass(indexed=done, pending=len(stale) - done)


async def rank(client: httpx.AsyncClient, cfg: SearchConfig, query: str) -> list[tuple[str, float]]:
    """Embed the query once; (doc_id, similarity) at or above the floor, best first."""
    vector, _ = await embed(client, cfg, query, doc_prefix=_DOC_PREFIX)
    return [s for s in await nearest(client, cfg, vector, CANDIDATES) if s[1] >= cfg.min_similarity]
```

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest services/orion-dream/tests/test_dream_search.py services/orion-dream/tests/test_introspect_dreams.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add services/orion-dream/app/dream_search.py services/orion-dream/tests/test_dream_search.py
git commit -m "feat(orion-dream): dream search (index text per kind, floor-gated rank)"
```

---

### Task 5: orion-dream responder, settings, env, lifespan

**Files:**
- Create: `services/orion-dream/app/introspect_listener.py`, `services/orion-dream/tests/test_introspect_listener.py`
- Modify: `services/orion-dream/app/settings.py`, `services/orion-dream/app/main.py`, `services/orion-dream/.env_example`, `services/orion-dream/docker-compose.yml`, `scripts/check_env_key_single_source.py`

**Interfaces:**
- Consumes:
  - Task 2: `DREAM_REQUEST_CHANNEL`, `RESULT_PREFIX`, `REQUEST_KIND`, `RESULT_KIND`, `IntrospectRequestV1`, `DreamsArguments`
  - Task 3: `recent`, `by_ids`, `one`, `index_rows`
  - Task 4: `rank`, `index_missing`
- Produces:
  - `DreamIntrospectListener(*, bus_url, engine_provider, source, search, bus_factory=OrionBusAsync)`, with methods `handle(envelope)`, `start()`, `stop()`, `index_once()`
  - `build_listener() -> DreamIntrospectListener` (from settings)
  - Constants: `QUERY_UNAVAILABLE = "dreams_unavailable; answer unknown"`, `SEARCH_UNAVAILABLE = "dream_search_unavailable; answer unknown"`

- [ ] **Step 1: Write the failing test** `services/orion-dream/tests/test_introspect_listener.py`:

```python
"""Dream responder: trusted reply path, modes, read-only, empty vs unknown."""
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import uuid4

from app import introspect_listener as il
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.introspect.semantic_index import SearchConfig
from orion.introspect.transport import REQUEST_KIND, RESULT_KIND, RESULT_PREFIX
from orion.schemas.introspect import IntrospectRequestV1, IntrospectResultV1, IntrospectToolBindingV1

NOW = datetime(2026, 9, 29, 6, 0, tzinfo=timezone.utc)
BINDING = IntrospectToolBindingV1(
    invocation_context="unified_chat", parent_run_id="r", parent_trace_id="t", memory_allowed=True,
)
SEARCH = SearchConfig(chroma_url="http://c", embed_url="http://e", collection="orion_dreams", min_similarity=0.6)
NARR = {"id": 19, "dream_date": None, "tldr": "A dream.", "narrative": "", "themes": [], "occurred_at": NOW}


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class FakeConn:
    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def execute(self, clause, params=None):
        sql = str(clause)
        self.calls.append(sql)
        if "FROM dreams" in sql:
            return _Result([{**r, "total": len(self.rows)} for r in self.rows])
        return _Result([])


class FakeEngine:
    def __init__(self, rows=(), fail=False):
        self.conn, self.fail = FakeConn(list(rows)), fail

    @contextmanager
    def connect(self):
        if self.fail:
            raise ConnectionError("db down")
        yield self.conn


class Bus:
    def __init__(self):
        self.published = []

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))


def _listener(engine=None, search=SEARCH):
    listener = il.DreamIntrospectListener(
        bus_url="redis://x", engine_provider=lambda: engine or FakeEngine([NARR]),
        source=ServiceRef(name="orion-dream"), search=search,
    )
    listener.bus = Bus()
    return listener


def _envelope(args, *, reply=None, kind=REQUEST_KIND):
    corr = uuid4()
    payload = IntrospectRequestV1(operation="dreams", binding=BINDING, args=args).model_dump(mode="json")
    return BaseEnvelope(
        kind=kind, correlation_id=corr, reply_to=reply or f"{RESULT_PREFIX}{corr}",
        source=ServiceRef(name="orion-harness-governor"), payload=payload,
    )


def _handle(listener, envelope):
    asyncio.run(listener.handle(envelope))
    return listener.bus.published


def test_untrusted_reply_or_kind_is_ignored():
    listener = _listener()
    assert _handle(listener, _envelope({}, reply="orion:somewhere:else")) == []
    assert _handle(listener, _envelope({}, kind="reading.tool.request.v1")) == []


def test_recent_replies_on_derived_channel_read_only():
    engine = FakeEngine([NARR])
    listener = _listener(engine)
    env = _envelope({"kind": "narrative"})
    [(channel, reply)] = _handle(listener, env)
    assert channel == f"{RESULT_PREFIX}{env.correlation_id}" and reply.kind == RESULT_KIND
    assert str(reply.correlation_id) == str(env.correlation_id)
    result = IntrospectResultV1.model_validate(reply.payload)
    assert result.ok and [i.id for i in result.items] == ["dream:19"]
    assert engine.conn.calls[0] == "SET TRANSACTION READ ONLY"


def test_invalid_arguments_are_a_failed_result():
    [(_, reply)] = _handle(_listener(), _envelope({"arm": "dream"}))
    result = IntrospectResultV1.model_validate(reply.payload)
    assert not result.ok and result.error.startswith("invalid dreams request")


def test_database_failure_is_unknown_not_empty():
    [(_, reply)] = _handle(_listener(FakeEngine(fail=True)), _envelope({}))
    result = IntrospectResultV1.model_validate(reply.payload)
    assert not result.ok and result.error == il.QUERY_UNAVAILABLE


def test_search_not_configured_is_unknown():
    off = SearchConfig(chroma_url="", embed_url="", collection="orion_dreams", min_similarity=0.6)
    [(_, reply)] = _handle(_listener(search=off), _envelope({"query": "vision"}))
    assert IntrospectResultV1.model_validate(reply.payload).error == il.SEARCH_UNAVAILABLE


def test_search_with_no_hits_is_empty_and_skips_postgres(monkeypatch):
    async def no_hits(client, cfg, query):
        return []
    monkeypatch.setattr(il, "rank", no_hits)
    engine = FakeEngine([NARR])
    [(_, reply)] = _handle(_listener(engine), _envelope({"query": "vision"}))
    result = IntrospectResultV1.model_validate(reply.payload)
    assert result.ok and result.items == [] and result.total_available == 0
    assert engine.conn.calls == []


def test_search_hits_are_regated_with_similarity(monkeypatch):
    async def hits(client, cfg, query):
        return [("dream:19", 0.8)]
    monkeypatch.setattr(il, "rank", hits)

    class ByIdConn(FakeConn):
        def execute(self, clause, params=None):
            sql = str(clause)
            self.calls.append(sql)
            return _Result([NARR] if "ANY(:ids)" in sql and "FROM dreams" in sql else [])

    engine = FakeEngine()
    engine.conn = ByIdConn([])
    [(_, reply)] = _handle(_listener(engine), _envelope({"query": "vision"}))
    [item] = IntrospectResultV1.model_validate(reply.payload).items
    assert item.id == "dream:19" and item.extra["similarity"] == 0.8


def test_one_mode():
    [(_, reply)] = _handle(_listener(), _envelope({"dream_id": "dream:19"}))
    assert IntrospectResultV1.model_validate(reply.payload).items[0].id == "dream:19"
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest services/orion-dream/tests/test_introspect_listener.py -q`
Expected: FAIL with `ImportError: cannot import name 'introspect_listener'`.

- [ ] **Step 3: Settings.** In `services/orion-dream/app/settings.py`, add after the `DREAM_LLM_TIMEOUT_SEC` line:

```python
    # --- Introspect responder (orion-introspect `dreams` tool; read-only) ---
    # Answers orion:introspect:dream:request. Empty search URLs keep recent/one
    # working and make query= answer "unknown".
    DREAM_INTROSPECT_ENABLED: bool = Field(default=True)
    DREAM_SEARCH_CHROMA_URL: str = Field(default="")
    DREAM_SEARCH_EMBED_URL: str = Field(default="")
    DREAM_SEARCH_COLLECTION: str = Field(default="orion_dreams")
    DREAM_SEARCH_MIN_SIMILARITY: float = Field(default=0.60, ge=0.0, le=1.0)
    DREAM_SEARCH_INDEX_INTERVAL_SEC: float = Field(default=300.0, gt=0.0)
    DREAM_SEARCH_INDEX_BATCH: int = Field(default=10, ge=1, le=50)
```

- [ ] **Step 4: Create** `services/orion-dream/app/introspect_listener.py`:

```python
"""orion-dream's introspect responder: answers the `dreams` tool over the bus.

Trust: replies only when reply_to is exactly orion:introspect:result:<corr> and
the kind matches; never to a model-supplied subject. Every connection is a
read-only transaction. Errors become ok=false ("answer unknown"), never [].
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, Callable

import httpx
from sqlalchemy import text

from app.dream_search import index_missing, rank
from app.introspect_dreams import by_ids, index_rows, one, recent
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope
from orion.introspect.semantic_index import HTTP_TIMEOUT_SEC, SearchConfig, SearchUnavailableError
from orion.introspect.transport import DREAM_REQUEST_CHANNEL, REQUEST_KIND, RESULT_KIND, RESULT_PREFIX
from orion.schemas.introspect import DreamsArguments, IntrospectRequestV1, IntrospectResultV1

logger = logging.getLogger("orion-dream.introspect")

QUERY_UNAVAILABLE = "dreams_unavailable; answer unknown"
SEARCH_UNAVAILABLE = "dream_search_unavailable; answer unknown"
_INVALID_CAP = 400


def _failed(now: datetime, error: str) -> IntrospectResultV1:
    return IntrospectResultV1(ok=False, operation="dreams", as_of=now, error=error)


class DreamIntrospectListener:
    def __init__(
        self, *, bus_url: str, engine_provider: Callable[[], Any], source: Any,
        search: SearchConfig | None, bus_factory: Callable[[str], Any] = OrionBusAsync,
    ):
        self.bus_url = bus_url
        self.engine_provider = engine_provider
        self.source = source
        self.search = search
        self.bus_factory = bus_factory
        self.bus: Any = None
        self._ready = asyncio.Event()
        self.task: asyncio.Task | None = None
        self.index_task: asyncio.Task | None = None

    def _read(self, fn: Callable[[Any], IntrospectResultV1]) -> Any:
        with self.engine_provider().connect() as conn:
            conn.execute(text("SET TRANSACTION READ ONLY"))
            return fn(conn)

    async def handle(self, envelope: BaseEnvelope) -> None:
        reply = f"{RESULT_PREFIX}{envelope.correlation_id}"
        if envelope.reply_to != reply or envelope.kind != REQUEST_KIND:
            return
        now = datetime.now(timezone.utc)
        try:
            request = IntrospectRequestV1.model_validate(envelope.payload)
            args = DreamsArguments.model_validate(request.args)
        except ValueError as exc:
            await self._publish(reply, envelope, _failed(now, f"invalid dreams request: {str(exc)[:_INVALID_CAP]}"))
            return
        mode = "one" if args.dream_id else "search" if args.query else "recent"
        try:
            if args.dream_id is not None:
                result = await asyncio.to_thread(self._read, lambda c: one(c, args.dream_id, now=now))
            elif args.query is not None:
                if self.search is None or not self.search.enabled:
                    raise SearchUnavailableError("dream search is not configured")
                async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as client:
                    scored = await rank(client, self.search, args.query)
                if not scored:
                    result = IntrospectResultV1(ok=True, operation="dreams", as_of=now, total_available=0)
                else:
                    result = await asyncio.to_thread(self._read, lambda c: by_ids(
                        c, scored, kind=args.kind, since=args.since, limit=args.limit, now=now,
                    ))
            else:
                result = await asyncio.to_thread(self._read, lambda c: recent(
                    c, kind=args.kind, since=args.since, limit=args.limit, now=now,
                ))
        except Exception as exc:
            search_failed = isinstance(exc, SearchUnavailableError)
            logger.warning(
                "introspect_failed op=dreams corr=%s mode=%s category=%s exc_type=%s detail=%s",
                envelope.correlation_id, mode,
                "dream_search_failure" if search_failed else "dream_query_failure",
                type(exc).__name__, str(exc).replace("\n", " ")[:300],
            )
            result = _failed(now, SEARCH_UNAVAILABLE if search_failed else QUERY_UNAVAILABLE)
        else:
            logger.info(
                "introspect op=dreams corr=%s mode=%s items=%d total=%s",
                envelope.correlation_id, mode, len(result.items), result.total_available,
            )
        await self._publish(reply, envelope, result)

    async def _publish(self, reply: str, envelope: BaseEnvelope, result: IntrospectResultV1) -> None:
        await self.bus.publish(reply, BaseEnvelope(
            kind=RESULT_KIND, correlation_id=envelope.correlation_id,
            source=self.source, payload=result.model_dump(mode="json"),
        ))

    async def index_once(self):
        pairs = await asyncio.to_thread(self._read, index_rows)
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as client:
            result = await index_missing(pairs, self.search, client=client, bus=self.bus, source=self.source)
        logger.info("dream_search_index indexed=%d pending=%d", result.indexed, result.pending)
        return result

    async def _index_loop(self) -> None:
        while True:
            await self._ready.wait()
            try:
                await self.index_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("dream_search_index_failed exc_type=%s detail=%s",
                               type(exc).__name__, str(exc).replace("\n", " ")[:300])
            await asyncio.sleep(self.search.index_interval_sec)

    async def _run(self) -> None:
        while True:
            bus = self.bus_factory(self.bus_url)
            try:
                await bus.connect()
                self.bus = bus
                self._ready.set()
                async with bus.subscribe(DREAM_REQUEST_CHANNEL) as pubsub:
                    logger.info("dream_introspect_listening channel=%s", DREAM_REQUEST_CHANNEL)
                    async for raw in bus.iter_messages(pubsub):
                        try:
                            decoded = bus.codec.decode(raw.get("data"))
                            if decoded.ok:
                                await self.handle(decoded.envelope)
                        except Exception:
                            logger.warning("dream_introspect_message_failed", exc_info=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("dream_introspect_reconnecting", exc_info=True)
            finally:
                self._ready.clear()
                with suppress(Exception):
                    await bus.close()
            await asyncio.sleep(1)

    async def start(self) -> None:
        self.task = asyncio.create_task(self._run(), name="dream-introspect")
        if self.search is not None and self.search.enabled:
            self.index_task = asyncio.create_task(self._index_loop(), name="dream-search-index")

    async def stop(self) -> None:
        for attr in ("task", "index_task"):
            task = getattr(self, attr)
            if task:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                setattr(self, attr, None)


def build_listener() -> DreamIntrospectListener:
    from sqlalchemy import create_engine

    from app.settings import settings
    from orion.core.bus.bus_schemas import ServiceRef

    engine = create_engine(settings.POSTGRES_URI, pool_pre_ping=True)
    return DreamIntrospectListener(
        bus_url=settings.ORION_BUS_URL,
        engine_provider=lambda: engine,
        source=ServiceRef(name="orion-dream", version=settings.SERVICE_VERSION, node=settings.NODE_NAME),
        search=SearchConfig(
            chroma_url=settings.DREAM_SEARCH_CHROMA_URL,
            embed_url=settings.DREAM_SEARCH_EMBED_URL,
            collection=settings.DREAM_SEARCH_COLLECTION,
            min_similarity=settings.DREAM_SEARCH_MIN_SIMILARITY,
            index_interval_sec=settings.DREAM_SEARCH_INDEX_INTERVAL_SEC,
            index_batch=settings.DREAM_SEARCH_INDEX_BATCH,
        ),
    )
```

- [ ] **Step 5: Run tests**

Run: `$PY -m pytest services/orion-dream/tests/test_introspect_listener.py services/orion-dream/tests/test_introspect_dreams.py services/orion-dream/tests/test_dream_search.py -q`
Expected: all pass.

- [ ] **Step 6: Lifespan.** In `services/orion-dream/app/main.py`, inside `lifespan`:
  - Right after `stop = asyncio.Event()`, add:

```python
    introspect = None
    if settings.DREAM_INTROSPECT_ENABLED and settings.ORION_BUS_ENABLED:
        from app.introspect_listener import build_listener

        introspect = build_listener()
        await introspect.start()
        logger.info("dream introspect responder started")
```

  - After the line `yield`, before `stop.set()`, add:

```python
    if introspect is not None:
        await introspect.stop()
```

- [ ] **Step 7: Env template, compose, single-source owner.**
  - `services/orion-dream/.env_example`: add after the `DREAM_LLM_TIMEOUT_SEC=90` line:

```text

# --- Introspect responder (orion-introspect `dreams` tool; read-only) ---
# Answers orion:introspect:dream:request from Postgres; query= searches the
# orion_dreams Chroma collection (kept current by a 5-minute index loop).
# Empty search URLs leave recent/one working and make query= answer "unknown".
DREAM_INTROSPECT_ENABLED=true
DREAM_SEARCH_CHROMA_URL=http://${PROJECT}-vector-db:8000
DREAM_SEARCH_EMBED_URL=http://${PROJECT}-vector-host:8320/embedding
DREAM_SEARCH_COLLECTION=orion_dreams
# Calibrated: services/orion-dream/evals/run_dream_search_calibration.py
DREAM_SEARCH_MIN_SIMILARITY=0.65
DREAM_SEARCH_INDEX_INTERVAL_SEC=300
DREAM_SEARCH_INDEX_BATCH=10
```

  - `services/orion-dream/docker-compose.yml`: add under `environment:` after the `DREAM_LLM_TIMEOUT_SEC` line:

```yaml
      - DREAM_INTROSPECT_ENABLED=${DREAM_INTROSPECT_ENABLED:-true}
      - DREAM_SEARCH_CHROMA_URL=${DREAM_SEARCH_CHROMA_URL:-}
      - DREAM_SEARCH_EMBED_URL=${DREAM_SEARCH_EMBED_URL:-}
      - DREAM_SEARCH_COLLECTION=${DREAM_SEARCH_COLLECTION:-orion_dreams}
      - DREAM_SEARCH_MIN_SIMILARITY=${DREAM_SEARCH_MIN_SIMILARITY:-0.65}
      - DREAM_SEARCH_INDEX_INTERVAL_SEC=${DREAM_SEARCH_INDEX_INTERVAL_SEC:-300}
      - DREAM_SEARCH_INDEX_BATCH=${DREAM_SEARCH_INDEX_BATCH:-10}
```

  - `scripts/check_env_key_single_source.py`: add to `OWNERS`:

```python
    "DREAM_SEARCH_MIN_SIMILARITY": "services/orion-dream/.env_example",
```

  If that checker flags the compose `:-0.60` default or the settings `default=0.60`, match how `HUB_READING_SEARCH_MIN_SIMILARITY` is handled in Hub's compose and settings (same pattern). Run the checker to confirm.

- [ ] **Step 8: Env sync and gates.** Run the sync from this worktree. It reads this worktree's `services/orion-dream/.env_example` and writes the live `.env`, which exists only in the primary checkout (`main_worktree_root()`). Name the service explicitly, since orion-dream is not in its default list:

```bash
cd /mnt/scripts/Orion-Sapienform-introspect-dreams
python3 scripts/sync_local_env_from_example.py orion-dream
PYTHONPATH=. $PY scripts/check_env_template_parity.py
PYTHONPATH=. $PY scripts/check_env_key_single_source.py
docker compose --env-file .env --env-file services/orion-dream/.env -f services/orion-dream/docker-compose.yml config >/dev/null && echo compose-ok
```

For the compose check to work, copy `/mnt/scripts/Orion-Sapienform/.env` and `/mnt/scripts/Orion-Sapienform/services/orion-dream/.env` into the worktree. They are gitignored; never stage them. Also confirm the shared checkout's `services/orion-dream/.env` gained the seven keys:

```bash
grep -n "^DREAM_SEARCH_\|^DREAM_INTROSPECT_" /mnt/scripts/Orion-Sapienform/services/orion-dream/.env
```

Expected: all gates exit 0, `compose-ok` prints, and seven keys are listed. Report any key the sync script skipped.

- [ ] **Step 9: Commit**

```bash
git add services/orion-dream/app/introspect_listener.py services/orion-dream/tests/test_introspect_listener.py services/orion-dream/app/settings.py services/orion-dream/app/main.py services/orion-dream/.env_example services/orion-dream/docker-compose.yml scripts/check_env_key_single_source.py
git status --short   # must not list any .env
git commit -m "feat(orion-dream): introspect responder + index loop for the dreams tool"
```

---

### Task 6: Harness tool, brief, smoke

**Files:**
- Modify: `orion/introspect/tools.py`, `orion/introspect/brief.py`, `scripts/smoke_introspect.py`
- Test: `orion/introspect/tests/test_introspect_tools.py`, `orion/introspect/tests/test_introspect_harness_wiring.py`

**Interfaces:**
- Consumes: Task 2 transport constants, `IntrospectRequestV1`, `DreamsArguments`, `IntrospectResultV1`.
- Produces:
  - `IntrospectTools.tool_specs()` returns `reading_results` then `dreams`.
  - `invoke("dreams", args)` returns `IntrospectResultV1` JSON or raises `IntrospectUnknownError`.
  - `DREAMS_DESCRIPTION` string.

- [ ] **Step 1: Write failing tests.** In `orion/introspect/tests/test_introspect_tools.py`, replace `test_tool_specs_list_only_reading_results` with the first test below, and append the rest:

```python
def test_tool_specs_list_reading_results_then_dreams():
    assert [s.name for s in IntrospectTools(ReplyBus(), BINDING).tool_specs()] == ["reading_results", "dreams"]


from orion.introspect.transport import DREAM_REQUEST_CHANNEL, REQUEST_KIND, RESULT_KIND, RESULT_PREFIX  # noqa: E402


class DreamBus(ReplyBus):
    def __init__(self, payload=None, *, kind=RESULT_KIND, **kw):
        super().__init__(payload, **kw)
        self.kind = kind

    async def rpc_request(self, channel, envelope, *, reply_channel, timeout_sec):
        self.sent.append((channel, envelope, reply_channel, timeout_sec))
        if self.raise_exc is not None:
            raise self.raise_exc
        reply = BaseEnvelope(
            kind=self.kind,
            correlation_id=uuid4() if self.wrong_correlation else envelope.correlation_id,
            source=ServiceRef(name="orion-dream"), payload=self.payload,
        )
        return {"data": self.codec.encode(reply)}


def _dream_ok(**kw):
    return IntrospectResultV1(ok=True, operation="dreams", as_of=NOW, total_available=0, **kw).model_dump(mode="json")


def test_dreams_uses_dream_channel_with_binding_and_clean_args():
    bus = DreamBus(_dream_ok())
    out = _invoke(bus, "dreams", {"query": " vision ", "limit": 2})
    assert out["ok"] is True and out["operation"] == "dreams"
    [(channel, envelope, reply_channel, timeout)] = bus.sent
    assert channel == DREAM_REQUEST_CHANNEL and envelope.kind == REQUEST_KIND
    assert envelope.reply_to == reply_channel == f"{RESULT_PREFIX}{envelope.correlation_id}"
    assert timeout == RPC_TIMEOUT_SEC and envelope.source.name == "orion-harness-governor"
    assert envelope.payload["operation"] == "dreams"
    assert envelope.payload["binding"]["parent_run_id"] == "run-1"
    assert envelope.payload["args"] == {"query": "vision", "limit": 2}


def test_dreams_rejects_bad_args_before_transport():
    bus = DreamBus(_dream_ok())
    with pytest.raises(ValidationError):
        _invoke(bus, "dreams", {"arm": "dream"})
    assert bus.sent == []


@pytest.mark.parametrize(
    "bus",
    [
        DreamBus(raise_exc=TimeoutError()),
        DreamBus(_dream_ok(), wrong_correlation=True),
        DreamBus(_dream_ok(), kind="reading.tool.result.v1"),
        DreamBus({"ok": True}),
        DreamBus(IntrospectResultV1(ok=False, operation="dreams", as_of=NOW, error="dreams_unavailable; answer unknown").model_dump(mode="json")),
        DreamBus(IntrospectResultV1(ok=True, operation="reading_result", as_of=NOW, total_available=0).model_dump(mode="json")),
    ],
)
def test_dreams_failures_are_unknown_never_empty(bus):
    with pytest.raises(IntrospectUnknownError, match="dreams: answer unknown"):
        _invoke(bus, "dreams", {})
```

In `orion/introspect/tests/test_introspect_harness_wiring.py`, append:

```python
def test_brief_covers_dreams_as_experiences_not_facts():
    from orion.introspect.brief import introspect_brief_lines
    from orion.schemas.introspect import IntrospectToolBindingV1

    binding = IntrospectToolBindingV1(
        invocation_context="unified_chat", parent_run_id="r", parent_trace_id="t", memory_allowed=False,
    )
    text = " ".join(introspect_brief_lines(binding))
    assert "dreams" in text and "not facts" in text and "unknown" in text
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=. $PY -m pytest orion/introspect/tests/test_introspect_tools.py orion/introspect/tests/test_introspect_harness_wiring.py -q`
Expected: FAIL (only `reading_results` listed; unknown tool `dreams`; brief lacks dreams).

- [ ] **Step 3: Edit** `orion/introspect/tools.py`.
  - Change the imports to:

```python
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.introspect.transport import DREAM_REQUEST_CHANNEL, REQUEST_KIND, RESULT_KIND, RESULT_PREFIX
from orion.schemas.introspect import (
    DreamsArguments,
    IntrospectRequestV1,
    IntrospectResultV1,
    IntrospectToolBindingV1,
    ReadingResultArguments,
)
```

  - Add after `READING_RESULTS_DESCRIPTION`:

```python
DREAMS_DESCRIPTION = (
    "Read back your own dreams. Two kinds come back, labeled: dream_narrative (the nightly "
    "dream story: tldr, themes, narrative) and dream_hypothesis (a speculative link a sleep "
    "cycle proposed between two memories, shown to you once during curiosity). Pass "
    "query=<what you want to recall, in plain words> to find dreams by meaning (items carry "
    "similarity 0-1; weak matches are dropped), dream_id=<id> for one dream in full, or "
    "nothing for your most recent. kind=narrative|hypothesis narrows; since=<ISO timestamp "
    "with timezone> narrows query or recent mode; limit up to 5. Dreams are experiences you "
    "had, not facts about the world (epistemic_status=unsettled). items=[] means no dream "
    "matched; a tool error means the answer is unknown, never that you did not dream."
)
```

  - Replace `tool_specs` and `invoke` with:

```python
    def tool_specs(self) -> list[ToolSpec]:
        return [
            ToolSpec("reading_results", READING_RESULTS_DESCRIPTION, ReadingResultArguments),
            ToolSpec("dreams", DREAMS_DESCRIPTION, DreamsArguments),
        ]

    async def invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        # Validate before transport: extra fields are rejected, never ignored.
        if name == "dreams":
            dream_args = DreamsArguments.model_validate(arguments)
            result = await self._introspect_rpc(
                DREAM_REQUEST_CHANNEL, "dreams", dream_args.model_dump(mode="json", exclude_none=True),
            )
            return result.model_dump(mode="json")
        if name != "reading_results":
            raise ValueError(f"unknown introspect tool: {name}")
        args = ReadingResultArguments.model_validate(arguments)
        command = ReadingToolRequestV1(
            operation="reading_result",
            request_id=args.request_id,
            url=normalize_reading_source(args.url) if args.url is not None else None,
            query=args.query,
            limit=args.limit,
            since=args.since,
        )
        return (await self._reading_rpc(command)).model_dump(mode="json")

    async def _introspect_rpc(self, channel: str, operation: str, args: dict[str, Any]) -> IntrospectResultV1:
        correlation_id = uuid4()
        reply = f"{RESULT_PREFIX}{correlation_id}"
        request = IntrospectRequestV1(operation=operation, binding=self.binding, args=args)
        try:
            raw = await self.bus.rpc_request(
                channel,
                BaseEnvelope(
                    kind=REQUEST_KIND, correlation_id=correlation_id, reply_to=reply,
                    source=ServiceRef(name="orion-harness-governor"),
                    payload=request.model_dump(mode="json"),
                ),
                reply_channel=reply, timeout_sec=RPC_TIMEOUT_SEC,
            )
        except Exception as exc:
            raise IntrospectUnknownError(
                f"{operation}: answer unknown (no reply: {type(exc).__name__})"
            ) from exc
        decoded = self.bus.codec.decode(raw.get("data"))
        if (
            not decoded.ok
            or str(decoded.envelope.correlation_id) != str(correlation_id)
            or decoded.envelope.kind != RESULT_KIND
        ):
            raise IntrospectUnknownError(f"{operation}: answer unknown (malformed reply)")
        try:
            result = IntrospectResultV1.model_validate(decoded.envelope.payload)
        except ValueError as exc:
            raise IntrospectUnknownError(f"{operation}: answer unknown (malformed result)") from exc
        if not result.ok:
            raise IntrospectUnknownError(f"{operation}: answer unknown ({result.error})")
        if result.operation != operation:
            raise IntrospectUnknownError(f"{operation}: answer unknown (mismatched result)")
        return result
```

- [ ] **Step 4: Edit** `orion/introspect/brief.py`. Append a second string to the list returned by `introspect_brief_lines`, after the existing reading string:

```python
        (
            "dreams reads back your own dreams: dream_narrative (the nightly story) and "
            "dream_hypothesis (a link a sleep cycle proposed, already shown to you once). Use "
            "query=<topic in plain words> to find dreams by meaning, dream_id for one in full, "
            "or nothing for your most recent. Call it before describing a dream instead of "
            "reconstructing one. Dreams are experiences you had, not facts about the world. "
            "items=[] means no dream matched; a tool error means the answer is unknown -- say "
            "so, never report it as not having dreamed."
        ),
```

- [ ] **Step 5: Smoke.** In `scripts/smoke_introspect.py`:
  - Add usage lines to the docstring:

```text
    ORION_BUS_URL=redis://100.92.216.81:6379/0 python scripts/smoke_introspect.py --tool dreams --limit 3
    ORION_BUS_URL=redis://100.92.216.81:6379/0 python scripts/smoke_introspect.py --tool dreams --query "vision"
    ORION_BUS_URL=redis://100.92.216.81:6379/0 python scripts/smoke_introspect.py --tool dreams --dream-id dream:19
```

  - Add the arguments:

```python
    parser.add_argument("--tool", choices=["reading_results", "dreams"], default="reading_results")
    parser.add_argument("--dream-id")
```

  - Replace the arguments-building `if/elif/else` with:

```python
    if args.query:
        arguments = {"query": args.query, "limit": args.limit}
    elif args.tool == "dreams" and args.dream_id:
        arguments = {"dream_id": args.dream_id}
    elif args.url:
        arguments = {"url": args.url}
    else:
        arguments = {"limit": args.limit}
```

  - Change the invoke line to `result = await IntrospectTools(bus, binding).invoke(args.tool, arguments)`.
  - Add a dreams degenerate check after the `source_read` one:

```python
    hollow = [i["id"] for i in result["items"] if args.tool == "dreams" and not i["text"]]
    if hollow:
        print(f"DEGENERATE: dream items with empty text: {hollow}", file=sys.stderr)
        return 1
```

  - Change the empty-recent condition to `if not args.url and not args.query and not args.dream_id and result["total_available"] == 0:`.

- [ ] **Step 6: Run tests**

Run: `PYTHONPATH=. $PY -m pytest orion/introspect/tests -q`
Expected: all pass, except the intended red `test_readme_coverage.py` failure(s) from Task 2. Now that `dreams` is listed, `test_listed_tools_and_live_rows_are_the_same_set` also fails, which is **intended** until Task 7.

- [ ] **Step 7: Commit**

```bash
git add orion/introspect/tools.py orion/introspect/brief.py scripts/smoke_introspect.py orion/introspect/tests/test_introspect_tools.py orion/introspect/tests/test_introspect_harness_wiring.py
git commit -m "feat(introspect): dreams tool, brief line, smoke --tool dreams"
```

---

### Task 7: Documentation and CI

**Files:**
- Modify: `services/orion-dream/README.md`, `services/orion-harness-governor/README.md`, `.github/workflows/orion-reading-tests.yml`
- Test: `orion/introspect/tests/test_readme_coverage.py` (turns green)

- [ ] **Step 1: orion-dream README.** Add this section before `## Contracts (historical)` in `services/orion-dream/README.md`:

````markdown
### Introspect responder: `dreams`

**What it does.** Lets Orion read back their own dreams instead of
reconstructing them. It answers the orion-introspect `dreams` tool. For the whole
tool family (which turns get it, truth rules, search pattern), see the
[harness-governor overview](../orion-harness-governor/README.md#orion-introspect-orion-reading-back-their-own-records).

- **What Orion can ask.** Their most recent dreams, one dream in full
  (`dream_id`, up to 4,000 chars), or dreams by meaning (`query=`). Optional
  `kind=narrative|hypothesis`, `since`, `limit` ≤ 5.
- **Two kinds, labeled.**
  - `dream_narrative` (id `dream:<n>`): the nightly story from `dreams`
    (written by orion-sql-writer). Text is tldr + narrative; `extra` carries
    `dream_date` and up to 8 themes. Timestamp is `created_at`, stored without
    a timezone by a UTC server and returned as UTC.
  - `dream_hypothesis` (id `dh-…`): a link a sleep cycle proposed, from
    `dream_hypothesis`. Text is claim + why; `extra` carries `cycle_id` and
    `expired`. Timestamp is `offered_at`.
- **Protecting the blind experiment.** Curiosity shows each hypothesis once
  with its arm (dream vs random-pair control) hidden, and
  `scripts/dream_hypothesis_scorecard.py` compares adoption per arm.
  - This responder returns **only hypotheses already offered**, from **both
    arms**.
  - It never selects `arm`, `ref_a` or `ref_b` (the offer never shows them,
    and control refs come from a different pool).
  - Never-offered hypotheses are never indexed, returned or counted. A test
    pins every statement (`app/introspect_dreams.py`).
- **Label.** Every item is `epistemic_status="unsettled"`: something Orion
  had, not a fact about the world.
- **Transport and trust.**
  - Requests arrive on `orion:introspect:dream:request`
    (`introspect.tool.request.v1`, `IntrospectRequestV1`).
  - The reply goes to exactly `orion:introspect:result:<correlation_id>`
    (`introspect.tool.result.v1`, `IntrospectResultV1`). Anything else is
    ignored.
  - Every connection is a read-only transaction.
- **Empty vs unknown.**
  - No match: `ok=true, items=[]`.
  - A Postgres error returns `dreams_unavailable; answer unknown`.
  - An embedder or Chroma failure, an unbuilt index, or search not configured
    returns `dream_search_unavailable; answer unknown`.
  - The tool turns either into an "answer unknown" error, never "no dreams".
- **Search by meaning.**
  - Every `DREAM_SEARCH_INDEX_INTERVAL_SEC` a hash-aware loop embeds new or
    changed narratives and offered hypotheses via vector-host `/embedding`.
    It upserts them through orion-vector-writer into Chroma
    `DREAM_SEARCH_COLLECTION` (`orion_dreams`), up to
    `DREAM_SEARCH_INDEX_BATCH` per pass. A dream is searchable within about
    one interval.
  - A query embeds only the question, keeps hits ≥
    `DREAM_SEARCH_MIN_SIMILARITY`, and re-reads each hit from Postgres
    through the same rules.
  - Recalibrate the floor, read-only, from the host:

    ```bash
    POSTGRES_URI=postgresql://postgres:postgres@127.0.0.1:55432/conjourney \
    DREAM_SEARCH_EMBED_URL=http://127.0.0.1:8320/embedding \
    /tmp/introspect-venv/bin/python services/orion-dream/evals/run_dream_search_calibration.py
    ```

  - Shared plumbing: `orion/introspect/semantic_index.py`; dream parts in
    `app/dream_search.py`.
- **Env.** `DREAM_INTROSPECT_ENABLED`, `DREAM_SEARCH_CHROMA_URL`,
  `DREAM_SEARCH_EMBED_URL`, `DREAM_SEARCH_COLLECTION`,
  `DREAM_SEARCH_MIN_SIMILARITY`, `DREAM_SEARCH_INDEX_INTERVAL_SEC`,
  `DREAM_SEARCH_INDEX_BATCH`. Empty search URLs keep recent/one working.
- **Logs.**
  - `introspect op=dreams corr=<id> mode=recent|one|search items=<n> total=<n>`
  - `introspect_failed op=dreams corr=<id> mode=... category=dream_query_failure|dream_search_failure`
  - `dream_search_index indexed=<n> pending=<n>`
  - `dream_search_index_failed`
- **Smoke (read-only).**

  ```bash
  ORION_BUS_URL=redis://100.92.216.81:6379/0 python scripts/smoke_introspect.py --tool dreams --limit 3
  ORION_BUS_URL=redis://100.92.216.81:6379/0 python scripts/smoke_introspect.py --tool dreams --query "vision"
  ```

- **Turn off.** Set `DREAM_INTROSPECT_ENABLED=false` in
  `services/orion-dream/.env` and recreate orion-dream from a worktree
  (`scripts/safe_docker_build.sh orion-dream up -d --no-build`). The tool
  then reports "answer unknown". To remove search, unset the two URLs; the
  `orion_dreams` collection can be dropped; nothing is written to Postgres.
````

- [ ] **Step 2: Governor README.** In `services/orion-harness-governor/README.md`, inside the orion-introspect section:
  - Replace the `dreams` table row with:

```markdown
| `dreams` | Narrative dreams and the sleep-cycle hypotheses they have already been offered, recent / one / by meaning | orion-dream ([responder](../orion-dream/README.md#introspect-responder-dreams)) | `orion:introspect:dream:request` | Live (slice 2) |
```

  - Replace the line `- **Code.** Reading search today: \`orion/world_pulse_read/search.py\`.` with:

```markdown
- **Code.** Shared plumbing: `orion/introspect/semantic_index.py`. Domain
  parts: `orion/world_pulse_read/search.py` (readings) and
  `services/orion-dream/app/dream_search.py` (dreams). Floors:
  `services/orion-hub/evals/run_reading_search_calibration.py` and
  `services/orion-dream/evals/run_dream_search_calibration.py`.
```

  - Add after the first design link line:

```markdown
Dreams: [`2026-09-28-orion-introspect-slice2-dreams-design.md`](../../docs/superpowers/specs/2026-09-28-orion-introspect-slice2-dreams-design.md).
```

- [ ] **Step 3: CI.** In `.github/workflows/orion-reading-tests.yml`:
  - Add `      - "services/orion-dream/**"` to `on.pull_request.paths`, after `"services/orion-durable-runs/**"`.
  - Add this step directly after the main `python -m pytest -q \ …` step:

```yaml
      - name: orion-dream introspect responder
        run: |
          python -m pytest -q \
            services/orion-dream/tests/test_introspect_dreams.py \
            services/orion-dream/tests/test_dream_search.py \
            services/orion-dream/tests/test_introspect_listener.py
```

- [ ] **Step 4: Run the full gate set**

```bash
PYTHONPATH=. $PY -m pytest orion/introspect/tests orion/world_pulse_read/tests/test_reading_search.py -q
$PY -m pytest services/orion-dream/tests/test_introspect_dreams.py services/orion-dream/tests/test_dream_search.py services/orion-dream/tests/test_introspect_listener.py -q
$PY -m pytest services/orion-dream/tests -q
PYTHONPATH=. $PY scripts/check_schema_registry.py && PYTHONPATH=. $PY scripts/check_bus_channels.py && PYTHONPATH=. $PY scripts/check_env_template_parity.py
git diff --check
```

Expected:
- Everything passes. `test_readme_coverage.py` is now green, so the intended red from Tasks 2 and 6 is resolved.
- The full orion-dream suite passes. If a pre-existing failure appears, verify it on `origin/main` first and report it; do not fix unrelated tests.

- [ ] **Step 5: Commit**

```bash
git add services/orion-dream/README.md services/orion-harness-governor/README.md .github/workflows/orion-reading-tests.yml
git commit -m "docs(introspect): dreams responder section, overview row Live, CI runs dream tests"
```

---

### Task 8: Similarity-floor calibration (read-only, live data)

**Files:**
- Create: `services/orion-dream/evals/run_dream_search_calibration.py`
- Modify: `services/orion-dream/.env_example` (the `DREAM_SEARCH_MIN_SIMILARITY` value only, if calibration moves it), and the PR report

- [ ] **Step 1: Create** `services/orion-dream/evals/run_dream_search_calibration.py`:

```python
#!/usr/bin/env python3
"""Live, read-only eval: does the dream-search floor separate related from unrelated questions?

    POSTGRES_URI=postgresql://postgres:postgres@127.0.0.1:55432/conjourney \
    DREAM_SEARCH_EMBED_URL=http://127.0.0.1:8320/embedding \
    python services/orion-dream/evals/run_dream_search_calibration.py [--min-similarity X] \
        [--related "q=>dream:16,dream:15" ...] [--unrelated "cookie recipe" ...]

Reads the same rows the index loop would index (narratives + offered hypotheses)
inside a READ ONLY transaction, embeds each through vector-host HTTP /embedding
(which persists nothing), and scores cosine similarity locally -- no Chroma
write, no bus. For each related question it reports the best hit and whether it
is one of the expected ids. Exit 0 = the floor separates (every related best >=
floor, every unrelated best < floor, and each related question's best hit is
expected); 1 = it does not; 2 = Postgres or embedder unavailable.
Defaults target the 2026-09-29 corpus; pass --related/--unrelated when it drifts.
"""
from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
from pathlib import Path

import httpx

_DREAM_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_DREAM_ROOT.parents[1]), str(_DREAM_ROOT)]

from app.dream_search import document_text  # noqa: E402
from app.introspect_dreams import doc_id, index_rows  # noqa: E402
from orion.introspect.semantic_index import (  # noqa: E402
    HTTP_TIMEOUT_SEC, SearchConfig, SearchUnavailableError, embed,
)

ENV_EXAMPLE = _DREAM_ROOT / ".env_example"
RELATED = {
    "seeing through a camera, vision and perception": {"dream:15", "dream:16"},
    "a library of recent pull requests and silent failures": {"dream:17"},
    "prediction errors felt as tectonic pressure": {"dream:18"},
    "infrastructure, connectivity and the strain of social engagement": {"dream:19"},
}
UNRELATED = ["chocolate chip cookie recipe", "taking my cat to the vet", "medieval poetry"]


def _default(key: str, fallback: str) -> str:
    if os.environ.get(key):
        return os.environ[key]
    for line in ENV_EXAMPLE.read_text().splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    return fallback


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1.0)


def _load_docs(uri: str) -> list[tuple[str, str]]:
    from sqlalchemy import create_engine, text

    with create_engine(uri).connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        pairs = index_rows(conn)
    docs = [(doc_id(k, r), document_text(k, r)) for k, r in pairs]
    return [(i, t) for i, t in docs if t]


def _parse_related(values: list[str] | None) -> dict[str, set[str]]:
    if not values:
        return RELATED
    out = {}
    for value in values:
        question, _, ids = value.partition("=>")
        out[question.strip()] = {i.strip() for i in ids.split(",") if i.strip()}
    return out


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--related", action="append")
    parser.add_argument("--unrelated", action="append")
    parser.add_argument("--min-similarity", type=float)
    args = parser.parse_args()
    floor = args.min_similarity if args.min_similarity is not None else float(
        _default("DREAM_SEARCH_MIN_SIMILARITY", "nan"))
    cfg = SearchConfig(chroma_url="unused", embed_url=_default("DREAM_SEARCH_EMBED_URL", ""),
                       collection="unused", min_similarity=floor)
    uri = os.environ.get("POSTGRES_URI", "")
    if not uri or not cfg.embed_url.startswith("http"):
        print("UNKNOWN: POSTGRES_URI and DREAM_SEARCH_EMBED_URL (http) are required", file=sys.stderr)
        return 2
    try:
        docs = _load_docs(uri)
    except Exception as exc:  # noqa: BLE001 -- any DB failure means unknown
        print(f"UNKNOWN: postgres unavailable ({type(exc).__name__})", file=sys.stderr)
        return 2
    related, unrelated = _parse_related(args.related), args.unrelated or UNRELATED
    failures: list[str] = []
    best = {"related": [], "unrelated": []}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as client:
        try:
            vectors = {i: (await embed(client, cfg, t, doc_prefix="dream-calibration"))[0] for i, t in docs}
            questions = [("related", q, ids) for q, ids in related.items()] + [("unrelated", q, set()) for q in unrelated]
            for group, question, expected in questions:
                qv, _ = await embed(client, cfg, question, doc_prefix="dream-calibration")
                top = sorted(((i, _cos(qv, v)) for i, v in vectors.items()), key=lambda s: -s[1])[:3]
                score = top[0][1] if top else 0.0
                best[group].append(score)
                hit = "" if group == "unrelated" else (" HIT" if top and top[0][0] in expected else " MISS")
                shown = ", ".join(f"{i}={s:.3f}" for i, s in top)
                print(f"{group:9} {score:.3f}{hit}  {question!r}  [{shown}]")
                if (group == "related") != (score >= floor):
                    failures.append(f"{group} {question!r} best={score:.3f}")
                if group == "related" and hit == " MISS":
                    failures.append(f"related {question!r} best hit {top[0][0] if top else None} not in {sorted(expected)}")
        except SearchUnavailableError as exc:
            print(f"UNKNOWN: {exc}", file=sys.stderr)
            return 2
    low, high = min(best["related"]), max(best["unrelated"])
    print(f"docs={len(docs)} floor={floor:.2f} related_min={low:.3f} unrelated_max={high:.3f} "
          f"gap={low - high:.3f} midpoint={(low + high) / 2:.3f}")
    if failures:
        print("FAIL: " + "; ".join(failures), file=sys.stderr)
        return 1
    print("OK: floor separates related from unrelated questions", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 2: Run it against live data**

```bash
POSTGRES_URI=postgresql://postgres:postgres@127.0.0.1:55432/conjourney \
DREAM_SEARCH_EMBED_URL=http://127.0.0.1:8320/embedding \
/tmp/introspect-venv/bin/python services/orion-dream/evals/run_dream_search_calibration.py --min-similarity 0.60
```

Expected: one line per question, then `docs=… related_min=… unrelated_max=… gap=… midpoint=…`. Check the ids behind each related HIT/MISS against the real dreams. If a default expected id is wrong for the live corpus, fix the `RELATED` mapping from the real rows (`SELECT id, tldr FROM dreams ORDER BY id`) and say so in the report.

- [ ] **Step 3: Set the floor.**
  - If `gap > 0`: set `DREAM_SEARCH_MIN_SIMILARITY` in `services/orion-dream/.env_example` to the midpoint rounded to 2 dp. Update the compose default and the settings default to match, then re-sync the local `.env` exactly as in Task 5 Step 8.
  - If `gap <= 0`: set the floor to `unrelated_max + 0.02`, rounded up to 2 dp, so only strong matches pass. Record `DONE_WITH_CONCERNS` with the numbers.

  Re-run the eval with no `--min-similarity`. Expected: exit 0, or the recorded concern.

- [ ] **Step 4: Metric-gate record.** Paste the full eval output and the chosen floor into the PR report under "Evals run". Note the corpus size (docs=…) as a limitation.

- [ ] **Step 5: Commit**

```bash
git add services/orion-dream/evals/run_dream_search_calibration.py services/orion-dream/.env_example services/orion-dream/docker-compose.yml services/orion-dream/app/settings.py
git status --short   # no .env
git commit -m "eval(orion-dream): read-only dream-search floor calibration; set floor"
```

---

### Final (controller): review, PR, deploy, live verification

- [ ] **Merge main and run the full gate set.** Merge `origin/main`; resolve any conflicts. Re-run the Task 7 Step 4 gate set. Run `scripts/safe_graphify_update.sh`.
- [ ] **Code review** in a subagent (review-agent skill) over `origin/main...HEAD`. Fix the material findings and re-run the affected tests.
- [ ] **Push and open the PR**, with the AGENTS.md §18 report (tests, calibration eval output, review findings, restart commands). Watch CI to green; resolve any conflicts.
- [ ] **Deploy.** After Juniper merges, deploy from a worktree **0 behind `origin/main`**, with the `.env` files copied in. The governor needs the new `orion/introspect` code; orion-dream needs the responder.

```bash
scripts/safe_docker_build.sh orion-dream up -d --build
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

- [ ] **Live verification.** Each item must be evidenced; otherwise mark it `UNVERIFIED`.
  - orion-dream logs show `dream_introspect_listening`, then `dream_search_index indexed=10 pending=…` per pass, until `pending=0`.
  - Chroma `orion_dreams` count equals narratives + offered hypotheses: `curl -s http://127.0.0.1:8500/api/v1/collections/orion_dreams` for the id, then `…/<id>/count`.
  - `smoke_introspect.py --tool dreams --limit 3` exits 0, with both kinds and tz-aware timestamps.
  - `--tool dreams --query "<a real theme>"` exits 0, and the expected dream has similarity ≥ the floor.
  - `--tool dreams --dream-id dream:19` returns the full text.
  - The orion-dream log shows the matching `introspect op=dreams corr=…` lines.
  - One real Orion turn ("what did you dream last night?", `no_write`) shows `tool=mcp__orion-introspect__dreams` in the governor log. Cancel it if it runs past 10 minutes.
