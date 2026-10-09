# Orion Introspect — Slice 1b (semantic search over reading results) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `reading_results(query="...")` finds what Orion learned by meaning, not by exact URL/request id.

**Architecture:** The Hub embeds each verified reading once (a bounded, hash-aware index loop inside `ReadingListener`) and publishes `VectorUpsertV1` to `orion:vector:semantic:upsert`, which orion-vector-writer stores in Chroma collection `orion_reading_results`. A query embeds only the question (vector-host HTTP `/embedding`), asks Chroma REST for the nearest 20, keeps hits at or above a calibrated cosine floor, and re-reads/re-gates each hit from Postgres.

**Tech Stack:** Python 3.12, pydantic v2, asyncpg, httpx 0.27 (already a Hub dependency), Chroma 0.4.24 REST API, pytest.

**Spec:** `docs/superpowers/specs/2026-09-28-orion-introspect-slice1b-semantic-search.md` (includes the metric-gate record for the similarity floor).

## Global Constraints

- Worktree `/mnt/scripts/Orion-Sapienform-introspect-slice1b`, branch `feat/introspect-slice1b` (stacked on `feat/introspect-slice1` / PR #2381). Never commit from `/mnt/scripts/Orion-Sapienform`.
- Test interpreter: `PY=/tmp/introspect-venv/bin/python`; run from the worktree root with `PYTHONPATH=.`. Hub tests also need `services/orion-hub` on the path — use exactly the commands given.
- Real-Postgres tests: prefix `RUN_READING_POSTGRES=1`.
- Empty ≠ unknown: search ran and nothing cleared the floor → `ok=True, items=[], total_available=0`. Embedder/Chroma failure, missing collection, malformed reply, or search not configured → `SearchUnavailableError` in the Hub → tool error containing `answer unknown`.
- Chroma is an index, never the record: every hit is re-read from `world_pulse_read_seed` and must pass the same verified-reading gate as slice 1 (`_VERIFIED_WHERE` in SQL + `_source_read`/`_learned` in Python).
- Nothing but the query is embedded at ask time.
- Only vector-host's HTTP `/embedding` endpoint is used (pure; the bus `orion:embedding:generate` path persists vectors into another collection).
- Bounds: `QUERY_CAP = 500` chars; `CANDIDATES = 20`; `INDEX_TEXT_CHARS = 1800`; `INDEX_SCAN_LIMIT = 1000`; `HTTP_TIMEOUT_SEC = 5.0`; output bounds unchanged from slice 1 (5 items).
- Env keys (Hub): `HUB_READING_SEARCH_CHROMA_URL` (`http://127.0.0.1:8500`), `HUB_READING_SEARCH_EMBED_URL` (`http://127.0.0.1:8320/embedding`), `HUB_READING_SEARCH_COLLECTION` (`orion_reading_results`), `HUB_READING_SEARCH_MIN_SIMILARITY` (`0.60`, single-source owner `services/orion-hub/.env_example`), `HUB_READING_SEARCH_INDEX_INTERVAL_SEC` (`300`), `HUB_READING_SEARCH_INDEX_BATCH` (`10`). Settings defaults for the two URLs are empty (feature off unless configured).
- Never `git commit --no-verify`. Never stage `.env`. Do not deploy anything (controller does, only when the branch is 0 commits behind `origin/main`).

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `orion/schemas/introspect.py` | Modify | `QUERY_CAP`; `ReadingResultArguments.query` |
| `orion/schemas/reading.py` | Modify | `ReadingToolRequestV1.query` (omitted from wire when null) |
| `orion/bus/channels.yaml` | Modify | `orion-hub` produces `orion:vector:semantic:upsert` |
| `orion/world_pulse_read/introspect.py` | Modify | Extract `_VERIFIED_WHERE` shared with search |
| `orion/world_pulse_read/search.py` | Create | Config, embed, Chroma REST, `similarity`, `search_readings`, `verified_rows`, `index_missing_readings` |
| `orion/world_pulse_read/tests/test_reading_search.py` | Create | Unit tests (httpx.MockTransport, fake conn) |
| `services/orion-hub/tests/test_reading_postgres.py` | Modify | Real-SQL index + search test |
| `services/orion-hub/scripts/reading_listener.py` | Modify | Query dispatch, search error mapping, index loop |
| `services/orion-hub/scripts/main.py` | Modify | Build `ReadingSearchConfig` from settings |
| `services/orion-hub/app/settings.py` | Modify | Six keys |
| `services/orion-hub/.env_example` | Modify | Six keys, documented |
| `services/orion-hub/README.md` | Modify | Reading search paragraph |
| `services/orion-hub/tests/test_reading_ingress.py` | Modify | Listener query/unknown/index-loop tests |
| `scripts/check_env_key_single_source.py` | Modify | Owner for `HUB_READING_SEARCH_MIN_SIMILARITY` |
| `orion/introspect/tools.py` | Modify | Pass `query`; description leads with search |
| `orion/introspect/brief.py` | Modify | Brief leads with search |
| `orion/introspect/tests/test_introspect_tools.py`, `test_introspect_harness_wiring.py`, `test_introspect_schemas.py` | Modify | Query coverage |
| `scripts/smoke_introspect.py` | Modify | `--query` |
| `services/orion-hub/evals/run_reading_search_calibration.py` | Create | Live floor calibration eval |

---

### Task 1: Contracts — `query` argument and channel producer

**Files:**
- Modify: `orion/schemas/introspect.py`, `orion/schemas/reading.py`, `orion/bus/channels.yaml`
- Test: `orion/introspect/tests/test_introspect_schemas.py`

- [ ] **Step 1: Write failing tests.** Append to `orion/introspect/tests/test_introspect_schemas.py`:

```python
def test_reading_result_arguments_accept_query_and_strip_it():
    args = ReadingResultArguments(query="  graphics cards  ", since=NOW)
    assert args.query == "graphics cards"
    assert args.since == NOW


@pytest.mark.parametrize(
    "fields",
    [
        {"query": "   "},
        {"query": "x" * 501},
        {"query": "gpus", "url": "https://example.org/a"},
        {"query": "gpus", "request_id": "00000000-0000-0000-0000-000000000001"},
    ],
)
def test_reading_result_arguments_reject_bad_query(fields):
    with pytest.raises(ValidationError):
        ReadingResultArguments(**fields)


def test_reading_tool_request_carries_query_only_for_reading_result():
    wire = ReadingToolRequestV1(operation="reading_result", query="gpus", limit=2).model_dump(mode="json")
    assert wire["query"] == "gpus"
    assert ReadingToolRequestV1.model_validate(wire).query == "gpus"
    with pytest.raises(ValidationError):
        ReadingToolRequestV1(operation="reading_status", url="https://example.org/a", query="gpus")
    with pytest.raises(ValidationError):
        ReadingToolRequestV1(operation="reading_result", url="https://example.org/a", query="gpus")


def test_null_query_never_reaches_the_wire():
    # A pre-1b Hub forbids unknown keys, even null ones.
    recent = ReadingToolRequestV1(operation="reading_result", limit=3).model_dump(mode="json")
    assert "query" not in recent
    status = ReadingToolRequestV1(operation="reading_status", url="https://example.org/a")
    assert set(status.model_dump(mode="json")) == {"operation", "request", "request_id", "url"}
```

- [ ] **Step 2: Run to verify failure.**

Run: `PYTHONPATH=. $PY -m pytest orion/introspect/tests/test_introspect_schemas.py -q`
Expected: the four new tests FAIL (`query` is an extra field / not defined).

- [ ] **Step 3: Implement in `orion/schemas/introspect.py`.** Add the constant next to the other bounds, add `field_validator` to the pydantic import, and replace `ReadingResultArguments` with:

```python
QUERY_CAP = 500
```

```python
class ReadingResultArguments(BaseModel):
    """Model-supplied arguments for the ``reading_results`` tool."""

    model_config = ConfigDict(extra="forbid")
    query: str | None = Field(default=None, min_length=1, max_length=QUERY_CAP)
    request_id: UUID | None = None
    url: str | None = Field(default=None, min_length=1, max_length=8192)
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=MAX_ITEMS)
    since: datetime | None = None

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value

    @model_validator(mode="after")
    def _selectors(self):
        if self.request_id is not None and self.url is not None:
            raise ValueError("reading_results takes at most one of request_id or url")
        if self.query is not None and (self.request_id is not None or self.url is not None):
            raise ValueError("query searches all readings; it cannot be combined with request_id or url")
        _require_tz(self.since, "since")
        if self.since is not None and (self.request_id is not None or self.url is not None):
            raise ValueError("since applies only to query or recent reads (no request_id or url)")
        return self
```

- [ ] **Step 4: Implement in `orion/schemas/reading.py`.** Import `QUERY_CAP` alongside `MAX_ITEMS`. Add the field after `url`:

```python
    query: str | None = Field(default=None, min_length=1, max_length=QUERY_CAP)
```

In `operation_arguments`, inside the final `else:` (reading_result) branch, after the `request is not None` check, add:

```python
            if self.query is not None and (self.request_id is not None or self.url is not None):
                raise ValueError("reading_result query cannot be combined with request_id or url")
```

Replace the non-result guard with:

```python
        if self.operation != "reading_result" and (
            self.limit is not None or self.since is not None or self.query is not None
        ):
            raise ValueError(f"{self.operation} takes no limit, since, or query")
```

and change the serializer's key tuple to `("limit", "since", "query")`.

- [ ] **Step 5: Channel producer.** In `orion/bus/channels.yaml`, entry `orion:vector:semantic:upsert`:

```yaml
    producer_services: ["orion-vector-host", "orion-hub"]
```

- [ ] **Step 6: Run tests and gates.**

Run: `PYTHONPATH=. $PY -m pytest orion/introspect/tests/test_introspect_schemas.py -q && $PY scripts/check_bus_channels.py && $PY scripts/check_schema_registry.py`
Expected: all pass. (If either script does not exist, say so and skip it.)

- [ ] **Step 7: Commit.**

```bash
git add orion/schemas/introspect.py orion/schemas/reading.py orion/bus/channels.yaml orion/introspect/tests/test_introspect_schemas.py
git commit -m "feat(introspect): query argument for semantic reading search"
```

---

### Task 2: Search module (index + query) over Chroma REST

**Files:**
- Modify: `orion/world_pulse_read/introspect.py`
- Create: `orion/world_pulse_read/search.py`, `orion/world_pulse_read/tests/test_reading_search.py` (create `orion/world_pulse_read/tests/__init__.py` only if sibling test dirs use one — check `ls orion/*/tests/__init__.py`)
- Modify: `services/orion-hub/tests/test_reading_postgres.py`

- [ ] **Step 1: Extract the verified-reading predicate.** In `orion/world_pulse_read/introspect.py`, replace the `_RECENT_SQL` block (keep its comment) with:

```python
# Rows whose Stage 1 handoff carries no tool-trace fetch are not readings (see
# read_evidence.py); a Stage 2 summary built on one is no better. CASE, not AND,
# because SQL does not guarantee short-circuit evaluation.
_VERIFIED_WHERE = """s.duplicate_of IS NULL
  AND s.status = 'done'
  AND CASE WHEN jsonb_typeof(s.handoff_json->'read_evidence') = 'array'
           THEN jsonb_array_length(s.handoff_json->'read_evidence') > 0
           ELSE false END"""

# count(*) OVER () is evaluated before LIMIT, so ``total`` is the full match count.
_RECENT_SQL = f"""
SELECT {_COLUMNS}, count(*) OVER () AS total
FROM world_pulse_read_seed s
WHERE {_VERIFIED_WHERE}
  AND ($2::timestamptz IS NULL OR {_OCCURRED} >= $2)
ORDER BY {_OCCURRED} DESC, s.seed_id DESC
LIMIT $1
"""
```

Run: `RUN_READING_POSTGRES=1 PYTHONPATH=.:services/orion-hub:services/orion-hub/tests $PY -m pytest services/orion-hub/tests/test_reading_postgres.py -q -k introspection`
Expected: PASS (pure refactor).

- [ ] **Step 2: Write failing unit tests.** Create `orion/world_pulse_read/tests/test_reading_search.py`:

```python
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
```

- [ ] **Step 3: Run to verify failure.**

Run: `PYTHONPATH=. $PY -m pytest orion/world_pulse_read/tests/test_reading_search.py -q`
Expected: collection error (`orion.world_pulse_read.search` does not exist).

- [ ] **Step 4: Implement `orion/world_pulse_read/search.py`:**

```python
"""Semantic search over verified readings for the reading_results introspect tool.

Each verified reading is embedded once, when it lands, and upserted through
orion-vector-writer; a question embeds only the query. Chroma is an index, not
the record: every hit is re-read from world_pulse_read_seed and re-gated.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import httpx

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.introspect import URL_CAP, IntrospectResultV1, clip_text
from orion.schemas.vector.schemas import EmbeddingGenerateV1, EmbeddingResultV1, VectorUpsertV1
from orion.world_pulse_read.introspect import (
    _COLUMNS,
    _OCCURRED,
    _VERIFIED_WHERE,
    _item,
    _learned,
    _source_read,
)

UPSERT_CHANNEL = "orion:vector:semantic:upsert"
UPSERT_KIND = "vector.upsert.v1"
CANDIDATES = 20
INDEX_TEXT_CHARS = 1800
INDEX_SCAN_LIMIT = 1000
HTTP_TIMEOUT_SEC = 5.0

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


class SearchUnavailableError(RuntimeError):
    """The embedder, index, or configuration could not answer; the result is unknown."""


@dataclass(frozen=True)
class ReadingSearchConfig:
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


def document_text(row: Any) -> str | None:
    learned = _learned(row, _source_read(row))
    if not learned:
        return None
    title = str(row["title"] or "").strip()
    return clip_text(f"{title}\n\n{learned}" if title else learned, INDEX_TEXT_CHARS)[0]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def similarity(distance: float, space: str) -> float:
    # Chroma's l2 is squared euclidean; for the unit vectors bge emits,
    # ||a-b||^2 = 2 - 2cos, so cos = 1 - d/2. cosine and ip report 1 - cos.
    if space == "l2":
        return 1.0 - distance / 2.0
    return 1.0 - distance


def _base(cfg: ReadingSearchConfig) -> str:
    return f"{cfg.chroma_url.rstrip('/')}/api/v1/collections"


async def _json(client: httpx.AsyncClient, method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
    try:
        resp = await client.request(method, url, **kwargs)
        return resp.status_code, resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise SearchUnavailableError(f"chroma unavailable: {type(exc).__name__}") from exc


async def embed(client: httpx.AsyncClient, cfg: ReadingSearchConfig, text: str) -> tuple[list[float], str | None]:
    req = EmbeddingGenerateV1(doc_id=f"reading-search-{uuid4()}", text=text)
    try:
        resp = await client.post(cfg.embed_url, json=req.model_dump(mode="json"))
        resp.raise_for_status()
        result = EmbeddingResultV1.model_validate(resp.json())
    except (httpx.HTTPError, ValueError) as exc:
        raise SearchUnavailableError(f"embedder unavailable: {type(exc).__name__}") from exc
    if not result.embedding:
        raise SearchUnavailableError("embedder returned no vector")
    return result.embedding, result.embedding_model


async def _collection(client: httpx.AsyncClient, cfg: ReadingSearchConfig) -> _Collection | None:
    status, body = await _json(client, "GET", f"{_base(cfg)}/{cfg.collection}")
    if status == 200 and isinstance(body, dict) and body.get("id"):
        space = str((body.get("metadata") or {}).get("hnsw:space") or "l2")
        return _Collection(id=str(body["id"]), space=space)
    # Chroma 0.4.24 answers a missing collection with HTTP 500 + ValueError text.
    if isinstance(body, dict) and "does not exist" in str(body.get("error") or ""):
        return None
    raise SearchUnavailableError(f"chroma collection lookup failed: HTTP {status}")


async def nearest(
    client: httpx.AsyncClient, cfg: ReadingSearchConfig, vector: list[float], n: int = CANDIDATES,
) -> list[tuple[str, float]]:
    """(seed_id, similarity) for the n nearest indexed readings, best first."""
    coll = await _collection(client, cfg)
    if coll is None:
        raise SearchUnavailableError("reading index not built yet")
    status, body = await _json(
        client, "POST", f"{_base(cfg)}/{coll.id}/query",
        json={"query_embeddings": [vector], "n_results": n, "include": ["distances"]},
    )
    if status != 200:
        raise SearchUnavailableError(f"chroma query failed: HTTP {status}")
    try:
        pairs = zip(body["ids"][0], body["distances"][0])
        scored = [(str(i), similarity(float(d), coll.space)) for i, d in pairs]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise SearchUnavailableError("malformed chroma query reply") from exc
    return sorted(scored, key=lambda s: -s[1])


async def search_readings(
    conn: Any,
    cfg: ReadingSearchConfig,
    *,
    client: httpx.AsyncClient,
    query: str,
    limit: int,
    since: datetime | None = None,
) -> IntrospectResultV1:
    as_of = datetime.now(timezone.utc)
    vector, _ = await embed(client, cfg, query)
    scored = [s for s in await nearest(client, cfg, vector) if s[1] >= cfg.min_similarity]
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


async def verified_rows(conn: Any, scan_limit: int = INDEX_SCAN_LIMIT) -> list[Any]:
    return list(await conn.fetch(_VERIFIED_ROWS_SQL, scan_limit))


async def _stored_hashes(client: httpx.AsyncClient, cfg: ReadingSearchConfig, ids: list[str]) -> dict[str, str]:
    coll = await _collection(client, cfg)
    if coll is None:
        return {}
    status, body = await _json(
        client, "POST", f"{_base(cfg)}/{coll.id}/get", json={"ids": ids, "include": ["metadatas"]},
    )
    if status != 200 or not isinstance(body, dict):
        raise SearchUnavailableError(f"chroma get failed: HTTP {status}")
    return {
        str(i): str((m or {}).get("content_hash") or "")
        for i, m in zip(body.get("ids") or [], body.get("metadatas") or [])
    }


async def index_missing_readings(
    rows: list[Any],
    cfg: ReadingSearchConfig,
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
    stored = await _stored_hashes(client, cfg, [str(row["seed_id"]) for row, _, _ in docs])
    stale = [d for d in docs if stored.get(str(d[0]["seed_id"])) != d[2]]
    batch = cfg.index_batch if batch is None else batch
    for row, text, digest in stale[:batch]:
        vector, model = await embed(client, cfg, text)
        occurred = row["landing_at"] or row["stage2_completed_at"] or row["handoff_at"] or row["created_at"]
        payload = VectorUpsertV1(
            doc_id=str(row["seed_id"]), collection=cfg.collection, embedding=vector,
            embedding_kind="semantic", embedding_model=model, embedding_dim=len(vector), text=text,
            meta={
                "content_hash": digest,
                "url": clip_text(row["url"], URL_CAP)[0],
                "request_id": str(row["request_id"] or ""),
                "occurred_at": occurred.isoformat() if occurred else "",
            },
        )
        await bus.publish(UPSERT_CHANNEL, BaseEnvelope(
            kind=UPSERT_KIND, source=source, payload=payload.model_dump(mode="json"),
        ))
    done = min(len(stale), batch)
    return IndexPass(indexed=done, pending=len(stale) - done)
```

- [ ] **Step 5: Run unit tests.**

Run: `PYTHONPATH=. $PY -m pytest orion/world_pulse_read/tests/test_reading_search.py -q`
Expected: all PASS.

- [ ] **Step 6: Real-Postgres test.** Append to `services/orion-hub/tests/test_reading_postgres.py` (reuses the file's `db`, `request`, `queue`; the mock transport is local to this test):

```python
def test_reading_search_index_and_query_use_real_sql_gate(local_pg):
    import httpx

    from orion.world_pulse_read.search import (
        ReadingSearchConfig, index_missing_readings, search_readings, verified_rows,
    )

    cfg = ReadingSearchConfig(
        chroma_url="http://chroma.test", embed_url="http://embed.test/embedding",
        collection="orion_reading_results", min_similarity=0.6,
    )

    async def run():
        conn, _ = await db(local_pg)
        done, hollow, queued = (request(f"https://example.org/{n}") for n in ("done", "hollow", "queued"))
        for r in (done, hollow, queued):
            await queue.enqueue_reading(conn, r)
        mark = """UPDATE world_pulse_read_seed SET status='done', stage2_status='done',
                  handoff_json=$2::jsonb, stage2_result_json=$3::jsonb, handoff_at=now(),
                  stage2_completed_at=now(), landing_at=now() WHERE request_id=$1"""
        await conn.execute(mark, done.request_id, json.dumps({"read_evidence": [
            {"tool_name": "WebFetch", "url": "https://example.org/done", "content_chars": 500}]}),
            json.dumps({"summary": "GPU supply is tight"}))
        await conn.execute(mark, hollow.request_id, json.dumps({"what_i_learned": "guess"}),
                           json.dumps({"summary": "built on nothing"}))
        seeds = {r["url"]: r["seed_id"] for r in await conn.fetch("SELECT url, seed_id FROM world_pulse_read_seed")}

        def handler(req: httpx.Request) -> httpx.Response:
            if req.url.host == "embed.test":
                return httpx.Response(200, json={"doc_id": "q", "embedding": [1.0, 0.0], "embedding_dim": 2})
            if req.url.path.endswith("/orion_reading_results"):
                return httpx.Response(200, json={"id": "cid", "metadata": None})
            if req.url.path.endswith("/get"):
                return httpx.Response(200, json={"ids": [], "metadatas": []})
            ids = [seeds[u] for u in ("https://example.org/done", "https://example.org/hollow", "https://example.org/queued")]
            return httpx.Response(200, json={"ids": [ids], "distances": [[0.2, 0.2, 0.2]]})

        published = []

        class Bus:
            async def publish(self, channel, envelope):
                published.append(envelope.payload["doc_id"])

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            rows = await verified_rows(conn)
            result = await index_missing_readings(rows, cfg, client=client, bus=Bus(), source=ServiceRef(name="orion-hub"))
            assert (result.indexed, result.pending) == (1, 0)
            assert published == [seeds["https://example.org/done"]]
            found = await search_readings(conn, cfg, client=client, query="graphics cards", limit=5)
            assert found.total_available == 1
            assert found.items[0].extra["url"] == "https://example.org/done"
            assert found.items[0].extra["similarity"] == 0.9
            assert found.items[0].text == "GPU supply is tight"
            future = await search_readings(conn, cfg, client=client, query="gpus", limit=5,
                                           since=datetime(2999, 1, 1, tzinfo=timezone.utc))
            assert future.items == [] and future.total_available == 0
        await conn.close()

    asyncio.run(run())
```

Check how the slice 1 introspection test ends (does it `await conn.close()`? does `db()` return a second value that must be closed?) and mirror it exactly.

Run: `RUN_READING_POSTGRES=1 PYTHONPATH=.:services/orion-hub:services/orion-hub/tests $PY -m pytest services/orion-hub/tests/test_reading_postgres.py -q -k "search or introspection"`
Expected: both PASS.

- [ ] **Step 7: Commit.**

```bash
git add orion/world_pulse_read/introspect.py orion/world_pulse_read/search.py orion/world_pulse_read/tests/test_reading_search.py services/orion-hub/tests/test_reading_postgres.py
git commit -m "feat(reading): semantic index and search over verified readings"
```

---

### Task 3: Hub wiring — settings, listener query dispatch, index loop

**Files:**
- Modify: `services/orion-hub/app/settings.py`, `services/orion-hub/.env_example`, `services/orion-hub/README.md`, `services/orion-hub/scripts/reading_listener.py`, `services/orion-hub/scripts/main.py`, `scripts/check_env_key_single_source.py`
- Test: `services/orion-hub/tests/test_reading_ingress.py`

- [ ] **Step 1: Write failing listener tests.** Append to `services/orion-hub/tests/test_reading_ingress.py`:

```python
def _search_cfg(**overrides):
    from orion.world_pulse_read.search import ReadingSearchConfig

    base = dict(chroma_url="http://chroma.test", embed_url="http://embed.test/embedding",
                collection="orion_reading_results", min_similarity=0.6)
    base.update(overrides)
    return ReadingSearchConfig(**base)


def test_query_routes_to_semantic_search(monkeypatch, caplog):
    import logging
    from datetime import datetime, timezone

    from orion.schemas.introspect import IntrospectResultV1

    seen = {}

    async def fake_search(conn, cfg, **kwargs):
        seen.update(kwargs)
        return IntrospectResultV1(ok=True, operation="reading_result",
                                  as_of=datetime.now(timezone.utc), total_available=0)

    def forbidden(*args, **kwargs):
        raise AssertionError("query mode must not use exact lookup")

    monkeypatch.setitem(ReadingListener.handle.__globals__, "search_readings", fake_search)
    monkeypatch.setitem(ReadingListener.handle.__globals__, "reading_results", forbidden)
    bus = RpcBus(_FakeConn())
    bus.listener.search = _search_cfg()
    with caplog.at_level(logging.INFO):
        out = asyncio.run(_introspect_tools(bus).invoke("reading_results", {"query": "graphics cards", "limit": 2}))
    assert out["ok"] is True and out["items"] == []
    assert seen["query"] == "graphics cards" and seen["limit"] == 2
    assert "introspect op=reading_result" in caplog.text and "mode=query" in caplog.text


@pytest.mark.parametrize("configured", [False, True])
def test_query_without_working_search_is_unknown(monkeypatch, configured):
    from orion.introspect.tools import IntrospectUnknownError
    from orion.world_pulse_read.search import SearchUnavailableError

    async def down(conn, cfg, **kwargs):
        raise SearchUnavailableError("reading index not built yet")

    monkeypatch.setitem(ReadingListener.handle.__globals__, "search_readings", down)
    bus = RpcBus(_FakeConn())
    bus.listener.search = _search_cfg() if configured else _search_cfg(chroma_url="")
    with pytest.raises(IntrospectUnknownError, match="answer unknown"):
        asyncio.run(_introspect_tools(bus).invoke("reading_results", {"query": "gpus"}))


def test_index_once_uses_released_rows_and_logs(monkeypatch, caplog):
    import logging

    from orion.world_pulse_read.search import IndexPass

    calls = {}

    async def fake_rows(conn, *args, **kwargs):
        return ["row"]

    async def fake_index(rows, cfg, **kwargs):
        calls["rows"] = rows
        calls["source"] = kwargs["source"]
        return IndexPass(indexed=1, pending=2)

    monkeypatch.setitem(ReadingListener.index_once.__globals__, "verified_rows", fake_rows)
    monkeypatch.setitem(ReadingListener.index_once.__globals__, "index_missing_readings", fake_index)
    bus = RpcBus(_FakeConn())
    bus.listener.search = _search_cfg()
    with caplog.at_level(logging.INFO):
        result = asyncio.run(bus.listener.index_once())
    assert result == IndexPass(indexed=1, pending=2)
    assert calls["rows"] == ["row"] and calls["source"].name == "orion-hub"
    assert "reading_search_index indexed=1 pending=2" in caplog.text


def test_index_loop_starts_only_when_search_enabled():
    async def run(search):
        listener = ReadingListener(lambda: None, ServiceRef(name="orion-hub"), search=search)

        class Bus:
            async def publish(self, *a):
                pass

        listener._run = lambda: asyncio.sleep(3600)
        await listener.start(Bus())
        started = listener.index_task is not None
        await listener.stop()
        return started

    assert asyncio.run(run(_search_cfg())) is True
    assert asyncio.run(run(_search_cfg(chroma_url=""))) is False
    assert asyncio.run(run(None)) is False
```

Note: `_run` is replaced by a coroutine function in the last test so `start()` does not subscribe to a real bus; if `start()` calls `self._run()` this works as written.

- [ ] **Step 2: Run to verify failure.**

Run: `PYTHONPATH=.:services/orion-hub:services/orion-hub/tests $PY -m pytest services/orion-hub/tests/test_reading_ingress.py -q -k "query or index"`
Expected: FAIL (`search` kwarg / `index_once` / `search_readings` missing).

- [ ] **Step 3: Implement the listener.** In `services/orion-hub/scripts/reading_listener.py`:

Imports:

```python
import httpx

from orion.world_pulse_read.search import (
    HTTP_TIMEOUT_SEC,
    ReadingSearchConfig,
    SearchUnavailableError,
    index_missing_readings,
    search_readings,
    verified_rows,
)
```

Constant next to `_SAFE_ERROR`:

```python
_SEARCH_UNAVAILABLE = "reading_search_unavailable; answer unknown"
```

In `_failure_category`, before the `reading_result` branch:

```python
    if phase == "reading_search":
        return "reading_search_failure"
```

Constructor:

```python
    def __init__(self, pool_provider, source_ref, search: ReadingSearchConfig | None = None):
        self.pool_provider = pool_provider
        self.source_ref = source_ref
        self.search = search
        self.task = None
        self.index_task = None
        self.bus = None
```

Replace the `else:` (reading_result) branch in `handle` with:

```python
                else:
                    if command.query is not None:
                        phase = "reading_search"
                        mode = "query"
                        if self.search is None or not self.search.enabled:
                            raise SearchUnavailableError("semantic reading search is not configured")
                        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as client:
                            introspection = await search_readings(
                                conn, self.search, client=client, query=command.query,
                                limit=command.limit or DEFAULT_LIMIT, since=command.since,
                            )
                    else:
                        phase = "reading_result"
                        mode = "lookup" if (command.request_id or command.url) else "recent"
                        introspection = await reading_results(
                            conn,
                            request_id=command.request_id,
                            url=command.url,
                            limit=command.limit or DEFAULT_LIMIT,
                            since=command.since,
                        )
                    result = introspection.model_dump(mode="json")
                    logger.info(
                        "introspect op=reading_result corr=%s items=%d total=%s mode=%s",
                        envelope.correlation_id,
                        len(introspection.items),
                        introspection.total_available,
                        mode,
                    )
```

In the `except Exception as exc:` block, replace the response line with:

```python
            error = _SEARCH_UNAVAILABLE if isinstance(exc, SearchUnavailableError) else _SAFE_ERROR
            response = ReadingToolResultV1(ok=False, error=error)
```

Replace `start`/`stop` and add the index loop:

```python
    async def start(self, bus):
        self.bus = bus
        self.task = asyncio.create_task(self._run(), name="hub-reading-tools")
        if self.search is not None and self.search.enabled:
            self.index_task = asyncio.create_task(self._index_loop(), name="hub-reading-search-index")

    async def stop(self):
        for attr in ("task", "index_task"):
            task = getattr(self, attr)
            if task:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                setattr(self, attr, None)

    async def index_once(self):
        pool = self.pool_provider()
        if pool is None:
            return None
        # Release the connection before embedding; a pass can take seconds.
        async with pool.acquire() as conn:
            rows = await verified_rows(conn)
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as client:
            result = await index_missing_readings(
                rows, self.search, client=client, bus=self.bus, source=self.source_ref,
            )
        logger.info("reading_search_index indexed=%d pending=%d", result.indexed, result.pending)
        return result

    async def _index_loop(self):
        while True:
            try:
                await self.index_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "reading_search_index_failed exc_type=%s detail=%s",
                    type(exc).__name__, _safe_exception_detail(exc),
                )
            await asyncio.sleep(self.search.index_interval_sec)
```

- [ ] **Step 4: Run listener tests.**

Run: `PYTHONPATH=.:services/orion-hub:services/orion-hub/tests $PY -m pytest services/orion-hub/tests/test_reading_ingress.py -q`
Expected: all PASS (including the pre-existing `reading_result` log test — the log line keeps its prefix).

- [ ] **Step 5: Settings.** In `services/orion-hub/app/settings.py`, directly after `HUB_READING_DURABLE_URL`:

```python
    # Semantic search over verified readings (reading_results query=...). Empty
    # CHROMA or EMBED URL = search off; query calls then answer "unknown".
    HUB_READING_SEARCH_CHROMA_URL: str = Field(default="", alias="HUB_READING_SEARCH_CHROMA_URL")
    HUB_READING_SEARCH_EMBED_URL: str = Field(default="", alias="HUB_READING_SEARCH_EMBED_URL")
    HUB_READING_SEARCH_COLLECTION: str = Field(
        default="orion_reading_results", alias="HUB_READING_SEARCH_COLLECTION"
    )
    HUB_READING_SEARCH_MIN_SIMILARITY: float = Field(
        default=0.60, ge=0.0, le=1.0, alias="HUB_READING_SEARCH_MIN_SIMILARITY"
    )
    HUB_READING_SEARCH_INDEX_INTERVAL_SEC: float = Field(
        default=300.0, gt=0, alias="HUB_READING_SEARCH_INDEX_INTERVAL_SEC"
    )
    HUB_READING_SEARCH_INDEX_BATCH: int = Field(
        default=10, ge=1, le=50, alias="HUB_READING_SEARCH_INDEX_BATCH"
    )
```

- [ ] **Step 6: `.env_example`.** In `services/orion-hub/.env_example`, directly after `HUB_READING_DURABLE_URL=...`:

```bash
# Semantic search over verified readings (orion-introspect reading_results query=...).
# Each verified reading is embedded once by the Hub's index loop and upserted via
# orion-vector-writer into Chroma; a query embeds only the question. Hub is
# network_mode: host, so use published host ports. Empty CHROMA/EMBED URL = off.
HUB_READING_SEARCH_CHROMA_URL=http://127.0.0.1:8500
HUB_READING_SEARCH_EMBED_URL=http://127.0.0.1:8320/embedding
HUB_READING_SEARCH_COLLECTION=orion_reading_results
# Cosine floor for a hit. Calibrated 2026-09-28: related questions 0.66-0.79,
# unrelated 0.47-0.54 (bge never rests near 0). Recalibrate with
# services/orion-hub/evals/run_reading_search_calibration.py.
HUB_READING_SEARCH_MIN_SIMILARITY=0.60
HUB_READING_SEARCH_INDEX_INTERVAL_SEC=300
HUB_READING_SEARCH_INDEX_BATCH=10
```

- [ ] **Step 7: Single-source owner.** In `scripts/check_env_key_single_source.py`, add to `OWNERS`:

```python
    "HUB_READING_SEARCH_MIN_SIMILARITY": "services/orion-hub/.env_example",
```

- [ ] **Step 8: Wire main.py.** In `services/orion-hub/scripts/main.py`, import `ReadingSearchConfig` from `orion.world_pulse_read.search` (next to the other `orion.world_pulse_read` imports) and change the `ReadingListener(...)` construction to:

```python
            reading_listener = ReadingListener(
                pool_provider=lambda: getattr(app.state, "memory_pg_pool", None),
                source_ref=world_pulse_read_pipeline._source_ref,
                search=ReadingSearchConfig(
                    chroma_url=settings.HUB_READING_SEARCH_CHROMA_URL,
                    embed_url=settings.HUB_READING_SEARCH_EMBED_URL,
                    collection=settings.HUB_READING_SEARCH_COLLECTION,
                    min_similarity=settings.HUB_READING_SEARCH_MIN_SIMILARITY,
                    index_interval_sec=settings.HUB_READING_SEARCH_INDEX_INTERVAL_SEC,
                    index_batch=settings.HUB_READING_SEARCH_INDEX_BATCH,
                ),
            )
```

- [ ] **Step 9: README.** In `services/orion-hub/README.md`, after the paragraph at the `reading_tool_failed` line (~3156), add:

```markdown
**Reading search (orion-introspect `reading_results query=...`).** A loop in `ReadingListener` embeds each verified reading once (title + learned text, via vector-host HTTP `/embedding`) and publishes `VectorUpsertV1` on `orion:vector:semantic:upsert`; orion-vector-writer stores it in Chroma `HUB_READING_SEARCH_COLLECTION`. The loop is hash-aware, so a Stage 2 summary replacing Stage 1 text is re-indexed, and it doubles as the backfill (`reading_search_index indexed=N pending=M` every `HUB_READING_SEARCH_INDEX_INTERVAL_SEC`). A query embeds only the question, keeps Chroma hits at or above `HUB_READING_SEARCH_MIN_SIMILARITY`, and re-reads each hit from Postgres through the same verified-reading gate. Embedder/Chroma failure or a not-yet-built index is logged as `reading_search_failure` and reported to the model as "answer unknown", never as no results. Recalibrate the floor with `python services/orion-hub/evals/run_reading_search_calibration.py`.
```

Also add `reading_search_failure` to the category list in that `reading_tool_failed` line.

- [ ] **Step 10: Env sync and gates.**

Run (from the worktree root):

```bash
python scripts/sync_local_env_from_example.py
$PY scripts/check_env_template_parity.py
$PY scripts/check_env_key_single_source.py
PYTHONPATH=.:services/orion-hub:services/orion-hub/tests $PY -m pytest services/orion-hub/tests/test_reading_ingress.py -q
```

Expected: sync reports the six new Hub keys added (report any skipped key); parity and single-source OK; tests pass. If a gate script is missing, say so.

- [ ] **Step 11: Commit.**

```bash
git add services/orion-hub/app/settings.py services/orion-hub/.env_example services/orion-hub/README.md services/orion-hub/scripts/reading_listener.py services/orion-hub/scripts/main.py services/orion-hub/tests/test_reading_ingress.py scripts/check_env_key_single_source.py
git status --short   # confirm no .env staged
git commit -m "feat(hub): answer reading_results queries and index verified readings"
```

---

### Task 4: Harness — pass `query`, lead the description and brief with search, smoke

**Files:**
- Modify: `orion/introspect/tools.py`, `orion/introspect/brief.py`, `scripts/smoke_introspect.py`
- Test: `orion/introspect/tests/test_introspect_tools.py`, `orion/introspect/tests/test_introspect_harness_wiring.py`, `orion/introspect/tests/test_introspect_mcp_server.py`

- [ ] **Step 1: Write failing tests.** Append to `orion/introspect/tests/test_introspect_tools.py`:

```python
def test_reading_results_forwards_query_without_url_normalization():
    bus = ReplyBus(_ok_payload())
    out = _invoke(bus, args={"query": "  graphics cards ", "limit": 2})
    assert out["ok"] is True
    [(_, envelope, _, _)] = bus.sent
    assert envelope.payload["query"] == "graphics cards"
    assert envelope.payload["limit"] == 2
    assert "url" not in envelope.payload or envelope.payload["url"] is None


def test_description_leads_with_semantic_query():
    [spec] = IntrospectTools(ReplyBus(), BINDING).tool_specs()
    assert spec.description.lower().startswith("search")
    assert "query" in spec.description and "similarity" in spec.description
    assert "query" in spec.arguments.model_json_schema()["properties"]
```

Append to `orion/introspect/tests/test_introspect_harness_wiring.py` (use the binding fixture/constant that file already uses; name it accordingly):

```python
def test_brief_tells_orion_to_search_by_meaning():
    from orion.introspect.brief import introspect_brief_lines
    from orion.schemas.introspect import IntrospectToolBindingV1

    binding = IntrospectToolBindingV1(
        invocation_context="unified_chat", parent_run_id="r", parent_trace_id="t", memory_allowed=True,
    )
    text = " ".join(introspect_brief_lines(binding))
    assert "query=" in text and "similarity" in text
    assert "answer is unknown" in text
```

- [ ] **Step 2: Run to verify failure.**

Run: `PYTHONPATH=. $PY -m pytest orion/introspect/tests -q`
Expected: the new tests FAIL (query not forwarded; description/brief unchanged).

- [ ] **Step 3: Implement `orion/introspect/tools.py`.** Add `query=args.query,` to the `ReadingToolRequestV1(...)` call in `invoke`, and replace the description:

```python
READING_RESULTS_DESCRIPTION = (
    "Search what you actually learned from sources read through your reading pipeline. "
    "Pass query=<what you want to recall, in plain words> to find readings by meaning; each "
    "item carries similarity (0-1) and only matches above a relevance floor come back. Or pass "
    "url or request_id for one reading, or nothing for your most recent finished reads. "
    "since=<ISO timestamp with timezone> narrows query or recent mode; limit up to 5. Each "
    "item's text is the learned summary; learned=false means no output exists yet, so report "
    "its reading_status instead. Results are source-attributed candidates, not settled beliefs. "
    "items=[] means nothing matched; a tool error means the answer is unknown, never that "
    "nothing happened."
)
```

- [ ] **Step 4: Implement `orion/introspect/brief.py`:**

```python
def introspect_brief_lines(binding: IntrospectToolBindingV1) -> list[str]:
    return [
        (
            "Introspect MCP (orion-introspect) is available: reading_results searches what you "
            "actually learned from sources read through your reading pipeline. Use query=<topic "
            "in plain words> to recall readings by meaning (items carry similarity; weak matches "
            "are dropped), url or request_id for one known reading, or nothing for your most "
            "recent finished reads. Call it before describing what you learned from reading "
            "instead of reconstructing it; reading_status says where a request is in the queue, "
            "reading_results says what came out of it. Results are source-attributed candidates, "
            "not settled beliefs. items=[] means nothing matched; a tool error means the answer "
            "is unknown -- say so, never report an error as nothing having happened. "
            "learned=false means no output exists yet: report its reading_status."
        ),
    ]
```

- [ ] **Step 5: Smoke `--query`.** In `scripts/smoke_introspect.py`: update the docstring usage line to also show `--query "graphics cards"`; add `parser.add_argument("--query")`; build arguments as:

```python
    if args.query:
        arguments = {"query": args.query, "limit": args.limit}
    elif args.url:
        arguments = {"url": args.url}
    else:
        arguments = {"limit": args.limit}
```

and after the degenerate check add:

```python
    unscored = [i["id"] for i in result["items"] if args.query and "similarity" not in i["extra"]]
    if unscored:
        print(f"DEGENERATE: query hits without similarity: {unscored}", file=sys.stderr)
        return 1
```

Keep the empty-recent-window check limited to recent mode (`not args.url and not args.query`).

- [ ] **Step 6: Run tests.**

Run: `PYTHONPATH=. $PY -m pytest orion/introspect/tests -q && $PY -m py_compile scripts/smoke_introspect.py`
Expected: all PASS. If a pre-existing test pins the old description/brief text, update it to the new wording (do not weaken what it asserts about unknown vs empty).

- [ ] **Step 7: Commit.**

```bash
git add orion/introspect/tools.py orion/introspect/brief.py orion/introspect/tests scripts/smoke_introspect.py
git commit -m "feat(introspect): reading_results searches by meaning"
```

---

### Task 5: Live calibration eval and full gates

**Files:**
- Create: `services/orion-hub/evals/run_reading_search_calibration.py`

- [ ] **Step 1: Write the eval script:**

```python
#!/usr/bin/env python3
"""Live eval: does the reading-search floor separate related from unrelated questions?

    python services/orion-hub/evals/run_reading_search_calibration.py \
        [--related "graphics cards" ...] [--unrelated "cookie recipe" ...] [--min-similarity X]

Embeds each question once (vector-host HTTP /embedding, which persists nothing),
asks Chroma for the nearest indexed readings, and prints the top similarities.
Exit 0 = every related question's best hit clears the floor and every unrelated
one stays below it; 1 = the floor does not separate them; 2 = index or embedder
unavailable. Defaults target the 2026-09-28 corpus (mostly GPU/Nvidia coverage);
pass --related/--unrelated when the corpus drifts.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import httpx

from orion.world_pulse_read.search import (
    HTTP_TIMEOUT_SEC, ReadingSearchConfig, SearchUnavailableError, embed, nearest,
)

ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env_example"
RELATED = ["graphics cards", "AI chip export controls", "websites blocking automated fetchers"]
UNRELATED = ["chocolate chip cookie recipe", "taking my cat to the vet", "medieval poetry"]


def _default(key: str, fallback: str) -> str:
    if os.environ.get(key):
        return os.environ[key]
    for line in ENV_EXAMPLE.read_text().splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    return fallback


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--related", action="append")
    parser.add_argument("--unrelated", action="append")
    parser.add_argument("--min-similarity", type=float)
    args = parser.parse_args()
    cfg = ReadingSearchConfig(
        chroma_url=_default("HUB_READING_SEARCH_CHROMA_URL", ""),
        embed_url=_default("HUB_READING_SEARCH_EMBED_URL", ""),
        collection=_default("HUB_READING_SEARCH_COLLECTION", "orion_reading_results"),
        min_similarity=args.min_similarity if args.min_similarity is not None
        else float(_default("HUB_READING_SEARCH_MIN_SIMILARITY", "nan")),
    )
    groups = [("related", args.related or RELATED), ("unrelated", args.unrelated or UNRELATED)]
    failures = []
    best: dict[str, list[float]] = {"related": [], "unrelated": []}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as client:
        try:
            for group, questions in groups:
                for question in questions:
                    vector, _ = await embed(client, cfg, question)
                    top = await nearest(client, cfg, vector, n=3)
                    score = top[0][1] if top else 0.0
                    best[group].append(score)
                    shown = ", ".join(f"{sid[:12]}={sim:.3f}" for sid, sim in top)
                    print(f"{group:9} {score:.3f}  {question!r}  [{shown}]")
                    if (group == "related") != (score >= cfg.min_similarity):
                        failures.append(f"{group} {question!r} top={score:.3f}")
        except SearchUnavailableError as exc:
            print(f"UNKNOWN: {exc}", file=sys.stderr)
            return 2
    low, high = min(best["related"]), max(best["unrelated"])
    print(f"floor={cfg.min_similarity:.2f} related_min={low:.3f} unrelated_max={high:.3f} gap={low - high:.3f}")
    if failures:
        print("FAIL: floor does not separate: " + "; ".join(failures), file=sys.stderr)
        return 1
    print("OK: floor separates related from unrelated questions", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 2: Offline sanity.**

Run: `PYTHONPATH=. $PY -m py_compile services/orion-hub/evals/run_reading_search_calibration.py && PYTHONPATH=. HUB_READING_SEARCH_CHROMA_URL=http://127.0.0.1:1 HUB_READING_SEARCH_EMBED_URL=http://127.0.0.1:1/embedding $PY services/orion-hub/evals/run_reading_search_calibration.py; echo exit=$?`
Expected: `UNKNOWN: ...` and `exit=2` (unreachable endpoints are unknown, not a pass).

(The live run happens after deploy, once the index loop has filled the collection — controller step.)

- [ ] **Step 3: Full focused suite and gates.**

```bash
PYTHONPATH=. $PY -m pytest orion/introspect/tests orion/world_pulse_read/tests/test_reading_search.py -q
PYTHONPATH=.:services/orion-hub:services/orion-hub/tests $PY -m pytest services/orion-hub/tests/test_reading_ingress.py services/orion-hub/tests/test_world_pulse_read_queue.py services/orion-hub/tests/test_world_pulse_read_pipeline.py -q
RUN_READING_POSTGRES=1 PYTHONPATH=.:services/orion-hub:services/orion-hub/tests $PY -m pytest services/orion-hub/tests/test_reading_postgres.py -q
git diff --check
$PY scripts/check_env_template_parity.py
$PY scripts/check_env_key_single_source.py
$PY scripts/check_bus_channels.py
$PY scripts/check_schema_registry.py
```

Expected: all pass (report exact counts). If a listed test file does not exist, drop it and say so.

- [ ] **Step 4: Commit.**

```bash
git add services/orion-hub/evals/run_reading_search_calibration.py
git commit -m "test(reading): live calibration eval for the search floor"
```

---

## Controller-only steps (not for implementer subagents)

1. Final whole-branch review subagent; fix findings.
2. `scripts/safe_graphify_update.sh`.
3. Deploy the Hub only when `git rev-list --count HEAD..origin/main` is 0 (ideally after #2381 merges and this branch is rebased onto main). Copy the primary checkout's synced `.env` and `services/orion-hub/.env` into the worktree first, then `scripts/safe_docker_build.sh orion-hub up -d --build`.
4. Live evidence: Hub log `reading_search_index indexed=... pending=0`; Chroma `orion_reading_results` count equals verified-reading count; `run_reading_search_calibration.py` exit 0; `ORION_BUS_URL=redis://100.92.216.81:6379/0 python scripts/smoke_introspect.py --query "graphics cards"` exit 0 with `similarity` on items.
5. Push, PR (base `feat/introspect-slice1` until #2381 merges, then main), CI check.
