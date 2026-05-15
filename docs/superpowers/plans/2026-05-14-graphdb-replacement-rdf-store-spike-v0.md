# GraphDB Replacement Spike v0 (RDF Store Abstraction + Fuseki) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a backend-neutral RDF persistence layer in `orion-rdf-writer`, keep GraphDB as the default backend and wire Apache Jena Fuseki as the first alternate backend with durable Athena storage, async write decoupling from the bus hot path, and a chat-history smoke readback proving `chat.history` → RDF → store → SPARQL without touching `rdf_builder.py` semantics, autonomy reads, substrate store, or concept induction.

**Architecture:** A small `RdfStoreClient` protocol plus `build_rdf_store_client(settings)` selects `GraphDbRdfStoreClient`, `FusekiRdfStoreClient`, or `GenericSparqlRdfStoreClient` (with `rdf4j` as a URL-gated alias). All bus-driven RDF still flows `build_triples_from_envelope` → enqueue or direct `_push_to_rdf_store` → client `write_graph`. An `asyncio` worker pool drains a bounded queue under a global in-flight semaphore, with retries, dead-letter NDJSON, and optional bus notifications via a publisher hook registered from `main.py` after the `Hunter` connects.

**Tech stack:** Python 3.12, FastAPI, httpx 0.28.x, pydantic-settings 2.x, Redis bus (`OrionBusAsync` / `Hunter`), Docker Compose, `stain/jena-fuseki` image, existing GraphDB HTTP repository statements API.

**Branch (create/use before code changes):** `graphdb-replacement-v0-chat-rdf-store`

---

## File map (create / modify / test)

| Path | Responsibility |
|------|----------------|
| `services/orion-rdf-writer/app/rdf_store.py` | **Create.** `RdfWriteResult`, `normalize_graph_name`, `RdfStoreClient` protocol, GraphDB/Fuseki/Generic clients, `build_rdf_store_client`, shared httpx limits helpers. |
| `services/orion-rdf-writer/app/settings.py` | **Modify.** Optional `GRAPHDB_URL`, new `RDF_STORE_*` and `RDF_WRITE_*` fields; keep all existing channel/env keys. |
| `services/orion-rdf-writer/app/service.py` | **Modify.** Replace `_push_to_graphdb` with `_push_to_rdf_store`, `RdfWriteJob`, queue, workers, retries, dead-letter, structured logs (`rdf_write_enqueued`, `rdf_write_committed`), optional bus error hook. |
| `services/orion-rdf-writer/app/main.py` | **Modify.** Lifespan: build store client, `start_rdf_write_workers`, register bus publisher from `hunter.bus`, `stop_rdf_write_workers` with bounded drain; extend `/health`. |
| `services/orion-rdf-writer/app/router.py` | **Modify.** Fix broken ingest (`build_triples` / wrong arity); call async `_push_to_rdf_store` (or shared internal async helper). |
| `services/orion-rdf-writer/docker-compose.yml` | **Modify.** Pass all new `RDF_STORE_*` and `RDF_WRITE_*` env vars; add `CHANNEL_WORLD_PULSE_GRAPH` (present in `settings.py` but missing from compose today). |
| `services/orion-rdf-writer/.env_example` | **Modify.** Document new vars + Fuseki/JVM/Athena knobs (per spec section H). |
| `.env_example` (repo root) | **Modify only if** you add mesh-wide operators knobs not mirrored in service `.env_example`; prefer keeping the service file as the RDF-writer contract per `AGENTS.md`. |
| `services/orion-rdf-store/README.md` | **Create.** Operator stack docs (Fuseki runbook merged here); clarifies this is not the Python writer; non-migrations list. |
| `services/orion-rdf-store/docker-compose.yml` | **Create.** `orion-athena-fuseki` only: `stain/jena-fuseki:latest`, bind `${FUSEKI_DATA_DIR:-/mnt/storage-lukewarm/rdf-store/fuseki}:/fuseki`, external `app-net`, `ADMIN_PASSWORD`, `JVM_ARGS`, healthcheck. |
| `services/orion-rdf-store/Makefile` | **Create.** Operator targets (`help`, `up`, `down`, `logs`, `preflight`, `config`, …). |
| `services/orion-rdf-store/.env_example` | **Create.** `NET`, `PROJECT`, `RDF_STORE_DATA_ROOT`, Fuseki/JVM, endpoint contract for the writer. |
| `scripts/smoke_chat_to_rdf_store.py` | **Create** (preferred over bloating the old script). Synthetic `chat.history` publish + SPARQL readback with retries; support GraphDB and Fuseki URL resolution from env. |
| `scripts/smoke_chat_to_rdf.py` | **Optional modify.** Add one-line comment pointing to the new store-aware smoke script (no behavior change) if you want discoverability. |
| `services/orion-rdf-writer/README.md` | **Modify.** Backends, async queue, scaling, smoke command, “chat is canary only” warning. |
| `services/orion-rdf-writer/tests/test_rdf_store.py` | **Create.** Normalization, factory, endpoint construction, GraphDB URL shape, secret-free health helpers if factored. |
| `services/orion-rdf-writer/tests/test_rdf_write_queue.py` | **Create.** Queue, semaphore, retry/backoff, dead-letter, shutdown/drain behavior (mock httpx). |
| `services/orion-rdf-writer/tests/test_service_rdf_store_integration.py` | **Create.** `handle_envelope` parametrized kinds → same write path; async vs direct mode. |
| `services/orion-rdf-writer/app/rdf_builder.py` | **Do not change semantics** (only if an unavoidable import cycle forces a one-line noop; avoid). |

**Explicitly out of scope:** `orion-recall` / hub substrate reads, concept induction, ontology edits, removing GraphDB, migrating non-writer consumers.

---

### Task 0: Branch and compile baseline

**Files:** (git only)

- [ ] **Step 1: Create branch**

```bash
git fetch origin 2>/dev/null || true
git checkout -b graphdb-replacement-v0-chat-rdf-store
```

- [ ] **Step 2: Baseline compile (expect pass on current tree)**

```bash
cd /mnt/scripts/Orion-Sapienform
python3 -m compileall services/orion-rdf-writer/app
```

Expected: no syntax errors.

---

### Task 1: Relax GraphDB URL + add RDF store / async settings

**Files:**
- Modify: `services/orion-rdf-writer/app/settings.py` (full class body merge; preserve `get_all_subscribe_channels` / `get_skip_kinds`)

- [ ] **Step 1: Replace `settings.py` GraphDB block and append new fields**

Use this complete `Settings` class (merge with your file’s existing channel definitions — the block below preserves every field name from the current `settings.py` in the repo and only extends it):

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List


class Settings(BaseSettings):
    # RDF + GraphDB (GRAPHDB_URL optional when RDF_STORE_BACKEND != graphdb)
    GRAPHDB_URL: str | None = Field(default=None, env="GRAPHDB_URL")
    GRAPHDB_REPO: str = Field(default="collapse", env="GRAPHDB_REPO")
    GRAPHDB_USER: str | None = Field(None, env="GRAPHDB_USER")
    GRAPHDB_PASS: str | None = Field(None, env="GRAPHDB_PASS")

    # Backend-neutral RDF store
    RDF_STORE_BACKEND: str = Field(default="graphdb", env="RDF_STORE_BACKEND")
    RDF_STORE_BASE_URL: str | None = Field(default=None, env="RDF_STORE_BASE_URL")
    RDF_STORE_DATASET: str = Field(default="orion", env="RDF_STORE_DATASET")
    RDF_STORE_QUERY_URL: str | None = Field(default=None, env="RDF_STORE_QUERY_URL")
    RDF_STORE_UPDATE_URL: str | None = Field(default=None, env="RDF_STORE_UPDATE_URL")
    RDF_STORE_GRAPH_STORE_URL: str | None = Field(default=None, env="RDF_STORE_GRAPH_STORE_URL")
    RDF_STORE_USER: str | None = Field(default=None, env="RDF_STORE_USER")
    RDF_STORE_PASS: str | None = Field(default=None, env="RDF_STORE_PASS")
    RDF_STORE_TIMEOUT_SEC: float = Field(default=10.0, env="RDF_STORE_TIMEOUT_SEC")

    RDF_WRITE_ASYNC_ENABLED: bool = Field(default=True, env="RDF_WRITE_ASYNC_ENABLED")
    RDF_WRITE_QUEUE_MAXSIZE: int = Field(default=5000, env="RDF_WRITE_QUEUE_MAXSIZE")
    RDF_WRITE_WORKERS: int = Field(default=8, env="RDF_WRITE_WORKERS")
    RDF_WRITE_MAX_IN_FLIGHT: int = Field(default=32, env="RDF_WRITE_MAX_IN_FLIGHT")
    RDF_WRITE_HTTP_MAX_CONNECTIONS: int = Field(default=64, env="RDF_WRITE_HTTP_MAX_CONNECTIONS")
    RDF_WRITE_HTTP_MAX_KEEPALIVE: int = Field(default=32, env="RDF_WRITE_HTTP_MAX_KEEPALIVE")
    RDF_WRITE_RETRY_ATTEMPTS: int = Field(default=3, env="RDF_WRITE_RETRY_ATTEMPTS")
    RDF_WRITE_RETRY_BASE_DELAY_SEC: float = Field(default=0.25, env="RDF_WRITE_RETRY_BASE_DELAY_SEC")
    RDF_WRITE_RETRY_MAX_DELAY_SEC: float = Field(default=5.0, env="RDF_WRITE_RETRY_MAX_DELAY_SEC")
    RDF_WRITE_DEAD_LETTER_ENABLED: bool = Field(default=True, env="RDF_WRITE_DEAD_LETTER_ENABLED")
    RDF_WRITE_DEAD_LETTER_PATH: str = Field(
        default="/tmp/orion-rdf-writer-deadletter.ndjson",
        env="RDF_WRITE_DEAD_LETTER_PATH",
    )

    # === ORION BUS (Shared Core) ===
    ORION_BUS_URL: str = Field(..., env="ORION_BUS_URL")
    ORION_BUS_ENABLED: bool = Field(default=True, env="ORION_BUS_ENABLED")
    ORION_BUS_ENFORCE_CATALOG: bool = Field(default=False, env="ORION_BUS_ENFORCE_CATALOG")

    # === LISTENER CHANNELS ===
    CHANNEL_RDF_ENQUEUE: str = Field(default="orion:rdf:enqueue", env="CHANNEL_RDF_ENQUEUE")
    CHANNEL_EVENTS_COLLAPSE: str = Field(default="orion:collapse:intake", env="CHANNEL_EVENTS_COLLAPSE")
    CHANNEL_EVENTS_TAGGED: str = Field(default="orion:tags:enriched", env="CHANNEL_EVENTS_TAGGED")
    CHANNEL_EVENTS_TAGGED_CHAT: str = Field(default="orion:tags:chat:enriched", env="CHANNEL_EVENTS_TAGGED_CHAT")
    CHANNEL_CORE_EVENTS: str = Field(default="orion:core:events", env="CHANNEL_CORE_EVENTS")
    CHANNEL_WORKER_RDF: str = Field(default="orion:rdf:worker", env="CHANNEL_WORKER_RDF")
    CHANNEL_COGNITION_TRACE_PUB: str = Field(default="orion:cognition:trace", env="CHANNEL_COGNITION_TRACE_PUB")
    CHANNEL_CHAT_HISTORY_TURN: str = Field(default="orion:chat:history:turn", env="CHANNEL_CHAT_HISTORY_TURN")
    CHANNEL_CHAT_HISTORY_LOG: str = Field(default="orion:chat:history:log", env="CHANNEL_CHAT_HISTORY_LOG")
    CHANNEL_MEMORY_IDENTITY_SNAPSHOT: str = Field(default="orion:memory:identity:snapshot", env="CHANNEL_MEMORY_IDENTITY_SNAPSHOT")
    CHANNEL_MEMORY_DRIVES_AUDIT: str = Field(default="orion:memory:drives:audit", env="CHANNEL_MEMORY_DRIVES_AUDIT")
    CHANNEL_MEMORY_GOALS_PROPOSED: str = Field(default="orion:memory:goals:proposed", env="CHANNEL_MEMORY_GOALS_PROPOSED")

    # === PUBLISH CHANNELS ===
    CHANNEL_RDF_CONFIRM: str = Field(default="orion:rdf:confirm", env="CHANNEL_RDF_CONFIRM")
    CHANNEL_RDF_ERROR: str = Field(default="orion:rdf:error", env="CHANNEL_RDF_ERROR")
    CORTEX_LOG_CHANNEL: str = Field(default="orion:cortex:telemetry", env="CORTEX_LOG_CHANNEL")

    SERVICE_NAME: str = Field(default="orion-rdf-writer", env="SERVICE_NAME")
    SERVICE_VERSION: str = Field(default="0.2.0", env="SERVICE_VERSION")
    NODE_NAME: str = Field(default="unknown")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    RDF_SKIP_KINDS: str = Field(default="", env="RDF_SKIP_KINDS")
    RDF_SKIP_REJECTED: bool = Field(default=True, env="RDF_SKIP_REJECTED")
    RDF_DURABLE_ONLY: bool = Field(default=False, env="RDF_DURABLE_ONLY")
    WORLD_PULSE_GRAPH_ENABLED: bool = Field(default=False, env="WORLD_PULSE_GRAPH_ENABLED")
    WORLD_PULSE_GRAPH_DRY_RUN: bool = Field(default=True, env="WORLD_PULSE_GRAPH_DRY_RUN")
    WORLD_PULSE_GRAPH_REQUIRE_POLICY_STAMP: bool = Field(default=True, env="WORLD_PULSE_GRAPH_REQUIRE_POLICY_STAMP")
    CHANNEL_WORLD_PULSE_GRAPH: str = Field(default="orion:world_pulse:graph:upsert", env="CHANNEL_WORLD_PULSE_GRAPH")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def get_all_subscribe_channels(self) -> List[str]:
        channels = [
            self.CHANNEL_RDF_ENQUEUE,
            "orion:rdf-collapse:enqueue",
            self.CHANNEL_EVENTS_COLLAPSE,
            self.CHANNEL_EVENTS_TAGGED,
            self.CHANNEL_EVENTS_TAGGED_CHAT,
            "orion:chat:social:stored",
            self.CHANNEL_CORE_EVENTS,
            self.CHANNEL_WORKER_RDF,
            self.CORTEX_LOG_CHANNEL,
            self.CHANNEL_COGNITION_TRACE_PUB,
            self.CHANNEL_CHAT_HISTORY_TURN,
            self.CHANNEL_CHAT_HISTORY_LOG,
            self.CHANNEL_MEMORY_IDENTITY_SNAPSHOT,
            self.CHANNEL_MEMORY_DRIVES_AUDIT,
            self.CHANNEL_MEMORY_GOALS_PROPOSED,
            "orion:metacog:trace",
            self.CHANNEL_WORLD_PULSE_GRAPH,
        ]
        seen = set()
        ordered: List[str] = []
        for channel in channels:
            channel = (channel or "").strip()
            if not channel or channel in seen:
                continue
            seen.add(channel)
            ordered.append(channel)
        return ordered

    def get_skip_kinds(self) -> List[str]:
        return [k.strip() for k in (self.RDF_SKIP_KINDS or "").split(",") if k.strip()]


settings = Settings()
```

- [ ] **Step 2: Update tests that `setdefault("GRAPHDB_URL", ...)`**

In `services/orion-rdf-writer/tests/test_world_pulse_graph_gates.py` and `services/orion-rdf-writer/tests/test_autonomy_materialization.py`, keep `GRAPHDB_URL` setdefault **or** switch to `RDF_STORE_BACKEND=fuseki` + dummy base in a follow-up task; minimal fix: leave setdefault so imports still work until factory validates backend.

- [ ] **Step 3: Run compile**

```bash
python3 -m compileall services/orion-rdf-writer/app
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add services/orion-rdf-writer/app/settings.py
git commit -m "$(cat <<'EOF'
feat(rdf-writer): add RDF store and async pipeline settings

Make GRAPHDB_URL optional for non-graphdb backends and declare RDF_STORE_*/RDF_WRITE_* knobs without changing channel contracts.
EOF
)"
```

---

### Task 2: Add `rdf_store.py` (normalization + clients + factory)

**Files:**
- Create: `services/orion-rdf-writer/app/rdf_store.py`

- [ ] **Step 1: Create the file with the following complete implementation**

```python
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse, urlunparse

import httpx

from app.settings import Settings


@dataclass(frozen=True)
class RdfWriteResult:
    backend: str
    graph_name: str | None
    normalized_graph_uri: str | None
    byte_count: int
    status_code: int | None = None
    endpoint: str | None = None
    elapsed_ms: float | None = None


class RdfStoreClient(Protocol):
    async def write_graph(self, content: str, graph_name: str | None = None) -> RdfWriteResult: ...

    async def health(self) -> dict[str, Any]: ...


def normalize_graph_name(graph_name: str | None) -> str | None:
    if graph_name is None:
        return None
    raw = str(graph_name).strip()
    if not raw:
        return None
    lower = raw.lower()
    if lower.startswith("http://") or lower.startswith("https://") or lower.startswith("urn:"):
        return raw
    mapping = {
        "orion:chat": "http://conjourney.net/graph/orion/chat",
        "orion:collapse": "http://conjourney.net/graph/orion/collapse",
        "orion:enrichment": "http://conjourney.net/graph/orion/enrichment",
        "orion:cognition": "http://conjourney.net/graph/orion/cognition",
        "orion:metacog": "http://conjourney.net/graph/orion/metacog",
        "orion:chat:social": "http://conjourney.net/graph/orion/chat/social",
        "orion:default": "http://conjourney.net/graph/orion/default",
    }
    if raw in mapping:
        return mapping[raw]
    safe = re.sub(r"[^A-Za-z0-9._:/-]+", "_", raw)
    safe = safe.replace(":", "/").strip("/")
    if not safe:
        safe = "unknown"
    return f"http://conjourney.net/graph/{safe}"


def _strip_credentials(url: str | None) -> str | None:
    if not url:
        return None
    p = urlparse(url)
    netloc = p.hostname or ""
    if p.port:
        netloc = f"{netloc}:{p.port}"
    return urlunparse((p.scheme, netloc, p.path, p.params, p.query, p.fragment))


def _httpx_auth(user: str | None, password: str | None) -> httpx.Auth | None:
    if user and password is not None:
        return (user, password)
    return None


def _httpx_limits(settings: Settings) -> httpx.Limits:
    return httpx.Limits(
        max_connections=int(settings.RDF_WRITE_HTTP_MAX_CONNECTIONS),
        max_keepalive_connections=int(settings.RDF_WRITE_HTTP_MAX_KEEPALIVE),
    )


def httpx_limits_for_settings(settings: Settings) -> httpx.Limits:
    """Used by the RDF write pipeline to size the shared AsyncClient connection pool."""
    return _httpx_limits(settings)


class GraphDbRdfStoreClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client
        if not settings.GRAPHDB_URL:
            raise ValueError("GRAPHDB_URL is required when RDF_STORE_BACKEND=graphdb")

    @property
    def backend(self) -> str:
        return "graphdb"

    async def write_graph(self, content: str, graph_name: str | None = None) -> RdfWriteResult:
        t0 = time.perf_counter()
        base_url = (
            f"{self._settings.GRAPHDB_URL.rstrip('/')}"
            f"/repositories/{self._settings.GRAPHDB_REPO}/statements"
        )
        params: dict[str, str] = {}
        if graph_name:
            params["context"] = f"<{graph_name}>"
        headers = {"Content-Type": "text/plain"}
        auth = _httpx_auth(self._settings.GRAPHDB_USER, self._settings.GRAPHDB_PASS)
        resp = await self._client.post(
            base_url,
            content=content,
            headers=headers,
            params=params,
            auth=auth,
        )
        resp.raise_for_status()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return RdfWriteResult(
            backend=self.backend,
            graph_name=graph_name,
            normalized_graph_uri=normalize_graph_name(graph_name),
            byte_count=len(content.encode("utf-8")),
            status_code=resp.status_code,
            endpoint=base_url,
            elapsed_ms=elapsed_ms,
        )

    async def health(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "endpoint": f"{self._settings.GRAPHDB_URL.rstrip('/')}/repositories/{self._settings.GRAPHDB_REPO}/statements",
            "repo": self._settings.GRAPHDB_REPO,
        }


class FusekiRdfStoreClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client
        base = (settings.RDF_STORE_BASE_URL or "http://orion-athena-fuseki:3030").rstrip("/")
        ds = settings.RDF_STORE_DATASET.strip().strip("/")
        self._query_url = settings.RDF_STORE_QUERY_URL or f"{base}/{ds}/query"
        self._update_url = settings.RDF_STORE_UPDATE_URL or f"{base}/{ds}/update"
        self._graph_store_url = settings.RDF_STORE_GRAPH_STORE_URL or f"{base}/{ds}/data"

    @property
    def backend(self) -> str:
        return "fuseki"

    async def write_graph(self, content: str, graph_name: str | None = None) -> RdfWriteResult:
        t0 = time.perf_counter()
        url = self._graph_store_url
        params: dict[str, str] = {}
        ng = normalize_graph_name(graph_name)
        if ng:
            params["graph"] = ng
        headers = {"Content-Type": "application/n-triples"}
        auth = _httpx_auth(self._settings.RDF_STORE_USER, self._settings.RDF_STORE_PASS)
        resp = await self._client.post(
            url,
            content=content,
            headers=headers,
            params=params,
            auth=auth,
        )
        resp.raise_for_status()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return RdfWriteResult(
            backend=self.backend,
            graph_name=graph_name,
            normalized_graph_uri=ng,
            byte_count=len(content.encode("utf-8")),
            status_code=resp.status_code,
            endpoint=url,
            elapsed_ms=elapsed_ms,
        )

    async def health(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "query_url": _strip_credentials(self._query_url),
            "update_url": _strip_credentials(self._update_url),
            "graph_store_url": _strip_credentials(self._graph_store_url),
            "dataset": self._settings.RDF_STORE_DATASET,
        }


class GenericSparqlRdfStoreClient:
    """
    Conservative adapter: prefer Graph Store HTTP POST; optional SPARQL UPDATE fallback.
    """

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client
        self._graph_store_url = settings.RDF_STORE_GRAPH_STORE_URL
        self._update_url = settings.RDF_STORE_UPDATE_URL

    @property
    def backend(self) -> str:
        return "generic"

    async def write_graph(self, content: str, graph_name: str | None = None) -> RdfWriteResult:
        t0 = time.perf_counter()
        ng = normalize_graph_name(graph_name)
        if self._graph_store_url:
            url = self._graph_store_url
            params: dict[str, str] = {}
            if ng:
                params["graph"] = ng
            headers = {"Content-Type": "application/n-triples"}
            auth = _httpx_auth(self._settings.RDF_STORE_USER, self._settings.RDF_STORE_PASS)
            resp = await self._client.post(
                url,
                content=content,
                headers=headers,
                params=params,
                auth=auth,
            )
            resp.raise_for_status()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return RdfWriteResult(
                backend=self.backend,
                graph_name=graph_name,
                normalized_graph_uri=ng,
                byte_count=len(content.encode("utf-8")),
                status_code=resp.status_code,
                endpoint=url,
                elapsed_ms=elapsed_ms,
            )
        if not self._update_url:
            raise ValueError("Generic RDF store requires RDF_STORE_GRAPH_STORE_URL or RDF_STORE_UPDATE_URL")
        if ng:
            body = f"INSERT DATA {{ GRAPH <{ng}> {{ {content} }} }}"
        else:
            body = f"INSERT DATA {{ {content} }}"
        headers = {"Content-Type": "application/sparql-update"}
        auth = _httpx_auth(self._settings.RDF_STORE_USER, self._settings.RDF_STORE_PASS)
        resp = await self._client.post(
            self._update_url,
            content=body,
            headers=headers,
            auth=auth,
        )
        resp.raise_for_status()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return RdfWriteResult(
            backend=self.backend,
            graph_name=graph_name,
            normalized_graph_uri=ng,
            byte_count=len(content.encode("utf-8")),
            status_code=resp.status_code,
            endpoint=self._update_url,
            elapsed_ms=elapsed_ms,
        )

    async def health(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "graph_store_url": _strip_credentials(self._graph_store_url),
            "update_url": _strip_credentials(self._update_url),
            "query_url": _strip_credentials(self._settings.RDF_STORE_QUERY_URL),
        }


def build_rdf_store_client(settings: Settings, client: httpx.AsyncClient) -> RdfStoreClient:
    b = (settings.RDF_STORE_BACKEND or "graphdb").strip().lower()
    if b == "graphdb":
        return GraphDbRdfStoreClient(settings, client)
    if b == "fuseki":
        return FusekiRdfStoreClient(settings, client)
    if b == "generic":
        return GenericSparqlRdfStoreClient(settings, client)
    if b == "rdf4j":
        if not settings.RDF_STORE_GRAPH_STORE_URL and not settings.RDF_STORE_UPDATE_URL:
            raise ValueError(
                "RDF_STORE_BACKEND=rdf4j requires RDF_STORE_GRAPH_STORE_URL and/or RDF_STORE_UPDATE_URL "
                "(alias to generic adapter in this spike)."
            )
        return GenericSparqlRdfStoreClient(settings, client)
    raise ValueError(f"Unknown RDF_STORE_BACKEND={settings.RDF_STORE_BACKEND!r} (expected graphdb|fuseki|generic|rdf4j)")
```

- [ ] **Step 2: Run unit tests for normalization only (after Task 11 creates tests, skip here and run full suite in Task 11)**  
For immediate check, run:

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=services/orion-rdf-writer:/mnt/scripts/Orion-Sapienform python3 -c "from app.rdf_store import normalize_graph_name as n; assert n('orion:chat')=='http://conjourney.net/graph/orion/chat'"
```

Expected: no output / exit 0

- [ ] **Step 3: Commit**

```bash
git add services/orion-rdf-writer/app/rdf_store.py
git commit -m "$(cat <<'EOF'
feat(rdf-writer): add RDF store clients and graph name normalization

Introduce GraphDB, Fuseki graph-store POST, generic SPARQL adapter, and backend factory without changing triple builders.
EOF
)"
```

---

### Task 3: Async write pipeline + service integration

**Files:**
- Modify: `services/orion-rdf-writer/app/service.py` (replace `_push_to_graphdb` and extend `handle_envelope`)
- Modify: `services/orion-rdf-writer/app/main.py` (wire client, workers, publisher hook, health)

**Design notes (implement exactly in code):**
- Module-level: `_write_queue: asyncio.Queue | None`, `_workers: list[asyncio.Task]`, `_sem: asyncio.Semaphore | None`, `_http_client: httpx.AsyncClient | None`, `_store: RdfStoreClient | None`, `_rdf_bus_publish: callable | None`, counters `_queue_puts`, `_in_flight` (or use queue.qsize()).
- `RdfWriteJob` fields per spec + `attempt` default 0.
- Retry on `httpx.HTTPError`, `httpx.TimeoutException`, `OSError`; do **not** retry HTTP 400.
- Backoff: `delay = min(RDF_WRITE_RETRY_MAX_DELAY_SEC, RDF_WRITE_RETRY_BASE_DELAY_SEC * (2 ** attempt))` with `await asyncio.sleep(delay)`.
- Dead-letter line (NDJSON): `{"kind":"rdf_write_failed","reason":"...","job":{...}}` — never include `GRAPHDB_PASS` / `RDF_STORE_PASS`.
- `register_rdf_write_publisher(pub: Callable[[str, dict], Awaitable[None]] | None)` called from `main.py` with `hunter.bus.publish` wrapper for `settings.CHANNEL_RDF_ERROR` using payload dict `{ "kind": "rdf.write.error", ... }` (if catalog rejects unknown kinds, publish a minimal dict without breaking codec — mirror how other services publish errors; if `publish` requires `BaseEnvelope`, build `BaseEnvelope` with `kind="rdf.write.error"`, `source=ServiceRef(name=settings.SERVICE_NAME, node=settings.NODE_NAME, version=settings.SERVICE_VERSION)`, `payload={...}`).
- `_push_to_rdf_store(content, graph, *, env_meta: dict)`: if `RDF_WRITE_ASYNC_ENABLED`, `queue.put_nowait` inside try/except `asyncio.QueueFull`; on full queue call `_deadletter("queue_full", ...)` + `rdf_write_backpressure` log + optional bus publish; **re-raise or swallow**: spec says do not pretend success — do **not** log `rdf_write_committed`, and do **not** log the old single-line “Written RDF” at enqueue time.
- When async: log `rdf_write_enqueued` with `kind`, `correlation_id`, `graph`, `bytes`.
- Worker: after successful `write_graph`, log `rdf_write_committed` with `status_code`, `elapsed_ms`, `endpoint` (sanitized via `_strip_credentials` from rdf_store).
- When `RDF_WRITE_ASYNC_ENABLED` is false: `await _write_direct(...)` from handler (still retries inside helper).
- **Graph name to GraphDB:** pass **raw** `graph` from builder to `GraphDbRdfStoreClient` (preserve `context=<{graph_name}>` behavior). **Fuseki** still normalizes inside client (already does).

- [ ] **Step 1: Implement `service.py`**

Replace the body of `services/orion-rdf-writer/app/service.py` with an implementation structured like this (fill in imports and helper bodies; keep `handle_envelope` signature):

```python
import asyncio
import json
import logging
import time
import hashlib
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

import httpx

from orion.core.bus.bus_schemas import BaseEnvelope

from app.rdf_builder import build_triples_from_envelope
from app import rdf_store as rdf_store_mod
from app.rdf_store import RdfStoreClient, build_rdf_store_client, RdfWriteResult
from app.settings import settings

logger = logging.getLogger(settings.SERVICE_NAME)

_rdf_bus_publish: Optional[Callable[[str, dict[str, Any]], Awaitable[None]]] = None


def register_rdf_write_publisher(fn: Optional[Callable[[str, dict[str, Any]], Awaitable[None]]]) -> None:
    global _rdf_bus_publish
    _rdf_bus_publish = fn


# ... keep _payload_fingerprint, _dedupe_cache, _should_dedupe unchanged ...


@dataclass
class RdfWriteJob:
    kind: str
    graph_name: str | None
    content: str
    correlation_id: str | None
    source: str | None
    created_at: float
    payload_fingerprint: str | None
    attempt: int = 0


_write_queue: asyncio.Queue[RdfWriteJob | None] | None = None
_worker_tasks: list[asyncio.Task[None]] = []
_http_client: httpx.AsyncClient | None = None
_store: RdfStoreClient | None = None
_inflight_sem: asyncio.Semaphore | None = None


async def init_rdf_write_pipeline() -> None:
    ...


async def shutdown_rdf_write_pipeline(*, drain_timeout_sec: float = 8.0) -> None:
    ...


async def _push_to_rdf_store(content: str, graph_name: str | None, *, env: BaseEnvelope) -> None:
    ...


async def handle_envelope(env: BaseEnvelope) -> None:
    ...
```

Implementation constraints:
- `init_rdf_write_pipeline` creates `_http_client = httpx.AsyncClient(timeout=settings.RDF_STORE_TIMEOUT_SEC, limits=rdf_store_mod.httpx_limits_for_settings(settings))` (defined in Task 2 `rdf_store.py`).
- `build_rdf_store_client(settings, _http_client)` stored in `_store`.
- `_write_queue = asyncio.Queue(maxsize=settings.RDF_WRITE_QUEUE_MAXSIZE)`.
- `_inflight_sem = asyncio.Semaphore(settings.RDF_WRITE_MAX_IN_FLIGHT)`.
- Spawn `settings.RDF_WRITE_WORKERS` tasks running `rdf_write_worker_loop(worker_id)`.
- Worker loop: `job = await queue.get()`; if job is None: `queue.task_done(); return`; try/finally `queue.task_done()`.

- [ ] **Step 2: Wire `main.py`**

```python
from app.service import handle_envelope, init_rdf_write_pipeline, shutdown_rdf_write_pipeline, register_rdf_write_publisher

@asynccontextmanager
async def lifespan(app: FastAPI):
    global hunter
    ...
    await init_rdf_write_pipeline()

    hunter = Hunter(...)
    await hunter.start_background()

    async def _pub(channel: str, payload: dict) -> None:
        await hunter.bus.publish(channel, payload)

    register_rdf_write_publisher(_pub)

    yield

    await shutdown_rdf_write_pipeline(drain_timeout_sec=8.0)
    register_rdf_write_publisher(None)
    if hunter:
        await hunter.stop()
```

- [ ] **Step 3: Extend `/health`**

Return JSON including: `rdf_store_backend`, `rdf_store_base_url` (use `rdf_store._strip_credentials(settings.RDF_STORE_BASE_URL or settings.GRAPHDB_URL)`), `rdf_store_dataset`, `rdf_write_async_enabled`, queue size/maxsize, workers, in-flight estimate, dead-letter flags, **never** passwords.

- [ ] **Step 4: Run compile**

```bash
python3 -m compileall services/orion-rdf-writer/app
```

- [ ] **Step 5: Commit**

```bash
git add services/orion-rdf-writer/app/service.py services/orion-rdf-writer/app/main.py services/orion-rdf-writer/app/rdf_store.py
git commit -m "$(cat <<'EOF'
feat(rdf-writer): async RDF write queue with retries and dead-letter

Decouple bus handlers from slow HTTP when async mode is on; add lifecycle hooks and richer health without exposing secrets.
EOF
)"
```

---

### Task 4: Fix HTTP ingest router

**Files:**
- Modify: `services/orion-rdf-writer/app/router.py`

Current file incorrectly references `build_triples` and passes three args to `_push_to_graphdb`. Replace `ingest_rdf` with:

```python
from fastapi import APIRouter, HTTPException

from app.rdf_builder import build_triples_from_envelope
from app.service import _push_to_rdf_store
from app.utils import logger
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from uuid import uuid4

router = APIRouter(prefix="/rdf", tags=["rdf"])


@router.post("/ingest")
async def ingest_rdf(payload: dict):
    try:
        kind = str(payload.get("kind") or "rdf.write.request")
        nt_data, graph_name = build_triples_from_envelope(kind, payload)
        if not nt_data:
            raise HTTPException(status_code=400, detail="no triples generated for payload")
        env = BaseEnvelope(
            kind=kind,
            source=ServiceRef(name="http-ingest", node=None, version=None),
            correlation_id=uuid4(),
            payload=payload if isinstance(payload, dict) else {},
        )
        await _push_to_rdf_store(nt_data, graph_name, env=env)
        return {"status": "ok", "id": payload.get("id"), "graph": graph_name}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("HTTP ingest failed")
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Run compile + targeted pytest after tests exist**

```bash
python3 -m compileall services/orion-rdf-writer/app
```

- [ ] **Commit**

```bash
git add services/orion-rdf-writer/app/router.py
git commit -m "$(cat <<'EOF'
fix(rdf-writer): repair /rdf/ingest triple build and RDF store write path

Use build_triples_from_envelope and the shared _push_to_rdf_store helper with a synthetic envelope for provenance fields.
EOF
)"
```

---

### Task 5: Docker Compose + `.env_example` + `services/orion-rdf-store/**`

**Files:**
- Modify: `services/orion-rdf-writer/docker-compose.yml`
- Modify: `services/orion-rdf-writer/.env_example`
- Create: `services/orion-rdf-store/README.md`
- Create: `services/orion-rdf-store/docker-compose.yml`
- Create: `services/orion-rdf-store/Makefile`
- Create: `services/orion-rdf-store/.env_example`

**`services/orion-rdf-writer/docker-compose.yml` additions** (under `rdf-writer.environment`, preserve all existing lines):

```yaml
      - CHANNEL_WORLD_PULSE_GRAPH=${CHANNEL_WORLD_PULSE_GRAPH}
      - RDF_STORE_BACKEND=${RDF_STORE_BACKEND:-graphdb}
      - RDF_STORE_BASE_URL=${RDF_STORE_BASE_URL:-}
      - RDF_STORE_DATASET=${RDF_STORE_DATASET:-orion}
      - RDF_STORE_QUERY_URL=${RDF_STORE_QUERY_URL:-}
      - RDF_STORE_UPDATE_URL=${RDF_STORE_UPDATE_URL:-}
      - RDF_STORE_GRAPH_STORE_URL=${RDF_STORE_GRAPH_STORE_URL:-}
      - RDF_STORE_USER=${RDF_STORE_USER:-}
      - RDF_STORE_PASS=${RDF_STORE_PASS:-}
      - RDF_STORE_TIMEOUT_SEC=${RDF_STORE_TIMEOUT_SEC:-10.0}
      - RDF_WRITE_ASYNC_ENABLED=${RDF_WRITE_ASYNC_ENABLED:-true}
      - RDF_WRITE_QUEUE_MAXSIZE=${RDF_WRITE_QUEUE_MAXSIZE:-5000}
      - RDF_WRITE_WORKERS=${RDF_WRITE_WORKERS:-8}
      - RDF_WRITE_MAX_IN_FLIGHT=${RDF_WRITE_MAX_IN_FLIGHT:-32}
      - RDF_WRITE_HTTP_MAX_CONNECTIONS=${RDF_WRITE_HTTP_MAX_CONNECTIONS:-64}
      - RDF_WRITE_HTTP_MAX_KEEPALIVE=${RDF_WRITE_HTTP_MAX_KEEPALIVE:-32}
      - RDF_WRITE_RETRY_ATTEMPTS=${RDF_WRITE_RETRY_ATTEMPTS:-3}
      - RDF_WRITE_RETRY_BASE_DELAY_SEC=${RDF_WRITE_RETRY_BASE_DELAY_SEC:-0.25}
      - RDF_WRITE_RETRY_MAX_DELAY_SEC=${RDF_WRITE_RETRY_MAX_DELAY_SEC:-5.0}
      - RDF_WRITE_DEAD_LETTER_ENABLED=${RDF_WRITE_DEAD_LETTER_ENABLED:-true}
      - RDF_WRITE_DEAD_LETTER_PATH=${RDF_WRITE_DEAD_LETTER_PATH:-/app/logs/orion-rdf-writer-deadletter.ndjson}
```

**`services/orion-rdf-store/docker-compose.yml`** (adapt to repo pattern: external `app-net`; no nested `fuseki/` path):

```yaml
services:
  orion-athena-fuseki:
    image: stain/jena-fuseki:latest
    container_name: orion-athena-fuseki
    restart: unless-stopped
    environment:
      ADMIN_PASSWORD: ${FUSEKI_ADMIN_PASSWORD:-orion}
      JVM_ARGS: ${FUSEKI_JVM_ARGS:-}
    volumes:
      - ${FUSEKI_DATA_DIR:-/mnt/storage-lukewarm/rdf-store/fuseki}:/fuseki
    ports:
      - "${FUSEKI_PORT:-3030}:3030"
    networks:
      - app-net

networks:
  app-net:
    name: ${NET:-app-net}
    external: true
```

**Note:** `stain/jena-fuseki` documents **`JVM_ARGS`** for heap/GC (see Docker Hub / `jena-docker` Dockerfile); this compose maps `FUSEKI_JVM_ARGS` into `JVM_ARGS`.

**`.env_example` append block** (exact lines from your spec section H — paste verbatim after GraphDB section).

- [ ] **Commit**

```bash
git add services/orion-rdf-writer/docker-compose.yml services/orion-rdf-writer/.env_example services/orion-rdf-store
git commit -m "$(cat <<'EOF'
chore(deploy): wire RDF store envs and add Fuseki compose fragment

Pass RDF_STORE_* and RDF_WRITE_* into rdf-writer; document Athena persistent paths for Fuseki on app-net.
EOF
)"
```

---

### Task 6: Smoke script `scripts/smoke_chat_to_rdf_store.py`

**Files:**
- Create: `scripts/smoke_chat_to_rdf_store.py`

- [ ] **Step 1: Add full script**

```python
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Optional

import requests

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from uuid import uuid4


def _query_url() -> Optional[str]:
    backend = (os.getenv("RDF_STORE_BACKEND") or "graphdb").strip().lower()
    if os.getenv("RDF_STORE_QUERY_URL"):
        return os.getenv("RDF_STORE_QUERY_URL")
    if backend == "fuseki":
        base = (os.getenv("RDF_STORE_BASE_URL") or "http://orion-athena-fuseki:3030").rstrip("/")
        ds = (os.getenv("RDF_STORE_DATASET") or "orion").strip("/")
        return f"{base}/{ds}/query"
    if os.getenv("RECALL_RDF_ENDPOINT_URL"):
        return os.getenv("RECALL_RDF_ENDPOINT_URL")
    graphdb_url = os.getenv("GRAPHDB_URL")
    graphdb_repo = os.getenv("GRAPHDB_REPO")
    if graphdb_url and graphdb_repo:
        return f"{graphdb_url.rstrip('/')}/repositories/{graphdb_repo}"
    return None


def _sparql(session_id: str) -> str:
    return f"""
    PREFIX orion: <http://conjourney.net/orion#>
    SELECT ?turn ?prompt ?response ?timestamp
    WHERE {{
      ?turn a orion:ChatTurn ;
            orion:sessionId "{session_id}" ;
            orion:prompt ?prompt ;
            orion:response ?response .
      OPTIONAL {{ ?turn orion:timestamp ?timestamp }}
    }}
    ORDER BY DESC(?timestamp)
    LIMIT 10
    """


async def main() -> int:
    redis_url = os.getenv("ORION_BUS_URL", "redis://localhost:6379/0")
    channel = os.getenv("CHANNEL_CHAT_HISTORY_TURN", "orion:chat:history:turn")
    session_id = "rdf-store-spike"

    endpoint = _query_url()
    if not endpoint:
        print("FAIL: missing query endpoint (set RDF_STORE_QUERY_URL or Fuseki/GraphDB vars).")
        return 2

    bus = OrionBusAsync(url=redis_url)
    await bus.connect()
    try:
        payload = {
            "id": "graphdb-replacement-smoke-001",
            "session_id": session_id,
            "prompt": "Can Orion write to the replacement RDF store?",
            "response": "Yes. This turn is stored outside GraphDB.",
            "timestamp": "2026-05-14T00:00:00Z",
            "correlation_id": "rdf-store-spike-001",
            "verb": "chat_general",
            "model": "smoke-model",
            "node": "athena",
        }
        env = BaseEnvelope(
            kind="chat.history",
            source=ServiceRef(name="smoke-chat-to-rdf-store", node=os.getenv("NODE_NAME"), version="0"),
            correlation_id=uuid4(),
            payload=payload,
        )
        await bus.publish(channel, env)
    finally:
        await bus.close()

    timeout = float(os.getenv("RDF_STORE_TIMEOUT_SEC", "10"))
    deadline = time.monotonic() + max(5.0, timeout * 3.0)
    auth_user = os.getenv("RDF_STORE_USER") or os.getenv("RECALL_RDF_USER") or os.getenv("GRAPHDB_USER") or "admin"
    auth_pass = os.getenv("RDF_STORE_PASS") or os.getenv("RECALL_RDF_PASS") or os.getenv("GRAPHDB_PASS") or "admin"
    auth = (auth_user, auth_pass)

    bindings: list[Any] = []
    while time.monotonic() < deadline:
        await asyncio.sleep(0.5)
        resp = requests.post(
            endpoint,
            data=_sparql(session_id),
            headers={
                "Content-Type": "application/sparql-query",
                "Accept": "application/sparql-results+json",
            },
            auth=auth,
            timeout=timeout,
        )
        if resp.status_code != 200:
            continue
        data = resp.json()
        bindings = data.get("results", {}).get("bindings", [])
        if bindings:
            break

    ok = False
    if bindings:
        b0 = bindings[0]
        prompt = b0.get("prompt", {}).get("value")
        response = b0.get("response", {}).get("value")
        ok = (
            prompt == "Can Orion write to the replacement RDF store?"
            and response == "Yes. This turn is stored outside GraphDB."
        )

    if ok:
        print("PASS")
        return 0

    print(
        "FAIL",
        {
            "backend": os.getenv("RDF_STORE_BACKEND"),
            "query_endpoint": endpoint,
            "dataset": os.getenv("RDF_STORE_DATASET"),
            "bindings_preview": bindings[:1],
        },
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

- [ ] **Step 2: Run (UNVERIFIED without live bus/store)**

```bash
PYTHONPATH=/mnt/scripts/Orion-Sapienform:/mnt/scripts/Orion-Sapienform/services/orion-rdf-writer ./venv/bin/python scripts/smoke_chat_to_rdf_store.py
```

- [ ] **Step 3: Commit**

```bash
git add scripts/smoke_chat_to_rdf_store.py
git commit -m "$(cat <<'EOF'
test(smoke): add chat.history RDF store readback script

Publish a synthetic chat turn and SPARQL-query the configured backend with async-write tolerant polling.
EOF
)"
```

---

### Task 7: Unit tests — `test_rdf_store.py`

**Files:**
- Create: `services/orion-rdf-writer/tests/test_rdf_store.py`

- [ ] **Step 1: Paste tests**

```python
from __future__ import annotations

import os
import sys
import asyncio
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("ORION_BUS_URL", "redis://example.test/0")
os.environ.setdefault("GRAPHDB_URL", "http://graphdb.example")
sys.path[:0] = [str(ROOT), str(SERVICE_ROOT)]

from app.rdf_store import (
    FusekiRdfStoreClient,
    GraphDbRdfStoreClient,
    GenericSparqlRdfStoreClient,
    build_rdf_store_client,
    normalize_graph_name,
    httpx_limits_for_settings,
)
from app.settings import Settings


def test_normalize_known_graphs() -> None:
    assert normalize_graph_name("orion:chat") == "http://conjourney.net/graph/orion/chat"
    assert normalize_graph_name("orion:chat:social") == "http://conjourney.net/graph/orion/chat/social"
    assert normalize_graph_name("orion:default") == "http://conjourney.net/graph/orion/default"


def test_normalize_absolute_unchanged() -> None:
    u = "http://example.com/g#x"
    assert normalize_graph_name(u) is u


def test_normalize_unknown_is_deterministic() -> None:
    assert normalize_graph_name("weird compact") == normalize_graph_name("weird compact")


def test_factory_default_graphdb_requires_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPHDB_URL", "")
    monkeypatch.setenv("RDF_STORE_BACKEND", "graphdb")
    s = Settings()
    import httpx

    with pytest.raises(ValueError):
        build_rdf_store_client(s, httpx.AsyncClient())


def test_factory_fuseki_does_not_require_graphdb_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GRAPHDB_URL", raising=False)
    monkeypatch.setenv("RDF_STORE_BACKEND", "fuseki")
    monkeypatch.setenv("RDF_STORE_BASE_URL", "http://orion-athena-fuseki:3030")
    s = Settings()
    import httpx

    c = build_rdf_store_client(s, httpx.AsyncClient())
    assert isinstance(c, FusekiRdfStoreClient)


def test_factory_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RDF_STORE_BACKEND", "nope")
    monkeypatch.setenv("GRAPHDB_URL", "http://g.example")
    s = Settings()
    import httpx

    with pytest.raises(ValueError, match="Unknown RDF_STORE_BACKEND"):
        build_rdf_store_client(s, httpx.AsyncClient())


def test_rdf4j_requires_explicit_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RDF_STORE_BACKEND", "rdf4j")
    monkeypatch.delenv("GRAPHDB_URL", raising=False)
    monkeypatch.delenv("RDF_STORE_GRAPH_STORE_URL", raising=False)
    monkeypatch.delenv("RDF_STORE_UPDATE_URL", raising=False)
    s = Settings()
    import httpx

    with pytest.raises(ValueError, match="rdf4j"):
        build_rdf_store_client(s, httpx.AsyncClient())


def test_fuseki_endpoint_defaults() -> None:
    s = Settings(
        RDF_STORE_BACKEND="fuseki",
        RDF_STORE_BASE_URL="http://orion-athena-fuseki:3030",
        RDF_STORE_DATASET="orion",
        ORION_BUS_URL="redis://example/0",
    )
    import httpx

    c = FusekiRdfStoreClient(s, httpx.AsyncClient())
    h = asyncio.run(c.health())
    assert h["graph_store_url"].endswith("/orion/data")
    assert h["query_url"].endswith("/orion/query")


def test_graphdb_endpoint_shape() -> None:
    s = Settings(
        RDF_STORE_BACKEND="graphdb",
        GRAPHDB_URL="http://gdb:7200",
        GRAPHDB_REPO="collapse",
        ORION_BUS_URL="redis://example/0",
    )
    import httpx

    c = GraphDbRdfStoreClient(s, httpx.AsyncClient())
    assert asyncio.run(c.health())["endpoint"].endswith("/repositories/collapse/statements")


def test_limits_helper_smoke() -> None:
    s = Settings(ORION_BUS_URL="redis://example/0", GRAPHDB_URL="http://g/")
    lim = httpx_limits_for_settings(s)
    assert lim.max_connections == 64
```

- [ ] **Step 2: Run**

```bash
cd /mnt/scripts/Orion-Sapienform
./scripts/test_service.sh orion-rdf-writer services/orion-rdf-writer/tests/test_rdf_store.py -q --tb=short
```

Expected: PASS (fix imports if `Settings()` construction needs all required envs).

- [ ] **Step 3: Commit**

```bash
git add services/orion-rdf-writer/tests/test_rdf_store.py
git commit -m "$(cat <<'EOF'
test(rdf-writer): cover RDF store normalization and factory wiring

Assert Fuseki default URLs, GraphDB repository path, and backend validation rules.
EOF
)"
```

---

### Task 8: Unit tests — `test_rdf_write_queue.py`

**Files:**
- Create: `services/orion-rdf-writer/tests/test_rdf_write_queue.py`

Use `httpx.MockTransport` to simulate 500 then 200 for retry tests; assert dead-letter file append with `tmp_path` monkeypatch for `RDF_WRITE_DEAD_LETTER_PATH`.

Minimum cases:
- `RDF_WRITE_ASYNC_ENABLED=true`: two rapid enqueues complete without handler awaiting second POST (use counter in mock transport + small sleep in handler side).
- Queue full: set `maxsize=1`, fill with slow job, third enqueue triggers dead-letter / error callback.
- `RDF_WRITE_ASYNC_ENABLED=false`: `handle_envelope` path awaits `write_graph` once (import service module with env).

Because `service.py` will use module-level queue, prefer **testing internal helpers** by extracting `_retrying_write` function in `service.py` for unit tests (keeps suite stable). Add in `service.py`:

```python
async def _retrying_write(store: RdfStoreClient, job: RdfWriteJob) -> RdfWriteResult: ...
```

Test `_retrying_write` directly with a fake store counting calls.

- [ ] **Implement tests + commit**

```bash
./scripts/test_service.sh orion-rdf-writer services/orion-rdf-writer/tests/test_rdf_write_queue.py -q --tb=short
```

---

### Task 9: Unit tests — `test_service_rdf_store_integration.py`

**Files:**
- Create: `services/orion-rdf-writer/tests/test_service_rdf_store_integration.py`

Pattern:
- `import importlib`, `importlib.import_module("app.service")` after env setup.
- `@pytest.mark.parametrize("kind", [...])` with kinds:
  - `chat.history`
  - `chat.history.message.v1`
  - `cognition.trace`
  - `metacognitive.trace.v1`
  - `tags.enriched`
  - `collapse.mirror.entry`
  - `rdf.write.request`
  - `world.pulse.graph.upsert.v1`
- Monkeypatch `app.service.build_triples_from_envelope` to return `("<urn:x> <urn:y> <urn:z> .\n", "orion:chat")` (or graph per case).
- Monkeypatch store `write_graph` to `AsyncMock` returning a dummy `RdfWriteResult`.
- Call `await handle_envelope(BaseEnvelope(...))` with minimal valid `ServiceRef` and payload `{}`.

For `world.pulse.graph.upsert.v1`, set `rdf_builder_mod.settings` flags as in `test_world_pulse_graph_gates.py` **or** rely on monkeypatch bypassing builder.

- [ ] **Run full service tests**

```bash
./scripts/test_service.sh orion-rdf-writer -q --tb=short
```

- [ ] **Commit**

```bash
git add services/orion-rdf-writer/tests/test_service_rdf_store_integration.py
git commit -m "$(cat <<'EOF'
test(rdf-writer): ensure all RDF envelope kinds hit the store client path

Parametrize representative kinds with a stubbed triple builder and capture write_graph calls.
EOF
)"
```

---

### Task 10: README updates

**Files:**
- Modify: `services/orion-rdf-writer/README.md`
- Ensure: `services/orion-rdf-store/README.md`, `services/orion-rdf-store/docker-compose.yml`, and `services/orion-rdf-store/.env_example` from Task 5 are complete prose (operator-focused).

Content checklist:
- RDF_STORE_BACKEND values
- GraphDB default/fallback
- Fuseki example env block + `orion-athena-fuseki` DNS
- Async queue semantics + backpressure honesty
- Smoke: `python scripts/smoke_chat_to_rdf_store.py`
- SPARQL readback query (same as README already has; bump LIMIT to 10)
- Explicit “chat is acceptance canary; writer is general-purpose”

- [ ] **Commit**

```bash
git add services/orion-rdf-writer/README.md services/orion-rdf-store
git commit -m "$(cat <<'EOF'
docs(rdf-store): document multi-backend RDF persistence and Fuseki spike

Explain async writer scaling, Athena disk layout, and non-migrated graph consumers.
EOF
)"
```

---

### Task 11: Final verification commands (required evidence)

- [ ] **Run**

```bash
cd /mnt/scripts/Orion-Sapienform
python3 -m compileall services/orion-rdf-writer/app
./scripts/test_service.sh orion-rdf-writer services/orion-rdf-writer/tests -q --tb=short
```

Expected: exit code 0.

---

## Self-review (spec coverage)

| Requirement | Task |
|-------------|------|
| A `rdf_store.py` contract + clients + factory | Task 2 |
| B settings + optional `GRAPHDB_URL` for Fuseki | Task 1 |
| C async queue / retry / dead-letter / logging semantics | Task 3 |
| D service integration, no `rdf_builder` semantic edits | Tasks 2–4 |
| E lifespan + health secret safety | Task 3 |
| F neutral `services/orion-rdf-store/` layout | Task 5 |
| G Fuseki compose + bind mount + app-net | Task 5 |
| H `.env_example` JVM/CPU | Task 5 |
| I compose passthrough | Task 5 |
| J smoke script | Task 6 |
| K unit tests | Tasks 7–9 |
| L READMEs | Task 10 |
| M acceptance criteria | Tasks 5–11 + manual stack smoke (report UNVERIFIED if not run on Athena) |
| N commands to report back | Task 11 + user’s Fuseki preflight exports |

**Placeholder scan:** No TBD sections above; any `verify image env` notes are explicit operator verification steps, not code placeholders.

**Type consistency:** `RdfWriteResult`, `RdfWriteJob`, and `register_rdf_write_publisher` signatures must match between `service.py` and tests.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-14-graphdb-replacement-rdf-store-spike-v0.md`. Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration. **REQUIRED SUB-SKILL:** superpowers:subagent-driven-development.

**2. Inline Execution** — Execute tasks in this session using executing-plans with batch checkpoints. **REQUIRED SUB-SKILL:** superpowers:executing-plans.

Which approach?

---

## Post-implementation report template (for workers)

- **Files changed:** (list paths)
- **Tests run:** `python3 -m compileall services/orion-rdf-writer/app` + `./scripts/test_service.sh orion-rdf-writer services/orion-rdf-writer/tests -q`
- **GraphDB smoke (legacy):** `PYTHONPATH=... ./venv/bin/python scripts/smoke_chat_to_rdf.py` (still valid when `GRAPHDB_URL`/`GRAPHDB_REPO` set)
- **Fuseki preflight:**

```bash
export RDF_STORE_DATA_ROOT=/mnt/storage-lukewarm/rdf-store
mkdir -p "${RDF_STORE_DATA_ROOT}/fuseki" "${RDF_STORE_DATA_ROOT}/fuseki-backups"
```

- **Fuseki env:** `RDF_STORE_BACKEND=fuseki`, `RDF_STORE_BASE_URL=http://orion-athena-fuseki:3030`, `RDF_STORE_DATASET=orion`, `FUSEKI_DATA_DIR=/mnt/storage-lukewarm/rdf-store/fuseki`
- **Fuseki smoke:** `PYTHONPATH=... ./venv/bin/python scripts/smoke_chat_to_rdf_store.py`
- **Also report:** exact compose file paths touched; persistent host path; Fuseki image (`stain/jena-fuseki:latest` unless changed after image doc check); service DNS `http://orion-athena-fuseki:3030`; remaining GraphDB assumptions (`context` query param, `text/plain`, `/repositories/{repo}/statements`); whether `rdf.write.error` uses dict publish vs `BaseEnvelope` and any catalog follow-up.
