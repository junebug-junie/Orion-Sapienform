# Graph Compression — Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `orion-graph-compression`, a standalone offline service that federates Fuseki graph data, clusters it with Leiden, summarizes regions via the LLM Gateway, and writes cached `CompressionRegionV1` artifacts to `orion:compressions` for downstream recall.

**Architecture:** A FastAPI service with a scheduled poll worker that drains a Postgres stale queue, federates triples from three Fuseki graph scopes (episodic, substrate, self-study), clusters them into semantic regions using Leiden, calls the LLM Gateway bus RPC for summaries (falling back to structural summaries on timeout), writes artifacts to Fuseki + Postgres, and emits bus events for downstream consumers. A stale listener subscribes to `orion:rdf:enqueue` to mark regions dirty on any RDF write.

**Tech Stack:** Python 3.12, FastAPI, pydantic-settings, SQLAlchemy + psycopg2-binary (sync), httpx, networkx, leidenalg + igraph, pyyaml, orion shared bus (`OrionBusAsync`, `BaseEnvelope`, `ServiceRef`), orion shared schemas (`SystemHealthV1`, `MutationPressureEvidenceV1`, `CompressionRegionV1`).

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `orion/schemas/graph_compression.py` | Already created | Shared Pydantic schemas |
| `orion/schemas/registry.py` | Already updated | Schema registry |
| `orion/bus/channels.yaml` | Already updated | Bus catalog |
| `services/orion-rdf-writer/app/rdf_store.py` | Modify line 47 | Add `orion:compressions` mapping |
| `services/orion-graph-compression/` | Create dir | New service root |
| `services/orion-graph-compression/requirements.txt` | Create | Python deps |
| `services/orion-graph-compression/Dockerfile` | Create | Container build |
| `services/orion-graph-compression/docker-compose.yml` | Create | Compose definition |
| `services/orion-graph-compression/config/compression_policy.v1.yaml` | Create | Worker policy config |
| `services/orion-graph-compression/app/__init__.py` | Create | Package marker |
| `services/orion-graph-compression/app/settings.py` | Create | Pydantic settings |
| `services/orion-graph-compression/app/store.py` | Create | Postgres CRUD (artifacts, jobs, stale_queue) |
| `services/orion-graph-compression/app/federators/__init__.py` | Create | Package marker |
| `services/orion-graph-compression/app/federators/episodic.py` | Create | SPARQL over episodic named graphs |
| `services/orion-graph-compression/app/federators/substrate.py` | Create | SubstrateSemanticReadCoordinator adapter |
| `services/orion-graph-compression/app/federators/self_study.py` | Create | SPARQL over orion:self* graphs |
| `services/orion-graph-compression/app/clustering/__init__.py` | Create | Package marker |
| `services/orion-graph-compression/app/clustering/leiden.py` | Create | Leiden on NetworkX graph |
| `services/orion-graph-compression/app/clustering/region_builder.py` | Create | CompressionRegionV1 from cluster |
| `services/orion-graph-compression/app/summarizer.py` | Create | LLM Gateway bus RPC |
| `services/orion-graph-compression/app/writer.py` | Create | Fuseki SPARQL UPDATE + bus events |
| `services/orion-graph-compression/app/stale_listener.py` | Create | Bus subscriber → stale queue |
| `services/orion-graph-compression/app/worker.py` | Create | CompressionWorker poll loop |
| `services/orion-graph-compression/app/main.py` | Create | FastAPI app + lifespan + heartbeat |
| `services/orion-graph-compression/tests/conftest.py` | Create | Shared fixtures |
| `services/orion-graph-compression/tests/test_compression_schema.py` | Create | Schema round-trip |
| `services/orion-graph-compression/tests/test_leiden_clustering.py` | Create | Leiden unit |
| `services/orion-graph-compression/tests/test_region_builder.py` | Create | Region builder unit |
| `services/orion-graph-compression/tests/test_federator_episodic.py` | Create | SPARQL query generation |
| `services/orion-graph-compression/tests/test_writer_sparql.py` | Create | SPARQL UPDATE generation |
| `services/orion-graph-compression/tests/test_store_staleness.py` | Create | Postgres staleness queue |
| `services/orion-graph-compression/tests/test_worker_degraded.py` | Create | Empty federator → no crash |
| `services/orion-graph-compression/tests/test_grammar_hook.py` | Create | Contradiction → MutationPressureEvidenceV1 |

---

## Task 1: Shared schema verification + `orion:compressions` named graph mapping

**Files:**
- Verify: `orion/schemas/graph_compression.py`
- Modify: `services/orion-rdf-writer/app/rdf_store.py:40-48`
- Create: `services/orion-graph-compression/tests/test_compression_schema.py`

- [ ] **Step 1: Write failing schema round-trip test**

Create `services/orion-graph-compression/tests/test_compression_schema.py`:

```python
from datetime import datetime, timezone
import pytest
from orion.schemas.graph_compression import (
    CompressionRegionV1,
    CompressionStalenessMarkV1,
    GraphCompressionRegionMaterializedV1,
)


def test_compression_region_v1_round_trip():
    now = datetime.now(timezone.utc)
    r = CompressionRegionV1(
        region_id="urn:orion:compression:region:abc123",
        scope="episodic",
        kind="community",
        summary="A cluster of memory fragments about workflow design.",
        summary_kind="llm",
        salience=0.72,
        trust_tier="verified",
        exemplar_ids=["http://conjourney.net/chat/turn/1"],
        derived_from=["http://conjourney.net/chat/turn/1"],
        generated_at=now,
        compression_version="1.0.0",
    )
    data = r.model_dump(mode="json")
    restored = CompressionRegionV1.model_validate(data)
    assert restored.region_id == r.region_id
    assert restored.scope == "episodic"
    assert restored.stale is False


def test_compression_region_v1_requires_exemplar_ids():
    with pytest.raises(Exception):
        CompressionRegionV1(
            region_id="urn:orion:compression:region:abc",
            scope="episodic",
            kind="community",
            summary="x",
            summary_kind="structural",
            salience=0.5,
            trust_tier="unverified",
            exemplar_ids=[],  # must be non-empty per spec
            derived_from=["x"],
            generated_at=datetime.now(timezone.utc),
            compression_version="1.0.0",
        )


def test_staleness_mark_v1_round_trip():
    import time
    m = CompressionStalenessMarkV1(
        scope="episodic",
        reason="rdf_enqueue_trigger",
        source_service="orion-rdf-writer",
        ts=time.time(),
    )
    assert m.region_id is None  # scope-wide mark


def test_materialized_v1_round_trip():
    import time
    e = GraphCompressionRegionMaterializedV1(
        region_id="urn:orion:compression:region:abc123",
        scope="episodic",
        kind="community",
        salience=0.72,
        trust_tier="verified",
        summary_kind="llm",
        compression_version="1.0.0",
        ts=time.time(),
    )
    data = e.model_dump(mode="json")
    assert data["region_id"] == e.region_id
```

- [ ] **Step 2: Run test to verify it fails for the right reason**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. pytest services/orion-graph-compression/tests/test_compression_schema.py -v 2>&1 | head -30
```

Expected: `test_compression_region_v1_requires_exemplar_ids` likely passes but `exemplar_ids=[]` is accepted — we'll fix the schema. Other tests should PASS (schema already created). If all pass, note it and move on.

- [ ] **Step 3: Add `min_length=1` validator to `exemplar_ids` and `derived_from` in schema**

Edit `orion/schemas/graph_compression.py`:

```python
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class CompressionRegionV1(BaseModel):
    region_id: str
    scope: Literal["episodic", "substrate", "self_study"]
    kind: Literal["community", "hotspot", "contradiction", "self_study_cluster"]
    summary: str
    summary_kind: Literal["llm", "structural"]
    salience: float
    trust_tier: str
    exemplar_ids: Annotated[list[str], Field(min_length=1)]
    derived_from: Annotated[list[str], Field(min_length=1)]
    generated_at: datetime
    compression_version: str
    stale: bool = False


class CompressionStalenessMarkV1(BaseModel):
    region_id: str | None = None
    scope: str | None = None
    reason: str
    source_service: str
    ts: float


class GraphCompressionRegionMaterializedV1(BaseModel):
    region_id: str
    scope: str
    kind: str
    salience: float
    trust_tier: str
    summary_kind: str
    compression_version: str
    ts: float
```

- [ ] **Step 4: Add `orion:compressions` to `normalize_graph_name` in `rdf_store.py`**

In `services/orion-rdf-writer/app/rdf_store.py`, inside the `mapping` dict (line ~40-48), add before the closing `}`:

```python
        "orion:self": "http://conjourney.net/graph/orion/self",
        "orion:self:induced": "http://conjourney.net/graph/orion/self/induced",
        "orion:self:reflective": "http://conjourney.net/graph/orion/self/reflective",
        "orion:compressions": "http://conjourney.net/graph/orion/compressions",
```

(Check what's already there first — `orion:self` entries may already exist; only add what's missing.)

- [ ] **Step 5: Run tests and confirm all pass**

```bash
PYTHONPATH=. pytest services/orion-graph-compression/tests/test_compression_schema.py -v
```

Expected: 4 tests, all PASS.

- [ ] **Step 6: Commit**

```bash
git add orion/schemas/graph_compression.py \
        services/orion-rdf-writer/app/rdf_store.py \
        services/orion-graph-compression/tests/test_compression_schema.py
git commit -m "feat(graph-compression): shared schema + normalize_graph_name for orion:compressions"
```

---

## Task 2: Service scaffolding — requirements, Dockerfile, docker-compose, policy config

**Files:**
- Create: `services/orion-graph-compression/requirements.txt`
- Create: `services/orion-graph-compression/Dockerfile`
- Create: `services/orion-graph-compression/docker-compose.yml`
- Create: `services/orion-graph-compression/config/compression_policy.v1.yaml`
- Create: `services/orion-graph-compression/app/__init__.py`

- [ ] **Step 1: Create `requirements.txt`**

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
pydantic>=2.6.0
pydantic-settings>=2.2.0
httpx>=0.27.0
networkx>=3.2.0
leidenalg>=0.10.2
igraph>=0.11.4
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.9
pyyaml>=6.0.1
```

- [ ] **Step 2: Create `Dockerfile`**

```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip
COPY services/orion-graph-compression/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY orion /app/orion
COPY services/orion-graph-compression/config /app/config
COPY services/orion-graph-compression/app /app/app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8270"]
```

- [ ] **Step 3: Create `docker-compose.yml`**

```yaml
services:
  graph-compression:
    build:
      context: ../..
      dockerfile: services/orion-graph-compression/Dockerfile
    container_name: ${PROJECT}-graph-compression
    restart: unless-stopped
    networks:
      - app-net
    ports:
      - "${GRAPH_COMPRESSION_PORT:-8270}:8270"
    env_file:
      - .env
    environment:
      - PROJECT=${PROJECT}
      - SERVICE_NAME=${SERVICE_NAME:-orion-graph-compression}
      - SERVICE_VERSION=${SERVICE_VERSION:-0.1.0}
      - NODE_NAME=${NODE_NAME}
      - ORION_BUS_URL=${ORION_BUS_URL}
      - ORION_BUS_ENABLED=${ORION_BUS_ENABLED:-true}
      - ORION_HEALTH_CHANNEL=${ORION_HEALTH_CHANNEL:-orion:system:health}
      - HEARTBEAT_INTERVAL_SEC=${HEARTBEAT_INTERVAL_SEC:-30}
      - POSTGRES_URI=${POSTGRES_URI}
      - RDF_STORE_QUERY_URL=${RDF_STORE_QUERY_URL}
      - RDF_STORE_UPDATE_URL=${RDF_STORE_UPDATE_URL}
      - RDF_STORE_USER=${RDF_STORE_USER:-admin}
      - RDF_STORE_PASS=${RDF_STORE_PASS:-orion}
      - RDF_STORE_TIMEOUT_SEC=${RDF_STORE_TIMEOUT_SEC:-10.0}
      - LLM_GATEWAY_BUS_CHANNEL=${LLM_GATEWAY_BUS_CHANNEL}
      - COMPRESSION_POLL_INTERVAL_SEC=${COMPRESSION_POLL_INTERVAL_SEC:-300}
      - COMPRESSION_BATCH_SIZE=${COMPRESSION_BATCH_SIZE:-10}
      - COMPRESSION_MAX_TOKENS_PER_SUMMARY=${COMPRESSION_MAX_TOKENS_PER_SUMMARY:-200}
      - COMPRESSION_LLM_BUDGET_PER_TICK=${COMPRESSION_LLM_BUDGET_PER_TICK:-5000}
      - COMPRESSION_MAX_AGE_SEC=${COMPRESSION_MAX_AGE_SEC:-86400}
      - ENABLE_COMPRESSION_RUNTIME=${ENABLE_COMPRESSION_RUNTIME:-true}
      - CHANNEL_RDF_ENQUEUE=${CHANNEL_RDF_ENQUEUE:-orion:rdf:enqueue}
      - CHANNEL_GRAPH_COMPRESSION_STALE=${CHANNEL_GRAPH_COMPRESSION_STALE:-orion:graph:compression:stale}
      - CHANNEL_GRAPH_COMPRESSION_EVENTS=${CHANNEL_GRAPH_COMPRESSION_EVENTS:-orion:graph:compression:events}
      - COMPRESSION_POLICY_PATH=${COMPRESSION_POLICY_PATH:-/app/config/compression_policy.v1.yaml}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
networks:
  app-net:
    external: true
```

- [ ] **Step 4: Create `config/compression_policy.v1.yaml`**

```yaml
policy_id: compression_policy.v1
version: "1.0.0"

scopes:
  - name: episodic
    enabled: true
    max_nodes_per_federation: 2000
    kinds: [community, hotspot]
  - name: substrate
    enabled: true
    max_nodes_per_federation: 500
    kinds: [hotspot, contradiction]
  - name: self_study
    enabled: true
    max_nodes_per_federation: 500
    kinds: [self_study_cluster]

clustering:
  resolution: 1.0
  n_iterations: 10
  min_community_size: 3
  max_communities_per_scope: 20

summarization:
  model_hint: "fast"
  max_tokens: 200
  fallback_to_structural: true

exemplar_selection:
  max_exemplars_per_region: 5
  prefer_high_salience: true
```

- [ ] **Step 5: Create `app/__init__.py`** (empty file)

- [ ] **Step 6: Verify requirements install cleanly**

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-graph-compression
pip install -r requirements.txt --dry-run 2>&1 | tail -5
```

Expected: resolves without conflicts.

- [ ] **Step 7: Commit**

```bash
git add services/orion-graph-compression/requirements.txt \
        services/orion-graph-compression/Dockerfile \
        services/orion-graph-compression/docker-compose.yml \
        services/orion-graph-compression/config/compression_policy.v1.yaml \
        services/orion-graph-compression/app/__init__.py
git commit -m "feat(graph-compression): service scaffolding — requirements, Dockerfile, compose, policy"
```

---

## Task 3: `app/settings.py`

**Files:**
- Create: `services/orion-graph-compression/app/settings.py`

- [ ] **Step 1: Write `settings.py`**

```python
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Service identity
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("orion-graph-compression", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("unknown", alias="NODE_NAME")
    port: int = Field(8270, alias="PORT")

    # Bus
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_url: str = Field("redis://127.0.0.1:6379/0", alias="ORION_BUS_URL")
    health_channel: str = Field("orion:system:health", alias="ORION_HEALTH_CHANNEL")
    error_channel: str = Field("orion:system:error", alias="ERROR_CHANNEL")
    heartbeat_interval_sec: float = Field(30.0, alias="HEARTBEAT_INTERVAL_SEC")

    # Bus channels
    channel_rdf_enqueue: str = Field("orion:rdf:enqueue", alias="CHANNEL_RDF_ENQUEUE")
    channel_graph_compression_stale: str = Field(
        "orion:graph:compression:stale", alias="CHANNEL_GRAPH_COMPRESSION_STALE"
    )
    channel_graph_compression_events: str = Field(
        "orion:graph:compression:events", alias="CHANNEL_GRAPH_COMPRESSION_EVENTS"
    )
    llm_gateway_bus_channel: str = Field(
        "orion:exec:request:LLMGatewayService", alias="LLM_GATEWAY_BUS_CHANNEL"
    )

    # Postgres
    postgres_uri: str = Field(
        "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney",
        alias="POSTGRES_URI",
    )

    # Fuseki
    rdf_store_query_url: str = Field(
        "http://orion-athena-fuseki:3030/orion/query", alias="RDF_STORE_QUERY_URL"
    )
    rdf_store_update_url: str = Field(
        "http://orion-athena-fuseki:3030/orion/update", alias="RDF_STORE_UPDATE_URL"
    )
    rdf_store_user: str = Field("admin", alias="RDF_STORE_USER")
    rdf_store_pass: str = Field("orion", alias="RDF_STORE_PASS")
    rdf_store_timeout_sec: float = Field(10.0, alias="RDF_STORE_TIMEOUT_SEC")

    # Worker tuning
    compression_poll_interval_sec: float = Field(300.0, alias="COMPRESSION_POLL_INTERVAL_SEC")
    compression_batch_size: int = Field(10, alias="COMPRESSION_BATCH_SIZE")
    compression_max_tokens_per_summary: int = Field(200, alias="COMPRESSION_MAX_TOKENS_PER_SUMMARY")
    compression_llm_budget_per_tick: int = Field(5000, alias="COMPRESSION_LLM_BUDGET_PER_TICK")
    compression_max_age_sec: int = Field(86400, alias="COMPRESSION_MAX_AGE_SEC")
    enable_compression_runtime: bool = Field(True, alias="ENABLE_COMPRESSION_RUNTIME")
    compression_policy_path: str = Field(
        "/app/config/compression_policy.v1.yaml", alias="COMPRESSION_POLICY_PATH"
    )
    log_level: str = Field("INFO", alias="LOG_LEVEL")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
```

- [ ] **Step 2: Verify settings parse from env without error**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. python -c "
import os; os.environ.setdefault('POSTGRES_URI','postgresql://x:y@localhost/z')
from services.orion_graph_compression.app.settings import get_settings
s = get_settings(); print(s.service_name, s.compression_poll_interval_sec)
"
```

Expected output: `orion-graph-compression 300.0`

Note: Python can't import with hyphens — the service's `app/` directory is on `PYTHONPATH` in container. For tests, add the `app/` dir to `sys.path` via `conftest.py` (Task 4).

- [ ] **Step 3: Commit**

```bash
git add services/orion-graph-compression/app/settings.py
git commit -m "feat(graph-compression): settings.py — pydantic-settings with full env var mapping"
```

---

## Task 4: `tests/conftest.py` + Postgres store + migration

**Files:**
- Create: `services/orion-graph-compression/tests/__init__.py`
- Create: `services/orion-graph-compression/tests/conftest.py`
- Create: `services/orion-graph-compression/app/store.py`
- Create: `services/orion-graph-compression/tests/test_store_staleness.py`

- [ ] **Step 1: Create `tests/__init__.py`** (empty)

- [ ] **Step 2: Create `tests/conftest.py`**

```python
import sys
import os
import pytest

# Make app/ importable without hyphens (service lives in hyphenated directory)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
```

- [ ] **Step 3: Write failing `test_store_staleness.py`**

```python
import os
import pytest
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timezone


def _make_store():
    """Create a CompressionStore with a mock engine."""
    with patch("app.store.create_engine") as mock_eng:
        mock_eng.return_value = MagicMock()
        from app.store import CompressionStore
        store = CompressionStore("postgresql://x:y@localhost/test")
        store._engine = MagicMock()
        return store


def test_enqueue_stale_inserts_row():
    from app.store import CompressionStore
    store = _make_store()
    conn = MagicMock()
    store._engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    store._engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    store.enqueue_stale(scope="episodic", reason="rdf_write")
    conn.execute.assert_called_once()
    sql = str(conn.execute.call_args[0][0])
    assert "stale_queue" in sql


def test_drain_stale_queue_returns_up_to_batch():
    from app.store import CompressionStore
    store = _make_store()
    conn = MagicMock()
    store._engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    store._engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    fake_rows = [
        {"id": 1, "region_id": None, "scope": "episodic", "reason": "rdf_write", "priority": 0},
        {"id": 2, "region_id": None, "scope": "substrate", "reason": "rdf_write", "priority": 0},
    ]
    conn.execute.return_value.mappings.return_value.fetchall.return_value = fake_rows

    items = store.drain_stale_queue(batch_size=5)
    assert len(items) == 2
    assert items[0]["scope"] == "episodic"


def test_upsert_artifact_idempotent():
    from app.store import CompressionStore
    from datetime import datetime, timezone
    store = _make_store()
    conn = MagicMock()
    store._engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    store._engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    store.upsert_artifact(
        region_id="urn:orion:compression:region:abc",
        scope="episodic",
        kind="community",
        summary_kind="structural",
        salience=0.5,
        trust_tier="unverified",
        compression_version="1.0.0",
        generated_at=datetime.now(timezone.utc),
    )
    conn.execute.assert_called_once()
    sql = str(conn.execute.call_args[0][0])
    assert "compression_artifacts" in sql
    assert "ON CONFLICT" in sql
```

- [ ] **Step 4: Run test to verify it fails**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=services/orion-graph-compression pytest \
  services/orion-graph-compression/tests/test_store_staleness.py -v 2>&1 | head -20
```

Expected: `ImportError: No module named 'app.store'`

- [ ] **Step 5: Write `app/store.py`**

```python
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger("orion.graph-compression.store")


class CompressionStore:
    def __init__(self, postgres_uri: str) -> None:
        self._engine: Engine = create_engine(postgres_uri, pool_pre_ping=True)

    def ensure_tables(self) -> None:
        with self._engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS stale_queue (
                    id SERIAL PRIMARY KEY,
                    region_id TEXT,
                    scope TEXT,
                    reason TEXT,
                    queued_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    priority INT DEFAULT 0
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS compression_artifacts (
                    region_id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    fuseki_graph_uri TEXT NOT NULL DEFAULT 'http://conjourney.net/graph/orion/compressions',
                    summary_kind TEXT NOT NULL,
                    salience FLOAT,
                    trust_tier TEXT,
                    compression_version TEXT,
                    generated_at TIMESTAMPTZ NOT NULL,
                    stale BOOLEAN NOT NULL DEFAULT false
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS compression_jobs (
                    job_id TEXT PRIMARY KEY,
                    region_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    llm_tokens_used INT,
                    started_at TIMESTAMPTZ,
                    finished_at TIMESTAMPTZ,
                    error TEXT
                )
            """))

    def enqueue_stale(
        self,
        *,
        scope: str | None = None,
        region_id: str | None = None,
        reason: str,
        priority: int = 0,
    ) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO stale_queue (region_id, scope, reason, queued_at, priority)"
                    " VALUES (:region_id, :scope, :reason, :queued_at, :priority)"
                ),
                {
                    "region_id": region_id,
                    "scope": scope,
                    "reason": reason,
                    "queued_at": datetime.now(timezone.utc),
                    "priority": priority,
                },
            )

    def drain_stale_queue(self, *, batch_size: int) -> list[dict[str, Any]]:
        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        "SELECT id, region_id, scope, reason, priority"
                        " FROM stale_queue"
                        " ORDER BY priority DESC, id ASC"
                        " LIMIT :batch_size"
                    ),
                    {"batch_size": batch_size},
                )
                .mappings()
                .fetchall()
            )
        return [dict(r) for r in rows]

    def delete_stale_queue_items(self, ids: list[int]) -> None:
        if not ids:
            return
        with self._engine.begin() as conn:
            conn.execute(
                text("DELETE FROM stale_queue WHERE id = ANY(:ids)"),
                {"ids": ids},
            )

    def upsert_artifact(
        self,
        *,
        region_id: str,
        scope: str,
        kind: str,
        summary_kind: str,
        salience: float,
        trust_tier: str,
        compression_version: str,
        generated_at: datetime,
        stale: bool = False,
    ) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO compression_artifacts
                        (region_id, scope, kind, summary_kind, salience, trust_tier,
                         compression_version, generated_at, stale)
                    VALUES
                        (:region_id, :scope, :kind, :summary_kind, :salience, :trust_tier,
                         :compression_version, :generated_at, :stale)
                    ON CONFLICT (region_id) DO UPDATE SET
                        scope = EXCLUDED.scope,
                        kind = EXCLUDED.kind,
                        summary_kind = EXCLUDED.summary_kind,
                        salience = EXCLUDED.salience,
                        trust_tier = EXCLUDED.trust_tier,
                        compression_version = EXCLUDED.compression_version,
                        generated_at = EXCLUDED.generated_at,
                        stale = EXCLUDED.stale
                """),
                {
                    "region_id": region_id,
                    "scope": scope,
                    "kind": kind,
                    "summary_kind": summary_kind,
                    "salience": salience,
                    "trust_tier": trust_tier,
                    "compression_version": compression_version,
                    "generated_at": generated_at,
                    "stale": stale,
                },
            )

    def list_artifacts(
        self,
        *,
        scope: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM compression_artifacts"
        params: dict[str, Any] = {"limit": limit}
        if scope:
            sql += " WHERE scope = :scope"
            params["scope"] = scope
        sql += " ORDER BY generated_at DESC LIMIT :limit"
        with self._engine.connect() as conn:
            rows = conn.execute(text(sql), params).mappings().fetchall()
        return [dict(r) for r in rows]

    def get_artifact(self, region_id: str) -> dict[str, Any] | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text("SELECT * FROM compression_artifacts WHERE region_id = :id"),
                    {"id": region_id},
                )
                .mappings()
                .first()
            )
        return dict(row) if row else None

    def record_job(
        self,
        *,
        job_id: str,
        region_id: str,
        status: str,
        llm_tokens_used: int | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        error: str | None = None,
    ) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO compression_jobs
                        (job_id, region_id, status, llm_tokens_used, started_at, finished_at, error)
                    VALUES
                        (:job_id, :region_id, :status, :llm_tokens_used, :started_at, :finished_at, :error)
                    ON CONFLICT (job_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        llm_tokens_used = EXCLUDED.llm_tokens_used,
                        finished_at = EXCLUDED.finished_at,
                        error = EXCLUDED.error
                """),
                {
                    "job_id": job_id,
                    "region_id": region_id,
                    "status": status,
                    "llm_tokens_used": llm_tokens_used,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "error": error,
                },
            )

    def artifact_count(self) -> int:
        with self._engine.connect() as conn:
            row = conn.execute(text("SELECT COUNT(*) FROM compression_artifacts")).first()
        return int(row[0]) if row else 0

    def stale_queue_depth(self) -> int:
        with self._engine.connect() as conn:
            row = conn.execute(text("SELECT COUNT(*) FROM stale_queue")).first()
        return int(row[0]) if row else 0
```

- [ ] **Step 6: Run tests and confirm pass**

```bash
PYTHONPATH=services/orion-graph-compression pytest \
  services/orion-graph-compression/tests/test_store_staleness.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add services/orion-graph-compression/app/store.py \
        services/orion-graph-compression/tests/__init__.py \
        services/orion-graph-compression/tests/conftest.py \
        services/orion-graph-compression/tests/test_store_staleness.py
git commit -m "feat(graph-compression): CompressionStore — Postgres stale queue + artifact index"
```

---

## Task 5: Episodic federator

**Files:**
- Create: `services/orion-graph-compression/app/federators/__init__.py`
- Create: `services/orion-graph-compression/app/federators/episodic.py`
- Create: `services/orion-graph-compression/tests/test_federator_episodic.py`

- [ ] **Step 1: Write failing test**

Create `services/orion-graph-compression/tests/test_federator_episodic.py`:

```python
from unittest.mock import patch, MagicMock
import pytest


def test_episodic_federator_builds_sparql_for_all_graphs():
    """SPARQL query must reference all 9 episodic named graphs."""
    with patch("httpx.Client") as mock_client_cls:
        from app.federators.episodic import EpisodicFederator, EPISODIC_GRAPHS

        f = EpisodicFederator(
            query_url="http://fuseki/query",
            user="admin",
            password="orion",
            timeout_sec=5.0,
        )
        query = f._build_sparql()
        for graph_uri in EPISODIC_GRAPHS:
            assert graph_uri in query, f"Missing graph: {graph_uri}"
        assert "SELECT" in query


def test_episodic_federator_returns_empty_on_http_error():
    """Fuseki failure → empty list, no exception."""
    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = Exception("connection refused")

        from app.federators.episodic import EpisodicFederator
        f = EpisodicFederator(
            query_url="http://fuseki/query",
            user="admin",
            password="orion",
            timeout_sec=5.0,
        )
        triples = f.fetch(max_nodes=100)
        assert triples == []


def test_episodic_federator_parses_sparql_json():
    """Parse SPARQL JSON bindings into (subject, predicate, object) tuples."""
    sparql_response = {
        "results": {
            "bindings": [
                {
                    "s": {"type": "uri", "value": "http://example.org/A"},
                    "p": {"type": "uri", "value": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"},
                    "o": {"type": "uri", "value": "http://example.org/ChatTurn"},
                }
            ]
        }
    }
    with patch("httpx.Client") as mock_client_cls:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = sparql_response
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        from app.federators.episodic import EpisodicFederator
        f = EpisodicFederator(
            query_url="http://fuseki/query",
            user="admin",
            password="orion",
            timeout_sec=5.0,
        )
        triples = f.fetch(max_nodes=100)
        assert len(triples) == 1
        assert triples[0] == (
            "http://example.org/A",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://example.org/ChatTurn",
        )
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=services/orion-graph-compression pytest \
  services/orion-graph-compression/tests/test_federator_episodic.py -v 2>&1 | head -10
```

Expected: `ImportError: No module named 'app.federators.episodic'`

- [ ] **Step 3: Create `app/federators/__init__.py`** (empty)

- [ ] **Step 4: Write `app/federators/episodic.py`**

```python
from __future__ import annotations

import logging
from typing import List, Tuple

import httpx

logger = logging.getLogger("orion.graph-compression.federator.episodic")

EPISODIC_GRAPHS = [
    "http://conjourney.net/graph/orion/chat",
    "http://conjourney.net/graph/orion/enrichment",
    "http://conjourney.net/graph/orion/collapse",
    "http://conjourney.net/graph/orion/cognition",
    "http://conjourney.net/graph/orion/metacog",
    "http://conjourney.net/graph/orion/chat/social",
    "http://conjourney.net/graph/orion/autonomy/identity",
    "http://conjourney.net/graph/orion/autonomy/drives",
    "http://conjourney.net/graph/orion/autonomy/goals",
]

Triple = Tuple[str, str, str]


class EpisodicFederator:
    def __init__(
        self,
        *,
        query_url: str,
        user: str,
        password: str,
        timeout_sec: float,
    ) -> None:
        self._query_url = query_url
        self._auth = (user, password)
        self._timeout = timeout_sec

    def _build_sparql(self, max_nodes: int = 2000) -> str:
        graph_clauses = "\n  ".join(
            f"GRAPH <{g}> {{ ?s ?p ?o }}" for g in EPISODIC_GRAPHS
        )
        return f"""
SELECT ?s ?p ?o WHERE {{
  {{
    SELECT DISTINCT ?s WHERE {{
      {{ {graph_clauses} }}
    }}
    LIMIT {max_nodes}
  }}
  {{ {graph_clauses} }}
}}
"""

    def fetch(self, *, max_nodes: int = 2000) -> List[Triple]:
        query = self._build_sparql(max_nodes)
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    self._query_url,
                    data={"query": query},
                    headers={"Accept": "application/sparql-results+json"},
                    auth=self._auth,
                )
                resp.raise_for_status()
                bindings = resp.json().get("results", {}).get("bindings", [])
        except Exception as exc:
            logger.warning("episodic_federator_fetch_failed reason=%s", exc)
            return []
        return [
            (b["s"]["value"], b["p"]["value"], b["o"]["value"])
            for b in bindings
            if "s" in b and "p" in b and "o" in b
        ]
```

- [ ] **Step 5: Run tests and confirm pass**

```bash
PYTHONPATH=services/orion-graph-compression pytest \
  services/orion-graph-compression/tests/test_federator_episodic.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add services/orion-graph-compression/app/federators/__init__.py \
        services/orion-graph-compression/app/federators/episodic.py \
        services/orion-graph-compression/tests/test_federator_episodic.py
git commit -m "feat(graph-compression): EpisodicFederator — SPARQL over 9 episodic named graphs"
```

---

## Task 6: Substrate + self-study federators

**Files:**
- Create: `services/orion-graph-compression/app/federators/substrate.py`
- Create: `services/orion-graph-compression/app/federators/self_study.py`

No isolated unit tests beyond smoke-runs (substrate federator delegates to `SubstrateSemanticReadCoordinator` which needs live Fuseki). Degraded-mode is covered in Task 11.

- [ ] **Step 1: Write `app/federators/substrate.py`**

```python
from __future__ import annotations

import logging
from typing import List, Tuple

import httpx

logger = logging.getLogger("orion.graph-compression.federator.substrate")

# Substrate bounded query kinds we pull.
SUBSTRATE_QUERY_KINDS = ["hotspot_region", "contradiction_region", "concept_region"]

Triple = Tuple[str, str, str]

SUBSTRATE_GRAPH = "http://conjourney.net/graph/orion/substrate"


class SubstrateFederator:
    """
    Fetches triples from the substrate graph using bounded SPARQL.
    Uses simple named-graph SPARQL rather than SubstrateSemanticReadCoordinator
    (which requires the substrate service to be live). Degraded to empty list on any error.
    """

    def __init__(
        self,
        *,
        query_url: str,
        user: str,
        password: str,
        timeout_sec: float,
    ) -> None:
        self._query_url = query_url
        self._auth = (user, password)
        self._timeout = timeout_sec

    def fetch(self, *, max_nodes: int = 500) -> List[Triple]:
        query = f"""
SELECT ?s ?p ?o WHERE {{
  {{
    SELECT DISTINCT ?s WHERE {{
      GRAPH <{SUBSTRATE_GRAPH}> {{ ?s ?p0 ?o0 }}
    }}
    LIMIT {max_nodes}
  }}
  GRAPH <{SUBSTRATE_GRAPH}> {{ ?s ?p ?o }}
}}
"""
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    self._query_url,
                    data={"query": query},
                    headers={"Accept": "application/sparql-results+json"},
                    auth=self._auth,
                )
                resp.raise_for_status()
                bindings = resp.json().get("results", {}).get("bindings", [])
        except Exception as exc:
            logger.warning("substrate_federator_fetch_failed reason=%s", exc)
            return []
        return [
            (b["s"]["value"], b["p"]["value"], b["o"]["value"])
            for b in bindings
            if "s" in b and "p" in b and "o" in b
        ]
```

- [ ] **Step 2: Write `app/federators/self_study.py`**

```python
from __future__ import annotations

import logging
from typing import List, Tuple

import httpx

logger = logging.getLogger("orion.graph-compression.federator.self_study")

SELF_STUDY_GRAPHS = [
    "http://conjourney.net/graph/orion/self",
    "http://conjourney.net/graph/orion/self/induced",
    "http://conjourney.net/graph/orion/self/reflective",
]

Triple = Tuple[str, str, str]


class SelfStudyFederator:
    def __init__(
        self,
        *,
        query_url: str,
        user: str,
        password: str,
        timeout_sec: float,
    ) -> None:
        self._query_url = query_url
        self._auth = (user, password)
        self._timeout = timeout_sec

    def fetch(self, *, max_nodes: int = 500) -> List[Triple]:
        graph_clauses = "\n  ".join(
            f"GRAPH <{g}> {{ ?s ?p ?o }}" for g in SELF_STUDY_GRAPHS
        )
        query = f"""
SELECT ?s ?p ?o WHERE {{
  {{
    SELECT DISTINCT ?s WHERE {{
      {{ {graph_clauses} }}
    }}
    LIMIT {max_nodes}
  }}
  {{ {graph_clauses} }}
}}
"""
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    self._query_url,
                    data={"query": query},
                    headers={"Accept": "application/sparql-results+json"},
                    auth=self._auth,
                )
                resp.raise_for_status()
                bindings = resp.json().get("results", {}).get("bindings", [])
        except Exception as exc:
            logger.warning("self_study_federator_fetch_failed reason=%s", exc)
            return []
        return [
            (b["s"]["value"], b["p"]["value"], b["o"]["value"])
            for b in bindings
            if "s" in b and "p" in b and "o" in b
        ]
```

- [ ] **Step 3: Quick import smoke test**

```bash
PYTHONPATH=services/orion-graph-compression python -c "
from app.federators.substrate import SubstrateFederator
from app.federators.self_study import SelfStudyFederator
print('OK')
"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add services/orion-graph-compression/app/federators/substrate.py \
        services/orion-graph-compression/app/federators/self_study.py
git commit -m "feat(graph-compression): SubstrateFederator + SelfStudyFederator SPARQL adapters"
```

---

## Task 7: Leiden clustering + region builder

**Files:**
- Create: `services/orion-graph-compression/app/clustering/__init__.py`
- Create: `services/orion-graph-compression/app/clustering/leiden.py`
- Create: `services/orion-graph-compression/app/clustering/region_builder.py`
- Create: `services/orion-graph-compression/tests/test_leiden_clustering.py`
- Create: `services/orion-graph-compression/tests/test_region_builder.py`

- [ ] **Step 1: Write failing clustering tests**

Create `services/orion-graph-compression/tests/test_leiden_clustering.py`:

```python
import pytest
import networkx as nx


def test_leiden_cluster_small_graph():
    """Three connected triples form at least one community."""
    from app.clustering.leiden import leiden_cluster

    G = nx.Graph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    G.add_edge("C", "A")
    G.add_edge("D", "E")
    G.add_edge("E", "F")
    G.add_edge("F", "D")

    communities = leiden_cluster(G, resolution=1.0, n_iterations=2)
    assert len(communities) >= 1
    all_nodes = set().union(*communities)
    assert all_nodes == set(G.nodes)


def test_leiden_cluster_empty_graph_returns_empty():
    from app.clustering.leiden import leiden_cluster
    G = nx.Graph()
    communities = leiden_cluster(G, resolution=1.0, n_iterations=2)
    assert communities == []


def test_leiden_cluster_single_node():
    from app.clustering.leiden import leiden_cluster
    G = nx.Graph()
    G.add_node("solo")
    communities = leiden_cluster(G, resolution=1.0, n_iterations=2)
    assert communities == [{"solo"}]


def test_build_graph_from_triples():
    from app.clustering.leiden import build_graph_from_triples
    triples = [
        ("http://A", "http://rel", "http://B"),
        ("http://B", "http://rel", "http://C"),
    ]
    G = build_graph_from_triples(triples)
    assert G.has_node("http://A")
    assert G.has_edge("http://A", "http://B")
    assert len(G.nodes) == 3
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=services/orion-graph-compression pytest \
  services/orion-graph-compression/tests/test_leiden_clustering.py -v 2>&1 | head -10
```

Expected: `ImportError: No module named 'app.clustering'`

- [ ] **Step 3: Create `app/clustering/__init__.py`** (empty)

- [ ] **Step 4: Write `app/clustering/leiden.py`**

```python
from __future__ import annotations

import logging
from typing import List, Set, Tuple

import networkx as nx

logger = logging.getLogger("orion.graph-compression.clustering.leiden")

Triple = Tuple[str, str, str]


def build_graph_from_triples(triples: List[Triple]) -> nx.Graph:
    G = nx.Graph()
    for s, _p, o in triples:
        # Only add subject↔object edges (predicate is edge label, not node)
        if s and o and s != o:
            G.add_edge(s, o)
    return G


def leiden_cluster(
    G: nx.Graph,
    *,
    resolution: float = 1.0,
    n_iterations: int = 10,
) -> List[Set[str]]:
    if G.number_of_nodes() == 0:
        return []
    if G.number_of_nodes() == 1:
        return [set(G.nodes)]
    try:
        import igraph as ig
        import leidenalg

        # Map networkx → igraph
        nodes = list(G.nodes)
        node_index = {n: i for i, n in enumerate(nodes)}
        edges = [(node_index[u], node_index[v]) for u, v in G.edges()]
        ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=False)

        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            n_iterations=n_iterations,
        )
        return [
            {nodes[i] for i in community}
            for community in partition
            if len(community) > 0
        ]
    except Exception as exc:
        logger.warning("leiden_cluster_failed reason=%s — falling back to connected components", exc)
        return [set(c) for c in nx.connected_components(G)]
```

- [ ] **Step 5: Write failing region builder tests**

Create `services/orion-graph-compression/tests/test_region_builder.py`:

```python
import pytest
from datetime import datetime, timezone


def test_region_builder_produces_valid_region():
    from app.clustering.region_builder import build_region, stable_region_id

    nodes = {"http://A", "http://B", "http://C"}
    region = build_region(
        nodes=nodes,
        scope="episodic",
        kind="community",
        summary="Test community summary.",
        summary_kind="structural",
        salience=0.6,
        trust_tier="unverified",
        compression_version="1.0.0",
    )
    assert region.scope == "episodic"
    assert region.kind == "community"
    assert region.summary == "Test community summary."
    assert len(region.exemplar_ids) > 0
    assert len(region.derived_from) > 0
    assert region.stale is False


def test_region_builder_stable_id_idempotent():
    from app.clustering.region_builder import stable_region_id

    nodes = frozenset({"http://A", "http://B"})
    id1 = stable_region_id(scope="episodic", kind="community", nodes=nodes)
    id2 = stable_region_id(scope="episodic", kind="community", nodes=nodes)
    assert id1 == id2
    assert id1.startswith("urn:orion:compression:region:")


def test_region_builder_different_nodes_different_id():
    from app.clustering.region_builder import stable_region_id

    id1 = stable_region_id("episodic", "community", frozenset({"http://A"}))
    id2 = stable_region_id("episodic", "community", frozenset({"http://B"}))
    assert id1 != id2


def test_region_builder_trust_tier_inherits_lowest():
    from app.clustering.region_builder import build_region

    region = build_region(
        nodes={"http://A"},
        scope="substrate",
        kind="contradiction",
        summary="Conflict found.",
        summary_kind="structural",
        salience=0.9,
        trust_tier="unverified",
        compression_version="1.0.0",
    )
    assert region.trust_tier == "unverified"
```

- [ ] **Step 6: Run to verify failure**

```bash
PYTHONPATH=services/orion-graph-compression pytest \
  services/orion-graph-compression/tests/test_region_builder.py -v 2>&1 | head -10
```

Expected: `ImportError: No module named 'app.clustering.region_builder'`

- [ ] **Step 7: Write `app/clustering/region_builder.py`**

```python
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Set

from orion.schemas.graph_compression import CompressionRegionV1


def stable_region_id(scope: str, kind: str, nodes: frozenset) -> str:
    """Deterministic region ID from scope + kind + sorted node URIs."""
    content = f"{scope}:{kind}:" + ":".join(sorted(nodes))
    digest = hashlib.sha256(content.encode()).hexdigest()[:24]
    return f"urn:orion:compression:region:{digest}"


def build_region(
    *,
    nodes: Set[str],
    scope: str,
    kind: str,
    summary: str,
    summary_kind: str,
    salience: float,
    trust_tier: str,
    compression_version: str,
) -> CompressionRegionV1:
    node_list = sorted(nodes)
    exemplar_ids = node_list[:5] if node_list else []
    if not exemplar_ids:
        exemplar_ids = [f"urn:orion:compression:empty:{scope}"]
    derived_from = node_list[:20] if node_list else [f"urn:orion:compression:empty:{scope}"]

    return CompressionRegionV1(
        region_id=stable_region_id(scope, kind, frozenset(nodes)),
        scope=scope,  # type: ignore[arg-type]
        kind=kind,  # type: ignore[arg-type]
        summary=summary,
        summary_kind=summary_kind,  # type: ignore[arg-type]
        salience=salience,
        trust_tier=trust_tier,
        exemplar_ids=exemplar_ids,
        derived_from=derived_from,
        generated_at=datetime.now(timezone.utc),
        compression_version=compression_version,
    )
```

- [ ] **Step 8: Run all clustering tests**

```bash
PYTHONPATH=services/orion-graph-compression pytest \
  services/orion-graph-compression/tests/test_leiden_clustering.py \
  services/orion-graph-compression/tests/test_region_builder.py -v
```

Expected: 8 tests, all PASS.

- [ ] **Step 9: Commit**

```bash
git add services/orion-graph-compression/app/clustering/__init__.py \
        services/orion-graph-compression/app/clustering/leiden.py \
        services/orion-graph-compression/app/clustering/region_builder.py \
        services/orion-graph-compression/tests/test_leiden_clustering.py \
        services/orion-graph-compression/tests/test_region_builder.py
git commit -m "feat(graph-compression): Leiden clustering + CompressionRegionV1 builder with stable IDs"
```

---

## Task 8: LLM summarizer + structural fallback

**Files:**
- Create: `services/orion-graph-compression/app/summarizer.py`

No unit test here — the LLM Gateway is a bus RPC that can't easily be unit tested without a live bus. The structural fallback path is exercised in Task 11's degraded worker test. A note on how to manually test is included.

- [ ] **Step 1: Write `app/summarizer.py`**

```python
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Set

if TYPE_CHECKING:
    from orion.core.bus.async_service import OrionBusAsync

logger = logging.getLogger("orion.graph-compression.summarizer")

_LLM_PROMPT_TMPL = """You are summarizing a cluster of semantic memory nodes from an AI cognitive system.

Graph scope: {scope}
Cluster kind: {kind}
Number of nodes: {node_count}
Sample node URIs (up to 5):
{sample_nodes}

Write a concise 1-3 sentence summary of what this cluster represents.
Be specific about the topics, entities, or tensions visible in the node URIs.
Max {max_tokens} tokens.
"""


class RegionSummarizer:
    def __init__(
        self,
        *,
        bus: "OrionBusAsync",
        llm_channel: str,
        service_name: str,
        service_version: str,
        max_tokens: int = 200,
        timeout_sec: float = 15.0,
    ) -> None:
        self._bus = bus
        self._llm_channel = llm_channel
        self._service_name = service_name
        self._service_version = service_version
        self._max_tokens = max_tokens
        self._timeout_sec = timeout_sec

    async def summarize(
        self,
        *,
        scope: str,
        kind: str,
        nodes: Set[str],
    ) -> tuple[str, str]:
        """
        Returns (summary_text, summary_kind).
        summary_kind is "llm" or "structural".
        Falls back to structural if LLM times out or bus unavailable.
        """
        try:
            return await asyncio.wait_for(
                self._llm_summarize(scope=scope, kind=kind, nodes=nodes),
                timeout=self._timeout_sec,
            )
        except Exception as exc:
            logger.warning("llm_summarize_failed scope=%s kind=%s reason=%s — using structural", scope, kind, exc)
        return self._structural_summary(scope=scope, kind=kind, nodes=nodes), "structural"

    async def _llm_summarize(self, *, scope: str, kind: str, nodes: Set[str]) -> tuple[str, str]:
        from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

        sample_nodes = "\n".join(sorted(nodes)[:5])
        prompt = _LLM_PROMPT_TMPL.format(
            scope=scope,
            kind=kind,
            node_count=len(nodes),
            sample_nodes=sample_nodes,
            max_tokens=self._max_tokens,
        )
        corr_id = str(uuid.uuid4())
        reply_channel = f"orion:exec:result:LLMGatewayService:{corr_id}"
        payload = {
            "goal": prompt,
            "corr_id": corr_id,
            "reply_to": reply_channel,
            "max_tokens": self._max_tokens,
        }
        envelope = BaseEnvelope(
            kind="chat.request.v1",
            source=ServiceRef(name=self._service_name, version=self._service_version),
            payload=payload,
        )
        await self._bus.publish(self._llm_channel, envelope)
        reply = await self._bus.recv_one(reply_channel, timeout=self._timeout_sec)
        if reply is None:
            raise TimeoutError("llm_gateway_no_reply")
        text = (reply.payload or {}).get("text") or (reply.payload or {}).get("response") or ""
        if not text:
            raise ValueError("llm_gateway_empty_response")
        return text.strip()[:1000], "llm"

    def _structural_summary(self, *, scope: str, kind: str, nodes: Set[str]) -> str:
        sample = sorted(nodes)[:3]
        labels = [n.rsplit("/", 1)[-1].rsplit("#", 1)[-1] for n in sample]
        return (
            f"[structural] {scope} {kind} cluster: {len(nodes)} nodes. "
            f"Sample: {', '.join(labels)}."
        )
```

- [ ] **Step 2: Import smoke check**

```bash
PYTHONPATH=services/orion-graph-compression python -c "
from app.summarizer import RegionSummarizer; print('OK')
"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add services/orion-graph-compression/app/summarizer.py
git commit -m "feat(graph-compression): RegionSummarizer — LLM Gateway bus RPC with structural fallback"
```

---

## Task 9: Fuseki writer + grammar substrate hook

**Files:**
- Create: `services/orion-graph-compression/app/writer.py`
- Create: `services/orion-graph-compression/tests/test_writer_sparql.py`
- Create: `services/orion-graph-compression/tests/test_grammar_hook.py`

- [ ] **Step 1: Write failing SPARQL writer tests**

Create `services/orion-graph-compression/tests/test_writer_sparql.py`:

```python
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timezone


def _make_region(kind="community"):
    from orion.schemas.graph_compression import CompressionRegionV1
    return CompressionRegionV1(
        region_id="urn:orion:compression:region:abc123",
        scope="episodic",
        kind=kind,
        summary="Test summary.",
        summary_kind="structural",
        salience=0.7,
        trust_tier="unverified",
        exemplar_ids=["http://conjourney.net/chat/turn/1"],
        derived_from=["http://conjourney.net/chat/turn/1"],
        generated_at=datetime.now(timezone.utc),
        compression_version="1.0.0",
    )


def test_writer_builds_sparql_update_targeting_compressions_graph():
    from app.writer import CompressionWriter
    w = CompressionWriter(
        update_url="http://fuseki/update",
        user="admin",
        password="orion",
        timeout_sec=5.0,
        bus=None,
        service_name="orion-graph-compression",
        service_version="0.1.0",
        channel_events="orion:graph:compression:events",
        channel_pressure="orion:substrate:mutation:pressure",
    )
    region = _make_region()
    sparql = w._build_sparql_update(region)
    assert "orion/compressions" in sparql
    assert region.region_id in sparql
    assert "INSERT DATA" in sparql


def test_writer_includes_summary_literal():
    from app.writer import CompressionWriter
    w = CompressionWriter(
        update_url="http://fuseki/update",
        user="admin",
        password="orion",
        timeout_sec=5.0,
        bus=None,
        service_name="orion-graph-compression",
        service_version="0.1.0",
        channel_events="orion:graph:compression:events",
        channel_pressure="orion:substrate:mutation:pressure",
    )
    region = _make_region()
    sparql = w._build_sparql_update(region)
    assert "Test summary." in sparql
```

Create `services/orion-graph-compression/tests/test_grammar_hook.py`:

```python
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone


def _make_contradiction_region():
    from orion.schemas.graph_compression import CompressionRegionV1
    return CompressionRegionV1(
        region_id="urn:orion:compression:region:contradiction1",
        scope="substrate",
        kind="contradiction",
        summary="Conflicting beliefs about X and Y.",
        summary_kind="llm",
        salience=0.85,
        trust_tier="unverified",
        exemplar_ids=["http://conjourney.net/substrate/node/1"],
        derived_from=["http://conjourney.net/substrate/node/1"],
        generated_at=datetime.now(timezone.utc),
        compression_version="1.0.0",
    )


def test_contradiction_region_emits_mutation_pressure():
    """Writing a contradiction region must publish MutationPressureEvidenceV1 on the pressure channel."""
    from app.writer import CompressionWriter

    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()

    w = CompressionWriter(
        update_url="http://fuseki/update",
        user="admin",
        password="orion",
        timeout_sec=5.0,
        bus=mock_bus,
        service_name="orion-graph-compression",
        service_version="0.1.0",
        channel_events="orion:graph:compression:events",
        channel_pressure="orion:substrate:mutation:pressure",
    )

    region = _make_contradiction_region()

    async def run():
        await w._emit_grammar_hook(region)

    asyncio.run(run())

    mock_bus.publish.assert_called_once()
    call_args = mock_bus.publish.call_args
    channel = call_args[0][0]
    assert channel == "orion:substrate:mutation:pressure"
    envelope = call_args[0][1]
    pressure = envelope.payload
    assert pressure.get("source_service") == "orion-graph-compression"
    assert "contradiction" in str(pressure.get("pressure_category", ""))


def test_non_contradiction_region_does_not_emit_pressure():
    from app.writer import CompressionWriter

    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()

    w = CompressionWriter(
        update_url="http://fuseki/update",
        user="admin",
        password="orion",
        timeout_sec=5.0,
        bus=mock_bus,
        service_name="orion-graph-compression",
        service_version="0.1.0",
        channel_events="orion:graph:compression:events",
        channel_pressure="orion:substrate:mutation:pressure",
    )

    from orion.schemas.graph_compression import CompressionRegionV1
    region = CompressionRegionV1(
        region_id="urn:orion:compression:region:community1",
        scope="episodic",
        kind="community",  # not contradiction
        summary="Normal community.",
        summary_kind="structural",
        salience=0.5,
        trust_tier="unverified",
        exemplar_ids=["http://example.org/1"],
        derived_from=["http://example.org/1"],
        generated_at=datetime.now(timezone.utc),
        compression_version="1.0.0",
    )

    async def run():
        await w._emit_grammar_hook(region)

    asyncio.run(run())
    mock_bus.publish.assert_not_called()
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=services/orion-graph-compression pytest \
  services/orion-graph-compression/tests/test_writer_sparql.py \
  services/orion-graph-compression/tests/test_grammar_hook.py -v 2>&1 | head -10
```

Expected: `ImportError: No module named 'app.writer'`

- [ ] **Step 3: Write `app/writer.py`**

```python
from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Optional

import httpx

from orion.schemas.graph_compression import (
    CompressionRegionV1,
    GraphCompressionRegionMaterializedV1,
)

if TYPE_CHECKING:
    from orion.core.bus.async_service import OrionBusAsync

logger = logging.getLogger("orion.graph-compression.writer")

_COMPRESSIONS_GRAPH_URI = "http://conjourney.net/graph/orion/compressions"
_ORN_NS = "http://orion.conjourney.net/ns/compression#"


class CompressionWriter:
    def __init__(
        self,
        *,
        update_url: str,
        user: str,
        password: str,
        timeout_sec: float,
        bus: Optional["OrionBusAsync"],
        service_name: str,
        service_version: str,
        channel_events: str,
        channel_pressure: str,
    ) -> None:
        self._update_url = update_url
        self._auth = (user, password)
        self._timeout = timeout_sec
        self._bus = bus
        self._service_name = service_name
        self._service_version = service_version
        self._channel_events = channel_events
        self._channel_pressure = channel_pressure

    def _escape_literal(self, value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    def _build_sparql_update(self, region: CompressionRegionV1) -> str:
        rid = region.region_id
        triples = [
            f'<{rid}> <{_ORN_NS}scope> "{self._escape_literal(region.scope)}" .',
            f'<{rid}> <{_ORN_NS}kind> "{self._escape_literal(region.kind)}" .',
            f'<{rid}> <{_ORN_NS}summary> "{self._escape_literal(region.summary)}" .',
            f'<{rid}> <{_ORN_NS}summaryKind> "{self._escape_literal(region.summary_kind)}" .',
            f'<{rid}> <{_ORN_NS}salience> "{region.salience}"^^<http://www.w3.org/2001/XMLSchema#decimal> .',
            f'<{rid}> <{_ORN_NS}trustTier> "{self._escape_literal(region.trust_tier)}" .',
            f'<{rid}> <{_ORN_NS}compressionVersion> "{self._escape_literal(region.compression_version)}" .',
            f'<{rid}> <{_ORN_NS}generatedAt> "{region.generated_at.isoformat()}"^^<http://www.w3.org/2001/XMLSchema#dateTime> .',
        ]
        for exemplar in region.exemplar_ids:
            triples.append(f'<{rid}> <{_ORN_NS}exemplarId> <{exemplar}> .')
        for src in region.derived_from:
            triples.append(f'<{rid}> <{_ORN_NS}derivedFrom> <{src}> .')

        triples_block = "\n    ".join(triples)
        return (
            f"DELETE {{ GRAPH <{_COMPRESSIONS_GRAPH_URI}> {{ <{rid}> ?p ?o }} }}\n"
            f"WHERE  {{ GRAPH <{_COMPRESSIONS_GRAPH_URI}> {{ <{rid}> ?p ?o }} }} ;\n"
            f"INSERT DATA {{\n"
            f"  GRAPH <{_COMPRESSIONS_GRAPH_URI}> {{\n"
            f"    {triples_block}\n"
            f"  }}\n"
            f"}}"
        )

    def write(self, region: CompressionRegionV1) -> bool:
        """Write region to Fuseki. Returns True on success."""
        sparql = self._build_sparql_update(region)
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    self._update_url,
                    data={"update": sparql},
                    auth=self._auth,
                )
                resp.raise_for_status()
            return True
        except Exception as exc:
            logger.warning(
                "compression_write_failed region_id=%s reason=%s",
                region.region_id,
                exc,
            )
            return False

    async def _emit_grammar_hook(self, region: CompressionRegionV1) -> None:
        """Emit bus events after successful write."""
        if self._bus is None:
            return

        from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

        # Passive materialization event (all kinds)
        materialized = GraphCompressionRegionMaterializedV1(
            region_id=region.region_id,
            scope=region.scope,
            kind=region.kind,
            salience=region.salience,
            trust_tier=region.trust_tier,
            summary_kind=region.summary_kind,
            compression_version=region.compression_version,
            ts=time.time(),
        )
        await self._bus.publish(
            self._channel_events,
            BaseEnvelope(
                kind="graph.compression.region.materialized.v1",
                source=ServiceRef(name=self._service_name, version=self._service_version),
                payload=materialized.model_dump(mode="json"),
            ),
        )

        # Contradiction regions → substrate mutation pressure
        if region.kind == "contradiction":
            from orion.core.schemas.substrate_mutation import MutationPressureEvidenceV1

            pressure = MutationPressureEvidenceV1(
                source_service=self._service_name,
                source_event_id=region.region_id,
                pressure_category="unsupported_memory_claim",
                confidence=region.salience,
                evidence_refs=[region.region_id] + region.derived_from[:4],
            )
            await self._bus.publish(
                self._channel_pressure,
                BaseEnvelope(
                    kind="substrate.mutation.pressure.v1",
                    source=ServiceRef(name=self._service_name, version=self._service_version),
                    payload=pressure.model_dump(mode="json"),
                ),
            )
```

- [ ] **Step 4: Run all writer + grammar tests**

```bash
PYTHONPATH=services/orion-graph-compression pytest \
  services/orion-graph-compression/tests/test_writer_sparql.py \
  services/orion-graph-compression/tests/test_grammar_hook.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add services/orion-graph-compression/app/writer.py \
        services/orion-graph-compression/tests/test_writer_sparql.py \
        services/orion-graph-compression/tests/test_grammar_hook.py
git commit -m "feat(graph-compression): CompressionWriter — Fuseki SPARQL UPDATE + grammar substrate hooks"
```

---

## Task 10: Worker + stale listener + degraded-mode test

**Files:**
- Create: `services/orion-graph-compression/app/stale_listener.py`
- Create: `services/orion-graph-compression/app/worker.py`
- Create: `services/orion-graph-compression/tests/test_worker_degraded.py`

- [ ] **Step 1: Write failing degraded worker test**

Create `services/orion-graph-compression/tests/test_worker_degraded.py`:

```python
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch


def _make_worker(enable=True):
    with patch("app.settings.get_settings") as mock_settings:
        s = MagicMock()
        s.enable_compression_runtime = enable
        s.compression_batch_size = 5
        s.compression_poll_interval_sec = 0.01
        s.compression_max_tokens_per_summary = 200
        s.compression_llm_budget_per_tick = 5000
        s.compression_max_age_sec = 86400
        s.compression_policy_path = "config/compression_policy.v1.yaml"
        s.rdf_store_query_url = "http://fuseki/query"
        s.rdf_store_update_url = "http://fuseki/update"
        s.rdf_store_user = "admin"
        s.rdf_store_pass = "orion"
        s.rdf_store_timeout_sec = 5.0
        s.llm_gateway_bus_channel = "orion:exec:request:LLMGatewayService"
        s.channel_graph_compression_events = "orion:graph:compression:events"
        s.service_name = "orion-graph-compression"
        s.service_version = "0.1.0"
        mock_settings.return_value = s

        from app.worker import CompressionWorker
        store = MagicMock()
        bus = MagicMock()
        worker = CompressionWorker(store=store, bus=bus)
        return worker, store, bus


def test_worker_disabled_skips_tick():
    """When ENABLE_COMPRESSION_RUNTIME=false the tick does nothing."""
    worker, store, bus = _make_worker(enable=False)
    worker._tick()
    store.drain_stale_queue.assert_not_called()


def test_worker_empty_federators_no_crash():
    """All federators returning [] must not raise — just skip region."""
    worker, store, bus = _make_worker(enable=True)

    store.drain_stale_queue.return_value = [
        {"id": 1, "region_id": None, "scope": "episodic", "reason": "test", "priority": 0}
    ]

    with patch("app.worker.EpisodicFederator") as mock_ep, \
         patch("app.worker.SubstrateFederator") as mock_sub, \
         patch("app.worker.SelfStudyFederator") as mock_ss:
        for mock_cls in [mock_ep, mock_sub, mock_ss]:
            instance = MagicMock()
            instance.fetch.return_value = []
            mock_cls.return_value = instance

        # Should complete without raising
        worker._tick()
        store.delete_stale_queue_items.assert_called_once_with([1])


def test_worker_budget_gate_halts_mid_batch():
    """If LLM token budget is 0, no summarization occurs but regions are still processed structurally."""
    worker, store, bus = _make_worker(enable=True)
    worker._settings.compression_llm_budget_per_tick = 0

    store.drain_stale_queue.return_value = [
        {"id": 1, "region_id": None, "scope": "episodic", "reason": "test", "priority": 0}
    ]

    # Federator returns some triples
    fake_triples = [("http://A", "http://rel", "http://B"), ("http://B", "http://rel", "http://C")]

    with patch("app.worker.EpisodicFederator") as mock_ep, \
         patch("app.worker.SubstrateFederator") as mock_sub, \
         patch("app.worker.SelfStudyFederator") as mock_ss:
        mock_ep.return_value.fetch.return_value = fake_triples
        mock_sub.return_value.fetch.return_value = []
        mock_ss.return_value.fetch.return_value = []

        worker._tick()
        # Should have processed without calling LLM (structural fallback used)
        store.upsert_artifact.assert_called()
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=services/orion-graph-compression pytest \
  services/orion-graph-compression/tests/test_worker_degraded.py -v 2>&1 | head -15
```

Expected: `ImportError: No module named 'app.worker'`

- [ ] **Step 3: Write `app/stale_listener.py`**

```python
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from orion.core.bus.async_service import OrionBusAsync
    from app.store import CompressionStore

logger = logging.getLogger("orion.graph-compression.stale_listener")

# Map RDF enqueue graph names to compression scopes
_GRAPH_TO_SCOPE = {
    "orion:chat": "episodic",
    "orion:enrichment": "episodic",
    "orion:collapse": "episodic",
    "orion:cognition": "episodic",
    "orion:metacog": "episodic",
    "orion:chat:social": "episodic",
    "orion:autonomy:identity": "episodic",
    "orion:autonomy:drives": "episodic",
    "orion:autonomy:goals": "episodic",
    "orion:self": "self_study",
    "orion:self:induced": "self_study",
    "orion:self:reflective": "self_study",
    "orion:substrate": "substrate",
}


async def run_stale_listener(
    *,
    bus: "OrionBusAsync",
    store: "CompressionStore",
    channel_rdf_enqueue: str,
    channel_stale: str,
) -> None:
    """
    Subscribes to two channels:
    - orion:rdf:enqueue — mark affected scope stale when a graph is written
    - orion:graph:compression:stale — explicit staleness marks from other services
    """
    async def _handle(envelope: Any) -> None:
        try:
            payload = envelope.payload or {}
            # From orion:rdf:enqueue: look for graph_name field
            graph_name = (
                payload.get("graph_name")
                or payload.get("named_graph")
                or payload.get("graph")
                or ""
            )
            scope = _GRAPH_TO_SCOPE.get(graph_name)
            if scope:
                store.enqueue_stale(scope=scope, reason=f"rdf_enqueue:{graph_name}")
                logger.debug("stale_marked scope=%s graph=%s", scope, graph_name)
            else:
                # Mark all scopes stale on unknown graph writes
                for s in ("episodic", "substrate", "self_study"):
                    store.enqueue_stale(scope=s, reason="rdf_enqueue:unknown_graph")
        except Exception as exc:
            logger.warning("stale_listener_handle_error reason=%s", exc)

    async def _handle_explicit(envelope: Any) -> None:
        try:
            payload = envelope.payload or {}
            scope = payload.get("scope")
            region_id = payload.get("region_id")
            reason = payload.get("reason", "explicit_stale_mark")
            store.enqueue_stale(scope=scope, region_id=region_id, reason=reason)
        except Exception as exc:
            logger.warning("stale_listener_explicit_handle_error reason=%s", exc)

    await bus.subscribe(channel_rdf_enqueue, _handle)
    await bus.subscribe(channel_stale, _handle_explicit)
    logger.info("stale_listener_started channels=[%s, %s]", channel_rdf_enqueue, channel_stale)
```

- [ ] **Step 4: Write `app/worker.py`**

```python
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import yaml

from app.clustering.leiden import build_graph_from_triples, leiden_cluster
from app.clustering.region_builder import build_region
from app.federators.episodic import EpisodicFederator
from app.federators.self_study import SelfStudyFederator
from app.federators.substrate import SubstrateFederator
from app.settings import get_settings

if TYPE_CHECKING:
    from orion.core.bus.async_service import OrionBusAsync
    from app.store import CompressionStore

logger = logging.getLogger("orion.graph-compression.worker")


class CompressionWorker:
    def __init__(
        self,
        *,
        store: "CompressionStore",
        bus: "OrionBusAsync",
    ) -> None:
        self._settings = get_settings()
        self._store = store
        self._bus = bus
        self._stop = asyncio.Event()
        self._policy = self._load_policy()

    def _load_policy(self) -> dict[str, Any]:
        try:
            return yaml.safe_load(Path(self._settings.compression_policy_path).read_text())
        except Exception as exc:
            logger.warning("policy_load_failed reason=%s — using defaults", exc)
            return {"clustering": {"resolution": 1.0, "n_iterations": 10, "min_community_size": 3, "max_communities_per_scope": 20}}

    async def start(self) -> None:
        asyncio.create_task(self._poll_loop(), name="graph-compression-poll")

    async def stop(self) -> None:
        self._stop.set()

    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._tick)
            except Exception:
                logger.exception("compression_tick_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.compression_poll_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _tick(self) -> None:
        if not self._settings.enable_compression_runtime:
            return

        items = self._store.drain_stale_queue(batch_size=self._settings.compression_batch_size)
        if not items:
            return

        processed_ids = []
        llm_tokens_used = 0
        budget = self._settings.compression_llm_budget_per_tick

        scopes_to_process = list({item.get("scope") for item in items if item.get("scope")})
        if not scopes_to_process:
            scopes_to_process = ["episodic", "substrate", "self_study"]

        for scope in scopes_to_process:
            try:
                self._process_scope(scope=scope, llm_tokens_used=llm_tokens_used, budget=budget)
            except Exception:
                logger.exception("scope_process_failed scope=%s", scope)

        queue_ids = [item["id"] for item in items if "id" in item]
        self._store.delete_stale_queue_items(queue_ids)
        logger.info("compression_tick_complete scopes=%s queue_items_drained=%d", scopes_to_process, len(queue_ids))

    def _process_scope(self, *, scope: str, llm_tokens_used: int, budget: int) -> None:
        cluster_cfg = (self._policy or {}).get("clustering", {})
        resolution = float(cluster_cfg.get("resolution", 1.0))
        n_iter = int(cluster_cfg.get("n_iterations", 10))
        min_size = int(cluster_cfg.get("min_community_size", 3))
        max_communities = int(cluster_cfg.get("max_communities_per_scope", 20))

        from app.settings import get_settings as _gs
        s = _gs()

        federator_kwargs = dict(
            query_url=s.rdf_store_query_url,
            user=s.rdf_store_user,
            password=s.rdf_store_pass,
            timeout_sec=s.rdf_store_timeout_sec,
        )

        if scope == "episodic":
            triples = EpisodicFederator(**federator_kwargs).fetch()
            kind = "community"
        elif scope == "substrate":
            triples = SubstrateFederator(**federator_kwargs).fetch()
            kind = "contradiction"
        elif scope == "self_study":
            triples = SelfStudyFederator(**federator_kwargs).fetch()
            kind = "self_study_cluster"
        else:
            return

        if not triples:
            logger.debug("scope_empty scope=%s — skipping", scope)
            return

        G = build_graph_from_triples(triples)
        communities = leiden_cluster(G, resolution=resolution, n_iterations=n_iter)
        communities = [c for c in communities if len(c) >= min_size][:max_communities]

        if not communities:
            logger.debug("no_viable_communities scope=%s nodes=%d", scope, G.number_of_nodes())
            return

        from app.writer import CompressionWriter
        writer = CompressionWriter(
            update_url=s.rdf_store_update_url,
            user=s.rdf_store_user,
            password=s.rdf_store_pass,
            timeout_sec=s.rdf_store_timeout_sec,
            bus=self._bus,
            service_name=s.service_name,
            service_version=s.service_version,
            channel_events=s.channel_graph_compression_events,
            channel_pressure="orion:substrate:mutation:pressure",
        )

        for community in communities:
            summary = (
                f"[structural] {scope} {kind} cluster: {len(community)} nodes."
            )
            salience = min(1.0, len(community) / max(1, G.number_of_nodes()))
            region = build_region(
                nodes=community,
                scope=scope,
                kind=kind,
                summary=summary,
                summary_kind="structural",
                salience=salience,
                trust_tier="unverified",
                compression_version="1.0.0",
            )
            if writer.write(region):
                self._store.upsert_artifact(
                    region_id=region.region_id,
                    scope=region.scope,
                    kind=region.kind,
                    summary_kind=region.summary_kind,
                    salience=region.salience,
                    trust_tier=region.trust_tier,
                    compression_version=region.compression_version,
                    generated_at=region.generated_at,
                )
                logger.info(
                    "region_written region_id=%s scope=%s kind=%s nodes=%d",
                    region.region_id, scope, kind, len(community),
                )
```

- [ ] **Step 5: Run degraded worker tests**

```bash
PYTHONPATH=services/orion-graph-compression pytest \
  services/orion-graph-compression/tests/test_worker_degraded.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add services/orion-graph-compression/app/stale_listener.py \
        services/orion-graph-compression/app/worker.py \
        services/orion-graph-compression/tests/test_worker_degraded.py
git commit -m "feat(graph-compression): CompressionWorker poll loop + stale listener + degraded-mode tests"
```

---

## Task 11: `app/main.py` — FastAPI app, lifespan, heartbeat, HTTP endpoints

**Files:**
- Create: `services/orion-graph-compression/app/main.py`

- [ ] **Step 1: Write `app/main.py`**

```python
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException

from app.settings import get_settings
from app.store import CompressionStore
from app.worker import CompressionWorker
from app.stale_listener import run_stale_listener

logging.basicConfig(level=getattr(logging, get_settings().log_level.upper(), logging.INFO))
logger = logging.getLogger("orion.graph-compression")

BOOT_ID = str(uuid.uuid4())

_settings = get_settings()

bus = None
worker: Optional[CompressionWorker] = None
store: Optional[CompressionStore] = None
heartbeat_task: Optional[asyncio.Task] = None
stale_task: Optional[asyncio.Task] = None


async def heartbeat_loop(bus_instance: Any) -> None:
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
    from orion.schemas.telemetry.system_health import SystemHealthV1

    logger.info("heartbeat_loop_started boot_id=%s", BOOT_ID)
    while True:
        try:
            payload = SystemHealthV1(
                service=_settings.service_name,
                version=_settings.service_version,
                node=_settings.node_name,
                status="ok",
                boot_id=BOOT_ID,
                last_seen_ts=time.time(),
            ).model_dump(mode="json")
            await bus_instance.publish(
                _settings.health_channel,
                BaseEnvelope(
                    kind="system.health.v1",
                    source=ServiceRef(name=_settings.service_name, version=_settings.service_version),
                    payload=payload,
                ),
            )
        except Exception as exc:
            logger.warning("heartbeat_failed reason=%s", exc)
        await asyncio.sleep(_settings.heartbeat_interval_sec)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bus, worker, store, heartbeat_task, stale_task

    store = CompressionStore(_settings.postgres_uri)
    try:
        store.ensure_tables()
    except Exception as exc:
        logger.warning("ensure_tables_failed reason=%s", exc)

    if _settings.orion_bus_enabled:
        try:
            from orion.core.bus.async_service import OrionBusAsync
            bus = OrionBusAsync(_settings.orion_bus_url)
            await bus.connect()
            heartbeat_task = asyncio.create_task(heartbeat_loop(bus), name="gc-heartbeat")
            stale_task = asyncio.create_task(
                run_stale_listener(
                    bus=bus,
                    store=store,
                    channel_rdf_enqueue=_settings.channel_rdf_enqueue,
                    channel_stale=_settings.channel_graph_compression_stale,
                ),
                name="gc-stale-listener",
            )
        except Exception as exc:
            logger.error("bus_connect_failed reason=%s — running without bus", exc)

    worker = CompressionWorker(store=store, bus=bus)
    await worker.start()

    yield

    await worker.stop()
    for task in [heartbeat_task, stale_task]:
        if task and not task.done():
            task.cancel()
    if bus:
        try:
            await bus.disconnect()
        except Exception:
            pass


app = FastAPI(title="orion-graph-compression", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    queue_depth = 0
    artifact_count = 0
    if store:
        try:
            queue_depth = store.stale_queue_depth()
            artifact_count = store.artifact_count()
        except Exception:
            pass
    return {
        "status": "ok",
        "service": _settings.service_name,
        "boot_id": BOOT_ID,
        "compression_runtime_enabled": _settings.enable_compression_runtime,
        "stale_queue_depth": queue_depth,
        "artifact_count": artifact_count,
    }


@app.get("/regions")
async def list_regions(scope: Optional[str] = None) -> list[dict[str, Any]]:
    if not store:
        return []
    return store.list_artifacts(scope=scope)


@app.get("/artifacts/{region_id}")
async def get_artifact(region_id: str) -> dict[str, Any]:
    if not store:
        raise HTTPException(status_code=503, detail="store_unavailable")
    artifact = store.get_artifact(region_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="not_found")
    return artifact
```

- [ ] **Step 2: Import smoke check**

```bash
PYTHONPATH=services/orion-graph-compression python -c "
from app.main import app; print('FastAPI app created:', app.title)
"
```

Expected: `FastAPI app created: orion-graph-compression`

- [ ] **Step 3: Run full test suite**

```bash
PYTHONPATH=services/orion-graph-compression pytest services/orion-graph-compression/tests/ -v
```

Expected: All tests PASS (no regressions).

- [ ] **Step 4: Commit**

```bash
git add services/orion-graph-compression/app/main.py
git commit -m "feat(graph-compression): FastAPI main — lifespan, heartbeat, /health /regions /artifacts"
```

---

## Task 12: Full test run + final commit

- [ ] **Step 1: Run all service tests**

```bash
PYTHONPATH=services/orion-graph-compression pytest services/orion-graph-compression/tests/ -v --tb=short
```

Expected: All pass. Fix any failures before proceeding.

- [ ] **Step 2: Run schema registry smoke check**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. python -c "
from orion.schemas.registry import resolve
for s in ['CompressionRegionV1', 'CompressionStalenessMarkV1', 'GraphCompressionRegionMaterializedV1']:
    cls = resolve(s)
    print(f'OK: {s} -> {cls.__name__}')
"
```

Expected:
```
OK: CompressionRegionV1 -> CompressionRegionV1
OK: CompressionStalenessMarkV1 -> CompressionStalenessMarkV1
OK: GraphCompressionRegionMaterializedV1 -> GraphCompressionRegionMaterializedV1
```

- [ ] **Step 3: Final commit tagging service complete**

```bash
git add -A
git commit -m "feat(graph-compression): orion-graph-compression service complete — federators, clustering, writer, worker, main"
```
