# Graph Compression v1 — Design Spec

**Date:** 2026-06-06  
**Status:** Approved for implementation planning  
**Scope:** `orion-graph-compression` service + `orion-recall` adapter extension

---

## 1. Problem statement

Orion's Fuseki store contains rich semantic graph data across multiple named graphs (episodic chat/enrichment, substrate, self-study, autonomy, cognition, etc.), but the recall path only reaches it via **keyword-filtered SPARQL** bounded to 8–16 fragments per turn. This means:

- **Global questions** ("what are my dominant preoccupations?", "what tensions am I carrying?") have no grounded answer — the model generates prose unsupported by the actual graph.
- **Cross-domain coherence** is impossible — chat enrichment, substrate contradictions, and self-study records never appear together in a single context window.
- **Keyword gaps** mean structurally important but lexically distant nodes are invisible to recall.

The goal of this feature is a **federated offline compression layer**: a service that periodically reads all RDF scopes, clusters them into semantic regions, generates cached summaries, and exposes them to recall as a first-class `graph_compression` backend. LLMs get pre-formed views of Orion's cognitive terrain, not just random triple matches.

---

## 2. Approach

**Native Orion implementation** — no LlamaIndex, no Microsoft GraphRAG runtime dependency. Borrows the GraphRAG community-summary pattern (Leiden clustering, region reports, global/local query modes) but implements it using:

- Fuseki SPARQL (existing `orion/graph/sparql_client.py` conventions)
- `leidenalg` + `igraph` for clustering
- `networkx` for graph construction from triples
- LLM Gateway bus RPC for summarization (on-bus, same as every other verb)
- Postgres + Fuseki for artifact storage (semantic content in Fuseki, operational metadata in Postgres)

---

## 3. Architecture and data flow

```text
[RDF writes / substrate commits]
         │
         ▼
orion:graph:compression:stale ──► orion-graph-compression
                                         │ (stale listener)
                                         ▼
                                  Postgres stale_queue
                                         │
                              ┌──────────▼──────────┐
                              │  CompressionWorker  │  (scheduled poll)
                              │  - drain stale queue│
                              │  - federate 3 scopes│
                              │  - Leiden cluster   │
                              │  - LLM summarize    │
                              │  - write artifacts  │
                              └──────┬──────┬───────┘
                                     │      │
                              Fuseki │      │ Postgres
                          orion:     │      │ compression_artifacts
                          compressions      │ compression_jobs
                                     │      │
                              ┌──────▼──────▼───────┐
                              │  orion-recall        │
                              │  graph_compression   │
                              │  adapter             │
                              └──────────┬───────────┘
                                         │ memory_bundle.rendered
                                         ▼
                                    LLM Gateway
```

**Non-goals (v1):**
- SQL timeline as compression input
- Hub UI for browsing compressions (health endpoint only)
- Agent tools for arbitrary SPARQL
- Trust-tier promotion from compression artifacts
- Grammar substrate reducer wiring (architecture hook present, reducer deferred to v2)

---

## 4. Named graph scope

### Episodic federator (`EpisodicFederator`)
Covers all episodic/memory named graphs in Fuseki:
- `http://conjourney.net/graph/orion/chat`
- `http://conjourney.net/graph/orion/enrichment`
- `http://conjourney.net/graph/orion/collapse`
- `http://conjourney.net/graph/orion/cognition`
- `http://conjourney.net/graph/orion/metacog`
- `http://conjourney.net/graph/orion/chat/social`
- `http://conjourney.net/graph/orion/autonomy/identity`
- `http://conjourney.net/graph/orion/autonomy/drives`
- `http://conjourney.net/graph/orion/autonomy/goals`

### Substrate federator (`SubstrateFederator`)
Uses existing `SubstrateSemanticReadCoordinator` bounded queries:
- `hotspot_region`
- `contradiction_region`
- `concept_region`
- `provenance_neighborhood`

### Self-study federator (`SelfStudyFederator`)
Named graphs:
- `http://conjourney.net/graph/orion/self` (`orion:self`)
- `http://conjourney.net/graph/orion/self/induced` (`orion:self:induced`)
- `http://conjourney.net/graph/orion/self/reflective` (`orion:self:reflective`)

### Output graph (new)
- `orion:compressions` → `http://conjourney.net/graph/orion/compressions`
- Requires adding to `normalize_graph_name()` in `services/orion-rdf-writer/app/rdf_store.py`

---

## 5. Service: `services/orion-graph-compression`

### File tree

```text
services/orion-graph-compression/
  Dockerfile
  docker-compose.yml
  .env_example
  requirements.txt
  app/
    main.py             ← FastAPI lifespan, heartbeat, /health, /regions, /artifacts/{id}
    settings.py
    worker.py           ← CompressionWorker poll loop + stale queue drain
    store.py            ← CompressionStore (Postgres)
    stale_listener.py   ← bus subscriber → stale marks in Postgres
    federators/
      __init__.py
      episodic.py       ← SPARQL over episodic named graphs
      substrate.py      ← SubstrateSemanticReadCoordinator adapter
      self_study.py     ← SPARQL over orion:self* graphs
    clustering/
      __init__.py
      leiden.py         ← Leiden on NetworkX graph from federated triples
      region_builder.py ← CompressionRegionV1 from cluster + exemplar selection
    summarizer.py       ← LLM Gateway bus RPC for region summaries
    writer.py           ← Fuseki SPARQL UPDATE to orion:compressions
  config/
    compression_policy.v1.yaml
orion/schemas/graph_compression.py  ← shared lib (new)
```

### `main.py` shape

Follows `orion-whisper-tts` / `orion-consolidation-runtime` conventions:

```python
BOOT_ID = str(uuid.uuid4())
bus: OrionBusAsync | None = None
worker: CompressionWorker | None = None
heartbeat_task: asyncio.Task | None = None
stale_listener_task: asyncio.Task | None = None

async def heartbeat_loop(bus_instance):
    """Publish SystemHealthV1 on orion:system:health every HEARTBEAT_INTERVAL_SEC."""
    while True:
        payload = SystemHealthV1(
            service=settings.service_name,
            version=settings.service_version,
            node=settings.node_name,
            status="ok",
            boot_id=BOOT_ID,
            last_seen_ts=time.time(),
        ).model_dump(mode="json")
        await bus_instance.publish(
            settings.health_channel,
            BaseEnvelope(kind="system.health.v1", source=..., payload=payload),
        )
        await asyncio.sleep(settings.heartbeat_interval_sec)

@asynccontextmanager
async def lifespan(app):
    # connect bus, start worker, start stale_listener, start heartbeat
    yield
    # cancel tasks, disconnect bus

app = FastAPI(title="orion-graph-compression", lifespan=lifespan)

@app.get("/health")  # includes runtime state, queue depth, last job, fuseki reachable
@app.get("/regions")  # artifact index from Postgres
@app.get("/artifacts/{region_id}")  # artifact metadata + summary preview
```

### `docker-compose.yml`

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
      - CHANNEL_STALE_INTAKE=${CHANNEL_STALE_INTAKE:-orion:graph:compression:stale}
      - CHANNEL_COMPRESSION_EVENTS=${CHANNEL_COMPRESSION_EVENTS:-orion:graph:compression:events}
      - COMPRESSION_POLICY_PATH=${COMPRESSION_POLICY_PATH:-/app/config/compression_policy.v1.yaml}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
networks:
  app-net:
    external: true
```

### `Dockerfile`

```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip
COPY services/orion-graph-compression/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY orion /app/orion
COPY config/compression /app/config
COPY services/orion-graph-compression /app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8270"]
```

### `requirements.txt` (key deps)
```
fastapi
uvicorn
pydantic>=2
httpx
networkx
leidenalg
igraph
sqlalchemy
psycopg2-binary
pyyaml
```

---

## 6. Shared schemas: `orion/schemas/graph_compression.py`

```python
class CompressionRegionV1(BaseModel):
    region_id: str          # urn:orion:compression:region:{stable_hash}
    scope: Literal["episodic", "substrate", "self_study"]
    kind: Literal["community", "hotspot", "contradiction", "self_study_cluster"]
    summary: str            # LLM or structural prose, <= 200 tokens
    summary_kind: Literal["llm", "structural"]  # structural = deterministic fallback
    salience: float         # inherited from source region / Leiden modularity
    trust_tier: str         # inherits lowest trust tier from source nodes
    exemplar_ids: list[str] # URIs of source ChatTurns / Claims / substrate nodes
    derived_from: list[str] # source triple / node URIs
    generated_at: datetime
    compression_version: str
    stale: bool = False

class CompressionStalenessMarkV1(BaseModel):
    """Bus payload: marks a graph region stale when source triples are written."""
    region_id: str | None = None   # None = mark all regions in scope stale
    scope: str | None = None
    reason: str
    source_service: str
    ts: float

class GraphCompressionRegionMaterializedV1(BaseModel):
    """Bus event published after each artifact write (passive grammar hook)."""
    region_id: str
    scope: str
    kind: str
    salience: float
    trust_tier: str
    summary_kind: str
    compression_version: str
    ts: float
```

---

## 7. Postgres schema

### `compression_artifacts`
```sql
region_id          TEXT PRIMARY KEY,
scope              TEXT NOT NULL,
kind               TEXT NOT NULL,
fuseki_graph_uri   TEXT NOT NULL,   -- always orion:compressions
summary_kind       TEXT NOT NULL,   -- llm | structural
salience           FLOAT,
trust_tier         TEXT,
compression_version TEXT,
generated_at       TIMESTAMPTZ NOT NULL,
stale              BOOLEAN NOT NULL DEFAULT false
```

### `compression_jobs`
```sql
job_id             TEXT PRIMARY KEY,
region_id          TEXT NOT NULL,
status             TEXT NOT NULL,  -- running | ok | failed
llm_tokens_used    INT,
started_at         TIMESTAMPTZ,
finished_at        TIMESTAMPTZ,
error              TEXT
```

### `stale_queue`
```sql
id                 SERIAL PRIMARY KEY,
region_id          TEXT,
scope              TEXT,
reason             TEXT,
queued_at          TIMESTAMPTZ NOT NULL,
priority           INT DEFAULT 0
```

---

## 8. Bus catalog additions

### New channels in `orion/bus/channels.yaml`

```yaml
- name: "orion:graph:compression:stale"
  kind: "event"
  schema_id: "CompressionStalenessMarkV1"
  message_kind: "graph.compression.stale.v1"
  producer_services: ["orion-rdf-writer", "orion-graph-compression"]
  consumer_services: ["orion-graph-compression"]
  stability: "experimental"
  since: "2026-06-06"

- name: "orion:graph:compression:events"
  kind: "event"
  schema_id: "GraphCompressionRegionMaterializedV1"
  message_kind: "graph.compression.region.materialized.v1"
  producer_services: ["orion-graph-compression"]
  consumer_services: ["*"]
  stability: "experimental"
  since: "2026-06-06"
```

### New entries in `orion/schemas/registry.py`

```python
from orion.schemas.graph_compression import (
    CompressionRegionV1,
    CompressionStalenessMarkV1,
    GraphCompressionRegionMaterializedV1,
)

# In _REGISTRY dict:
"CompressionRegionV1": CompressionRegionV1,
"CompressionStalenessMarkV1": CompressionStalenessMarkV1,
"GraphCompressionRegionMaterializedV1": GraphCompressionRegionMaterializedV1,
```

### Environment variables (channels + schemas)

Register channel names and schema/message-kind IDs in env so services and compose can reference them without hardcoding. **Update both `.env_example` and `.env` together** when adding new bus contracts.

| File | Variables added |
|------|-----------------|
| `.env_example` / `.env` (repo root) | `CHANNEL_GRAPH_COMPRESSION_*`, `SCHEMA_COMPRESSION_*`, `MESSAGE_KIND_GRAPH_COMPRESSION_*` |
| `services/orion-graph-compression/.env_example` / `.env` | Full service config including all channels, schemas, Fuseki, Postgres, worker tuning |
| `services/orion-rdf-writer/.env_example` / `.env` | `CHANNEL_GRAPH_COMPRESSION_STALE`, `SCHEMA_COMPRESSION_STALENESS_MARK_V1`, `MESSAGE_KIND_GRAPH_COMPRESSION_STALE` (publish on RDF confirm) |
| `services/orion-recall/.env_example` / `.env` | `RECALL_COMPRESSION_*` backend vars + passive `CHANNEL_GRAPH_COMPRESSION_EVENTS` / schema refs for future subscribers |

Root catalog block:

```bash
# Graph Compression (orion-graph-compression; orion/bus/channels.yaml)
CHANNEL_GRAPH_COMPRESSION_STALE=orion:graph:compression:stale
CHANNEL_GRAPH_COMPRESSION_EVENTS=orion:graph:compression:events
SCHEMA_COMPRESSION_REGION_V1=CompressionRegionV1
SCHEMA_COMPRESSION_STALENESS_MARK_V1=CompressionStalenessMarkV1
SCHEMA_GRAPH_COMPRESSION_REGION_MATERIALIZED_V1=GraphCompressionRegionMaterializedV1
MESSAGE_KIND_GRAPH_COMPRESSION_STALE=graph.compression.stale.v1
MESSAGE_KIND_GRAPH_COMPRESSION_REGION_MATERIALIZED=graph.compression.region.materialized.v1
```

---

## 9. Staleness and refresh model

**Stale marking (event-driven):**
- `stale_listener.py` subscribes to `orion:graph:compression:stale`
- Also subscribes to `orion:rdf:enqueue` (existing channel) — any RDF write marks affected scope stale in Postgres without a separate upstream change

**Batch refresh (scheduled):**
- `CompressionWorker` polls every `COMPRESSION_POLL_INTERVAL_SEC` (default 300s)
- Drains stale queue in priority order, bounded by `COMPRESSION_BATCH_SIZE` per tick
- Also refreshes any artifact older than `COMPRESSION_MAX_AGE_SEC` regardless of stale flag
- LLM token spend per tick capped by `COMPRESSION_LLM_BUDGET_PER_TICK`

**Idempotency:** `stable_compression_region_id()` hashes scope + kind + source node set → same region produces same `region_id`, preventing duplicates on retry.

---

## 10. Recall changes: `orion-recall`

### New module: `app/storage/graph_compression_adapter.py`

```python
def fetch_graph_compression_fragments(
    *,
    query_text: str,
    mode: Literal["global", "local", "unified"],
    max_global: int = 5,
    max_local: int = 5,
    scopes: list[str],
) -> list[dict]:
    """
    1. Query Postgres artifact index (fast lookup by scope/kind/salience).
    2. Keyword + salience rank artifacts against query_text.
    3. global mode: return top community/hotspot/contradiction summaries.
    4. local mode: return exemplar snippet URIs resolved from Fuseki.
    5. unified mode: fan out both, merge, dedupe.
    Returns fragments with:
      source="graph_compression"
      source_ref=region_id
      tags=["scope:...", "kind:...", "trust:...", "summary_kind:..."]
    """
```

### `_query_backends` extension in `worker.py`

```python
if _compression_enabled(profile) and bool(settings.RECALL_COMPRESSION_PG_DSN):
    compressions = fetch_graph_compression_fragments(
        query_text=fragment,
        mode=profile.get("compression_mode", "unified"),
        max_global=int(profile.get("compression_global_top_k", 5)),
        max_local=int(profile.get("compression_local_top_k", 5)),
        scopes=list(profile.get("compression_scopes") or ["episodic", "substrate", "self_study"]),
    )
    backend_counts["graph_compression"] = len(compressions)
    candidates.extend(compressions)
```

### New recall profiles

**`orion/recall/profiles/graph.compressions.global.v1.yaml`**
```yaml
profile: graph.compressions.global.v1
enable_rdf: false
enable_sql_timeline: false
enable_graph_compression: true
compression_mode: global
compression_global_top_k: 8
compression_local_top_k: 3
compression_scopes: [episodic, substrate, self_study]
render_budget_tokens: 512
max_total_items: 11
```

**`orion/recall/profiles/graph.compressions.local.v1.yaml`**
```yaml
profile: graph.compressions.local.v1
enable_rdf: true
rdf_top_k: 6
enable_graph_compression: true
compression_mode: local
compression_global_top_k: 2
compression_local_top_k: 8
compression_scopes: [episodic, substrate, self_study]
render_budget_tokens: 480
max_total_items: 16
```

**`orion/recall/profiles/graph.compressions.v1.yaml`** (default unified)
```yaml
profile: graph.compressions.v1
enable_rdf: true
rdf_top_k: 6
enable_graph_compression: true
compression_mode: unified
compression_global_top_k: 5
compression_local_top_k: 5
compression_scopes: [episodic, substrate, self_study]
render_budget_tokens: 640
max_total_items: 16
```

### `recall/.env_example` additions
```bash
# --- Graph Compression backend ---
RECALL_COMPRESSION_ENABLED=true
RECALL_COMPRESSION_PG_DSN=postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney
RECALL_COMPRESSION_RDF_QUERY_URL=http://orion-athena-fuseki:3030/orion/query
RECALL_COMPRESSION_RDF_USER=admin
RECALL_COMPRESSION_RDF_PASS=orion
```

---

## 11. Grammar substrate hook (v1-wired, v2-consumed)

### Contradiction → `MutationPressureEvidenceV1`

After writing a `contradiction`-kind artifact to Fuseki, `writer.py` publishes:

```python
pressure = MutationPressureEvidenceV1(
    pressure_category="unsupported_memory_claim",
    source_service="orion-graph-compression",
    evidence_ref=region.region_id,
    detail=f"Contradiction region: {region.summary[:120]}",
)
await bus.publish(
    settings.channel_substrate_mutation_pressure,
    BaseEnvelope(kind="substrate.mutation.pressure.v1", ...),
)
```

This feeds the substrate review queue with no changes to grammar substrate services.

### Passive materialization event

All artifacts emit `GraphCompressionRegionMaterializedV1` on `orion:graph:compression:events` after write. Grammar substrate (field digester / attention frame) can subscribe in v2 without any change to this service.

---

## 12. `normalize_graph_name` addition

In `services/orion-rdf-writer/app/rdf_store.py`, add to the `mapping` dict:

```python
"orion:compressions": "http://conjourney.net/graph/orion/compressions",
```

---

## 13. Error handling and degraded-mode posture

| Failure | Behavior |
|---------|----------|
| Fuseki query fails | Federator returns `[]`; region stays stale; worker continues to next region |
| Fuseki UPDATE fails | Job logged as `failed`; dead-letter entry after N consecutive failures; retry next tick |
| LLM Gateway timeout | Fall back to **structural summary** (entity labels + counts, no LLM); artifact written with `summary_kind: structural` |
| LLM token budget reached | Worker halts batch; remaining stale items held for next tick |
| Postgres unavailable | Worker skips tick; backs off exponentially |
| `ENABLE_COMPRESSION_RUNTIME=false` | Worker disabled; `/health` returns `ok`; recall adapter returns `[]` |
| Cold start (no artifacts yet) | Recall adapter returns `[]` gracefully; existing RDF/SQL backends unaffected |
| Compression adapter unavailable | Recall logs warning; other backends (RDF/SQL) still run; no recall failure propagation |

`/health` includes: `compression_runtime_enabled`, `stale_queue_depth`, `last_job_at`, `last_job_status`, `fuseki_reachable`, `artifact_count`.

---

## 14. Testing strategy

### `services/orion-graph-compression/tests/`

| Test file | Coverage |
|-----------|----------|
| `test_leiden_clustering.py` | Stable clusters on small NetworkX graphs; empty graph → `[]` without error |
| `test_region_builder.py` | `CompressionRegionV1` from cluster; trust tier inherits lowest source tier |
| `test_writer_sparql.py` | Valid SPARQL UPDATE generated for `orion:compressions` without live Fuseki |
| `test_store_staleness.py` | Enqueue/dequeue idempotency; budget gate halts mid-batch |
| `test_worker_degraded.py` | All-empty federators → no crash, correct log event |
| `test_federator_episodic.py` | SPARQL query generation for all episodic named graphs |
| `test_compression_schema.py` | `CompressionRegionV1` validates; rejects invalid trust tiers |
| `test_grammar_hook.py` | Contradiction-kind write emits `MutationPressureEvidenceV1` with correct fields |

### `services/orion-recall/tests/`

| Test file | Coverage |
|-----------|----------|
| `test_graph_compression_adapter.py` | Correct fragment shape; empty Postgres → `[]`; no exception on Fuseki miss |
| `test_recall_profiles_compression.py` | All three profiles parse; correct field defaults |
| `test_query_backends_compression.py` | Compression backend integrates without suppressing RDF/SQL backends |

### `orion/schemas/tests/`

| Test | Coverage |
|------|----------|
| `test_graph_compression_schema.py` | Round-trip clean; `derived_from` / `exemplar_ids` non-empty required |

### Integration smoke (gated by `TEST_FUSEKI_URL`)

End-to-end: federate → Leiden → structural summary → Fuseki write → recall adapter reads back. Validates full path without LLM cost.

---

## 15. Non-goals / v2 hooks

| Topic | v1 | v2 |
|-------|----|----|
| SQL timeline as compression input | ✗ | Add `SqlTimelineFederator` + new profile fields |
| Hub UI compression browser | `/health` + `/regions` only | Full Forge-tab-style browser |
| Grammar substrate reducer | Bus event emitted, no reducer | `orion-field-digester` wires reducer for `graph.compression.region.materialized.v1` |
| `FieldAttentionFrameV1` salience seeding | ✗ | Compressions feed attention frame scoring |
| `ProposalFrameV1` from contradiction regions | ✗ | Contradiction compressions become proposal candidates |
| Compression version diff / developmental trace | `generated_at` + `compression_version` stored | Diff view across versions |

---

## 16. Port assignment

`8270` — `GRAPH_COMPRESSION_PORT` (follows `orion-recall` on `8260`).
