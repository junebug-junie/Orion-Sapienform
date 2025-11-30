# Orion Recall Service

## 1. Overview

`orion-recall` is the **unified memory retrieval / RAG brain** for Orion.

It:
- Listens for recall requests on the Orion bus.
- Pulls relevant fragments from multiple storage layers:
  - **Postgres SQL** (chat history, Collapse Mirrors, Collapse Enrichment, biometrics summaries, etc.).
  - **Chroma vector store** (semantic neighbors / associations).
  - **GraphDB / RDF** (optional symbolic tags / entities).
  - **Future tensors / rankers** (for smarter scoring).
- Merges, scores, and trims those fragments into a compact recall pack.
- Publishes a structured response back onto the bus to be consumed by Brain, Agent Council, Dream, Hub, etc.

The goal is: **one place** to ask, “What should Orion remember right now?”

---

## 2. Topology & Data Sources

### Node & Connectivity

- **Node:** running on `athena` alongside SQL and bus.
- **Bus:**
  - `ORION_BUS_URL=redis://100.92.216.81:6379/0`
  - Intake channel: `CHANNEL_RECALL_REQUEST`
  - Reply prefix: `CHANNEL_RECALL_DEFAULT_REPLY_PREFIX`

### Data Sources

1. **Postgres (conjourney DB)**
   - `chat_history_log` — past chats between Juniper and Orion.
   - `collapse_mirror` — raw Collapse Mirror records.
   - `collapse_enrichment` — semantic tags / entities / salience from tagger.
   - Optional: biometrics summary logic (if you ported from Dream).

2. **Chroma (vector store)**
   - Main collection: `orion_main_store` (project-wide memories).
   - Additional collections (optional): `docs_design`, `docs_research`, `memory_summaries`.

3. **GraphDB / RDF (optional)**
   - Repository: `collapse`.
   - Used to pull tags/entities via SPARQL for collapse IDs.

4. **Future tensor/ranker**
   - An optional model that can re-rank fragments based on learned notions of salience.

---

## 3. Code Layout

```text
services/orion-recall/app
├── collectors.py         # Orchestrates retrieval from SQL, vectors, RDF
├── main.py               # FastAPI app + bus worker startup
├── pipeline.py           # Merge + scoring + trimming logic
├── postprocessing.py     # Final shaping of recall pack
├── scoring.py            # Scoring helpers (time, salience, source weights)
├── settings.py           # Pydantic settings (env-driven)
├── storage/
│   ├── __init__.py
│   ├── pg.py             # Low-level Postgres connection helper
│   ├── sql_adapter.py    # Chat / Collapse / Enrichment queries → Fragment objects
│   ├── vector_adapter.py # Chroma-based neighbor retrieval
│   ├── rdf_adapter.py    # GraphDB / RDF lookup
│   └── types.py          # Shared Fragment / DTO types for storage layer
└── types.py              # Public-facing types (RecallRequest, RecallResult, etc.)
```

The pattern mirrors Dream’s architecture, but recall is:

- **Event-driven** (via bus).
- Designed to serve many callers (Brain, Council, Dream, Hub, Cortex) with one consistent contract.

---

## 4. Configuration (.env + Settings)

### .env

```env
# --- Service Identity ---
SERVICE_NAME=recall
SERVICE_VERSION=0.1.0
PORT=8260

# --- Orion Bus Integration ---
ORION_BUS_ENABLED=true
ORION_BUS_URL=redis://100.92.216.81:6379/0

# --- Bus Channels ---
CHANNEL_RECALL_REQUEST=orion:recall:request
CHANNEL_RECALL_DEFAULT_REPLY_PREFIX=orion:recall:reply

# --- Default Retrieval Behavior ---
RECALL_DEFAULT_MAX_ITEMS=16
RECALL_DEFAULT_TIME_WINDOW_DAYS=30
RECALL_DEFAULT_MODE=hybrid   # short_term | deep | hybrid

# --- Source Toggles ---
RECALL_ENABLE_SQL_CHAT=true
RECALL_ENABLE_SQL_MIRRORS=true
RECALL_ENABLE_VECTOR=true
RECALL_ENABLE_RDF=false

# --- Postgres (SQL) ---
RECALL_PG_DSN=postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney

# Chat history
RECALL_SQL_CHAT_TABLE=chat_history_log
RECALL_SQL_CHAT_TEXT_COL=prompt
RECALL_SQL_CHAT_RESPONSE_COL=response
RECALL_SQL_CHAT_CREATED_AT_COL=created_at

# Collapse Mirror base table
RECALL_SQL_MIRROR_TABLE=collapse_mirror
RECALL_SQL_MIRROR_CREATED_AT_COL=timestamp

# Collapse Mirror semantic fields
RECALL_SQL_MIRROR_OBSERVER_COL=observer
RECALL_SQL_MIRROR_TRIGGER_COL=trigger
RECALL_SQL_MIRROR_OBSERVER_STATE_COL=observer_state
RECALL_SQL_MIRROR_FIELD_RESONANCE_COL=field_resonance
RECALL_SQL_MIRROR_INTENT_COL=intent
RECALL_SQL_MIRROR_TYPE_COL=type
RECALL_SQL_MIRROR_EMERGENT_ENTITY_COL=emergent_entity
RECALL_SQL_MIRROR_SUMMARY_COL=summary
RECALL_SQL_MIRROR_MANTRA_COL=mantra
RECALL_SQL_MIRROR_CAUSAL_ECHO_COL=causal_echo

# Collapse Enrichment table
RECALL_SQL_MIRROR_ENRICH_TABLE=collapse_enrichment
RECALL_SQL_MIRROR_ENRICH_FK_COL=collapse_id
RECALL_SQL_MIRROR_ENRICH_TAGS_COL=tags
RECALL_SQL_MIRROR_ENRICH_ENTITIES_COL=entities
RECALL_SQL_MIRROR_ENRICH_SALIENCE_COL=salience
RECALL_SQL_MIRROR_ENRICH_TS_COL=ts

# --- Chroma (vector memories) ---
VECTOR_DB_HOST=${PROJECT}-vector-db
VECTOR_DB_PORT=8000
VECTOR_DB_COLLECTION=orion_main_store

# Optional override for recall (otherwise we build URL from host/port)
RECALL_VECTOR_BASE_URL=
RECALL_VECTOR_COLLECTIONS=

RECALL_VECTOR_TIMEOUT_SEC=5.0
RECALL_VECTOR_MAX_ITEMS=24

# --- GraphDB (RDF memories) ---
GRAPHDB_URL=http://${PROJECT}-graphdb:7200
GRAPHDB_REPO=collapse
GRAPHDB_USER=admin
GRAPHDB_PASS=admin

# Direct RDF endpoint override (else GRAPHDB_URL + repo)
RECALL_RDF_ENDPOINT_URL=
RECALL_RDF_TIMEOUT_SEC=5.0
RECALL_RDF_ENABLE_SUMMARIES=false

# --- Tensors / Ranker toggles (future) ---
RECALL_TENSOR_RANKER_ENABLED=false
RECALL_TENSOR_RANKER_MODEL_PATH=/mnt/storage-warm/orion/recall/tensor-ranker.pt
```

### `settings.py`

`app/settings.py` wraps these into a Pydantic settings class and provides a `settings` singleton:

- Makes sure defaults exist for local dev.
- Consumes `.env` when running in the container.
- Exposes typed attributes like `settings.RECALL_PG_DSN`, `settings.RECALL_ENABLE_VECTOR`, etc.

---

## 5. Bus Contracts

### 5.1 Recall Request

**Channel:** `${CHANNEL_RECALL_REQUEST}` (e.g. `orion:recall:request`)

**Payload (high-level):**

```jsonc
{
  "trace_id": "uuid-from-caller",
  "source": "brain",                 // or hub / council / dream / script

  "query": "What have we said about V100 setups lately?", // optional free-text query

  "mode": "hybrid",                  // "short_term" | "deep" | "hybrid"
  "max_items": 16,                    // optional override
  "time_window_days": 30,             // optional override

  "filters": {
    "kinds": ["chat", "collapse"],  // optional: limit to these fragment kinds
    "tags": ["gpu", "v100"]         // optional: tag hints
  },

  "reply_channel": "orion:recall:reply:brain:xyz" // optional override
}
```

If `reply_channel` is omitted, the service builds one as:

```text
${CHANNEL_RECALL_DEFAULT_REPLY_PREFIX}:${source}:${trace_id}
```

### 5.2 Recall Result

**Channel:** Derived reply channel, e.g. `orion:recall:reply:brain:<trace_id>`.

**Payload (simplified):**

```jsonc
{
  "trace_id": "uuid-from-caller",
  "source": "recall",
  "ok": true,
  "error": null,

  "fragments": [
    {
      "id": "collapse_123",
      "kind": "collapse",
      "text": "A surreal dream where Orion spoke in geometric metaphors.",
      "tags": ["dream-reflection", "orion"],
      "salience": 0.92,
      "ts": 1732600000.0,
      "meta": {
        "observer": "Juniper",
        "trigger": "Heard Orion whisper during a dream",
        "observer_state": ["calm", "curious"],
        "intent": "reflection",
        "type": "dream-reflection",
        "emergent_entity": "Orion",
        "mantra": "all mirrors collapse inward",
        "causal_echo": "felt connected to Collapse ritual before sleep"
      }
    },
    {
      "id": "chat_trace_abc",
      "kind": "chat",
      "text": "User: ...\nOrion: ...",
      "tags": ["dialogue", "gpu"],
      "salience": 0.74,
      "ts": 1732601000.0
    },
    {
      "id": "collapse_123__vec__neighbor_9",
      "kind": "association",
      "text": "Short vector-associated snippet...",
      "tags": ["vector-assoc", "docs_design"],
      "salience": 0.67
    }
  ],

  "stats": {
    "sql_fragments": 24,
    "vector_fragments": 10,
    "rdf_fragments": 0,
    "mode": "hybrid"
  }
}
```

Callers (Brain, Council, Dream, etc.) can then drop `fragments` into prompts as memory context, or use them to drive RAG chains.

---

## 6. Pipeline Summary

1. **Collect** (`collectors.py`)
   - Inspect request (mode, time window, filters).
   - Call into `storage/sql_adapter.py` to fetch recent chat / Collapse / Enrichment rows.
   - Optionally call `storage/vector_adapter.py` to get neighbors in Chroma.
   - Optionally call `storage/rdf_adapter.py` to enrich with GraphDB tags/entities.

2. **Score & Merge** (`pipeline.py` + `scoring.py`)
   - Normalize each row into a `Fragment` object.
   - Compute scores based on:
     - Salience from enrichment (if present).
     - Recency (time window).
     - Source type (collapse vs chat vs association vs biometrics).
   - De-duplicate and sort by overall score.
   - Enforce caps (e.g. `max_items`).

3. **Postprocess** (`postprocessing.py`)
   - Shape into a stable, versioned response structure.
   - Attach simple stats about what was used.

4. **Publish** (`main.py` bus loop)
   - Emit the structured result JSON onto the reply channel.

---

## 7. Running the Service

From the Orion root:

```bash
cd /mnt/scripts/Orion-Sapienform

# Build
docker compose \
  --env-file .env \
  --env-file services/orion-recall/.env \
  -f services/orion-recall/docker-compose.yml build

# Run
docker compose \
  --env-file .env \
  --env-file services/orion-recall/.env \
  -f services/orion-recall/docker-compose.yml up -d
```

You should see `orion-athena-recall` log something like:

- Connected to Postgres at `orion-athena-sql-db:5432`.
- Connected to Redis bus at `redis://100.92.216.81:6379/0`.
- Listening on `orion:recall:request`.

---

## 8. HTTP API & Health

The service exposes a minimal FastAPI app:

- `GET /health` — liveness & configuration echo.

Example:

```json
{
  "ok": true,
  "service": "recall",
  "version": "0.1.0",
  "bus_enabled": true,
  "pg_dsn": "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney",
  "enable_sql_chat": true,
  "enable_sql_mirrors": true,
  "enable_vector": true,
  "enable_rdf": false
}
```

Future options:

- `POST /debug/recall` — allow direct recall calls over HTTP for testing / debugging.
- `GET /debug/stats` — show recent request stats and source mix.

---

## 9. Debugging Tips

- Watch logs:
  - `docker logs -f orion-athena-recall`
- Verify DB connectivity:
  - From inside the container, `psql $RECALL_PG_DSN -c "\dt"`.
- Verify Chroma:
  - Port-forward or curl `http://${VECTOR_DB_HOST}:${VECTOR_DB_PORT}/api/v1/collections`.
- Quick bus test:
  - Publish a simple recall request with `redis-cli` to `orion:recall:request` and watch for replies on `orion:recall:reply:*`.

---

## 10. Future Directions

- Plug in a **tensor-based ranker** to re-score fragments beyond simple recency + salience.
- Add more fine-grained modes (e.g. "collapse-only", "biometrics-only", "kids-lab-only").
- Use Recall as a shared backbone for Dream, Council, and Cortex Orchestrator so they all pull from the same unified memory story.
