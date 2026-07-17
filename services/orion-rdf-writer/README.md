# Orion RDF Writer

The **RDF Writer** service constructs the Knowledge Graph by converting incoming events and structured requests into RDF triples. It persists these triples through a small **`RdfStoreClient`** abstraction (GraphDB by default; Fuseki or generic SPARQL graph-store/update endpoints as alternates).

**Chat history is no longer written here (2026-07-17):** `chat.history` / `chat.history.message.v1` are Postgres-only via `orion-sql-writer` (`chat_message`, `chat_history_log`) — the Fuseki `orion:chat` copy covered only ~11-18% of real chat volume with almost none of the richer fields populated (live-checked). rdf-writer no longer subscribes to or handles either kind; do not use chat as a smoke path for this service (see "Store-aware chat smoke" note below).

### Backends (`RDF_STORE_BACKEND`)

| Value | Behavior |
| :--- | :--- |
| `graphdb` | GraphDB HTTP repository **statements** API (`text/plain` body, optional `context=<{graph}>`). Requires `GRAPHDB_URL`. |
| `fuseki` | Jena **Graph Store** HTTP POST to `{base}/{dataset}/data` with `graph=` query param; defaults target `http://orion-athena-fuseki:3030` / dataset `orion`. |
| `generic` | Same graph-store POST pattern against explicit URLs, or SPARQL UPDATE fallback when only `RDF_STORE_UPDATE_URL` is set. |
| `rdf4j` | URL-gated alias of `generic` (requires explicit graph-store and/or update URL). |

### Async queue and backpressure

When `RDF_WRITE_ASYNC_ENABLED=true` (default), writes are queued and drained by a worker pool with a global in-flight semaphore, retries with exponential backoff, and optional **NDJSON dead-letter** logging (`RDF_WRITE_DEAD_LETTER_*`). If the queue is full, the service **does not** log `rdf_write_committed`; it dead-letters, logs backpressure, optionally publishes `CHANNEL_RDF_ERROR`, and drops the hot-path work for that envelope (HTTP ingest maps queue saturation to **503**).

Tune `RDF_WRITE_*` and `RDF_STORE_TIMEOUT_SEC` before scaling bus traffic; for the Fuseki operator stack see `services/orion-rdf-store/README.md`.

## Contracts

### Consumed Channels
| Channel | Env Var | Kind(s) | Description |
| :--- | :--- | :--- | :--- |
| `orion:rdf:enqueue` | `CHANNEL_RDF_ENQUEUE` | `rdf.write.request` | Direct write requests. |
| `orion:collapse:intake` | `CHANNEL_EVENTS_COLLAPSE` | `collapse.mirror.entry` | Collapse entries (raw). |
| `orion:tags:enriched` | `CHANNEL_EVENTS_TAGGED` | `tags.enriched`, `telemetry.meta_tags` | Enriched metadata tags. |
| `orion:core:events` | `CHANNEL_CORE_EVENTS` | `orion.event` | Legacy events targeted for RDF. |
| `orion:rdf:worker` | `CHANNEL_WORKER_RDF` | `cortex.worker.rdf_build` | Worker tasks from Cortex. |

### Published Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion:rdf:confirm` | `CHANNEL_RDF_CONFIRM` | `rdf.write.confirm` | Write confirmation. |
| `orion:rdf:error` | `CHANNEL_RDF_ERROR` | `rdf.write.error` | Error reporting. |

### Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `CHANNEL_RDF_ENQUEUE` | `orion:rdf:enqueue` | Direct enqueue channel. |
| `CHANNEL_EVENTS_COLLAPSE` | `orion:collapse:intake` | Collapse event source. |
| `CHANNEL_EVENTS_TAGGED` | `orion:tags:enriched` | Tagged event source. |
| `GRAPHDB_URL` | Optional when backend ≠ `graphdb` | GraphDB base URL. |
| `RDF_STORE_*`, `RDF_WRITE_*` | See `.env_example` | Backend selection, HTTP pool sizing, async queue, retries, dead-letter. |

## Running & Testing

### Run via Docker
```bash
docker compose -f services/orion-rdf-writer/docker-compose.yml up -d rdf-writer
```

### Smoke Test
Check connection to GraphDB in logs.
```bash
docker compose -f services/orion-rdf-writer/docker-compose.yml logs -f rdf-writer | grep "Connected"
```

### Store-aware chat smoke (GraphDB or Fuseki) — STALE as of 2026-07-17

`scripts/smoke_chat_to_rdf_store.py` and `scripts/smoke_chat_to_rdf.py` publish a
synthetic `chat.history` turn and poll SPARQL for a resulting `orion:ChatTurn`.
Since rdf-writer no longer subscribes to `chat.history` / `chat.history.message.v1`
(see note above), both scripts will now always report `FAIL` — that is expected,
not a regression. Retiring or repointing them at a still-handled kind is a
recommended follow-up; do not use them to validate `orion-rdf-writer` behavior
until then.

### SPARQL Smoke Query (last 10 chat turns by sessionId) — reads historical data only

No longer populated by rdf-writer going forward; useful only for inspecting
whatever `orion:ChatTurn` data already exists in the store from before 2026-07-17.
```sparql
PREFIX orion: <http://conjourney.net/orion#>

SELECT ?turn ?prompt ?response ?timestamp
WHERE {
  ?turn a orion:ChatTurn ;
        orion:sessionId "session-id-here" ;
        orion:prompt ?prompt ;
        orion:response ?response .
  OPTIONAL { ?turn orion:timestamp ?timestamp }
}
ORDER BY DESC(?timestamp)
LIMIT 10
```
