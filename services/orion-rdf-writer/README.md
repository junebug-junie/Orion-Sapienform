# Orion RDF Writer

The **RDF Writer** service constructs the Knowledge Graph by converting incoming events and structured requests into RDF triples. It persists these triples through a small **`RdfStoreClient`** abstraction (GraphDB by default; Fuseki or generic SPARQL graph-store/update endpoints as alternates).

**Chat is an acceptance canary only:** `chat.history` is a convenient smoke path, but the writer is general-purpose across all subscribed kinds—do not treat chat-specific behavior as the whole contract.

### Backends (`RDF_STORE_BACKEND`)

| Value | Behavior |
| :--- | :--- |
| `graphdb` | GraphDB HTTP repository **statements** API (`text/plain` body, optional `context=<{graph}>`). Requires `GRAPHDB_URL`. |
| `fuseki` | Jena **Graph Store** HTTP POST to `{base}/{dataset}/data` with `graph=` query param; defaults target `http://orion-athena-fuseki:3030` / dataset `orion`. |
| `generic` | Same graph-store POST pattern against explicit URLs, or SPARQL UPDATE fallback when only `RDF_STORE_UPDATE_URL` is set. |
| `rdf4j` | URL-gated alias of `generic` (requires explicit graph-store and/or update URL). |

### Async queue and backpressure

When `RDF_WRITE_ASYNC_ENABLED=true` (default), writes are queued and drained by a worker pool with a global in-flight semaphore, retries with exponential backoff, and optional **NDJSON dead-letter** logging (`RDF_WRITE_DEAD_LETTER_*`). If the queue is full, the service **does not** log `rdf_write_committed`; it dead-letters, logs backpressure, optionally publishes `CHANNEL_RDF_ERROR`, and drops the hot-path work for that envelope (HTTP ingest maps queue saturation to **503**).

Tune `RDF_WRITE_*` and `RDF_STORE_TIMEOUT_SEC` before scaling bus traffic; see `services/rdf-store/README.md` for operator layout.

## Contracts

### Consumed Channels
| Channel | Env Var | Kind(s) | Description |
| :--- | :--- | :--- | :--- |
| `orion:rdf:enqueue` | `CHANNEL_RDF_ENQUEUE` | `rdf.write.request` | Direct write requests. |
| `orion:collapse:intake` | `CHANNEL_EVENTS_COLLAPSE` | `collapse.mirror.entry` | Collapse entries (raw). |
| `orion:tags:enriched` | `CHANNEL_EVENTS_TAGGED` | `tags.enriched`, `telemetry.meta_tags` | Enriched metadata tags. |
| `orion:core:events` | `CHANNEL_CORE_EVENTS` | `orion.event` | Legacy events targeted for RDF. |
| `orion:rdf:worker` | `CHANNEL_WORKER_RDF` | `cortex.worker.rdf_build` | Worker tasks from Cortex. |
| `orion:chat:history:turn` | `CHANNEL_CHAT_HISTORY_TURN` | `chat.history` | Chat turn history (prompt + response). |
| `orion:chat:history:log` | `CHANNEL_CHAT_HISTORY_LOG` | `chat.history.message.v1` | Chat message history (per-message). |

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
| `CHANNEL_CHAT_HISTORY_TURN` | `orion:chat:history:turn` | Chat turn history intake. |
| `CHANNEL_CHAT_HISTORY_LOG` | `orion:chat:history:log` | Chat message history intake. |
| `GRAPHDB_URL` | Optional when backend ≠ `graphdb` | GraphDB base URL. |
| `RDF_STORE_*`, `RDF_WRITE_*` | See `.env_example` | Backend selection, HTTP pool sizing, async queue, retries, dead-letter. |

## Running & Testing

### Run via Docker
```bash
docker-compose up -d orion-rdf-writer
```

### Smoke Test
Check connection to GraphDB in logs.
```bash
docker-compose logs -f orion-rdf-writer | grep "Connected"
```

### Store-aware chat smoke (GraphDB or Fuseki)

Publishes a synthetic `chat.history` turn on the bus and polls SPARQL until readback succeeds (tolerates async writer latency).

```bash
PYTHONPATH=/path/to/Orion-Sapienform:/path/to/Orion-Sapienform/services/orion-rdf-writer \
  ./venv/bin/python scripts/smoke_chat_to_rdf_store.py
```

### SPARQL Smoke Query (last 10 chat turns by sessionId)
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
