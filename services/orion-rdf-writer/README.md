# Orion RDF Writer

The **RDF Writer** service constructs the Knowledge Graph by converting incoming events and structured requests into RDF triples. It persists these triples into GraphDB (or equivalent SPARQL endpoint).

## Contracts

### Consumed Channels
| Channel | Env Var | Kind(s) | Description |
| :--- | :--- | :--- | :--- |
| `orion:rdf-collapse:enqueue` | `CHANNEL_RDF_ENQUEUE` | `rdf.write.request` | Direct write requests. |
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
| `CHANNEL_RDF_ENQUEUE` | `orion:rdf-collapse:enqueue` | Direct enqueue channel. |
| `CHANNEL_EVENTS_COLLAPSE` | `orion:collapse:intake` | Collapse event source. |
| `CHANNEL_EVENTS_TAGGED` | `orion:tags:enriched` | Tagged event source. |
| `CHANNEL_CHAT_HISTORY_TURN` | `orion:chat:history:turn` | Chat turn history intake. |
| `CHANNEL_CHAT_HISTORY_LOG` | `orion:chat:history:log` | Chat message history intake. |
| `GRAPHDB_URL` | (Required) | URL for the GraphDB endpoint. |

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
