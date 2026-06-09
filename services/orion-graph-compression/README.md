# orion-graph-compression

Offline service that federates Fuseki graph data, clusters it with Leiden community detection, summarizes regions via the LLM Gateway bus RPC, and writes cached `CompressionRegionV1` artifacts to `orion:compressions` for downstream recall.

## Architecture

```
orion:rdf:enqueue (bus)
        │
        ▼
stale_listener.py   ──► stale_queue (Postgres)
                                │
                                ▼
                    worker.py (_tick poll loop)
                         │
                 ┌───────┼───────┐
                 ▼       ▼       ▼
           Episodic  Substrate  SelfStudy
           Federator Federator  Federator
                 │       │       │
                 └───────┴───────┘
                         │
                 clustering/leiden.py
                         │
                 region_builder.py
                         │
                  summarizer.py (LLM Gateway RPC, structural fallback)
                         │
                    writer.py
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
    Fuseki orion:compressions   Postgres compression_artifacts
              │
    orion:graph:compression:events (bus)
    orion:substrate:mutation:pressure (bus, contradiction kind only)
```

## Service Port

`8270` (configurable via `GRAPH_COMPRESSION_PORT`)

## HTTP Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Service status, stale queue depth, artifact count |
| `GET /regions?scope=episodic` | List cached compression artifacts |
| `GET /artifacts/{region_id}` | Fetch a specific region artifact |

## Compression Scopes

| Scope | Source Graphs | Region Kind |
|-------|--------------|-------------|
| `episodic` | 9 episodic named graphs (chat, enrichment, collapse, cognition, metacog, chat/social, autonomy/*) | `community` |
| `substrate` | `orion:substrate` | `contradiction` |
| `self_study` | `orion:self`, `orion:self/induced`, `orion:self/reflective` | `self_study_cluster` |

## Bus Events

| Channel | Schema | When |
|---------|--------|------|
| `orion:graph:compression:events` | `GraphCompressionRegionMaterializedV1` | After each region written to Fuseki |
| `orion:substrate:mutation:pressure` | `MutationPressureEvidenceV1` | Only for `contradiction` kind regions |

## Key Env Vars

| Var | Default | Description |
|-----|---------|-------------|
| `ENABLE_COMPRESSION_RUNTIME` | `true` | Gate to disable worker entirely |
| `COMPRESSION_POLL_INTERVAL_SEC` | `300` | Worker poll interval |
| `COMPRESSION_BATCH_SIZE` | `10` | Stale queue drain batch size |
| `COMPRESSION_LLM_BUDGET_PER_TICK` | `5000` | Token budget per tick (future LLM path) |
| `POSTGRES_URI` | — | Postgres for artifact index + stale queue |
| `RDF_STORE_QUERY_URL` | — | Fuseki SPARQL query endpoint |
| `RDF_STORE_UPDATE_URL` | — | Fuseki SPARQL update endpoint |
| `COMPRESSION_POLICY_PATH` | `/app/config/compression_policy.v1.yaml` | Policy YAML path |

See `.env_example` for full list.

## Running Locally (Docker Compose)

```bash
cd services/orion-graph-compression
docker compose up --build
```

The Dockerfile builds from the repo root context so the shared `orion/` package is included.

## Tests

```bash
# From repo root, in the project venv:
PYTHONPATH=services/orion-graph-compression:. pytest services/orion-graph-compression/tests/ -v
```

25 tests covering: schema round-trips, Postgres store operations, federator SPARQL generation, Leiden clustering, region builder stable IDs, Fuseki SPARQL UPDATE generation, grammar substrate hook emission, and worker degraded-mode behaviour.

## Postgres Tables

- `stale_queue` — work queue; rows drain per tick and are deleted after processing
- `compression_artifacts` — idempotent artifact index (region_id PK, ON CONFLICT upsert)
- `compression_jobs` — optional job audit log
