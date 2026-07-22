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

`8271` (configurable via `PORT`; do not use `8270` — reserved by `orion-state-service`)

## HTTP Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Service status, stale queue depth, artifact count |
| `GET /regions?scope=episodic` | List cached compression artifacts |
| `GET /artifacts/{region_id}` | Fetch a specific region artifact |

## Compression Scopes

| Scope | Source Graphs | Region Kind |
|-------|--------------|-------------|
| `episodic` | 9 episodic named graphs (chat, enrichment, collapse, chat/social, autonomy/*). `orion:cognition`/`orion:metacog` are no longer written (2026-07-22: pure Postgres redundancy via orion-sql-writer, see rdf-writer's kill) -- still queried, will just read empty. | `community` |
| `substrate` | `orion:substrate` | `contradiction` |
| `self_study` | `orion:self`, `orion:self/induced`, `orion:self/reflective` | `self_study_cluster` |

### FalkorDB federators (additive, dark by default)

`app/federators/substrate_falkor.py` / `episodic_falkor.py` are Cypher-native
alternatives to the SPARQL federators above, reading the same shared
FalkorDB instance substrate-runtime/orion-recall/orion-meta-tags already
write to (`orion_substrate`, `orion_recall`). Gated behind
`GRAPH_COMPRESSION_SUBSTRATE_FALKOR_ENABLED`/`GRAPH_COMPRESSION_EPISODIC_FALKOR_ENABLED`
(both `false` by default). When on, results are **unioned** with the SPARQL
federator's output in `worker.py::_process_scope`, not swapped -- this can
only add clustering signal, never regress what's already there, while the
Falkor side is verified live.

Why this exists: substrate-runtime has been Falkor-primary
(`SUBSTRATE_STORE_BACKEND=falkor`) since PR #1153, so `SubstrateFederator`'s
SPARQL query has likely been reading stale/emptying data since that cutover
-- nothing currently re-populates `orion:substrate` in Fuseki. Separately,
`orion-rdf-writer`'s enrichment write (Entity/Mention/hasTag/hasEntity into
`orion:enrichment`) is the thing the `episodic` scope's live signal
increasingly depends on for chat-turn clustering, and orion-meta-tags'
Falkor writer (`orion_recall`) is the newer, live equivalent of that same
data. `episodic_falkor.py` covers only the `orion_recall` slice
(ChatTurn/Tag/Entity) -- collapse/chat-social have no Falkor writer yet and
still depend on the SPARQL federator.

Once verified live (community/cluster sanity check against real data,
matching the recall backfill's verification approach), the next step is
flipping these from additive-union to primary, then retiring the
now-redundant SPARQL side for the covered scopes -- not done in this patch.

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
| `FALKORDB_URI` | `redis://orion-athena-falkordb:6379` | Shared FalkorDB instance, read directly from env by `app/falkor_store.py` |
| `GRAPH_COMPRESSION_SUBSTRATE_FALKOR_ENABLED` | `false` | Additive Cypher-native substrate federator (see above) |
| `GRAPH_COMPRESSION_EPISODIC_FALKOR_ENABLED` | `false` | Additive Cypher-native episodic (recall) federator (see above) |

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
