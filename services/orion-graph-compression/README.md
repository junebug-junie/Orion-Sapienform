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
                 ┌───────┐
                 ▼       ▼
           Episodic  FalkorSubstrate
           Federator  Federator
                 │       │
                 └───────┘
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

| Scope | Source | Region Kind |
|-------|--------------|-------------|
| `episodic` | SPARQL: 8 episodic named graphs (chat, enrichment, collapse, autonomy/*). `orion:cognition`/`orion:metacog` are no longer written (2026-07-22: pure Postgres redundancy via orion-sql-writer, see rdf-writer's kill) -- still queried, will just read empty. `orion:chat:social` removed entirely 2026-07-23 (live-verified pure redundancy with Postgres `social_room_turns`, no other reader -- see federator section below). Additively unioned with Falkor (see below). | `community` |
| `substrate` | FalkorDB only (`orion_substrate` graph). SPARQL `SubstrateFederator` retired 2026-07-23 -- see below. | `hotspot` |

`self_study` was retired 2026-07-23 (previously read `orion:self`,
`orion:self/induced`, `orion:self/reflective`; region kind
`self_study_cluster`) -- live-verified zero triples in all three source
graphs, ever, no producer anywhere in the repo. Trivial kill, no Falkor
migration needed since there was nothing to migrate.

### FalkorDB federators

`app/federators/substrate_falkor.py` / `episodic_falkor.py` are Cypher-native
alternatives to the retired SPARQL federators, reading the same shared
FalkorDB instance substrate-runtime/orion-recall/orion-meta-tags already
write to (`orion_substrate`, `orion_recall`).

**`GRAPH_COMPRESSION_SUBSTRATE_FALKOR_ENABLED`** (`true` live as of
2026-07-23): the **sole source** for the `substrate` scope now -- there is no
SPARQL fallback left. Substrate-runtime has been Falkor-primary
(`SUBSTRATE_STORE_BACKEND=falkor`) since PR #1153; live-verified the SPARQL
`SubstrateFederator` was reading a frozen 126 stale triples with zero active
writers anywhere in the repo, so it was removed outright rather than kept as
a dead fallback. If this flag is off, the `substrate` scope simply produces
no clusters.

**`GRAPH_COMPRESSION_EPISODIC_FALKOR_ENABLED`** (dark by default, still
additive): results are **unioned** with the SPARQL `EpisodicFederator`'s
output in `worker.py::_process_scope`, not swapped -- `episodic_falkor.py`
covers only the `orion_recall` slice (ChatTurn/Tag/Entity); collapse has no
Falkor writer yet and still depends on the SPARQL federator. Not retired --
this scope's SPARQL side is genuinely still load-bearing for collapse
content Falkor doesn't cover, unlike `substrate`/`self_study` above.

`orion:chat:social` (social-turn content) was removed from the SPARQL
federator's graph list entirely, 2026-07-23 -- not "covered by Falkor
instead" so much as retired outright (live-verified pure redundancy with
Postgres `social_room_turns`: richer schema including actual prompt/response
text the Fuseki copy never had, no other reader anywhere). orion-meta-tags
does write social-turn tags/entities into the same `orion_recall` graph
`episodic_falkor.py` reads (unconditional for both `chat.history` and
`social.turn.stored.v1` since 2026-07-18), but live-verified thin as of this
writing (2 `social.turn.stored.v1` `ChatTurn` nodes, zero tag/entity edges,
vs 1,772 for `chat.history`) -- disclosed here rather than overclaimed as
equivalent coverage. Social-chat volume itself looks near-dormant (Postgres
`social_room_turns`' most recent row predates this cutover by ~20 days), so
this was judged low-risk to retire now rather than worth a dedicated
migration effort.

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
