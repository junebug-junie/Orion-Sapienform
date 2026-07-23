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
| `episodic` | SPARQL: 7 episodic named graphs (chat, enrichment, autonomy/*). `orion:cognition`/`orion:metacog` are no longer written (2026-07-22: pure Postgres redundancy via orion-sql-writer, see rdf-writer's kill) -- still queried, will just read empty (frozen historical content only). `orion:chat:social` and `orion:collapse` both removed entirely 2026-07-23 -- see federator section below. Additively unioned with Falkor (see below). | `community` |
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
covers the `orion_recall` slice (ChatTurn/Tag/Entity, plus CollapseEvent via
the collapse-triage Falkor writer, `RECALL_FALKOR_COLLAPSE_TRIAGE_ENABLED`
in orion-meta-tags). Cognition/metacog have no Falkor equivalent at all
(dead writers, frozen history) and remain genuinely SPARQL-only -- not
retired, that scope's SPARQL side is still load-bearing for that content,
unlike `substrate`/`self_study`/`chat:social`/`collapse` above.

`orion:chat:social` (social-turn content) and `orion:collapse` (raw
collapse-mirror-entry content) were both removed from the SPARQL
federator's graph list entirely, 2026-07-23 -- not "covered by Falkor
instead" so much as retired outright.

`orion:chat:social`: live-verified pure redundancy with Postgres
`social_room_turns` (richer schema including actual prompt/response text
the Fuseki copy never had, no other reader anywhere). orion-meta-tags
does write social-turn tags/entities into the same `orion_recall` graph
`episodic_falkor.py` reads (unconditional for both `chat.history` and
`social.turn.stored.v1` since 2026-07-18), but live-verified thin as of this
writing (2 `social.turn.stored.v1` `ChatTurn` nodes, zero tag/entity edges,
vs 1,772 for `chat.history`) -- disclosed here rather than overclaimed as
equivalent coverage. Social-chat volume itself looks near-dormant (Postgres
`social_room_turns`' most recent row predates this cutover by ~20 days), so
this was judged low-risk to retire now rather than worth a dedicated
migration effort.

`orion:collapse`: live SPARQL `COUNT` confirmed exactly **0** triples, ever
-- not low, zero, and structurally guaranteed to stay that way. `orion-rdf-
writer`'s `collapse.mirror.entry` handler (`_build_raw_collapse_graph`) has
its own `observer==Juniper`/dense gate, but that gate is unreachable: rdf-
writer only subscribes to `orion:collapse:intake`, which carries
`kind="collapse.mirror.intake"` from `orion-cortex-exec`. The only real
producer of `kind="collapse.mirror.entry"` is `orion-collapse-mirror`,
which publishes it to a *different* channel, `orion:collapse:triage` --
registered in `channels.yaml` with `orion-rdf-writer` absent from its
consumer list. So this dispatch branch never receives a matching envelope
at all, independent of the observer/dense gate -- a real channel/kind
mismatch bug in `orion-rdf-writer`, not fixed here (out of scope for a
series about retiring Fuseki *reads*, not reviving Fuseki *writes*), but
worth its own follow-up if anyone ever wants this path alive. Graph-name
resolution confirmed correct separately (`rdf_store.py::normalize_graph_name`
maps `"orion:collapse"` to the exact URI queried) -- not a naming-mismatch
bug either. Since it's provably always been empty, removing it from
`EpisodicFederator` changes nothing about any cluster ever formed. Collapse
tag/entity content (a separate concern from this raw-entry graph) has
*potential* Falkor coverage via the collapse-triage writer noted above, but
that flag (`RECALL_FALKOR_COLLAPSE_TRIAGE_ENABLED`) ships dark by default
in `orion-meta-tags` -- not claimed as active coverage here, just noted as
the mechanism that exists if/when it's flipped on.

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
