# orion-graphiti-adapter

Additive temporal graph projection service for `MemoryCrystallizationV1`.

- **Not** the RDF `memory_graph` path
- Postgres stores projection episodes/entities/edges
- Optional FalkorDB sync when `FALKORDB_ENABLED=true`

## API

- `POST /v1/episodes` — ingest crystallization as temporal episode
- `POST /v1/rebuild` — batch ingest crystallizations (same path as episodes)
- `GET /v1/neighborhood/{crystallization_id}` — graph neighborhood
- `POST /v1/search` — hybrid search (requires `GRAPHITI_BACKEND=graphiti_core`)
- `GET /health`

Hub `GraphitiAdapter` calls this service when `GRAPHITI_URL` is set.

## graphiti_core backend

`GRAPHITI_BACKEND=graphiti_core` is the shipped code-level default (`settings.py`,
`.env_example`) as of the 2026-07-13 hardening pass (see
`docs/superpowers/specs/2026-07-13-graphiti-core-backend-activation-spec.md`). `/v1/search` is
proven to find real data end to end (see the smoke table below), so `graphiti_core` is now the
default for fresh deployments too — this was previously a live-override-only decision, now
made. `FALKORDB_ENABLED` defaults to `true` alongside it. `CRYSTALLIZER_EMBED_HOST_URL` is
still required (`.env_example` ships the real container-DNS value for this node) since
`graphiti_core` cannot embed without it. `orion_postgres` remains the fallback/rollback
backend; neighborhood/BFS traversal is backend-agnostic and identical either way
(`get_neighborhood()` always delegates to the Postgres projection). Only `POST /v1/search`
(hybrid vector+graph) is gated by this flag.

Entities/edges are written via graphiti-core's own `EntityNode`/`EntityEdge` classes
(`uuid`-keyed, `RELATES_TO`-shaped, with a self-referential edge per crystallization so
link-less crystallizations stay searchable) — see the `2026-07-13-graphiti-core-backend-
activation-spec.md` "RELATES_TO schema" follow-up section for the full root cause and field
mapping. `fact`/`name`/`summary` text is always a deterministic string template built from
already-governed `subject`/`summary` fields, never LLM-generated.

## Bring up (FalkorDB + adapter)

FalkorDB is **not** in this compose file. Start the shared operator stack first:

```bash
cd services/orion-falkordb
cp .env_example .env
make preflight up health-probe
cd ../..
```

Then start the adapter:

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-graphiti-adapter/.env \
  -f services/orion-graphiti-adapter/docker-compose.yml up -d --build
```

Cutover from the old `--profile falkordb` sidecar: see [`services/orion-falkordb/README.md`](../orion-falkordb/README.md#cutover-from-graphiti---profile-falkordb-2026-07-16).

Rollback: `GRAPHITI_BACKEND=orion_postgres` and restart adapter (`/v1/search` → 501).

| Live smoke | Covers |
|---|---|
| `scripts/smoke_graphiti_links_e2e.sh` | Phase B link ingest + neighborhood BFS (backend-agnostic) |
| `scripts/smoke_graphiti_search_e2e.sh` | `/v1/search` against real FalkorDB — passing as of 2026-07-13 (`crystallization_ids` contains the seed, `trace.embed_used=true`) |

## FalkorDB

Run the shared FalkorDB operator stack from [`services/orion-falkordb`](../orion-falkordb/) (one container, separate graph names for graphiti vs substrate).

Set in `services/orion-graphiti-adapter/.env`:

- `FALKORDB_URI=redis://orion-${NODE_NAME}-falkordb:6379`
- `FALKORDB_GRAPH=graphiti_temporal`
- `FALKORDB_ENABLED=true`
