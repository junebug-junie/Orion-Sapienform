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

`GRAPHITI_BACKEND=graphiti_core` is the deployed runtime state for this Orion node's live
adapter container as of the 2026-07-13 activation pass (see
`docs/superpowers/specs/2026-07-13-graphiti-core-backend-activation-spec.md`). `settings.py`'s
and `.env_example`'s code-level default remains `orion_postgres` for fresh deployments —
`/v1/search` is now proven to find real data end to end (see the smoke table below), but
flipping the shipped default is a separate operator decision, not made in this patch.
`orion_postgres` is the fallback/rollback backend; neighborhood/BFS traversal is
backend-agnostic and identical either way (`get_neighborhood()` always delegates to the
Postgres projection). Only `POST /v1/search` (hybrid vector+graph) is gated by this flag.

Entities/edges are written via graphiti-core's own `EntityNode`/`EntityEdge` classes
(`uuid`-keyed, `RELATES_TO`-shaped, with a self-referential edge per crystallization so
link-less crystallizations stay searchable) — see the `2026-07-13-graphiti-core-backend-
activation-spec.md` "RELATES_TO schema" follow-up section for the full root cause and field
mapping. `fact`/`name`/`summary` text is always a deterministic string template built from
already-governed `subject`/`summary` fields, never LLM-generated.

```bash
docker compose --profile falkordb \
  --env-file .env --env-file services/orion-graphiti-adapter/.env \
  -f services/orion-graphiti-adapter/docker-compose.yml up -d --build
```

Rollback: `GRAPHITI_BACKEND=orion_postgres` and restart adapter (`/v1/search` → 501).

| Live smoke | Covers |
|---|---|
| `scripts/smoke_graphiti_links_e2e.sh` | Phase B link ingest + neighborhood BFS (backend-agnostic) |
| `scripts/smoke_graphiti_search_e2e.sh` | `/v1/search` against real FalkorDB — passing as of 2026-07-13 (`crystallization_ids` contains the seed, `trace.embed_used=true`) |

## FalkorDB profile (optional)

```bash
docker compose --profile falkordb \
  --env-file .env \
  -f services/orion-graphiti-adapter/docker-compose.yml up -d
```

Set `FALKORDB_ENABLED=true` in `services/orion-graphiti-adapter/.env`.
