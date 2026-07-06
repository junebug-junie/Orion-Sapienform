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

## graphiti_core backend smoke

```bash
GRAPHITI_BACKEND=graphiti_core FALKORDB_ENABLED=true \
docker compose --profile falkordb --env-file services/orion-graphiti-adapter/.env \
  -f services/orion-graphiti-adapter/docker-compose.yml up -d --build
```

Rollback: `GRAPHITI_BACKEND=orion_postgres` and restart adapter.

## FalkorDB profile (optional)

```bash
docker compose --profile falkordb \
  --env-file .env \
  -f services/orion-graphiti-adapter/docker-compose.yml up -d
```

Set `FALKORDB_ENABLED=true` in `services/orion-graphiti-adapter/.env`.
