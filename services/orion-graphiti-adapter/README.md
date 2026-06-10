# orion-graphiti-adapter

Additive temporal graph projection service for `MemoryCrystallizationV1`.

- **Not** the RDF `memory_graph` path
- Postgres stores projection episodes/entities/edges
- Optional FalkorDB sync when `FALKORDB_ENABLED=true`

## API

- `POST /v1/episodes` — ingest crystallization as temporal episode
- `GET /v1/neighborhood/{crystallization_id}` — graph neighborhood
- `GET /health`

Hub `GraphitiAdapter` calls this service when `GRAPHITI_URL` is set.
