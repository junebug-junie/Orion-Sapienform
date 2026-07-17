# Orion FalkorDB (operator stack)

This directory is the **operator / deployment stack** for Orion’s shared **FalkorDB** property-graph engine. It is **not** a Python Orion service: there is no `app/`, `settings.py`, or `requirements.txt` here—only compose and env templates around the upstream `falkordb/falkordb` image.

Same pattern as [`services/orion-rdf-store`](../orion-rdf-store/) (Fuseki) and [`services/orion-sql-db`](../orion-sql-db/) (Postgres): one engine container, many consumers on `app-net`.

## Shared ownership

One FalkorDB instance serves multiple workloads via **separate graph names**:

| Graph name | Consumer | Purpose |
|------------|----------|---------|
| `orion_graphiti` (planned canonical; adapter may still use `graphiti_temporal`) | [`orion-graphiti-adapter`](../orion-graphiti-adapter/) | Temporal crystallization projection / hybrid search |
| `orion_substrate` | [`orion-substrate-runtime`](../orion-substrate-runtime/) (future) | Substrate property-graph rail (`FalkorSubstrateStore`) |

Do not silently merge labels or properties across graphs. See `docs/superpowers/specs/2026-07-16-falkordb-property-graph-routing-design.md`.

## Preferred compose home

**Run FalkorDB from here**, not nested only under graphiti-adapter:

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-falkordb/.env \
  -f services/orion-falkordb/docker-compose.yml up -d
```

- **Container DNS:** `orion-${NODE_NAME}-falkordb` (e.g. `orion-athena-falkordb`)
- **Host port:** `6380` → container `6379` (matches existing graphiti-adapter profile)
- **Client URI:** `FALKORDB_URI=redis://orion-athena-falkordb:6379`

[`orion-graphiti-adapter`](../orion-graphiti-adapter/) still ships a `--profile falkordb` service for backward compatibility (single-node dev, legacy docs). For production and multi-consumer layouts, prefer this stack.

## Quick start

```bash
cd services/orion-falkordb
cp .env_example .env
# Confirm FALKORDB_DATA_DIR exists and is writable on the host
docker compose --env-file .env -f docker-compose.yml up -d
```

Point consumers at the shared URI:

- **graphiti-adapter:** `FALKORDB_URI`, `FALKORDB_GRAPH` (see its `.env_example`)
- **substrate-runtime (when enabled):** `FALKORDB_URI`, `FALKORDB_SUBSTRATE_GRAPH=orion_substrate`

## Persistence

Data bind-mounts to `FALKORDB_DATA_DIR` (default `/mnt/graphdb/falkordb`) → `/data` in the container. Snapshot that host path for backups while the container is stopped.

## What this is not

- **Not the RDF store.** SPARQL / Fuseki lives under [`orion-rdf-store`](../orion-rdf-store/).
- **Not graphiti-adapter.** That service owns ingestion API and projection logic; this stack only runs the graph database engine.
