# Orion FalkorDB (operator stack)

This directory is the **operator / deployment stack** for Orion’s shared **FalkorDB** property-graph engine. It is **not** a Python Orion service: there is no `app/`, `settings.py`, or `requirements.txt` here—only compose, env templates, Makefile targets, and shell scripts around the upstream `falkordb/falkordb` image.

Same pattern as [`services/orion-rdf-store`](../orion-rdf-store/) (Fuseki) and [`services/orion-sql-db`](../orion-sql-db/) (Postgres): one engine container, many consumers on `app-net`.

## Shared ownership

One FalkorDB instance serves multiple workloads via **separate graph names**:

| Graph name | Consumer | Purpose |
|------------|----------|---------|
| `graphiti_temporal` | [`orion-graphiti-adapter`](../orion-graphiti-adapter/) | Temporal crystallization projection / hybrid search |
| `orion_substrate` | [`orion-substrate-runtime`](../orion-substrate-runtime/) (future) | Substrate property-graph rail (`FalkorSubstrateStore`) |

Do not silently merge labels or properties across graphs. See `docs/superpowers/specs/2026-07-16-falkordb-property-graph-routing-design.md`.

Note: `GraphPersistenceRouter` / `GRAPH_PERSISTENCE_ROUTES_JSON` are library contracts for later workload routing. Substrate dual-run today is driven by `SUBSTRATE_STORE_BACKEND=routed` plus `SUBSTRATE_STORE_PRIMARY` / `SUBSTRATE_STORE_SHADOW`, not the JSON route table.

## Quick start

```bash
cd services/orion-falkordb
cp .env_example .env
make preflight
make up
make health-probe
```

From repo root:

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-falkordb/.env \
  -f services/orion-falkordb/docker-compose.yml up -d
```

- **Container DNS:** `orion-${NODE_NAME}-falkordb` (e.g. `orion-athena-falkordb`)
- **Host port:** `6380` → container `6379`
- **Client URI:** `FALKORDB_URI=redis://orion-athena-falkordb:6379`

Point consumers at the shared URI:

- **graphiti-adapter:** `FALKORDB_URI`, `FALKORDB_GRAPH` (see its `.env_example`)
- **substrate-runtime (when enabled):** `FALKORDB_URI`, `FALKORDB_SUBSTRATE_GRAPH=orion_substrate`

## Cutover from graphiti `--profile falkordb` (2026-07-16)

The co-located `falkordb` service was **removed** from `services/orion-graphiti-adapter/docker-compose.yml`. FalkorDB now lives only here.

**When:** moving from the old ephemeral graphiti profile sidecar to this durable shared stack (or after host data loss).

**Before:** the graphiti profile container had **no volume mount** — graph data was ephemeral. Expect an empty Falkor graph after cutover until you rebuild from Postgres.

### Steps

1. **Stop the legacy sidecar** (if still running from the old profile):

```bash
docker stop orion-athena-falkordb 2>/dev/null || true
docker rm orion-athena-falkordb 2>/dev/null || true
```

2. **Start the shared stack** (bind-mounts `/mnt/graphdb/falkordb`):

```bash
cd services/orion-falkordb
cp .env_example .env   # if needed
make preflight up health-probe
cd ../..
```

3. **Restart graphiti-adapter** (same `FALKORDB_URI`; container name unchanged):

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-graphiti-adapter/.env \
  -f services/orion-graphiti-adapter/docker-compose.yml up -d
```

4. **Verify adapter health:**

```bash
curl -fsS http://localhost:8640/health | jq '.backend, .falkordb_enabled'
```

5. **Rebuild Falkor projection from Postgres** (one-time after empty/ephemeral loss):

```bash
# Hub route — projects up to 200 active crystallizations
curl -fsS -X POST \
  -H "X-Orion-Session-Id: ${ORION_HUB_SESSION_ID}" \
  "http://localhost:8080/api/memory/crystallizations/projection/rebuild?limit=200"
```

The Hub route currently has a hard limit of 200 and no offset. If active
crystallizations exceed 200, stop and use adapter `POST /v1/rebuild` with
explicit item batches; repeating this Hub request only reprojects the same
top-ranked rows.

6. **Smoke search rail:**

```bash
GRAPHITI_ADAPTER_URL=http://localhost:8640 \
ORION_HUB_URL=http://localhost:8080 \
ORION_HUB_SESSION_ID=<session> \
  scripts/smoke_graphiti_search_e2e.sh
```

### Port collision note

Host port **6380** is owned by this stack. [`services/orion-bus/.env_example`](../orion-bus/.env_example) documents that bus-core standby uses **6381** on co-located hosts so it does not collide with FalkorDB.

## Persistence

Data bind-mounts to `FALKORDB_DATA_DIR` (default `/mnt/graphdb/falkordb`) →
FalkorDB's actual Redis persistence directory, `/var/lib/falkordb/data`.
Snapshot that host path for backups while the container is stopped.

On hosts where `/mnt/graphdb/falkordb` is not yet created, provision it once (typical Orion graphdb tier):

```bash
sudo mkdir -p /mnt/graphdb/falkordb
sudo chown "$(whoami):$(id -gn)" /mnt/graphdb/falkordb
```

Until then, a temporary writable path under the Fuseki data tree works for cutover testing (override `FALKORDB_DATA_DIR` in `.env`).

## What this is not

- **Not the RDF store.** SPARQL / Fuseki lives under [`orion-rdf-store`](../orion-rdf-store/).
- **Not graphiti-adapter.** That service owns ingestion API and projection logic; this stack only runs the graph database engine.
