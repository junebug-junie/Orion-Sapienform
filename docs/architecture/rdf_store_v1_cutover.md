# RDF Store V1 Cutover

Practical operator notes for the **V1 safety cutover**: one canonical bus-driven RDF write sink, explicit legacy quarantine, and narrowed blast radius for out-of-scope GraphDB special cases.

## Canonical writer

- **Service:** `services/orion-rdf-writer`
- **Role:** Only service that should perform **general** RDF materialization writes from the Orion bus (collapse/tags/chat/cortex/… → triples → store).
- **Implementation:** `app/rdf_store.py` (`GraphDbRdfStoreClient`, `FusekiRdfStoreClient`, `GenericSparqlRdfStoreClient`; `rdf4j` → generic), factory `build_rdf_store_client()`, queue in `app/service.py`.

## Backend environment (`orion-rdf-writer`)

| Variable | Purpose |
|----------|---------|
| `RDF_STORE_BACKEND` | `graphdb` \| `fuseki` \| `generic` \| `rdf4j` (alias of generic with URL requirements). |
| `RDF_STORE_BASE_URL` | Fuseki base (e.g. `http://orion-athena-fuseki:3030`). |
| `RDF_STORE_DATASET` | Fuseki dataset name (default `orion`). |
| `RDF_STORE_GRAPH_STORE_URL` | Graph Store HTTP POST target (Fuseki data URL or generic). |
| `RDF_STORE_QUERY_URL` | SPARQL query endpoint (optional for writes; used in health metadata). |
| `RDF_STORE_UPDATE_URL` | SPARQL UPDATE endpoint (generic adapter fallback). |
| `GRAPHDB_URL` / `GRAPHDB_REPO` / `GRAPHDB_USER` / `GRAPHDB_PASS` | Required when `RDF_STORE_BACKEND=graphdb` only. |

Copy from `services/orion-rdf-writer/.env_example` (including the **RDF Store V1 Cutover** block).

## Producer channels (unchanged)

- Default enqueue list: **`orion:rdf:enqueue`** (override with `CHANNEL_RDF_ENQUEUE`).
- Payload kind: **`rdf.write.request`** (`RdfWriteRequest` and related build kinds).
- The writer also subscribes to other catalog channels; see `Settings.get_all_subscribe_channels()` in `app/settings.py`.

## Legacy writer (quarantined)

- **Service:** `services/orion-gdb-client`
- **Default:** `GDB_CLIENT_ENABLED=false` — no GraphDB wait, no repository bootstrap, **no** bus listener, `/health` includes `"enabled": false`.
- **Enable:** `GDB_CLIENT_ENABLED=true` only for backfill or legacy tests. **Do not** run beside `orion-rdf-writer` for the same logical writes without understanding duplicate-write risk.

## Explicitly out of scope for V1 (still GraphDB or in-memory)

1. **Memory graph approval** — `orion/memory_graph/approve.py` + Hub `POST /api/memory/graph/approve`. GraphDB statements API + Postgres with RDF compensation. Hub returns **`503`** with detail **`memory_graph_approval_requires_graphdb`** when `GRAPHDB_URL` is unset.
2. **Substrate semantic store** — `orion/substrate/graphdb_store.py` `build_substrate_store_from_env()`. **GraphDB is used only when `SUBSTRATE_STORE_BACKEND=graphdb`**. Unset backend → **in-memory** even if `GRAPHDB_URL` is set (V1 blast-radius guard).

## Operator cutover checklist

1. Set `orion-rdf-writer` env: `RDF_STORE_BACKEND=fuseki` (or `generic`) and the matching `RDF_STORE_*` URLs (see `.env_example`).
2. Set `GDB_CLIENT_ENABLED=false` for `orion-gdb-client` (default) or remove the service from the running stack.
3. Set `SUBSTRATE_STORE_BACKEND=in_memory` unless you explicitly need durable substrate in GraphDB (`graphdb` + endpoint envs).
4. Confirm Hub / recall GraphDB vars still satisfy **memory graph approval** if you use that API.
5. Hit `orion-rdf-writer` **`GET /health`** — expect `rdf_store_backend`, queue snapshot fields, **no** secrets in URLs (credentials stripped).
6. Publish a test `rdf.write.request` on `orion:rdf:enqueue` and confirm `rdf_write_enqueued` / `rdf_write_committed` log lines and stable dead-letter file size.

## Rollback

1. Point `RDF_STORE_BACKEND=graphdb` and restore `GRAPHDB_URL` / `GRAPHDB_REPO` on `orion-rdf-writer`.
2. If you must restore the legacy parallel writer, set `GDB_CLIENT_ENABLED=true` on `orion-gdb-client` and ensure GraphDB is reachable (see service README). Prefer **not** duplicating producers indefinitely.

## Remaining V2 work (do not track as V1 deliverables)

- Backend-neutral memory graph approval store.
- Backend-neutral substrate graph store.
- Self-study / concept profile **read** cutover and GraphDB retirement once all direct read/write paths are migrated.
