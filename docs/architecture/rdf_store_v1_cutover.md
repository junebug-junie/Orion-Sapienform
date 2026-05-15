# RDF Store V1 Cutover

Practical operator notes for the **V1 safety cutover**: one canonical bus-driven RDF write sink, explicit legacy quarantine, and narrowed blast radius for out-of-scope GraphDB special cases.

## Canonical writer

- **Service:** `services/orion-rdf-writer`
- **Role:** Only service that should perform **general** RDF materialization writes from the Orion bus (collapse/tags/chat/cortex/… → triples → store).
- **Implementation:** `app/rdf_store.py` (`GraphDbRdfStoreClient`, `FusekiRdfStoreClient`, `GenericSparqlRdfStoreClient`; `rdf4j` → generic), factory `build_rdf_store_client()`, queue in `app/service.py`.

## Active graph backend (Fuseki / SPARQL)

**Fuseki is the active graph backend** for the Athena stack. Canonical HTTP surfaces:

| Surface | Environment variable |
|---------|------------------------|
| SPARQL query | `RDF_STORE_QUERY_URL` (e.g. `http://orion-athena-fuseki:3030/orion/query`) |
| SPARQL update | `RDF_STORE_UPDATE_URL` (e.g. `…/orion/update`) |
| Graph Store POST (bulk RDF) | `RDF_STORE_GRAPH_STORE_URL` (e.g. `…/orion/data`) |

Shared resolution and HTTP clients live in `orion/graph/backend_config.py` and `orion/graph/sparql_client.py`. **Global `GRAPHDB_URL` must never implicitly select GraphDB** for reads or writes; use `GRAPH_BACKEND=graphdb` or service-specific legacy flags only.

## Backend environment (`orion-rdf-writer`)

| Variable | Purpose |
|----------|---------|
| `RDF_STORE_BACKEND` | `fuseki` (default) \| `generic` \| `rdf4j` \| `graphdb` (legacy explicit). |
| `RDF_STORE_BASE_URL` | Fuseki base (e.g. `http://orion-athena-fuseki:3030`). |
| `RDF_STORE_DATASET` | Fuseki dataset name (default `orion`). |
| `RDF_STORE_GRAPH_STORE_URL` | Graph Store HTTP POST target (Fuseki data URL or generic). |
| `RDF_STORE_QUERY_URL` | SPARQL query endpoint (health + cross-service reads). |
| `RDF_STORE_UPDATE_URL` | SPARQL UPDATE endpoint. |
| `GRAPHDB_URL` / `GRAPHDB_REPO` / `GRAPHDB_USER` / `GRAPHDB_PASS` | **Legacy:** required only when `RDF_STORE_BACKEND=graphdb`. |

Copy from `services/orion-rdf-writer/.env_example` (including the **RDF Store V1 Cutover** block).

## Producer channels (unchanged)

- Default enqueue list: **`orion:rdf:enqueue`** (override with `CHANNEL_RDF_ENQUEUE`).
- Payload kind: **`rdf.write.request`** (`RdfWriteRequest` and related build kinds).
- The writer also subscribes to other catalog channels; see `Settings.get_all_subscribe_channels()` in `app/settings.py`.

## Legacy writer (quarantined)

- **Service:** `services/orion-gdb-client`
- **Default:** `GDB_CLIENT_ENABLED=false` — no GraphDB wait, no repository bootstrap, **no** bus listener, `/health` includes `"enabled": false`.
- **Enable:** `GDB_CLIENT_ENABLED=true` only for backfill or legacy tests. **Do not** run beside `orion-rdf-writer` for the same logical writes without understanding duplicate-write risk.

## Read paths (autonomy, recall, self-study, concept profile, substrate)

- **Autonomy / stance** — `AUTONOMY_GRAPH_BACKEND=auto` (default in cortex-exec settings) resolves a SPARQL query URL from `AUTONOMY_GRAPH_QUERY_URL`, then `RDF_STORE_QUERY_URL`, then Fuseki derivation. `AUTONOMY_GRAPH_BACKEND=graphdb` is **legacy only**. Quick lane caps: `AUTONOMY_QUICK_GRAPH_*`.
- **Recall RDF** — `RECALL_RDF_ENDPOINT_URL` / `RECALL_RDF_QUERY_URL` or `RDF_STORE_QUERY_URL`; GraphDB URL is **not** auto-derived unless `GRAPH_BACKEND=graphdb`.
- **Self-study / orionmem adapters** — same SPARQL resolution (`RDF_STORE_QUERY_URL` first); optional legacy GraphDB only when `GRAPH_BACKEND=graphdb`.
- **Concept profile graph** — `CONCEPT_PROFILE_GRAPHDB_ENDPOINT` alias chain includes `RDF_STORE_QUERY_URL`.
- **Substrate durable graph** — `SUBSTRATE_STORE_BACKEND=sparql` targets **Fuseki / SPARQL** over HTTP. Prefer `SUBSTRATE_GRAPH_QUERY_URL` + `SUBSTRATE_GRAPH_UPDATE_URL`; if either is unset, the builder falls back to `RDF_STORE_QUERY_URL` / `RDF_STORE_UPDATE_URL`. Named graph defaults to `SUBSTRATE_GRAPH_URI` (else `DEFAULT_SUBSTRATE_GRAPH_URI`). If `sparql` is selected but no query/update URL can be resolved, startup fails with `substrate_sparql_backend_unconfigured`. `SUBSTRATE_STORE_BACKEND=graphdb` remains explicit legacy only.
- **Memory graph approval** — `MEMORY_GRAPH_APPROVAL_BACKEND=auto` (default): writes use `RDF_STORE_GRAPH_STORE_URL` + `RDF_STORE_UPDATE_URL` (Fuseki graph store + SPARQL update). `MEMORY_GRAPH_APPROVAL_BACKEND=graphdb` uses legacy GraphDB statements API.

## Disabled / YAML fallback

`GRAPH_BACKEND=disabled` or missing SPARQL configuration yields **intentionally degraded** behavior (in-memory substrate, identity/YAML autonomy fallback, no RDF recall). That is **emergency / degraded mode**, not the Fuseki cutover target.

## Operator cutover checklist

1. Set `orion-rdf-writer` env: `RDF_STORE_BACKEND=fuseki` and the matching `RDF_STORE_*` URLs (see `.env_example`).
2. Set `GDB_CLIENT_ENABLED=false` for `orion-gdb-client` (default) or remove the service from the running stack.
3. Set `SUBSTRATE_STORE_BACKEND=sparql` when you need durable substrate (Fuseki). Provide substrate-specific query/update URLs or rely on the shared `RDF_STORE_QUERY_URL` / `RDF_STORE_UPDATE_URL` fallback; otherwise leave unset / `in_memory`.
4. Set `AUTONOMY_GRAPH_BACKEND=auto` (or omit) and provide `RDF_STORE_QUERY_URL` or Fuseki base + dataset for stance/autonomy reads.
5. Set Hub / recall `RDF_STORE_*` or `RECALL_RDF_*` query URLs for RDF recall; avoid implicit GraphDB.
6. For memory graph approval, set `RDF_STORE_GRAPH_STORE_URL` + `RDF_STORE_UPDATE_URL` (or legacy `MEMORY_GRAPH_APPROVAL_BACKEND=graphdb` + `GRAPHDB_URL`).
7. Hit `orion-rdf-writer` **`GET /health`** — expect `rdf_store_backend=fuseki`, queue snapshot fields, **no** secrets in URLs (credentials stripped).
8. Publish a test `rdf.write.request` on `orion:rdf:enqueue` and confirm `rdf_write_enqueued` / `rdf_write_committed` log lines and stable dead-letter file size.

## Rollback

1. Point `RDF_STORE_BACKEND=graphdb` and restore `GRAPHDB_URL` / `GRAPHDB_REPO` on `orion-rdf-writer`.
2. If you must restore the legacy parallel writer, set `GDB_CLIENT_ENABLED=true` on `orion-gdb-client` and ensure GraphDB is reachable (see service README). Prefer **not** duplicating producers indefinitely.

## Remaining V2 work (do not track as V1 deliverables)

- Further consolidation of per-service env aliases onto `orion.graph.backend_config`.
- Broader operator doc pass for non-Athena deployments.
