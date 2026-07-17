# Implementation plan: FalkorDB router guts (concept-induction deferred)

**Date:** 2026-07-16  
**Branch / worktree:** `docs/falkordb-property-graph-routing` → rename to `feat/falkordb-property-graph-routing`  
**Spec:** `docs/superpowers/specs/2026-07-16-falkordb-property-graph-routing-design.md`  
**Deferred:** Concept Atlas / topic-foundry / hub ingest wiring, `KgEdgeIngest` retirement, live `SUBSTRATE_STORE_BACKEND` flip on substrate-runtime, Fuseki decommission.

## Goal

Ship the library + store + operator stack so later phases can flip workloads without inventing contracts under fire.

## Global constraints

- Work only in `/mnt/scripts/Orion-Sapienform-falkordb-property-graph-routing`
- Thin seams; ride `SubstrateGraphStore` and `orion/graph/`
- No concept-induction / hub Concept Atlas route changes
- No rdf-writer kind changes
- No new microservice for the router (module only)
- One FalkorDB container; two graph names (`graphiti_temporal`, `orion_substrate`)
- Tests must pass without a live Falkor (mock driver/client)
- Commit per task; do not commit `.env`
- After code: prefer focused pytest; do not run `graphify update` unless asked

## Task map

| ID | Task | Depends | Parallel? |
|---|---|---|---|
| 1 | `GraphWriteIntentV1` + schema registry | — | yes with 2, 3, 6 |
| 2 | Route table + router module | — | yes |
| 3 | Property / metadata cathedral guard | — | yes |
| 4 | `FalkorSubstrateStore` + `build_substrate_store_from_env` | 1–3 | after 1–3 |
| 5 | `RoutedSubstrateGraphStore` (primary/shadow) | 4 | after 4 |
| 6 | `services/orion-falkordb/` operator stack | — | yes with 1–3 |

---

## Task 1 — GraphWriteIntentV1

**Create:**
- `orion/schemas/graph_write_intent.py` — `GraphWriteIntentV1` with:
  - `workload: str`
  - `operation: Literal["upsert_node","upsert_edge","delete_node","delete_edge","append_event"]`
  - `identity_key: str`
  - optional node/edge payloads as typed dicts or nested models (`extra="forbid"`)
  - `provenance` model (producer, source_refs list capped ≤32, observed_at)
  - `compatibility.rdf_graph_name: Optional[str]`
  - `routing_hint: Optional[str]`
- Register in `orion/schemas/registry.py`
- Tests: `orion/schemas/tests/test_graph_write_intent.py` — validate happy path + reject unknown fields + empty identity_key

**Acceptance:** `pytest orion/schemas/tests/test_graph_write_intent.py -q` passes; registry lists `GraphWriteIntentV1`.

---

## Task 2 — Route table + router

**Create:**
- `orion/graph/persistence_routes.py`
  - Parse `GRAPH_PERSISTENCE_ROUTES_JSON` (default empty → all unresolved)
  - `RouteTarget = Literal["falkor","sparql","rdf","in_memory","postgres","disabled"]`
  - `WorkloadRoute(primary, shadow="none")`
  - `resolve_workload_route(workload: str) -> WorkloadRoute`
  - Unknown workload → `primary="disabled"` + warning log (fail-safe, not Fuseki)
- `orion/graph/persistence_router.py`
  - `GraphPersistenceRouter` with `select(workload, routing_hint=None) -> WorkloadRoute`
  - Structured log: `graph_route_selected workload=… primary=… shadow=…`
- Tests: `orion/graph/tests/test_persistence_routes.py`

**Acceptance:** env JSON with `substrate.drive_state` primary falkor / shadow sparql resolves correctly; missing key → disabled.

---

## Task 3 — Property cathedral guard

**Create:**
- `orion/graph/property_guard.py`
  - Caps: `METADATA_MAX_KEYS=16`, `METADATA_MAX_BYTES=4096`
  - `sanitize_metadata(metadata: dict, *, fail_closed: bool=False) -> tuple[dict, list[str]]`
  - Drops oversize / excess keys; logs `property_cathedral_rejected`
  - `fail_closed=True` raises `PropertyCathedralError`
- Tests: `orion/graph/tests/test_property_guard.py`

**Acceptance:** 17th key dropped (or raises if fail_closed); >4KB value rejected; clean dict passes unchanged.

---

## Task 4 — FalkorSubstrateStore

**Create:**
- `orion/substrate/falkor_store.py`
  - Implements `SubstrateGraphStore` (same methods as `InMemorySubstrateGraphStore` / enough of the protocol for upsert/get/snapshot + query_* by delegating to in-memory cache hydrated from Falkor, OR pure Cypher — prefer: local cache + write-through like `GraphDBSubstrateStore` pattern)
  - Config from env: `FALKORDB_URI`, `FALKORDB_SUBSTRATE_GRAPH` (default `orion_substrate`)
  - Sync client preferred (redis `GRAPH.QUERY`); inject client for tests
  - On upsert: run `sanitize_metadata` on node/edge metadata
  - Labels: `SubstrateNode` / relationships use predicate as type (uppercase safe)
- Update `orion/substrate/graphdb_store.py::build_substrate_store_from_env`:
  - `falkor` / `falkordb` → `FalkorSubstrateStore`
  - missing URI → log + fall back in_memory (like graphdb missing endpoint)
- Tests: `orion/substrate/tests/test_falkor_store.py` with fake client recording queries; cathedral reject path; builder selection

**Acceptance:** upsert_node → snapshot round-trip via fake; builder returns Falkor store when env set; no live Falkor required.

**Do not:** flip substrate-runtime `.env` to falkor.

---

## Task 5 — RoutedSubstrateGraphStore

**Create:**
- `orion/substrate/routed_store.py`
  - Wraps primary + optional shadow stores
  - Workload for writes: constructor default `substrate.default` or per-call if we add optional kw — for SubstrateGraphStore protocol, use fixed workload key from config `SUBSTRATE_ROUTE_WORKLOAD` default `substrate.runtime`
  - Writes: primary then shadow (best-effort; shadow errors logged, never raise if primary ok)
  - Reads: primary only
- Builder: `SUBSTRATE_STORE_BACKEND=routed` requires `SUBSTRATE_STORE_PRIMARY` + optional `SUBSTRATE_STORE_SHADOW`
- Tests: `orion/substrate/tests/test_routed_store.py`

**Acceptance:** dual write hits both; shadow failure does not fail primary; reads only primary.

---

## Task 6 — orion-falkordb operator stack

**Create:** `services/orion-falkordb/` mirroring `orion-rdf-store` lightness:
- `docker-compose.yml` — service `falkordb`, container `orion-${NODE_NAME}-falkordb`, port 6380:6379, `app-net`, volume for data
- `.env_example` — `FALKORDB_URI=redis://orion-athena-falkordb:6379`, graph name docs
- `README.md` — ownership: shared by graphiti + future substrate; graph names `graphiti_temporal` / `orion_substrate`
- Update `services/orion-graphiti-adapter/docker-compose.yml` — **done:** falkordb profile removed; README points to shared stack

**Acceptance:** `docker compose … config` validates; no requirement to migrate running container in this task.

---

## Env keys (document in `.env_example` where a service already has substrate keys)

Add comments only to `services/orion-substrate-runtime/.env_example` for new keys (do not enable):

```text
# SUBSTRATE_STORE_BACKEND=falkor|routed|sparql|in_memory
# FALKORDB_URI=redis://orion-athena-falkordb:6379
# FALKORDB_SUBSTRATE_GRAPH=orion_substrate
# SUBSTRATE_STORE_PRIMARY=falkor
# SUBSTRATE_STORE_SHADOW=sparql
# GRAPH_PERSISTENCE_ROUTES_JSON=
```

Sync local `.env` only if `.env_example` keys are added (script).

---

## Out of scope (explicit)

- Hub `concept_atlas_routes.py` changes
- `orion/substrate/adapters/topic_foundry.py` changes
- Live cutover of substrate-runtime off SPARQL
- Graphiti schema changes
- Memgraph / Arcade / Ladybug

## Verification (orchestrator)

```bash
pytest orion/schemas/tests/test_graph_write_intent.py \
       orion/graph/tests/test_persistence_routes.py \
       orion/graph/tests/test_property_guard.py \
       orion/substrate/tests/test_falkor_store.py \
       orion/substrate/tests/test_routed_store.py -q
```
