# Graphiti-core backend activation (search rail only)

**Date:** 2026-07-13
**Status:** Approved for implementation — activation, not a build
**Scope:** Flip already-built `graphiti_core` backend live. **Do not touch** `orion/memory/consolidation_gate.py`, `low_info_social.py`, or any `MEMORY_CONSOLIDATION_*` threshold. That system is live and correct (48 active / 10 proposed / 27 rejected crystallizations) and is out of scope.

---

## Ground truth (verified, not assumed)

| Claim | Evidence |
|---|---|
| Graphiti Phase A/B/C already merged to `main` | `services/orion-graphiti-adapter/app/backends/{orion_postgres,graphiti_core}.py` exist, tested |
| Adapter is running now | `orion-athena-graphiti-adapter` container `Up`, backend=`orion_postgres` |
| Neighborhood/BFS is backend-agnostic | `graphiti_core.get_neighborhood()` (`backends/graphiti_core.py:198-203`) delegates to `pg_neighborhood` regardless of `GRAPHITI_BACKEND` — traversal already works today |
| **Only `/v1/search` is gated** by `GRAPHITI_BACKEND` | `main.py:170-172` — returns HTTP 501 unless `graphiti_core` |
| `graphiti_core` backend untested against real FalkorDB | `test_graphiti_core_backend.py` mocks the driver and the `graphiti_core` module entirely |
| Embed URL empty on adapter | `services/orion-graphiti-adapter/.env:21` `CRYSTALLIZER_EMBED_HOST_URL=` (siblings use `http://orion-athena-vector-host:8320/embedding`) |
| Approve already auto-projects to Graphiti | `crystallization_approve_proposal` → `project_crystallization(config=_projection_config())`, `graphiti_enabled=True` since `GRAPHITI_ENABLED=true`; `project_graphiti` defaults `True` unless overridden — **future approvals need no backfill action** |
| Env is unblocked | port 6380 free, `app-net` network exists, `falkordb/falkordb:latest` pulls clean |
| Real belief data exists | Postgres `memory_crystallizations`: 48 `active` |

**Reframed problem:** this is not "turn on graph memory" (graph traversal is already live). It is "turn on hybrid vector+graph search, verify it's real against real FalkorDB, and backfill it with Orion's 48 existing beliefs so it isn't an empty shell."

---

## Non-goals

- No change to `consolidation_gate.py`, `low_info_social.py`, `MEMORY_CONSOLIDATION_OUTPUT`/`MIN_NOVELTY`/`MIN_SIGNIFICANCE`
- No change to `orion_postgres` backend code, neighborhood BFS, or link projection (Phase B)
- No new schema, no new bus channel, no new crystallization kind
- No pinning/upgrading `graphiti-core` version in this pass (flag if 0.19.0 breaks against real FalkorDB — separate patch)

---

## Steps (execute in order; each has a verify)

### 1. Config flip
`services/orion-graphiti-adapter/.env` (+ mirror in `.env_example` if any key shape changes — it doesn't, keys already exist):
```
GRAPHITI_BACKEND=graphiti_core
FALKORDB_ENABLED=true
CRYSTALLIZER_EMBED_HOST_URL=http://orion-athena-vector-host:8320/embedding
```
Run `python scripts/sync_local_env_from_example.py`.
**Verify:** `git diff services/orion-graphiti-adapter/.env` shows exactly these 3 lines changed.

### 2. Bring up FalkorDB + restart adapter
```bash
docker compose --profile falkordb \
  --env-file .env --env-file services/orion-graphiti-adapter/.env \
  -f services/orion-graphiti-adapter/docker-compose.yml up -d --build
```
**Verify:** `curl -fsS http://localhost:8640/health | jq '.backend'` → `"graphiti_core"`

### 3. Backfill 48 active crystallizations (one-time)
Call Hub rebuild route (or adapter `/v1/rebuild` directly) over all `status=active` rows.
**Verify:** FalkorDB entity node count == `select count(*) from memory_crystallizations where status='active'` (48). Query via adapter debug/Cypher count, not assumption.

### 4. Live search smoke (new, non-mocked)
New script `scripts/smoke_graphiti_search_e2e.sh`, permanent (matches existing `smoke_graphiti_links_e2e.sh` convention — no special-casing):
- propose+approve 1 crystallization with a known subject string
- `POST /v1/search {"query": "<subject>"}`
- assert `crystallization_ids` non-empty AND `trace.embed_used == true`
**Verify:** script exits 0.

### 5. Chat-visible effect (not just adapter health)
Pick 2 already-linked crystallizations where the link (not either alone) answers a question. Call `retrieve_active_packet()` path (existing wiring, `retriever.py:79-81` already calls `/v1/search` when `health.backend == graphiti_core`).
**Verify:** response `graphiti_refs` contains both crystallization IDs.

### 6. Privacy check on real data
```sql
select crystallization_id from memory_crystallizations
where status='active' and metadata->>'sensitivity' = 'intimate';
```
**Verify:** none of these IDs appear in FalkorDB nodes or in any `/v1/search` result (existing skip logic at `graphiti_core.py:128-129`, `_filter_intimate_crystallization_ids`).

### 7. Rollback rehearsal
Flip `GRAPHITI_BACKEND=orion_postgres`, restart, confirm `/v1/search` → 501 again, neighborhood endpoint unaffected (never depended on the flag). Flip forward again.
**Verify:** both states produce expected status codes.

---

## Acceptance checks

- [ ] `/health` reports `backend: graphiti_core`, `embed_used: true` on search trace
- [ ] FalkorDB node count matches active crystallization count post-backfill
- [ ] `smoke_graphiti_search_e2e.sh` passes and is committed to `scripts/`
- [ ] `smoke_graphiti_links_e2e.sh` still passes (Phase B neighborhood unaffected)
- [ ] Active-packet fusion demonstrably surfaces a graph-only-reachable belief
- [ ] Zero `intimate` crystallizations reachable via FalkorDB or `/v1/search`
- [ ] Rollback to `orion_postgres` verified clean
- [ ] `consolidation_gate.py` / worker.py diff: empty

---

## Env/config changes

| Key | File | Before → After |
|---|---|---|
| `GRAPHITI_BACKEND` | `services/orion-graphiti-adapter/.env` | `orion_postgres` → `graphiti_core` |
| `FALKORDB_ENABLED` | `services/orion-graphiti-adapter/.env` | `false` → `true` |
| `CRYSTALLIZER_EMBED_HOST_URL` | `services/orion-graphiti-adapter/.env` | `` → `http://orion-athena-vector-host:8320/embedding` |

No `.env_example` shape changes (keys pre-exist). Local `.env` sync required per key change.

---

## Risks

| Severity | Risk | Mitigation |
|---|---|---|
| Medium | `graphiti-core==0.19.0` API drift vs `falkordb/falkordb:latest` (unpinned image tag) | Step 4 is the real test; pin image tag if it breaks |
| Medium | Dual-write divergence: FalkorDB write silently fails while Postgres dual-write in `ingest_episode` succeeds | Step 3 count-match check surfaces this; no code fix in this pass unless found |
| Low | Backfill misses future approvals | Not a risk — approve path already auto-projects (verified in Ground truth) |

---

## Restart required

```bash
docker compose --profile falkordb \
  --env-file .env --env-file services/orion-graphiti-adapter/.env \
  -f services/orion-graphiti-adapter/docker-compose.yml up -d --build
```

## Rollback

```bash
# set GRAPHITI_BACKEND=orion_postgres in services/orion-graphiti-adapter/.env
docker compose --env-file .env --env-file services/orion-graphiti-adapter/.env \
  -f services/orion-graphiti-adapter/docker-compose.yml up -d --build
```
No canonical Postgres crystallization data is touched by any step above (`canonical_mutated` invariant holds throughout).

---

## Design decision: RELATES_TO schema (2026-07-13 follow-up)

**Problem:** this activation pass shipped a live `graphiti_core` backend that ran without
crashing but returned zero results from `/v1/search` for real data — documented as
known-failing in `scripts/smoke_graphiti_search_e2e.sh`. Root-caused and fixed same day.

### Root cause (verified against `graphiti-core==0.19.0` source in the running container)

| Bug | Old behavior | Why `Graphiti.search()` never saw it |
|---|---|---|
| Wrong node identity property | `ingest_episode()` wrote raw Cypher `MERGE (e:Entity {id: $entity_id, ...})` | Every graphiti-core query (incl. internal `search()`) matches nodes on `.uuid`, not a custom `.id`; our entities had no `.uuid` at all |
| Wrong edge shape | Raw Cypher `(:Entity)-[:HAS_EPISODE]->(:Episode)` and `(:Entity)-[:RELATED]->(:Entity)` | `Graphiti.search()` only reads `(:Entity)-[e:RELATES_TO {uuid, group_id, name, fact, fact_embedding, episodes, created_at, ...}]->(:Entity)` via fulltext+vector hybrid search over `fact` |
| No fulltext index | Adapter never called `build_indices_and_constraints()` | `/v1/search`'s `CALL db.idx.fulltext.queryRelationships(...)` had nothing to query |
| `_extract_crystallization_ids()` fallback shape mismatch | Read `getattr(item, "source_node", None)` (nested object with its own `.crystallization_id`) | Real `Graphiti.search()` results are `EntityEdge` objects with `.source_node_uuid`/`.target_node_uuid` as plain strings — no nested node object exists |

### Fix — field mapping (graphiti-core's own write API, still no LLM extraction)

| Orion field | graphiti-core field | Notes |
|---|---|---|
| `gent_{crystallization_id}` | `EntityNode.uuid` | Identity key graphiti-core's own queries match on |
| `subject` | `EntityNode.name` | — |
| `{crystallization_id, sensitivity}` | `EntityNode.attributes` | Flattened onto the node as top-level properties by `.save()` on FalkorDB (`entity_data.update(self.attributes)`) — `_filter_intimate_crystallization_ids()`'s `n.crystallization_id`/`n.sensitivity` Cypher needed no change |
| `"orion"` (new constant `ORION_GROUP_ID`) | `EntityNode.group_id` / `EntityEdge.group_id` | Single shared value; `search()`'s `group_ids` filter is unset today, so this only needs to be consistent, not meaningful — no multi-tenant scheme added |
| self-referential edge, `name="describes"`, `fact=f"{subject}: {summary[:280]}"` | `EntityEdge` with `source_node_uuid == target_node_uuid == entity_id` | Written for **every** ingested crystallization, not just linked ones — `Graphiti.search()` only ever returns edges, never bare nodes, and most real crystallizations have zero `CrystallizationLinkV1` links |
| each `CrystallizationLinkV1` | `EntityEdge` with `name=relation`, `fact=f"{subject} {relation} {target_subject}"` | `target_subject` looked up from `graphiti_episodes.subject` via `pg_pool` (link payload carries no subject text); falls back to `f"related crystallization {target_id}"` if the target was never ingested |

`fact`/`name`/`summary` are always a deterministic string template over already-governed
`subject`/`summary` fields — never an LLM call (hard constraint, unchanged from the original
Phase C design doc).

**Target-entity stub write:** `EntityEdge.save()` on FalkorDB `MATCH`es both endpoints (it does
not create them) — a link to a not-yet-ingested crystallization needs a stub `EntityNode` first
or the edge write silently no-ops. `EntityNode.save()` is a full property replace (`SET n =
$entity_data`), so the stub is only written when an existence check finds no prior node —
otherwise it would clobber an already-real target's `summary`/`attributes`.

### Second bug, same file: FalkorDB single-entity `.save()` doesn't cast embeddings to `Vectorf32`

Discovered during live verification (not predicted up front): `graphiti-core==0.19.0`'s
non-bulk `EntityNode.save()`/`EntityEdge.save()` queries for the FalkorDB provider do
`SET n/e = $data` with a plain Python list for `name_embedding`/`fact_embedding` — FalkorDB
stores that as a Cypher `List`, not its native `Vectorf32` type. Only the separate *bulk* save
query builders (`get_entity_node_save_bulk_query`, `get_entity_edge_save_bulk_query`, used by
graphiti-core's own `add_episode_bulk` path, not exposed as a method on `EntityNode`/
`EntityEdge`) wrap embeddings in `vecf32(...)`. Without a fix, every embedded search hit
`ResponseError: Type mismatch: expected Null or Vectorf32 but was List` inside
`edge_similarity_search`'s cosine-distance query — a full `/v1/search` 500, not a silent
empty-result.

**Fix:** after each `.save()`, if an embedding was set, issue one narrow follow-up Cypher call
(`SET n.name_embedding = vecf32($embedding)` / `SET e.fact_embedding = vecf32($embedding)`)
keyed on the same `uuid`. Same schema, same property names graphiti-core itself writes — this
casts the type the library's own non-bulk save path forgot to, it does not introduce a
different shape.

### Index bootstrap idempotency

FalkorDB's `CREATE FULLTEXT INDEX` has no `IF NOT EXISTS` guard (unlike Neo4j) — a second
`build_indices_and_constraints()` call against an already-indexed graph raises `already
indexed`. `FalkorDriver.execute_query` itself already catches that substring and returns
`None`; `ensure_graphiti_indices()` (`backends/graphiti_core.py`) adds a broader catch plus a
process-local `_indices_ready` flag on top, mirroring the `GRAPHITI_AUTO_APPLY_SCHEMA` /
`apply_graphiti_schema()` pattern already used for the Postgres projection schema. New setting:
`GRAPHITI_AUTO_BUILD_INDICES` (default `true`), called once from `main.py`'s `lifespan` when
`GRAPHITI_BACKEND=graphiti_core` and `FALKORDB_ENABLED=true`.

### Live verification (this host, 2026-07-13)

| Check | Before fix | After fix |
|---|---|---|
| `MATCH ()-[r:RELATES_TO]->() RETURN count(r)` | `0` | `10` (grows with real ingest/sync traffic) |
| `scripts/smoke_graphiti_search_e2e.sh` | FAIL (empty `crystallization_ids`), then 500 (`Vectorf32` bug) mid-fix | PASS, repeatable across 3 consecutive runs — `crystallization_ids` contains seed, `trace.embed_used=true` |
| `scripts/smoke_graphiti_links_e2e.sh` | PASS | PASS (no regression) |
| Intimate crystallization (`governance.sensitivity=intimate`) synced via `/api/memory/graphiti/sync/{id}` | — | Not written to FalkorDB (`MATCH (n) WHERE n.crystallization_id = '<id>'` empty); not present in any `/v1/search` result |

**Unrelated finding, fixed in the same pass because it blocked verification:**
`scripts/smoke_graphiti_search_e2e.sh`'s original fixed-template subject/summary (only a
timestamp varying) has word-level Jaccard similarity ~0.75 against its own prior runs once a
few have accumulated — `orion/memory/crystallization/detection.py::detect_duplicates`
(threshold `0.72`) flags it as a duplicate candidate, and validate sets
`validation_status=invalid`, blocking approve with HTTP 400. Not a graphiti-core bug; the
script now varies two independent random tokens per run (Jaccard ~0.5) and checks the approve
HTTP status explicitly instead of swallowing failures.
