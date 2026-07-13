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
