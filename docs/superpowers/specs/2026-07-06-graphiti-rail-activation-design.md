# Graphiti rail activation — sequential vertical slices (A → B → C)

**Date:** 2026-07-06  
**Status:** Approved for implementation planning  
**Operator update (2026-07-16):** Commands using graphiti-adapter's
`--profile falkordb` are historical. The sidecar was removed; use
`services/orion-falkordb/README.md` for current bring-up and cutover steps.
**Problem:** The Graphiti seam (Phases A–E, commit `ab40f277`) is architecturally sound but operationally dormant — default env wiring may skip auto-projection on approve, smoke never exercises the rail, adapter has no tests, and the graph is one node per crystallization with no cross-link traversal.

---

## Root cause (evidence)

| Symptom | Choke point |
|---------|-------------|
| Approve auto-project skips Graphiti when only `GRAPHITI_ADAPTER_URL` is set | `_projection_config()` uses `GRAPHITI_URL` only; `_graphiti()` falls back to `GRAPHITI_ADAPTER_URL` |
| No runtime proof | `scripts/smoke_memory_crystallization_e2e.sh` stops at active-packet; never calls `/api/memory/graphiti/*` |
| No adapter gate tests | `services/orion-graphiti-adapter/tests/` does not exist |
| Neighborhood returns only seed crystallization | `upsert_episode()` writes entity→episode only; `memory_crystallization_links` never projected |
| Not real Graphiti library | `orion-graphiti-adapter` is Orion-owned Postgres + optional FalkorDB Cypher |

Canonical crystallizations remain correct in Postgres. The projection rail is the gap.

---

## Goals

- **Phase A:** Graphiti rail is live on approve with default `.env_example` settings; smoke proves sync + neighborhood.
- **Phase B:** Cross-crystallization edges from `memory_crystallization_links` flow into projection and multi-hop retrieval.
- **Phase C:** `graphiti-core` backend behind feature flag provides hybrid retrieval using **prescribed ontology** (no LLM re-extraction of crystallization text).

## Non-goals

- Replacing RDF `/api/memory/graph/*`
- Context-exec writing Graphiti, Chroma, RDF, or SQL
- Zep managed platform
- LLM-based entity extraction from raw conversation text (crystallizations are already governed structured memory)
- Neo4j as a new required dependency (FalkorDB already in adapter compose profile)
- Keyword cathedrals or new cognitive taxonomies beyond existing `CrystallizationRelation` literals

---

## Architecture (stable contract)

```text
Hub project_crystallization / retrieve_active_packet
        │
        ▼
GraphitiAdapter  (orion/memory/crystallization/projection_graphiti.py)
        │  POST /v1/episodes  {crystallization fields + links[]}
        │  GET  /v1/neighborhood/{id}?depth=N
        │  POST /v1/search      (Phase C only)
        ▼
orion-graphiti-adapter
        ├── GRAPHITI_BACKEND=orion_postgres   (Phase A–B)
        └── GRAPHITI_BACKEND=graphiti_core      (Phase C, FalkorDB driver)
```

**Invariant:** `canonical_mutated` is always `false`. Projection only updates `projection_refs` on the crystallization record.

**Choke points:**

| File | Role |
|------|------|
| `services/orion-hub/scripts/crystallization_routes.py` | `_graphiti()`, `_projection_config()`, Hub API routes |
| `orion/memory/crystallization/projection_graphiti.py` | `GraphitiAdapter` HTTP client |
| `orion/memory/crystallization/projector.py` | `project_crystallization()` multi-rail project |
| `orion/memory/crystallization/retriever.py` | `retrieve_active_packet()` graphiti neighborhood/search fusion |
| `services/orion-graphiti-adapter/app/main.py` | Episode ingest, neighborhood, search, rebuild |
| `services/orion-graphiti-adapter/app/store.py` | Postgres projection tables |

---

## Design pattern: sequential vertical slices

Each phase ships a complete, testable rail before the next starts. Three PRs, three gates.

```text
Phase A (PR 1)  → smoke green; shelf dusted off
Phase B (PR 2)  → links + multi-hop; graph semantically useful
Phase C (PR 3)  → graphiti-core backend; hybrid retrieval
```

Hub and retriever contract freeze after Phase A except: passing `links` in episode payload (B) and optional search rail (C).

---

## Phase A — Operational hardening

### A1. URL unification

Add `resolve_graphiti_adapter_url(settings) -> str` in `orion/memory/crystallization/projection_graphiti.py` (or a thin `orion/memory/crystallization/graphiti_config.py` if import cycles arise):

```python
def resolve_graphiti_adapter_url(settings) -> str:
    return (
        getattr(settings, "GRAPHITI_ADAPTER_URL", "") or
        getattr(settings, "GRAPHITI_URL", "") or
        ""
    ).strip()
```

Use in both `_graphiti()` and `_projection_config()` in `crystallization_routes.py`.

- `graphiti_enabled`: `bool(GRAPHITI_ENABLED) or bool(resolve_graphiti_adapter_url(s))`
- `graphiti_url`: `resolve_graphiti_adapter_url(s)`

Keep `GRAPHITI_URL` as deprecated fallback in `.env_example` comments; primary key remains `GRAPHITI_ADAPTER_URL`.

### A2. Adapter unit tests

Create `services/orion-graphiti-adapter/tests/`:

| Test | Assert |
|------|--------|
| `test_ingest_episode_returns_ids` | POST `/v1/episodes` → episode/entity/edge IDs; `canonical_mutated: false` |
| `test_neighborhood_after_ingest` | Ingest then GET neighborhood → nodes ≥ 1 |
| `test_health_without_postgres` | Health reports `postgres: false` when pool unset |

Use testcontainers or mocked asyncpg pool per service conventions; prefer in-process FastAPI TestClient with patched pool if that's the local pattern.

### A3. Hub contract regression test

`services/orion-hub/tests/test_graphiti_config_parity.py`:

- When only `GRAPHITI_ADAPTER_URL` is set, `_projection_config().graphiti_url` equals `_graphiti(mock_request).url`.

### A4. Smoke extension

Extend `scripts/smoke_memory_crystallization_e2e.sh` after approve:

1. `GET /api/memory/crystallizations/projection/health` — log `graphiti_enabled`
2. `POST /api/memory/graphiti/sync/{CID}` — assert HTTP 200
3. When adapter expected up: `graphiti.episode_ids` non-empty in response
4. `GET /api/memory/graphiti/neighborhood/{CID}` — `nodes` length ≥ 1
5. `GET /api/memory/graphiti/health` — `enabled: true`, `url_configured: true`

Smoke steps 2–4 are **skipped with WARN** (not FAIL) when `GRAPHITI_SKIP=1` or health shows adapter disabled — allows crystallization smoke without adapter container. Default run with adapter up must PASS.

### Phase A acceptance

- [ ] `pytest services/orion-graphiti-adapter/tests -q` passes
- [ ] Hub graphiti config parity test passes
- [ ] `scripts/smoke_memory_crystallization_e2e.sh` PASS with adapter container running
- [ ] Approve response `projection.graphiti` non-empty when `GRAPHITI_ENABLED=true` and adapter URL set

---

## Phase B — Richer Orion-owned graph

### B1. Link projection on ingest

Extend adapter `EpisodeIngestV1`:

```python
class CrystallizationLinkIngestV1(BaseModel):
    target_crystallization_id: str
    relation: str  # CrystallizationRelation literal values
    confidence: float = 0.5

class EpisodeIngestV1(BaseModel):
    # ... existing fields ...
    links: list[CrystallizationLinkIngestV1] = Field(default_factory=list)
```

Hub `GraphitiAdapter.sync_crystallization(_async)` includes:

```python
"links": [
    {"target_crystallization_id": l.target_crystallization_id, "relation": l.relation, "confidence": l.confidence}
    for l in crystallization.links
]
```

`upsert_episode()` in `store.py`:

- For each link, ensure target entity node exists (stub entity if target not yet ingested)
- Insert `graphiti_edges` with `edge_id = ged_{from}_{to}_{relation}`, `relation` from link
- Store `confidence` and `note` in edge metadata

### B2. Multi-hop neighborhood

- `GET /v1/neighborhood/{crystallization_id}?depth=1|2` (default 1, max 2 for v1)
- BFS over `graphiti_edges` from seed entity/episode nodes
- Return merged `nodes` and `edges` arrays (same shape as today for retriever compat)

Retriever (`retriever.py`):

- Pass `depth=2` query param when calling neighborhood
- Collect `crystallization_id` from all returned nodes into `graphiti_refs` and `extra_crystallization_ids`

### B3. FalkorDB link sync

In `falkordb.py`, extend `sync_to_falkordb()`:

- Accept optional `links` list
- MERGE link edges between entity nodes using relation as edge type (sanitized)
- Keep best-effort semantics; failures logged, do not fail Postgres ingest

Document in `services/orion-graphiti-adapter/README.md`:

```bash
docker compose --profile falkordb -f services/orion-graphiti-adapter/docker-compose.yml up -d
```

`FALKORDB_ENABLED` remains `false` in `.env_example` default; operator opts in.

### B4. UI (minimal)

`services/orion-hub/static/js/memory-crystallization-ui.js`:

- Show `projection_refs.graphiti_episode_ids.length` (and edge count) in detail panel
- Optional button: "Sync Graphiti" → `POST /api/memory/graphiti/sync/{id}`

### Phase B acceptance

- [ ] Adapter test: ingest two crystallizations with `supports` link; neighborhood from seed returns both crystallization IDs at depth 2
- [ ] Hub test: sync payload includes links from crystallization model
- [ ] Smoke variant or focused script: two-crystallization link scenario PASS
- [ ] Active-packet with seed returns linked crystallization in `graphiti_refs`

---

## Phase C — Real Graphiti (`graphiti-core`)

### C1. Backend selection

Add to `services/orion-graphiti-adapter/app/settings.py`:

```python
GRAPHITI_BACKEND: Literal["orion_postgres", "graphiti_core"] = "orion_postgres"
```

Route handlers delegate to backend module:

- `backends/orion_postgres.py` — current `store.py` logic (Phase A–B)
- `backends/graphiti_core.py` — `graphiti-core` FalkorDB driver

Rollback: set `GRAPHITI_BACKEND=orion_postgres` and restart adapter.

### C2. Prescribed ontology (no LLM extraction)

Crystallizations are already governed memory. Phase C maps them directly:

| Orion | Graphiti |
|-------|----------|
| `MemoryCrystallizationV1` | Episode node with bi-temporal metadata from `created_at`, `updated_at`, governance timestamps |
| `subject` | Entity name |
| `CrystallizationLinkV1.relation` | Typed edge between entity nodes |
| `projection_refs` | Provenance pointer only; not written into Graphiti as authority |

Do **not** call Graphiti conversation-ingest APIs that run LLM entity extraction on `summary` text. Use graphiti-core driver write APIs with explicit node/edge payloads.

Dependencies: add `graphiti-core[falkordb]` to adapter `requirements.txt`; pin version in same changeset.

### C3. Hybrid retrieval endpoint

`POST /v1/search`:

```json
{
  "query": "memory architecture",
  "seed_crystallization_id": "crys_...",
  "limit": 10
}
```

Response:

```json
{
  "crystallization_ids": ["crys_...", "crys_..."],
  "trace": {"backend": "graphiti_core", "rails": ["vector", "graph"]}
}
```

Embed query via existing Orion embed HTTP (`CRYSTALLIZER_EMBED_HOST_URL` passed from Hub or configured on adapter).

Retriever: when adapter backend is `graphiti_core` (reported in `/health`), call `/v1/search` instead of `/v1/neighborhood` for semantic expansion; still use neighborhood for explicit seed expansion.

### C4. Backfill / rebuild

`POST /v1/rebuild` on adapter:

- Reads active crystallizations from Hub-provided batch or adapter pulls via Hub internal API (prefer Hub-initiated rebuild calling adapter per-item to keep canonical source in Hub Postgres)
- Hub `/api/memory/crystallizations/projection/rebuild` unchanged; each item triggers adapter ingest

### C5. Privacy

Skip or redact crystallizations with `governance.sensitivity == "intimate"` from FalkorDB/Graphiti search payloads. Log skip count in projection trace. `private` and `public` project normally.

### Phase C acceptance

- [ ] With `GRAPHITI_BACKEND=graphiti_core` and FalkorDB profile up, search returns seed + linked crystallization
- [ ] `canonical_mutated` remains false on all paths
- [ ] Rollback to `orion_postgres` restores Phase B behavior without data loss in canonical Postgres
- [ ] Eval or smoke documents `GRAPHITI_BACKEND` flag and skip path when FalkorDB unavailable

---

## Error handling

| Failure | Behavior |
|---------|----------|
| Adapter down on project | Log warning; `ProjectionResult.errors` append `graphiti_projection_failed:*`; approve succeeds |
| Adapter down on retrieval | Empty `graphiti_refs`; trace notes `graphiti_unavailable` |
| FalkorDB sync failure | Postgres projection succeeds; `falkordb.synced: false` in response |
| Invalid link target | Create stub entity node; edge still written |

---

## Env / config changes (by phase)

### Phase A

| Key | Change |
|-----|--------|
| `GRAPHITI_ADAPTER_URL` | Primary; documented as required when `GRAPHITI_ENABLED=true` |
| `GRAPHITI_URL` | Deprecated fallback comment |

### Phase B

| Key | Change |
|-----|--------|
| `FALKORDB_ENABLED` | Documented optional profile; not default-on |

### Phase C

| Key | Change |
|-----|--------|
| `GRAPHITI_BACKEND` | `orion_postgres` (default) \| `graphiti_core` |
| `CRYSTALLIZER_EMBED_HOST_URL` | Required on adapter when `graphiti_core` (or adapter-local embed URL) |

Run `python scripts/sync_local_env_from_example.py` after each phase's `.env_example` changes.

---

## Testing matrix

| Layer | Phase A | Phase B | Phase C |
|-------|---------|---------|---------|
| Adapter unit | ingest, neighborhood, health | link edges, depth-2 BFS | backend switch, prescribed ingest, search |
| Hub contract | URL parity | links in sync payload | retriever uses search when backend=graphiti_core |
| Root tests | `test_graphiti_cannot_mutate_canonical` (existing) | link projection refs | search rail in active-packet trace |
| Smoke | sync + neighborhood | two-crys link script | optional `GRAPHITI_BACKEND=graphiti_core` profile |

---

## Restart required (per phase)

```bash
# Phase A–B
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-graphiti-adapter/.env \
  -f services/orion-graphiti-adapter/docker-compose.yml up -d --build

# Phase C (add FalkorDB profile when testing graphiti_core)
docker compose --profile falkordb \
  --env-file .env --env-file services/orion-graphiti-adapter/.env \
  -f services/orion-graphiti-adapter/docker-compose.yml up -d --build
```

---

## Risks

| Severity | Risk | Mitigation |
|----------|------|------------|
| Medium | Phase C `graphiti-core` API drift | Pin version; adapter backend module isolates churn |
| Low | FalkorDB ops burden | Optional profile; Postgres backend remains default |
| Low | Dual neighborhood implementations (`links.py` vs Graphiti) | Document: canonical links in Postgres; Graphiti is projection; B syncs links on ingest |
| Medium | Embed dependency for Phase C search | Reuse existing crystallizer embed path; fail soft on retrieval |

---

## Recommended implementation order

1. **PR 1 (Phase A):** URL unification, adapter tests, hub parity test, smoke extension
2. **PR 2 (Phase B):** Link payload, multi-hop neighborhood, FalkorDB link sync, UI counts
3. **PR 3 (Phase C):** Backend flag, graphiti-core module, search endpoint, privacy skip, rebuild wiring

Invoke `writing-plans` skill after spec approval to produce `docs/superpowers/plans/2026-07-06-graphiti-rail-activation.md`.
