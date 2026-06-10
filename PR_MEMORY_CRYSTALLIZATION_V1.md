# Memory Crystallization v1 — `MemoryCrystallizationV1` governed cognitive memory

## Summary

Adds the governed cognitive memory layer from the Orion Cognitive Memory Core spec as a thin, inspectable vertical slice:

```text
MemoryCardV1 stays the turn-facing recall artifact.
GrammarEventV1 stays the substrate trace artifact.
MemoryCrystallizationV1 is the new governed cognitive memory artifact.
Chroma stays semantic recall projection (via existing vector path).
RDF memory_graph stays the approved graph path (untouched).
Graphiti/FalkorDB enters as an additive, disabled-by-default temporal projection.
Postgres preserves canonical crystallizations.
The governor decides. The user inspects.
```

## What's in this PR

### 1. Schemas (`orion/schemas/memory_crystallization.py`)

- `MemoryCrystallizationV1` — kind (`semantic|episode|procedure|stance|open_loop|contradiction|attractor|decision|failure_mode`), status lifecycle (`proposed → active|rejected|quarantined`, then `superseded|deprecated|archived`), claims, evidence refs, links, grammar envelope, governance block, projection refs.
- `MemoryGrammarEnvelopeV1` — memory-local adornment that *references* existing `GrammarEventV1`/`GrammarAtomV1`/`GrammarEdgeV1` ids. No competing grammar law layer.
- `ActiveMemoryPacketV1` — compact retriever output grouped by stance / project_state / procedures / open_loops / contradictions / warnings / attractors.
- Registered in `orion/schemas/registry.py` (`MemoryCrystallizationV1`, `ActiveMemoryPacketV1`).

### 2. Bus channels (`orion/bus/channels.yaml`)

Seven new channels, no existing channels renamed:

```text
orion:memory:crystallization:proposed     memory.crystallization.proposed.v1
orion:memory:crystallization:validated    memory.crystallization.validated.v1
orion:memory:crystallization:approved     memory.crystallization.approved.v1
orion:memory:crystallization:rejected     memory.crystallization.rejected.v1
orion:memory:crystallization:quarantined  memory.crystallization.quarantined.v1
orion:memory:crystallization:project      memory.crystallization.project.v1
orion:memory:crystallization:retrieved    memory.crystallization.retrieved.v1  (ActiveMemoryPacketV1)
```

Existing `orion:memory:vector:upsert` is reused as-is for the Chroma projection.

### 3. Core package (`orion/memory/crystallization/`)

- `validator.py` — spec §22 rules: non-empty evidence/scope/summary, stance/procedure/decision require `planning_effects`, stance requires manual review, contradiction requires ≥2 distinct targets, proposals must not carry projection refs.
- `governor.py` — pure transitions (`validate`, `approve`, `reject`, `quarantine`, `supersede`, `set_status`) returning new artifacts + audit history entries. Only `approve()` reaches `active`; supersession preserves the old artifact and adds an explicit `supersedes` link; contradictions are represented, not erased.
- `salience.py` — deterministic kind/evidence/confidence prior (operator can override at approval).
- `proposer.py` — proposal assembly from memory cards / grammar events; output is always `proposed` + unvalidated.
- `projection_cards.py` — only `active` (and explicitly marked `superseded`) crystallizations project to `MemoryCardV1` with `subschema.crystallization_ref`; rejected/quarantined/proposed raise.
- `projection_chroma.py` — `VectorDocumentUpsertV1` builder (collection `orion_memory_crystallizations`, kind `memory.crystallization`); active-only; Chroma is rebuildable, Postgres wins.
- `projection_graphiti.py` — additive temporal episode builder; pre-canonical inputs labeled `canonical: false`; the adapter returns updated projection refs and can never mutate canonical state.
- `active_packet.py` — packet assembly; non-active artifacts are excluded and the exclusion is traced.
- `repository.py` + `sql/memory_crystallizations.sql` — idempotent DDL (applied at boot, memory_cards idiom) for `memory_crystallizations`, `_claims`, `_sources`, `_links`, `_history`, `_retrieval_events`. Governance writes are transactional (status change + audit row commit together) with optimistic status guards so stale writers can never revert governed state. Projection refs merge via a row-locked targeted update, not a doc clobber.

### 4. Service (`services/orion-memory-crystallizer/`, port 8634)

FastAPI worker following the graph-compression template (Dockerfile from repo root, per-service requirements, external `app-net`, `.env_example` beside service):

```http
POST  /api/memory/crystallizations/propose
GET   /api/memory/crystallizations/proposals[/{id}]
POST  /api/memory/crystallizations/proposals/{id}/validate|approve|reject|quarantine
GET   /api/memory/crystallizations[/{id}][/history][/links]
POST  /api/memory/crystallizations/{id}/status|supersede|links
POST  /api/memory/crystallizations/{id}/project/card|chroma|graphiti
GET   /api/memory/graphiti/health
POST  /api/memory/active-packet
GET   /api/memory/retrieval-events/{id}
```

- Bus ingest worker on the proposed channel: validates, persists, refuses to overwrite governed state, skips its own echoes.
- All psycopg2 calls run via `asyncio.to_thread`; card projection uses the existing async `insert_card` DAL when a Postgres pool is available.
- Graphiti routes live under `/api/memory/graphiti/*` — the RDF `/api/memory/graph/*` stack is untouched.
- Lifecycle changes (quarantine-from-active, supersede, deprecate/archive) publish on the `project` channel so derived surfaces can refresh or retract.

### 5. Wiring

- `README.md` — service added to memory services list + §14 inventory.
- `scripts/sync_local_env_from_example.py` — `CRYSTALLIZER_`/`GRAPHITI_`/`FALKORDB_` prefixes + service in defaults; sync run.
- `tests/test_memory_crystallization_bus_catalog.py` — registry/channel contract test (goals-catalog pattern), including a guard that `orion:memory:vector:upsert` is unchanged.

## Non-goals (deliberate)

- No `MemoryCardV2`; `orion/core/contracts/memory_cards.py` untouched.
- No new grammar reducer/organ; grammar artifacts are referenced by id only.
- No pgvector; Chroma via the existing `memory.vector.upsert.v1` path.
- No RDF memory_graph changes; RDF projection is deferred (refs supported in schema).
- No Observatory UI yet; the API exposes everything an inbox/detail view needs.
- Graphiti/FalkorDB deployment itself (adapter is config-gated, `GRAPHITI_ENABLED=false`).

## Spec deviations worth knowing

- `crystallization_id` is `text` (`crys_<hex>`), not `uuid`, matching the Python contract.
- A lossless `doc jsonb` column backs round-trips; claims/sources/links tables are queryable projections rebuilt per write.
- `memory_crystallization_projection_refs`/`_quarantine` tables from §11.1 are folded into the `projection_refs` jsonb column and the `quarantined` status respectively.
- Proposed (pre-canonical) artifacts may project to Graphiti per spec §10.3, always labeled `canonical: false`.

## Verification

```text
PYTHONPATH=. ./venv/bin/python -m pytest services/orion-memory-crystallizer/tests tests/test_memory_crystallization_bus_catalog.py -q
→ 55 passed

PYTHONPATH=. ./venv/bin/python -m pytest tests/test_channel_prefix_guardrail.py tests/test_autonomy_goals_bus_catalog.py tests/test_recall_diagnostic_schema_registry.py -q
→ 10 passed (existing catalog guardrails intact)

python -m compileall orion/memory services/orion-memory-crystallizer → OK
FastAPI app imports; route table verified (proposals routes precede /{cid})
```

Covered invariants: proposal cannot become active without the governor; stance requires evidence + planning_effects; contradiction requires ≥2 targets; rejected/quarantined never project; superseded card projection carries an explicit marker; Chroma payload is a valid `VectorDocumentUpsertV1`; Graphiti adapter cannot mutate canonical state; supersession preserves the old artifact; worker never overwrites governed state; packet excludes and traces non-active artifacts.

## Remaining risks

- DDL and repository SQL have not run against a live Postgres in this environment (no DB available); they execute at service boot via `apply_memory_crystallizations_schema` and follow the proven memory_cards idiom, but live-stack verification is pending: **UNVERIFIED** for the live runtime path.
- Chroma projection publishes without embeddings unless `CRYSTALLIZER_EMBED_HOST_URL` is set (vector-writer skips embedding-less upserts by design).
- De-projection is event-driven only (project channel); no consumer yet rebuilds stale cards/Chroma docs automatically.

## Open design questions (tracked as spec §25 open loops, not guessed)

- Should RecallService query crystallizations directly or only via card projection?
- Graphiti ingest: proposed + active (current, labeled) or active-only?
- Which kinds should auto-approve under policy vs require manual review (currently all manual)?
- Where does the Observatory live (Hub tab vs separate service)?
