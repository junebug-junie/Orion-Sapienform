# Orion Memory Cards v1 — Design Spec

**Date:** 2026-05-01  
**Status:** Approved, ready for implementation planning  
**Branch:** `feat/memory-cards-v1` (to be created from `cognitive_substrate/v4`)  
**Author:** Juniper + Oríon brainstorming session

---

## 1. Purpose

Memory Cards is a persistent, operator-curated fact store that gives Oríon durable, structured knowledge about Juniper that survives session boundaries. It solves two problems the existing recall stack cannot:

1. **Zero-latency always-on facts** — high-priority cards (e.g., "Lives in Ogden, UT") are prepended to every prompt without waiting for a retrieval query.
2. **Structured biographical memory** — cards carry typed fields (anchor class, temporal horizon, confidence, visibility scope) that the existing vector/SQL/RDF backends cannot represent.

Cards are data, not prompt/identity mutations. The auto-extractor ships disabled by default.

---

## 2. Ground Rules

These apply across all phases and cannot be overridden by phase-level decisions:

- **No production default changes.** `RECALL_DEFAULT_PROFILE` and all 12 existing profiles are untouched except `self.factual.v1.yaml` (weight rewire described in Phase 1). Nothing else.
- **No autonomous prompt/identity/personality mutation surface.** Cards are data records.
- **Auto-extractor defaults off.** `ORION_AUTO_EXTRACTOR_ENABLED=false`.
- **Stage 2 LLM extraction not in v1.** Deferred to v1.5. Setting raises `NotImplementedError` if enabled.
- **No new external runtime deps.** Reuse existing FastAPI / asyncpg / psycopg2 / chromadb / pydantic-v2 / tailwind-CDN / Jinja stack.
- **Test discipline.** Every code-bearing phase ends with `pytest` green + manual smoke. No phase advances if acceptance criteria fail.

---

## 3. Architecture

### Shared library pattern

Memory Cards follows the existing `orion/` shared-library pattern. The DAL and contracts live in the orion root package, importable by all services — same as `orion.core.contracts.*`, `orion.core.bus.*`, `orion.cognition.*`.

```
orion/core/storage/          ← new package (create __init__.py)
    memory_cards.py          ← async DAL (asyncpg) + apply_memory_cards_schema()
    memory_extraction.py     ← pure Stage 1 regex extractor, no I/O

orion/core/contracts/
    memory_cards.py          ← pydantic v2 models (new file)
    recall.py                ← existing; RecallQueryV1 gains lane + profile_explicit
```

### Services touched

| Phase | Service | Change type |
|---|---|---|
| 0 | orion package | New package + SQL DDL |
| 1 | orion-recall | Cards backend + new profile + self.factual rewire |
| 2 | orion-recall | Intent routing + vector fan-out + soft truncation |
| 3 | orion-cortex-orch | Always-on injection |
| 4 | orion-hub | API routes + Memory tab UI |
| 5 | orion-cortex-orch, orion-recall | Auto-extractor + operator distiller |
| 6 | — | Safety verification + docs |

### Database connection

All three consumers (recall, cortex-orch, hub) share `RECALL_PG_DSN`, pointing at the existing `conjourney` Postgres database. No new database.

### Lane: visibility scoping

`RecallQueryV1` gains `lane: Optional[str] = None` (backward-compatible). The orchestrator (`orion-cortex-orch`) sets this explicitly from request context before dispatching to recall. Social-room requests → `lane="social"`; standard chat → `lane="chat"`; other contexts → `None` (all non-sensitive cards pass through).

Cards declare `visibility_scope text[]` (e.g., `['all']`, `['chat', 'intimate']`). Filter logic: include card iff `'all' in visibility_scope OR lane in visibility_scope OR lane is None`.

---

## 4. Data Model

### `memory_cards` table

```sql
CREATE TABLE IF NOT EXISTS memory_cards (
    card_id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    slug             text UNIQUE NOT NULL,
    types            text[] NOT NULL,
    anchor_class     text,
    status           text NOT NULL DEFAULT 'pending_review',
    confidence       text NOT NULL DEFAULT 'likely',
    sensitivity      text NOT NULL DEFAULT 'private',
    priority         text NOT NULL DEFAULT 'episodic_detail',
    visibility_scope text[] NOT NULL DEFAULT '{chat}',
    time_horizon     jsonb,
    provenance       text NOT NULL,
    trust_source     text,
    project          text,
    title            text NOT NULL,
    summary          text NOT NULL,
    still_true       text[],
    anchors          text[],
    tags             text[],
    evidence         jsonb NOT NULL DEFAULT '[]'::jsonb,
    subschema        jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now()
);
```

### `memory_card_edges` table

```sql
CREATE TABLE IF NOT EXISTS memory_card_edges (
    edge_id      uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    from_card_id uuid NOT NULL REFERENCES memory_cards(card_id) ON DELETE CASCADE,
    to_card_id   uuid NOT NULL REFERENCES memory_cards(card_id) ON DELETE CASCADE,
    edge_type    text NOT NULL,
    metadata     jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (from_card_id, to_card_id, edge_type)
);
```

### `memory_card_history` table

```sql
CREATE TABLE IF NOT EXISTS memory_card_history (
    history_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    card_id    uuid REFERENCES memory_cards(card_id) ON DELETE SET NULL,
    edge_id    uuid REFERENCES memory_card_edges(edge_id) ON DELETE SET NULL,
    op         text NOT NULL,
    actor      text NOT NULL,
    before     jsonb,
    after      jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);
```

### Indices

```sql
CREATE INDEX IF NOT EXISTS idx_mc_anchors  ON memory_cards USING GIN (anchors);
CREATE INDEX IF NOT EXISTS idx_mc_tags     ON memory_cards USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_mc_types    ON memory_cards USING GIN (types);
CREATE INDEX IF NOT EXISTS idx_mc_status   ON memory_cards (status);
CREATE INDEX IF NOT EXISTS idx_mc_priority ON memory_cards (priority);
CREATE INDEX IF NOT EXISTS idx_mc_prov     ON memory_cards (provenance);
CREATE INDEX IF NOT EXISTS idx_mce_from    ON memory_card_edges (from_card_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_mce_to      ON memory_card_edges (to_card_id, edge_type);
```

### Enum values

**status:** `pending_review`, `active`, `rejected`, `superseded`, `archived`, `deprecated`  
**confidence:** `certain`, `likely`, `possible`, `uncertain`  
**sensitivity:** `public`, `private`, `intimate`  
**priority:** `always_inject`, `high_recall`, `episodic_detail`, `archival`  
**provenance:** `operator_highlight`, `operator_distiller`, `auto_extractor`, `imported`  
**anchor_class:** `person`, `place`, `project`, `event`, `concept`, `relationship`, `health`, `preference`, `belief`  
**edge_type (17):** `relates_to`, `contradicts`, `supersedes`, `supports`, `parent_of`, `child_of`, `precedes`, `follows`, `co_occurs_with`, `derived_from`, `evidence_for`, `evidence_against`, `tagged_as`, `instance_of`, `example_of`, `analogy_of`, `associated_with`  
**op (history):** `create`, `update`, `status_change`, `edge_add`, `edge_remove`, `reverse_auto_promotion`

---

## 5. Contracts (`orion/core/contracts/memory_cards.py`)

Pydantic v2, `extra="forbid"` throughout.

**`TimeHorizonV1`**: `kind: Literal["timeless", "era_bound", "current", "expiring"]`, `start: Optional[str]`, `end: Optional[str]`, `as_of: Optional[str]`. Cross-validator: `era_bound` requires `start`.

**`EvidenceItemV1`**: `source: str`, `excerpt: Optional[str]`, `ts: Optional[str]`.

**`MemoryCardV1`**: all fields above as typed pydantic fields. `types` validated against the allowed token set. `time_horizon: Optional[TimeHorizonV1]`. `evidence: list[EvidenceItemV1]`. `subschema: dict[str, Any]` (open). Cross-validator: `anchor_class` required iff `'anchor' in types`.

**`MemoryCardEdgeV1`**: `edge_id`, `from_card_id`, `to_card_id`, `edge_type: Literal[...17 values]`, `metadata: dict[str, Any]`.

**`MemoryCardHistoryEntryV1`**: `history_id`, `card_id`, `edge_id`, `op: Literal[...6 values]`, `actor`, `before: Optional[dict]`, `after: Optional[dict]`, `created_at`.

**`RecallQueryV1` amendments** (in `orion/core/contracts/recall.py`):
- Add `lane: Optional[str] = None` — visibility scope selector, set by orchestrator
- Add `profile_explicit: bool = False` — when True, intent router skips profile override

Both fields default to backward-compatible values; no caller changes required.

---

## 6. Shared DAL (`orion/core/storage/memory_cards.py`)

```python
# Schema bootstrap
def apply_memory_cards_schema(dsn: str) -> None: ...  # psycopg2, idempotent

# Card CRUD
async def insert_card(pool, card: MemoryCardV1, *, actor: str, op: str = "create") -> uuid
async def update_card(pool, card_id, patch: dict, *, actor: str) -> MemoryCardV1
async def get_card(pool, card_id_or_slug: str) -> MemoryCardV1 | None
async def list_cards(pool, *, status=None, types=None, anchor_class=None,
                     project=None, priority=None, limit=200, offset=0) -> list[MemoryCardV1]

# Edge CRUD
async def add_edge(pool, edge: MemoryCardEdgeV1, *, actor: str) -> uuid
async def remove_edge(pool, edge_id: str, *, actor: str) -> None
async def list_edges(pool, *, card_id: str,
                     direction: Literal["out","in","both"] = "both") -> list[MemoryCardEdgeV1]

# History
async def list_history(pool, *, card_id=None, edge_id=None, limit=500) -> list[MemoryCardHistoryEntryV1]
async def reverse_history(pool, history_id: str, *, actor: str) -> MemoryCardHistoryEntryV1
```

All writes record a history row in the same transaction. `add_edge` runs BFS cycle detection for `parent_of`/`child_of` before inserting — documented as best-effort, raises `ValueError` on detected cycle. Pool helper lazily reads `RECALL_PG_DSN`.

---

## 7. Phase 1 — Recall Integration

### Cards adapter (`services/orion-recall/app/cards_adapter.py`)

```python
async def fetch_card_fragments(
    fragment: str,
    profile: dict,
    *,
    lane: str | None,
) -> list[dict]
```

**Scoring** against `fragment` (note: field was `.text` in older plan — corrected here):
- Anchor token match: +2.0
- Title token match: +1.0
- Summary token match: +0.5
- Tag match: +0.3

**Status filter:** exclude `rejected`, `archived`, `deprecated`.  
**Visibility filter:** `'all' in visibility_scope OR lane in visibility_scope OR lane is None`.  
**1-hop expansion:** for each top-K card, fetch outgoing `relates_to`, `child_of`, `supports` edges where `metadata->>'confidence' >= 'likely'`; neighbors scored at 0.5× parent score, capped at `RECALL_CARDS_MAX_NEIGHBORS`, deduplicated.

Rendered fragment format: `"[card:{anchor_class or types[0]}] {title} — {summary}"`.

### Fusion wiring (`services/orion-recall/app/fusion.py`)

Add `"cards": 0.0` to `DEFAULT_BACKEND_WEIGHTS`. No change to `TRANSCRIPT_SOURCES`.

### Worker wiring (`services/orion-recall/app/worker.py`)

`_query_backends` gains `lane: str | None` parameter (threaded from `process_recall` via `q.lane`). Cards fetch triggered when `profile["cards_top_k"] > 0 OR backend_weights.get("cards", 0) > 0`. Timeout-guarded by `RECALL_CARDS_TIMEOUT_SEC`; on timeout returns empty list, logs warning, does not fail the request.

### New settings (`services/orion-recall/app/settings.py`)

```python
RECALL_ENABLE_CARDS: bool = False           # env RECALL_ENABLE_CARDS
RECALL_CARDS_TIMEOUT_SEC: float = 0.250    # env RECALL_CARDS_TIMEOUT_SEC
RECALL_CARDS_MAX_NEIGHBORS: int = 6        # env RECALL_CARDS_MAX_NEIGHBORS
```

### New profile (`orion/recall/profiles/biographical.v1.yaml`)

```yaml
profile: biographical.v1
vector_top_k: 4
rdf_top_k: 0
sql_top_k: 0
cards_top_k: 12
max_per_source: 8
max_total_items: 12
render_budget_tokens: 280
backend_weights:
  cards: 1.5
  vector: 0.3
  sql_timeline: 0.0
  sql_chat: 0.0
  rdf_chat: 0.0
  rdf: 0.0
score_weight: 0.7
text_similarity_weight: 0.2
recency_weight: 0.0
enable_recency: false
```

> Note for implementation: Verify that `score_weight`, `text_similarity_weight`, `recency_weight`, and `enable_recency` are read by the fusion code before including them. If not present in the current fusion reader, omit them. `backend_weights` at top level is confirmed required — matches the existing profile schema (`self.factual.v1.yaml`).

### `self.factual.v1.yaml` rewire

Only two values change:
- `backend_weights.rdf: 0.4` (was `1.2`)
- Add `backend_weights.cards: 1.5`

All other fields untouched. This is the **only** existing profile file modified across all six phases.

---

## 8. Phase 2 — Recall Engine Improvements

These benefit all profiles, not just cards.

### Intent router (`services/orion-recall/app/intent.py`)

```python
class IntentClassification(BaseModel):
    intent: Literal["biographical","episodic_recent","episodic_historical",
                    "factual_project","associative","reflective","unknown"]
    confidence: float
    rationale: str

def classify_intent_v1(query_text: str) -> IntentClassification: ...
def resolve_profile_for_intent(intent: str, *, fallback_profile: str) -> str: ...
```

Stage 1 only (keyword/regex tables). Wired into `process_recall` before `get_profile(q.profile)`. Skipped when `q.profile_explicit is True`. Emits `recall.intent.v1` telemetry envelope (hashed query text, classified intent, selected profile, override flag). New setting: `RECALL_INTENT_ROUTING_ENABLED: bool = True`.

### Vector fan-out (`services/orion-recall/app/storage/vector_adapter.py`)

Replace sequential per-collection loop with `asyncio.gather` (each collection wrapped in `asyncio.to_thread` since Chroma client is sync). Embedding computed once, reused across collections. Per-collection top-K = `max(1, vector_top_k // n_collections)`. Deduplicate on `(collection, doc_id)` then content fingerprint. New env: `RECALL_VECTOR_EXCLUDE_COLLECTIONS` (regex string). Each fragment tagged `meta.collection`.

### Render budget soft truncation

`render_items` return type changes from `str` to `tuple[str, list[dict]]`. **Three callers updated:** `fusion.py:592`, `service.py`, `recall_v2.py`. When `diagnostic=True`, dropped items written to `MemoryBundleStatsV1.diagnostic["budget_dropped"]` (already an open `dict[str, Any]`, no contract change). When `diagnostic=False` and N > 0, appends `"\n[+N more items dropped due to budget]"` to rendered text. Controlled by `RECALL_RENDER_BUDGET_INDICATOR: bool = True`.

---

## 9. Phase 3 — Always-On Injection

### `services/orion-cortex-orch/app/memory_inject.py`

```python
async def fetch_always_inject_block(
    *,
    lane: str,
    token_budget: int,
    pool,
) -> str:
```

Queries `priority='always_inject' AND status='active'` filtered by `visibility_scope @> '{all}' OR lane = ANY(visibility_scope)`. Orders by `updated_at DESC` (freshest curated state first). Renders bullet list: `"- {summary} ({anchor_class or types[0]}, {time_horizon hint if present})"`. Greedy fill until token budget (word count × 1.3 estimate). Wraps in `"[Known facts about Juniper]\n{bullets}\n"`. Returns `""` if no cards or pool unavailable. Internal timeout 250ms; on timeout returns `""` and logs warning.

### Wiring into `conversation_front.py`

`context["known_facts_block"]` added from `fetch_always_inject_block(lane=lane, token_budget=settings.ALWAYS_INJECT_TOKEN_BUDGET, pool=pool)`. Lane comes from the same request context the orchestrator already derives. Prompt templates render `known_facts_block` above `memory_digest`. **Key invariant:** when `known_facts_block == ""`, prompt is byte-identical to pre-Phase-3 baseline (snapshot-tested).

### New settings (`services/orion-cortex-orch/app/settings.py`)

```python
ORION_ALWAYS_INJECT_TOKEN_BUDGET: int = 300   # env ORION_ALWAYS_INJECT_TOKEN_BUDGET
ORION_ALWAYS_INJECT_ENABLED: bool = True       # env ORION_ALWAYS_INJECT_ENABLED
```

---

## 10. Phase 4 — Hub API + Memory Tab

### Hub Postgres pool

`services/orion-hub/scripts/main.py` lifespan adds asyncpg pool reading `RECALL_PG_DSN`. `services/orion-hub/app/settings.py` gains `RECALL_PG_DSN: str` (same env var as recall service).

### API routes (extend `services/orion-hub/scripts/api_routes.py`)

All routes behind `ensure_session`:

| Method | Path | Description |
|---|---|---|
| POST | `/api/memory/cards` | Create card (manual highlight) |
| GET | `/api/memory/cards` | List with filters (status, types, anchor_class, project, priority, limit, offset) |
| GET | `/api/memory/cards/{id_or_slug}` | Single card + edges + recent history |
| PATCH | `/api/memory/cards/{id_or_slug}` | Update, writes history |
| POST | `/api/memory/cards/{id}/status` | Status change with optional reason |
| POST | `/api/memory/edges` | Add edge |
| DELETE | `/api/memory/edges/{id}` | Remove edge |
| GET | `/api/memory/history` | List history (card_id or edge_id filter) |
| POST | `/api/memory/history/{id}/reverse` | Reverse a history entry |
| GET | `/api/memory/cards/{id}/neighborhood` | Flat neighbor list grouped by edge_type (hops=1 default) |
| POST | `/api/memory/sessions/{id}/distill` | 501 stub until Phase 5 |

### Memory tab (`services/orion-hub/templates/index.html`)

New nav button `memoryTabButton` with `data-hash-target="memory"`. New panel `<section id="memory" data-panel="memory" hidden>` with three sub-panel toggles: Review Queue / All Cards / Activity Log. Tailwind classes consistent with existing tabs.

### `services/orion-hub/static/js/memory.js` (new file)

- **Fetcher module** wrapping all API endpoints against `API_BASE_URL`
- **Review Queue:** `status=pending_review` cards; approve / reject / edit buttons; bulk-select
- **All Cards:** filterable list; click → detail panel (card fields, edges grouped by edge_type in/out, 10-entry history, edit-modal trigger)
- **Activity Log:** reverse-chronological history; per-row "Reverse this" on `op=create` rows where `actor=auto_extractor`
- **Highlight-to-Remember modal:** triggered by text selection in chat output area; Stage 1 regex pre-fills `types` and `anchor_class` client-side; edges via search-as-you-type against `/api/memory/cards?...`; commit → `POST /api/memory/cards`; draft → localStorage only
- **Neighborhood graph:** Cytoscape loaded via CDN (no build step, no package.json change)

---

## 11. Phase 5 — Auto-Extractor + Operator Distiller

### `orion/core/storage/memory_extraction.py` (pure, no I/O)

```python
def extract_candidates(turn_text: str, *, speaker: str = "user") -> list[CandidateCard]: ...
def fingerprint(card: MemoryCardV1) -> str: ...  # stable hash of (anchor_class, normalized_summary)
```

Stage 1 regex tables per spec §5 patterns. Fully unit-testable in isolation.

### Auto-extractor (`services/orion-cortex-orch/app/memory_extractor.py`)

New `Hunter` registered in `main.py` subscribed to `orion:chat:history:turn`. Handler skips non-user turns. For each candidate:
1. Compute fingerprint → check history for prior rejection → skip if found
2. Check for existing card with same anchor + summary → skip if duplicate
3. Check for contradiction against `always_inject` cards → force `status=pending_review` + propose `contradicts` edge
4. Single occurrence → `confidence=likely, status=pending_review`
5. Repetition ≥ `ORION_AUTO_EXTRACTOR_AUTO_PROMOTE_THRESHOLD` distinct sessions → `confidence=certain, status=active`

Insert via DAL with `actor=auto_extractor`, `provenance=auto_extractor`.

New settings:
```python
ORION_AUTO_EXTRACTOR_ENABLED: bool = False           # env ORION_AUTO_EXTRACTOR_ENABLED
ORION_AUTO_EXTRACTOR_STAGE2_ENABLED: bool = False    # raises NotImplementedError if True (v1.5)
ORION_AUTO_EXTRACTOR_AUTO_PROMOTE_THRESHOLD: int = 2
```

### Operator distiller CLI (`services/orion-recall/scripts/distill_memory_cards.py`)

Argparse: `--transcript PATH`, `--project NAME`, `--since YYYY-MM-DD`, `--today YYYY-MM-DD`, `--dry-run` / `--apply`. Auto-detects JSONL format. `--dry-run` prints YAML; `--apply` writes via DAL (`actor=operator_distiller`, `provenance=operator_distiller`, `status=active`, `confidence=certain`). Phase 4's 501 stub wires to this via `subprocess.run` when Phase 5 ships.

---

## 12. Phase 6 — Safety Verification + Docs

### Safety checklist (in PR description, with commands + observed outputs)

1. `RECALL_DEFAULT_PROFILE` unchanged — `rg -n "RECALL_DEFAULT_PROFILE" services/orion-recall/app/settings.py`
2. Only `self.factual.v1.yaml` modified + `biographical.v1.yaml` added — `git diff main..feat/memory-cards-v1 -- orion/recall/profiles/`
3. No autonomous prompt/identity mutation surface — `rg -i "self_model|identity_mutate|prompt_mutate" services/`
4. Auto-extractor defaults off — `rg -n "ORION_AUTO_EXTRACTOR_ENABLED" services/orion-cortex-orch/app/settings.py`
5. Stage 2 raises `NotImplementedError` — test coverage confirmed
6. Always-inject budget capped + configurable — default 300 tokens, runbook caps operator guidance
7. Visibility scope enforced — `test_visibility_intimate_excluded_from_social_lane` passes
8. All mutations logged in history — DAL roundtrip test asserts history row per write
9. Auto-promotions reversible — Phase 4 Activity Log "Reverse this" confirmed in smoke
10. No new external deps — `git diff main..feat/memory-cards-v1 -- services/*/requirements.txt`
11. `chat.general.v1` retrieval unchanged when card store empty — Phase 1 + Phase 3 snapshot tests

### Operator runbook

`docs/superpowers/runbooks/2026-05-01-memory-cards-v1-runbook.md` covering: enabling the auto-extractor, setting `priority=always_inject`, reversing an auto-promoted card, adding visibility scopes, troubleshooting empty injection block / missing intent routing / fan-out latency.

### E2E smoke script

`scripts/smoke_memory_cards_e2e.sh`: apply schema → insert cards via DAL → enable always-inject + intent routing → run Hub Memory tab → recall query → confirm cards in bundle → reverse an auto-promotion → confirm restored state.

---

## 13. Path Reference

| Concern | Path |
|---|---|
| SQL DDL | `services/orion-recall/sql/memory_cards.sql` |
| Pydantic contracts | `orion/core/contracts/memory_cards.py` |
| RecallQueryV1 amendments | `orion/core/contracts/recall.py` |
| DAL (async) | `orion/core/storage/memory_cards.py` |
| Stage 1 extractor (pure) | `orion/core/storage/memory_extraction.py` |
| Cards adapter | `services/orion-recall/app/cards_adapter.py` |
| Intent router | `services/orion-recall/app/intent.py` |
| New biographical profile | `orion/recall/profiles/biographical.v1.yaml` |
| Modified factual profile | `orion/recall/profiles/self.factual.v1.yaml` |
| Always-inject helper | `services/orion-cortex-orch/app/memory_inject.py` |
| Auto-extractor worker | `services/orion-cortex-orch/app/memory_extractor.py` |
| Operator distiller CLI | `services/orion-recall/scripts/distill_memory_cards.py` |
| Hub API routes | `services/orion-hub/scripts/api_routes.py` (extend) |
| Hub Memory tab markup | `services/orion-hub/templates/index.html` (extend) |
| Hub Memory JS | `services/orion-hub/static/js/memory.js` (new) |
| DAL tests | `services/orion-recall/tests/test_memory_cards_dal.py` |
| Contract tests | `tests/test_memory_cards_contracts.py` |
| Cards backend tests | `services/orion-recall/tests/test_cards_backend.py` |
| Intent routing tests | `services/orion-recall/tests/test_intent_routing.py` |
| Inject tests | `services/orion-cortex-orch/tests/test_memory_inject.py` |
| Hub API tests | `services/orion-hub/tests/test_memory_api.py` |
| Extraction unit tests | `tests/test_memory_extraction.py` |
| Extractor integration tests | `services/orion-cortex-orch/tests/test_memory_extractor.py` |
| Distiller CLI tests | `services/orion-recall/tests/test_distill_cli.py` |
| E2E smoke script | `scripts/smoke_memory_cards_e2e.sh` |
| Operator runbook | `docs/superpowers/runbooks/2026-05-01-memory-cards-v1-runbook.md` |

---

## 14. Explicitly Not in v1

Per deliberate deferral to v1.5/v2:

- Per-card semantic embeddings; cross-encoder reranker; RDF vector-seeded traversal
- Stage 2 LLM extraction; Stage 2 LLM intent classification
- Per-lane distillers (dream / journal / metacog)
- GraphDB projection
- Markdown export mirror
- Playwright-automated Hub UI tests (manual smoke only in v1)

---

## 15. Key Corrections from Prior Plan

The stale plan (2026-04-30) was written against a different repo state. Corrections captured in this spec:

| Stale plan | This spec |
|---|---|
| `query.text` | `q.fragment` (correct field name in `RecallQueryV1`) |
| `RecallQueryV1.lane` assumed to exist | Added as `lane: Optional[str] = None` in Phase 0 |
| `orion/core/storage/` assumed to exist | New package, create `__init__.py` |
| `services/orion-recall/scripts/` assumed to exist | New directory |
| Hub postgres pool assumed to exist | Added in Phase 4 lifespan |
| `render_items` returns `str` | Phase 2 changes to `tuple[str, list[dict]]`; 3 callers updated |
| `feat/memory-cards-v1` branch/worktree assumed to exist | Create from `cognitive_substrate/v4` |
