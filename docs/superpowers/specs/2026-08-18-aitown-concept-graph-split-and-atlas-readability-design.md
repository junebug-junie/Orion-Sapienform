# AI Town concept-graph split + Concept Atlas readability — design spec

Status: DESIGN, not implemented. Ground truth verified live 2026-08-18
against `main` via direct file reads and a live Postgres query, not
inference from code alone.

Builds on `docs/superpowers/specs/2026-07-15-concept-atlas-graph-pipeline-design.md`
(the pipeline this spec touches) and surfaces a real defect that shipped
spec never caught: it defined the topic-foundry dataset as `chat_history_log`
with no source filter, on the assumption the table was Orion/Juniper chat.
It isn't, mostly.

**Update (same day, 2026-08-18):** on seeing the corpus-mix numbers below,
Juniper's reaction was "these should have their own table" — a real,
considered decision to physically split AI Town chat data out of
`chat_history_log` entirely, made aware that this consciously supersedes a
prior explicit decision on the same tradeoff
(`docs/superpowers/specs/2026-07-31-recall-aitown-source-tagging-design.md`
— tag AI Town rows in place rather than separate them, because "if those
memories just come in raw without any tag to differentiate them, orion
will take it as our history"). See the new **Physical chat-history table
split** section below — that decision surfaced real, load-bearing
complexity in `orion-sql-writer`'s write path (a concurrency-hardened
atomic-upsert system, not a simple insert) that materially raises this
migration's scope beyond what a first read of "split it" suggests.

## Arsonist summary

Hub's Concept Atlas — the read-only interpretability view over Orion's
"organically clustered" concept graph — is training on a corpus that is 90%
AI Town NPC dialogue, not Orion/Juniper conversation, and has been since the
pipeline first ran. Nothing routes, filters, or separates AI Town traffic
from the rest of `chat_history_log`; the earlier 2026-07-15 spec assumed a
single homogeneous chat corpus and never named AI Town at all. Concepts like
"Electrical Tech," "storm and memory," "Lighting and storytelling" showing up
as Orion's own "god nodes" are a direct, mechanical consequence, not noise or
a fluke.

Separately, and independently of the corpus problem, the Concept Atlas UI's
force-directed layout has no label-collision handling and no anchor-scope
filter, so it becomes unreadable well before node count gets large. Both
problems compound: fixing the corpus without fixing the layout still leaves
a graph that turns to mush past a few dozen nodes.

## Current architecture

- **Corpus**: `chat_history_log` (Postgres). Live query, 2026-08-18:

  | `source` | rows |
  |---|---|
  | `orion-embodiment` (AI Town) | 1,577 |
  | `hub_orion` | 115 |
  | *(blank)* | 24 |
  | `hub_ws` | 23 |
  | `hub_http` | 2 |
  | `hub` | 2 |

- **Dataset definition**: `services/orion-hub/scripts/concept_atlas_routes.py`
  `_ensure_topic_foundry_dataset_and_model()` (~line 126) creates exactly one
  topic-foundry dataset (`orion-hub-autonomous-dataset`) via
  `source_table="chat_history_log"`, `text_columns=["prompt","response"]`,
  **no `where_sql`**.
- **Topic-foundry already supports a source filter and it is unused**:
  `services/orion-topic-foundry/app/models.py`'s `DatasetCreateRequest`
  already has `where_sql: Optional[str]` / `where_params: Optional[dict]`,
  and `app/services/data_access.py:fetch_dataset_rows()` (~line 36-39)
  already appends `where_sql` into the query's `WHERE` clause alongside the
  time-window bounds. This is a real, working, already-shipped capability
  the Hub-side dataset config simply never passes.
- **Scheduler**: `services/orion-hub/scripts/main.py`
  `_run_substrate_topic_foundry_scheduler()` — one tick, three steps
  (trigger training → trigger enrichment → ingest), on
  `SUBSTRATE_TOPIC_FOUNDRY_SCHEDULER_INTERVAL_SEC` (default 86400s/daily).
- **Ingestion / graph target**: `concept_atlas_ingest_topic_foundry()` writes
  into the single FalkorDB graph named by `FALKORDB_SUBSTRATE_GRAPH`
  (default `orion_substrate`) via `orion/substrate/adapters/topic_foundry.py`,
  which defaults every organically-clustered concept to `anchor_scope="world"`.
  FalkorDB (Redis-backed) can hold multiple independently-named graphs in one
  instance at no extra infra cost — `FALKORDB_SUBSTRATE_GRAPH` is just a
  string, not a fixed resource.
- **UI**: `services/orion-hub/static/js/concept-atlas.js` renders via
  Cytoscape.js, `layout: { name: "cose", animate: false, padding: 24 }`
  (line ~304) — no `nodeDimensionsIncludeLabels`, no overlap/spacing tuning.
  `promotion_state` has a working client-side filter
  (`applyClientPromotionStateFilter`).
  **Correction (2026-08-18, later same day):** this doc originally claimed
  `anchor_scope` was never wired into an actual filter — wrong, caught
  before implementing a duplicate. `caFilterScope`
  (`templates/concept_atlas.html:26`) already round-trips as a real
  server-side `scope` query param on `/api/substrate/concepts/network`
  (`concept-atlas.js`'s `currentFilters()`/`fetchNetwork()`). The only real
  gap here was the layout's label-collision handling — fixed same day, see
  "Recommended next patch" / Track A.
- **Analysis already computed**: activation-weighted degree → top-5 "god
  nodes," computed fresh per request
  (`concept_atlas_routes.py:48`, `_GOD_NODE_TOP_N`). That's the only graph
  analysis that exists today.
- **Analysis that does not exist**: connected components, shortest-path /
  betweenness, community/cluster coloring propagated from topic-foundry's
  own HDBSCAN cluster assignments.

## Physical chat-history table split (2026-08-18 update)

Consciously supersedes the 2026-07-31 tagging decision. That decision's
solution to "Orion can't tell AI Town content from real history" was to tag
rows in place (`services/orion-recall/app/chat_source_tagging.py`) so
consumers could differentiate while AI Town content stayed visible where
needed. A physical split changes the *shape* of that solution: AI Town
content moves to its own table; any consumer that still needs to see it
does so via an explicit read of that table, not implicit presence in the
shared one. The underlying need the July decision was solving for
(differentiability) is still satisfied — more strongly, in fact, since
physical separation can't be un-tagged by a bug the way a JSONB field can
be dropped or mis-copied.

### Real complexity found, not assumed

1. **Producer side has no existing channel-level separation to route on.**
   Per `orion/bus/channels.yaml`, both `orion:chat:history:turn`
   (`ChatHistoryTurnV1`) and `orion:chat:social:turn`
   (`SocialRoomTurnV1`) have `producer_services: ["orion-hub",
   "orion-embodiment"]` — Hub and AI Town publish to the *same* channels
   today. Splitting requires either a new AI-Town-specific channel, or
   (simpler, no producer changes) branching inside the existing consumer
   on the `client_meta` tag already present in the payload.

2. **`orion-sql-writer`'s write path is a hardened, concurrency-sensitive
   system, not a simple insert — this is the real risk surface.**
   `services/orion-sql-writer/app/worker.py:upsert_chat_history_row()`
   is a single atomic `INSERT ... ON CONFLICT DO UPDATE`. Its own docstring
   explains why: a prior `SELECT`-then-`INSERT` pattern lost "roughly one
   Hub turn in five" to a real race — a turn's three contributing bus
   events are dispatched as independent parallel chassis tasks
   (`orion/core/bus/bus_service_chassis.py`) via `asyncio.to_thread`, and
   under the old pattern all three could `SELECT`-miss together, then race
   to `INSERT`. `ChatHistoryLogSQL` is referenced **24 separate times** in
   this one file: the atomic upsert itself, a separate
   `_apply_spark_meta_patch()` path with its own correlation-id
   lookup+update, a generic `sql_model_cls` dispatch table, and several
   duplicate-detection/debug-logging branches keyed specifically on this
   model class. A table-routing change has to thread through all of it
   correctly or risks reintroducing the exact race class this file was
   already hardened against once.

3. **~50 files read `chat_history_log`** (`orion-recall`, `orion/memory/crystallization/`,
   `orion-memory-consolidation`, `orion-dream`, the chat-history compactor,
   the discussion window, the journaler — full list via `rg -l
   chat_history_log`). A subset have deliberate, already-shipped
   AI-Town-aware handling built on the July tagging mechanism:
   `chat_source_tagging.py` itself, `scripts/smoke_aitown_crystallization_gate.py`,
   `scripts/bulk_reject_aitown_proposals.py` (a crystallization gate keyed
   specifically on the aitown tag). These need to keep seeing AI Town
   content post-split, correctly sourced from the new table — moving the
   data out from under them silently would break that gating, not just
   "stop showing AI Town concepts" the way the topic-foundry case would.

### Canonical detection signal (confirmed, not assumed)

`client_meta.external_room.platform == 'aitown'`
(`chat_source_tagging.py::chat_source_platform()`) is the sanctioned
signal — not the `source` column, which was only ever a coincidental
proxy. Live-checked 2026-08-18: 100% correlated with `source='orion-embodiment'`
today (1,577/1,577 rows both directions), so either works as a filter
predicate *right now* — but the tag is what the rest of the codebase
already builds around (chat_source_tagging.py, the crystallization gate)
and is the signal that stays correct if a second AI-Town-adjacent producer
ever appears.

### Proposed phased migration

- **Phase 0 (ships now, independent, not wasted work either way):**
  tactical `where_sql` filter on topic-foundry's *existing* Orion dataset
  (`client_meta -> 'external_room' ->> 'platform' IS DISTINCT FROM 'aitown'`)
  — stops new AI-Town concepts entering Orion's graph immediately, before
  any of Phases 1-4 land. If the table split completes later, this dataset
  naturally ends up reading Orion-only rows and the filter becomes
  redundant/removable, not wasted.
- **Phase 1 (new table, dual-write, zero consumer-visible change):**
  create `aitown_chat_history_log` (mirrors `chat_history_log`'s schema).
  In `orion-sql-writer`, branch on the `client_meta` platform tag at write
  time and write AI Town rows to the new table *in addition to* the
  existing one — every current consumer keeps working exactly as today.
  This is the piece that has to carefully thread `upsert_chat_history_row()`,
  `_apply_spark_meta_patch()`, and the `sql_model_cls` dispatch table
  without disturbing the atomic-upsert guarantee the July race-condition
  fix depends on.
- **Phase 2 (consumer audit + migration, reviewed per-consumer, not
  batch-decided here):** for each of the ~50 files: (a) doesn't need AI
  Town data → leave alone, it naturally stops seeing new AI Town rows once
  Phase 3 cuts over; (b) needs AI Town data → add an explicit read from
  `aitown_chat_history_log` alongside its existing `chat_history_log` read.
  The recall/crystallization consumers with deliberate AI-Town handling
  are almost certainly bucket (b) — this phase is real investigative work
  per file, not something to pre-decide in this doc.
- **Phase 3 (cutover):** stop writing AI Town rows into `chat_history_log`
  — single-write to `aitown_chat_history_log` only. Only safe once every
  bucket-(b) consumer from Phase 2 has actually shipped.
- **Phase 4 (historical data + cleanup):** decide whether to
  backfill-migrate the existing 1,577 historical AI-Town rows out of
  `chat_history_log`, or leave them as frozen historical residue with only
  new rows going to the new table going forward. Snapshot-first per this
  repo's backfill protocol (CLAUDE.md §14) either way.

## Missing questions

- Should the AI Town dataset's rolling window (`SUBSTRATE_TOPIC_FOUNDRY_WINDOW_DAYS`)
  and schedule cadence be independent from Orion's? AI Town produces ~14x the
  chat volume per unit time (1,577 vs 141 rows in the same table today), so
  the same 30-day/daily defaults may cluster very differently for each.
- Does AI Town's graph need its own Concept Atlas page immediately, or is a
  reused `/api/substrate/concepts/network?graph=aitown`-style parameter on
  the existing route enough for a first cut, with a dedicated page as a
  follow-up once someone's actually looking at it regularly?
- Should `orion_substrate_aitown` ever feed back into Orion's own cognition
  (a `concept_induced`-tier-style adapter reading AI Town concepts into
  `chat_stance`), or should it stay purely an interpretability/observability
  view with no downstream cognitive consumer? Not addressed here — explicit
  non-goal below — but worth a real answer before anyone builds toward it.
- Table split, Phase 4: backfill-migrate the 1,577 existing historical rows,
  or leave them in place as residue? Affects whether any currently-shipped
  AI-Town-aware consumer (the crystallization gate, recall) sees a visible
  discontinuity in its historical view at cutover.
- Table split, Phase 2: is there an existing inventory/owner for each of the
  ~50 `chat_history_log` readers, or does this spec's author need to build
  that inventory as Phase 2's first step? Affects how much of Phase 2 is
  "look up an answer" vs. "go trace 50 files."

## Proposed schema / API changes

**AI Town split** — additive, no breaking changes to the existing pipeline:

- New env: `FALKORDB_AITOWN_SUBSTRATE_GRAPH` (default e.g.
  `orion_substrate_aitown`), consumed by a second `SubstrateGraphStore`
  instance built the same way `SUBSTRATE_SEMANTIC_STORE` already is
  (`build_substrate_store_from_env()`-style, just a different graph name).
- `_ensure_topic_foundry_dataset_and_model()` generalized to take a
  `(dataset_name, model_name, where_sql)` triple instead of hardcoded
  module-level constants, called twice:
  - Orion dataset: `where_sql = "client_meta -> 'external_room' ->> 'platform' IS DISTINCT FROM 'aitown'"`
  - AI Town dataset: `where_sql = "client_meta -> 'external_room' ->> 'platform' = 'aitown'"`
  - (Both use the canonical `chat_source_tagging.py` signal, not the
    coincidentally-correlated `source` column — see the table-split section
    below for why that distinction matters.)
- New sibling `concept_atlas_ingest_topic_foundry_aitown()` (or a
  `graph_target` parameter on the existing function) writing into the new
  graph instead of `orion_substrate`.
- Scheduler tick gains one more (trigger/enrich/ingest) step-group for the
  AI Town dataset, same async-fire-and-forget shape as the existing one.
- No `SubstrateAnchorScopeV1` schema change — AI Town concepts keep
  `anchor_scope="world"` (still the correct organically-clustered default),
  the *graph they live in* is what's different, not their node shape.

**Readability** — additive, UI + backend, no schema changes:

- ~~Cytoscape layout: add `nodeDimensionsIncludeLabels`~~ **Shipped same
  day** (`nodeDimensionsIncludeLabels: true` + `componentSpacing: 80` in
  `concept-atlas.js`'s cose config) alongside the Track A corpus filter —
  see fix branch/PR. `nodeRepulsion`/`idealEdgeLength` tuning intentionally
  left alone: no way to visually verify a chosen magnitude is actually
  better without a live render, so guessing at absolute numbers risked
  making it worse; revisit with real before/after screenshots.
- ~~Wire `anchor_scope` into an actual client-side filter~~ **Not needed —
  already existed**, see the corrected "Current architecture" note above.
- Default label visibility to god-nodes-only (or above a zoom threshold),
  full labels on hover/click — data already carries `god_node: bool` per
  node (`concept_atlas_routes.py:715`), this is a pure frontend change.
- New backend: connected-components (pure BFS/union-find over the
  already-fetched node/edge list — no new dependency), returned alongside
  the existing `god_node_count` in `concept_atlas_network()`'s response, and
  used client-side to lay out components independently instead of one shared
  force simulation.
- Community coloring: propagate topic-foundry's existing HDBSCAN cluster
  assignment (already computed, already available via topics/keywords) into
  node `metadata`/style instead of computing a second clustering pass.

## Files likely to touch

- `services/orion-hub/scripts/concept_atlas_routes.py` — dataset/model
  constants → parameterized, second dataset+model+ingestion path, new
  connected-components computation.
- `services/orion-hub/scripts/main.py` — scheduler tick gains the AI Town
  dataset's trigger/enrich/ingest steps.
- `services/orion-hub/app/settings.py`, `.env_example`, `.env`,
  `docker-compose.yml` — `FALKORDB_AITOWN_SUBSTRATE_GRAPH` (and possibly
  independent `SUBSTRATE_TOPIC_FOUNDRY_AITOWN_WINDOW_DAYS`/interval if the
  cadence question above resolves toward "independent").
- `services/orion-hub/static/js/concept-atlas.js`,
  `services/orion-hub/templates/concept_atlas.html` — layout tuning,
  anchor_scope filter, label-visibility default, (if a dedicated AI Town
  page ships) a new route/template pair reusing the same JS module.
- One-off cleanup script (not a permanent script) to purge
  `orion-embodiment`-sourced nodes already ingested into `orion_substrate`
  today, once the corpus fix ships — otherwise the existing graph stays
  polluted even after new ingestion stops adding to the problem.
- `services/orion-hub/tests/test_concept_atlas_routes.py`,
  `test_concept_atlas_ingest_topic_foundry.py`,
  `test_topic_foundry_scheduler.py` — cover both the dual-dataset wiring and
  the new connected-components computation.

**Table split (Track B), additionally:**

- `services/orion-sql-writer/app/models/` — new `aitown_chat_history_log`
  table/model, mirroring `ChatHistoryLogSQL`.
- `services/orion-sql-writer/app/worker.py` — all 24 `ChatHistoryLogSQL`
  reference sites need review; `upsert_chat_history_row()` and
  `_apply_spark_meta_patch()` are the two that must branch correctly by
  platform tag without disturbing the atomic-upsert guarantee.
- A new Postgres migration for the new table.
- Per Phase 2: each bucket-(b) consumer file, individually — not
  enumerable here, that enumeration *is* Phase 2's deliverable.
- `services/orion-sql-writer/tests/` — new dual-write tests, plus
  regression coverage proving the original race-condition fix
  (`upsert_chat_history_row`'s docstring) still holds with the new branch
  in place.

## Non-goals

- No `concept_induced`-tier adapter feeding AI Town concepts into
  `chat_stance`/Orion's own cognition — this spec is interpretability-only
  for the AI Town side, per the open question above.
- No shortest-path/betweenness centrality in this pass — real gap, but no
  concrete question exists yet that needs it; adding it speculatively is
  exactly the keyword-cathedral risk CLAUDE.md warns against. Revisit once
  someone actually wants to ask "how does concept A relate to concept B."
- No retroactive re-clustering of AI Town's full chat history on day one —
  same rolling-window behavior the existing scheduler already has.
- No change to `orion.spark.concept_induction`'s dead spaCy pipeline or the
  `concept_induction_pass` substrate rewire (PR #1714) — unrelated seam.
- Table split's Phase 2 per-consumer bucket-(a)/(b) decisions are explicitly
  NOT pre-made for all ~50 files in this doc — real investigative work
  belonging to Phase 2 itself.
- No cutover (Phase 3) without every bucket-(b) consumer from Phase 2
  already shipped and verified — this is a hard sequencing constraint, not
  a suggestion.

## Acceptance checks

- Live query: the Orion dataset's ingested node count, re-run after the
  corpus filter ships, drops to reflect only `hub_*` sources — spot-check a
  handful of "god nodes" by label and confirm none read as AI Town
  world-building topics.
- Live query: a new `orion_substrate_aitown` graph exists in FalkorDB with a
  nonzero node count after the first scheduler tick following deploy.
- Concept Atlas UI: with the same node count as today's screenshot,
  labels no longer directly overlap; `anchor_scope` filter toggle exists and
  visibly changes the rendered node set.
- `concept_atlas_network()`'s response includes a connected-components field;
  spot-check against the actual rendered graph that component boundaries
  match visually-separated clusters.
- Table split, Phase 1 (dual-write): row counts in `aitown_chat_history_log`
  track 1:1 with new aitown-tagged rows landing in `chat_history_log`, with
  zero change in row counts or behavior for any existing consumer.
- Table split, Phase 3 (cutover): every Phase-2 bucket-(b) consumer's
  behavior (crystallization gate output, recall differentiation) diffed
  before/after cutover and confirmed unchanged.

## Recommended next patch

Two independent tracks — the concept-graph fix does not block on the table
split, and vice versa:

**Track A — concept graph (smallest, ships fastest):**

1. ~~**Corpus fix**~~ **shipped same day**: `where_sql` added to
   `_TOPIC_FOUNDRY_WHERE_SQL` on a renamed dataset/model
   (`-v2` suffix — topic-foundry's dataset/model routes are create-only, no
   update endpoint, so the old unfiltered names had to be superseded by new
   ones rather than patched in place), using the canonical `client_meta`
   tag. Stops new AI-Town concepts entering Orion's graph from the next
   scheduler tick onward. Old `orion-hub-autonomous-dataset`/
   `orion-hub-autonomous` left in place, unreferenced (no delete endpoint
   either).
1b. ~~**Layout label-collision fix**~~ **shipped same day**, see the
   Readability section above.
2. **Cleanup pass** (not yet done — a delete, deliberately left for an
   explicit go-ahead rather than run unattended): purge already-ingested
   AI-Town-sourced nodes from `orion_substrate` (one-off script,
   snapshot-first per this repo's backfill protocol). Without this, the
   *already-ingested* god nodes stay AI-Town-derived until the next full
   retrain cycle naturally ages them out via decay, or this runs.
3. **AI Town's own concept graph**: second dataset/model/FalkorDB-graph/
   ingestion path, per the earlier schema section — reads from
   `aitown_chat_history_log` once Track B's Phase 1 exists, or from
   `chat_history_log` filtered to the aitown tag in the interim.
4. **Readability, remaining**: connected-components → community coloring,
   roughly in that order of effort-to-value. `nodeRepulsion`/
   `idealEdgeLength` layout tuning also still open — needs a live
   before/after render to tune responsibly, not a guessed number.
5. **AI Town Concept Atlas page**: only once someone's actually looking at
   the new graph regularly.

**Track B — physical table split (larger, real risk surface, needs its own
review checkpoints between phases, not a single PR):**

1. Phase 1: new `aitown_chat_history_log` table + dual-write in
   `orion-sql-writer`. Reviewed and shipped on its own before Phase 2 starts.
2. Phase 2: per-consumer audit of the ~50 `chat_history_log` readers,
   bucket (a)/(b), migrate bucket-(b) consumers. Likely several PRs, not one
   — this is the bulk of the real work and the bulk of the real risk.
3. Phase 3: cutover. Gated on every bucket-(b) consumer from Phase 2 being
   shipped and verified first.
4. Phase 4: historical-data decision + cleanup.
