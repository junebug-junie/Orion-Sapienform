# AI Town concept-graph split + Concept Atlas readability — design spec

Status: DESIGN, not implemented. Ground truth verified live 2026-08-18
against `main` via direct file reads and a live Postgres query, not
inference from code alone.

Builds on `docs/superpowers/specs/2026-07-15-concept-atlas-graph-pipeline-design.md`
(the pipeline this spec touches) and surfaces a real defect that shipped
spec never caught: it defined the topic-foundry dataset as `chat_history_log`
with no source filter, on the assumption the table was Orion/Juniper chat.
It isn't, mostly.

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
  (`applyClientPromotionStateFilter`); `anchor_scope` is fetched and shown as
  a stat breakdown (`renderBreakdown(ANCHOR_SCOPE_BREAKDOWN, ...)`) but is
  never wired into an actual node filter.
- **Analysis already computed**: activation-weighted degree → top-5 "god
  nodes," computed fresh per request
  (`concept_atlas_routes.py:48`, `_GOD_NODE_TOP_N`). That's the only graph
  analysis that exists today.
- **Analysis that does not exist**: connected components, shortest-path /
  betweenness, community/cluster coloring propagated from topic-foundry's
  own HDBSCAN cluster assignments.

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

## Proposed schema / API changes

**AI Town split** — additive, no breaking changes to the existing pipeline:

- New env: `FALKORDB_AITOWN_SUBSTRATE_GRAPH` (default e.g.
  `orion_substrate_aitown`), consumed by a second `SubstrateGraphStore`
  instance built the same way `SUBSTRATE_SEMANTIC_STORE` already is
  (`build_substrate_store_from_env()`-style, just a different graph name).
- `_ensure_topic_foundry_dataset_and_model()` generalized to take a
  `(dataset_name, model_name, where_sql)` triple instead of hardcoded
  module-level constants, called twice:
  - Orion dataset: `where_sql = "source NOT IN ('orion-embodiment')"`
  - AI Town dataset: `where_sql = "source = 'orion-embodiment'"`
- New sibling `concept_atlas_ingest_topic_foundry_aitown()` (or a
  `graph_target` parameter on the existing function) writing into the new
  graph instead of `orion_substrate`.
- Scheduler tick gains one more (trigger/enrich/ingest) step-group for the
  AI Town dataset, same async-fire-and-forget shape as the existing one.
- No `SubstrateAnchorScopeV1` schema change — AI Town concepts keep
  `anchor_scope="world"` (still the correct organically-clustered default),
  the *graph they live in* is what's different, not their node shape.

**Readability** — additive, UI + backend, no schema changes:

- Cytoscape layout: add `nodeDimensionsIncludeLabels: true` plus tuned
  `nodeRepulsion`/`idealEdgeLength`/`componentSpacing` for denser graphs.
- Wire `anchor_scope` into an actual client-side filter (mirrors the existing
  `applyClientPromotionStateFilter` pattern exactly).
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

## Recommended next patch

Smallest useful slice, in order:

1. **Corpus fix only** (no new graph yet): add `where_sql` to the existing
   single dataset (`source NOT IN ('orion-embodiment')`), ship, verify next
   day's scheduler tick re-trains on a clean corpus. This alone stops new
   pollution and is a ~10-line change given `where_sql` already works
   end-to-end in topic-foundry.
2. **Cleanup pass**: purge already-ingested AI-Town-sourced nodes from
   `orion_substrate` (one-off script, snapshot-first per this repo's
   backfill protocol).
3. **AI Town's own graph**: second dataset/model/graph/ingestion path, per
   the schema section above.
4. **Readability**: layout tuning + `anchor_scope` filter (cheap, can ship
   independently/in parallel with 1-3) → connected-components → community
   coloring, roughly in that order of effort-to-value.
5. **AI Town Concept Atlas page**: only once someone's actually looking at
   the new graph regularly.
