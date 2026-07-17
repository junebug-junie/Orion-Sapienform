# Concept graph pipeline + Concept Atlas Hub tab — design spec

Status: DESIGN, not implemented. Ground truth verified live 2026-07-15
against `main` via direct file reads (not grep-only) for every load-bearing
claim below.

**Update (2026-07-17):** all 8 phases below shipped — Wave 1 (phases 1/2/3/5)
via #1079, Wave 2 (phases 4/6/7/8) via #1086. Follow-on hardening landed
separately, not as additional phases: seed-startup wiring (#1092), an
`.env_example` sync (#1094), a topic-foundry smoke-script fix (#1096), the
topic-foundry ingestion route (#1097), and its identity-merge fix (#1101).
This spec's own non-goal ("no new graph
infrastructure ... do not wire into `GraphDBSubstrateStore`/SPARQL") held —
the persistence backend that actually shipped is FalkorDB, not SPARQL/RDF,
per `docs/superpowers/specs/2026-07-16-falkordb-property-graph-routing-design.md`
(PR #1105) — a decision made after this spec was written, superseding the
"in-memory for now" framing in the original Phase 8/Concept Atlas sections
below. Live-verified: Concept Atlas's concept graph persists in FalkorDB
across Hub restarts, not just Hub process memory.

Scope decided by Juniper: replace concept-induction's NLP extractor with the
already-built, already-unsupervised `orion-topic-foundry` clustering
pipeline, fix the concept-identity merge bug, add a thin typed-edge layer on
top (tuning options A baseline + B and C as real candidates, D explicitly
deferred), retire a confirmed-fake stub signal adapter, and ship a read-only
"Concept Atlas" Hub tab for operator interpretability. No new graph
infrastructure — reuse the in-memory substrate store that is already the
default.

## Arsonist summary

Orion already has almost everything this effort needs, built by three
different past efforts that never got connected to each other:

1. `orion/substrate/` — a real node/edge/decay/promotion-state schema and an
   `InMemorySubstrateGraphStore`, wired to exactly one producer: the spaCy
   noun-chunk extractor in `orion/spark/concept_induction/`, which is
   currently switched off (`CONCEPT_AUTONOMOUS_TRIGGER_ENABLED=false` in
   `services/orion-spark-concept-induction/.env:35`) and has a known,
   unfixed defect (no stopword/POS filtering).
2. `services/orion-topic-foundry/` — a real, maintained, unsupervised
   embedding→UMAP→HDBSCAN→BERTopic clustering service with model
   promotion (`candidate/active/archived`) and drift detection, reading
   directly from `chat_history_log`
   (`app/pipelines/chat_corpus_builder/repository.py:26`). This already
   solves concept induction's hardest problem — forming candidate concepts
   without a predefined label vocabulary — but its output (`KgEdgeIngestV1`
   on `orion:kg:edge:ingest.v1`) only reaches `orion-rdf-writer`/
   `orion-graphdb` (`orion/bus/channels.yaml:340-346`), never
   `orion/substrate/`.
3. The one place topic-foundry's output does reach the signal layer,
   `orion/signals/adapters/topic_foundry.py::TopicFoundryAdapter.adapt()`,
   is a confirmed stub: it hardcodes
   `dimensions={"level": 0.5, "confidence": 0.5}` and its own `notes` field
   says `"stub adapter — not yet implemented"`. Independently corroborated
   client-side: `organ-signals-graph-ui.js` ships a
   `STUB_ORGAN_IDS` set that already lists `topic_foundry` and
   `concept_induction` by name, with the identical placeholder constant
   `GENERIC_PLACEHOLDER_DIMS = { level: 0.5, confidence: 0.5 }`. Someone
   already knew.
4. `orion/substrate/reconcile.py::SubstrateIdentityResolver.canonical_node_key`
   resolves concept identity by exact-lowercased-string match on `label`
   (line 47-49) — no embedding fallback. Two paraphrases of the same
   concept become two permanent nodes. This bug exists independent of which
   extractor feeds it, so swapping extractors alone would not have fixed
   it.

The sharp reframing from the original ask: this is not "build a concept
graph with organic growth, decay, and HITL review" — that already exists as
a schema+store+materializer+review-queue. It is "connect a clustering
engine that already works to a graph that already works, fix one identity
bug, kill one fake signal, and add a thin typed-relationship layer that
clustering structurally cannot produce on its own."

## Current architecture

- **Node candidate formation (to be repointed)**: `orion/spark/concept_induction/extractor.py::SpacyConceptExtractor`
  — spaCy `noun_chunks`, no `is_stop`/`PRON` filtering (confirmed absent
  from the file). Feeds `orion/spark/concept_induction/inducer.py` →
  `ConceptProfile` (`orion/core/schemas/concept_induction.py:103-119`).
- **Existing intra-batch clustering**: `orion/spark/concept_induction/clusterer.py::ConceptClusterer`
  — real cosine-similarity clustering (threshold 0.8) with a Jaccard
  string fallback, but scoped to one extraction window only; its output is
  a plain label string, memoryless across time.
- **Substrate schema**: `orion/core/schemas/cognitive_substrate.py` —
  `SubstrateNodeKindV1` includes `"concept"` (L11-23);
  `SubstrateAnchorScopeV1` = `orion/juniper/relationship/world/session`
  (L42); `ConceptNodeV1` (L152-157); `SubstrateEdgeV1` with predicates
  `supports/contradicts/refines/co_occurs_with` (L24-40, L232-243);
  `SubstratePromotionStateV1` = `proposed/provisional/canonical/deprecated/rejected`
  (L41); `SubstrateActivationV1` with `decay_half_life_seconds`/
  `decay_floor` (L78-84) — present in schema, no confirmed live writer.
- **Store selection**: `orion/substrate/graphdb_store.py::build_substrate_store_from_env()`
  (L822-849) defaults to `InMemorySubstrateGraphStore` unless
  `SUBSTRATE_STORE_BACKEND` explicitly requests `sparql`/`graphdb`. Docstring
  states this default exists specifically "to avoid accidental duplicate
  RDF surfaces during backend-neutral RDF writer cutover" — the codebase's
  own authors already flagged the redundancy risk this spec is navigating
  around.
- **Existing producer registration**: `orion/substrate/adapters/concept_induction.py::map_concept_profile_to_substrate()`
  registered in `services/orion-cortex-exec/app/chat_stance.py::_build_unification_registry()`
  (L163-170) as an always-on (`pull_on_cold=True`) lane — the wrong
  consumption shape for salience-gated recall, see Phase 6 below.
- **HITL review**: `orion/substrate/review_bootstrap.py`, `review_queue.py`,
  `review_runtime.py`, `review_schedule.py`, `review_telemetry.py` exist;
  whether an operator-facing Hub route already consumes them is unconfirmed
  (see Missing questions).
- **Topic-foundry pipeline**: `services/orion-topic-foundry/app/topic_engine.py`
  — `VectorHostEmbeddingModel` → `UMAP` (`n_components` default 5, not 2 —
  `app/settings.py:53`) → `HDBSCAN` → representation
  (`KeyBERTInspired`/`MaximalMarginalRelevance`/`PartOfSpeech`/`ctfidf`/`llm`).
  Model lifecycle: `POST /models/{model_id}/promote` to
  `candidate/active/archived`. Training triggered via `POST /runs/train`
  (no cron/scheduler found — only the drift daemon runs continuously,
  checking an existing active model, not retraining). Dataset source is
  real: `chat_history_log`. **Unconfirmed**: whether it has ever completed
  a training run against real data (see Missing questions / Phase 0).
- **Existing Hub operator surface for topic-foundry**: a full "Topic Studio"
  tab already exists (`index.html` `data-panel="topic-studio"`, L1330+),
  proxying `/api/topic-foundry` — dataset builder, model status
  (pg/embedding/model-dir health), debug drawer for preview/train/enrich/
  segments. This is a **management/control surface**, not an
  interpretability view of the concept graph's current state. Concept
  Atlas (Phase 8) must not duplicate it — link out to it instead.
- **Established Hub graph-viz patterns** (two, both real, serving different
  isolation needs):
  - *Inline panel + IIFE module*: Organ Signals
    (`organ-signals-graph-ui.js`, Cytoscape.js via CDN). Exposes
    `window.OrionOrganSignalsGraphUI = { attach, ... }`; `attach()` returns
    a controller `{ refresh, destroy, getLastPayload }`. `app.js::ensureOrganSignalsGraph()`
    lazily creates the controller once and calls `.refresh()` on tab-show
    (`app.js:1018-1023`); panel visibility is a CSS `hidden` class toggle,
    Cytoscape instance persists in the DOM across tab switches.
    **Known gap, do not replicate**: `destroy()` is defined (stops the
    poll interval, calls `cy.destroy()`) but `app.js` never calls it on
    tab-hide — if auto-refresh was checked, the `setInterval` keeps firing
    while the tab is hidden.
  - *iframe-embedded standalone page*: Substrate Atlas
    (`substrate_atlas.html` + `substrate-atlas.js`, also Cytoscape.js).
    Fully separate document; parent pings
    `contentWindow.OrionSubstrateAtlas.activate()` on tab-show
    (`app.js:996-1005`). Strongest possible isolation — a JS error in the
    embedded graph cannot touch `app.js`'s global scope at all, by
    construction, since it's a different document.
  - Note: "Substrate Atlas" is the grammar/schema_kernel trace explorer
    (`orion/schemas/grammar.py`), **not** the `cognitive_substrate.py`
    concept graph this spec is about, despite the name collision. Concept
    Atlas must be named distinctly (see below) to not compound this
    confusion.

## Missing questions

- Has `orion-topic-foundry` ever completed a training run against real
  `chat_history_log` data, or is it built-and-never-invoked? (Phase 0,
  blocks everything downstream — a live `GET /runs` check, not a code
  change.)
- Does `GraphDBSubstrateStore` point at the same physical SPARQL/Fuseki
  backend as `orion-rdf-writer`, or a separate one? (Answered enough to
  proceed: default is in-memory, so this only matters if/when a future
  patch opts into `SUBSTRATE_STORE_BACKEND=graphdb` — not blocking this
  spec.)
- Is there a distinct `EntityNodeV1` in `cognitive_substrate.py` separate
  from `ConceptNodeV1`? Determines whether Orion/Juniper seed as entities
  or concepts. Must resolve before Phase 1 seeding, not before this spec.
- Does an operator-facing Hub route already exist for
  `orion/substrate/review_queue.py`, or is HITL review purely backend
  today? Blocks Phase 7, not earlier phases.
- What co-occurrence threshold actually separates signal from noise for
  Layer 3 classification (Tuning options B/C below)? Cannot be answered
  without Phase 0's real run data — this is explicitly an empirical
  question the phased rollout is designed to answer, not a pre-decision.

## Proposed schema / API changes

- **New**: `orion/substrate/adapters/topic_foundry.py` — converts
  topic-foundry cluster/topic/keyword records into `ConceptNodeV1` +
  free `co_occurs_with` `SubstrateEdgeV1` records (same-segment
  co-membership), written at `promotion_state="proposed"`. Parallel in
  shape to the existing `concept_induction.py` adapter.
- **Changed**: `orion/substrate/reconcile.py::SubstrateIdentityResolver.canonical_node_key`
  — for `node_kind == "concept"`, add an embedding-nearest-neighbor lookup
  (same cosine-threshold pattern `ConceptClusterer` already validates,
  0.8 starting point) against existing canonical/provisional concept
  embeddings, falling back to the current exact-string key only when no
  embedding is available.
- **Changed**: `orion/core/schemas/cognitive_substrate.py::ConceptNodeV1` —
  needs an embedding/centroid field if one does not already exist, so
  `reconcile.py` has something to compare against without a new embedding
  call (topic-foundry already computes cluster centroids).
- **New**: Layer 3 relation classification — triggered on
  `promotion_state` transition `proposed → provisional` for node pairs
  whose `co_occurs_with` edge crosses the tuning threshold (Options A/B/C
  below, D deferred). One scoped LLM call per qualifying pair, classifying
  `supports/contradicts/refines`, not raw per-turn extraction.
- **Changed**: `orion/signals/adapters/topic_foundry.py::TopicFoundryAdapter.adapt()`
  — either wire to a real value (e.g. count/salience of concepts newly
  promoted this cycle, or drift magnitude already computed by
  `services/orion-topic-foundry/app/services/drift.py`) or remove the
  entry from `ORGAN_REGISTRY` entirely. Leaving the constant-0.5 stub live
  while this effort ships is not acceptable per this repo's own
  no-empty-shell-cognition rule.
- **New**: `GET /api/substrate/concepts/summary` — counts by
  `promotion_state`, by `anchor_scope`, edge counts by predicate, an
  "at risk" list (decayed activation nearing `decay_floor`).
- **New**: `GET /api/substrate/concepts/network?scope=&min_activation=&focus=`
  — nodes+edges JSON for Cytoscape, same response shape family as
  `/api/signals/active`, backed by `InMemorySubstrateGraphStore.query_concept_region()`.
  Server computes degree/activation-weighted-degree at request time and
  flags top-N nodes as `god_node: true` (store is in-memory and small; no
  precomputed ranking job needed yet).
- **New (thin proxy, or reuse existing)**: read-only summary of the latest
  topic-foundry run (`topics_summary.json`/`topics_keywords.json`) for the
  Concept Atlas clustering card — check whether `/api/topic-foundry`
  proxying already used by Topic Studio can be reused as-is before adding
  a new route.

## Files likely to touch

- `orion/substrate/adapters/topic_foundry.py` (new)
- `orion/substrate/reconcile.py`
- `orion/substrate/store.py` (embedding index for nearest-neighbor lookup)
- `orion/core/schemas/cognitive_substrate.py`
- `services/orion-cortex-exec/app/chat_stance.py` (registry entry —
  gate, don't `pull_on_cold`, see Phase 6)
- `orion/signals/adapters/topic_foundry.py`
- `orion/signals/registry.py` (`ORGAN_REGISTRY`, if removing the stub
  entry instead of fixing it)
- `services/orion-recall/app/collectors/concept_region.py` (new,
  salience-gated recall, modeled on `active_packet.py::fetch_active_packet_fragments`)
- `services/orion-hub/scripts/api_routes.py` (new summary/network routes)
- `services/orion-hub/templates/index.html` (new `data-panel="concept-atlas"`
  section + `#hubPrimaryNav` entry)
- `services/orion-hub/static/js/concept-atlas.js` (new, own file per
  established convention — see Phase 8 for the isolation-pattern decision)
- `orion/spark/concept_induction/rdf_materialization.py`, `graph_mapper.py`,
  `graph_query.py` (audit for deletion — competing, likely-dead second
  graph-projection path found inside `concept_induction/` itself)
- `orion/substrate/seed_concepts.yaml` (new, golden concept fixture)
- new `docs/superpowers/specs/` follow-up or `orion/substrate/decay_reducer.py`
  if Phase 4's decay wiring needs its own module

## Non-goals

- No new graph infrastructure. `InMemorySubstrateGraphStore` (already the
  default) is the target store. Do not wire this effort into
  `GraphDBSubstrateStore`/SPARQL or `orion-rdf-writer` — that would
  recreate the exact "accidental duplicate RDF surface" the store's own
  default-selection code already warns against.
- No RDF/OWL formal ontology reasoning, no federated SPARQL queries — not
  asked for, real infra cost, solves a problem this effort doesn't have.
- No per-turn LLM extraction call. Node formation stays batch/clustering-
  based; LLM calls are reserved for Layer 3 relation classification,
  scoped to already-established, already-co-occurring node pairs.
- No GLiNER, no predefined entity-type label vocabulary — the whole point
  of routing through topic-foundry's clustering is to avoid needing one.
- Tuning option D (piggyback relation-classification purely on the
  promotion-lifecycle transition, no co-occurrence signal) is explicitly
  **deferred**, not selected. Build A/B/C behind a flag for comparison;
  revisit D only if B/C prove not worth their complexity.
- Concept Atlas is read-only interpretability. It must not duplicate
  Topic Studio's dataset/model/training management surface — link to it,
  don't rebuild it.
- No temporal scrubber in this patch. The underlying data
  (`materialization_lineage` timestamps already appended by
  `reconcile.py::merge_node`/`merge_edge`, capped at 50/100 entries) is
  already being recorded, so a future temporal view is not blocked on new
  instrumentation — but it is out of scope here.

## Acceptance checks

- Phase 0: a real topic-foundry `run_id` exists against real
  `chat_history_log` data, with non-garbage `topics_summary.json` —
  verified by eyeball, not just "the endpoint responds."
- Phase 1 (seed): `query_concept_region()` returns the 3 seeded golden
  concept nodes at `promotion_state="canonical"`.
- Phase 2 (adapter): a topic-foundry run's clusters produce
  `ConceptNodeV1` rows in the substrate store at `promotion_state="proposed"`,
  each carrying evidence refs back to source chat turns (no empty-shell
  nodes — every node must cite an `EvidenceNodeV1`/evidence ref).
  Free `co_occurs_with` edges appear for concepts sharing a segment.
- Phase 3 (reconcile fix): two paraphrases of the same concept
  (synthetic test: "surface encodings" / "surface-level representations"
  with a known-similar embedding pair) resolve to one node, not two —
  regression test required.
- Phase 4 (Layer 3): for a manufactured co-occurrence-heavy pair (e.g.
  seeded Orion/Juniper), each of A/B/C produces a relation classification
  within one promotion cycle; results are comparable side by side (same
  pair, three threshold strategies, logged separately).
- Phase 5 (stub kill): `TopicFoundryAdapter.adapt()` either emits a value
  that changes when its input changes (test with two different payloads,
  assert different output) or the organ is removed from `ORGAN_REGISTRY`
  and `STUB_ORGAN_IDS` client-side list is updated to match.
- Phase 6 (salience-gated recall): a turn containing a seeded concept's
  label triggers a context-block injection; a turn without it does not;
  neither turn causes a permanent stance-registry write.
- Phase 8 (Concept Atlas): tab is reachable from `#hubPrimaryNav`; opening
  it does not throw in the console; switching to another tab and back
  does not duplicate Cytoscape instances or leak intervals (verified via
  the `destroy()`-on-hide fix noted in Non-goals/Current architecture);
  the four cards (summary, network, clustering, filters) each load from
  their own endpoint independently — one card's failure must not blank
  the others.

## Recommended next patch

**Phase 0 first, as a live check, not a code change.** Everything else is
contingent on topic-foundry's clustering actually being trustworthy on
real Orion chat data — if it's never been run for real, that becomes the
actual first patch (register a dataset, run a training pass, eyeball the
output) before any adapter work starts.

Then, in order (each phase independently shippable and checkable, per
Juniper's stated preference for phased checkpoints over one upfront plan
approval):

1. Phase 0 — verify topic-foundry has real run output.
2. Phase 1 — seed golden concepts (Orion, Juniper, Orion-Juniper-relationship)
   directly, independent of everything else.
3. Phase 2 — `orion/substrate/adapters/topic_foundry.py`, node conversion +
   free co-occurrence edges.
4. Phase 3 — fix `reconcile.py` identity resolution (embedding
   nearest-neighbor).
5. Phase 4 — Layer 3 relation classification, tuning options A (baseline)
   + B + C built behind a flag for empirical comparison; D documented as
   deferred future work, not built.
6. Phase 5 — kill or fix the `TopicFoundryAdapter` stub.
7. Phase 6 — salience-gated recall collector (`concept_region.py`),
   modeled on `active_packet.py`, registered as a gated lane, not folded
   into `chat_stance.py`'s always-on registry.
8. Phase 7 — confirm/build the HITL review Hub surface.
9. Phase 8 — Concept Atlas Hub tab: own `concept-atlas.js` file (own
   namespace, e.g. `window.OrionConceptAtlas`), decision needed at build
   time between the two established isolation patterns —
   **recommend the Substrate Atlas iframe pattern** (strongest isolation,
   matches the explicit "clicking out of the tab doesn't kill Hub and vice
   versa" requirement by construction) over Organ Signals' inline-panel
   pattern, but fix the observed gap either way: call the equivalent of
   `destroy()`/`stopPolling()` on tab-hide, don't just leave it defined
   and uncalled like the Organ Signals reference implementation does
   today. Four cards (summary stats, network drill-down with god-node
   flagging, clustering — read-only, links out to Topic Studio for
   management — and global filters including a focus-concept selector);
   temporal view explicitly deferred but not blocked, per Non-goals.

Also flagged for cleanup, not blocking: `orion/spark/concept_induction/rdf_materialization.py`/
`graph_mapper.py`/`graph_query.py` appear to be a second, competing,
likely-dead graph-projection attempt inside `concept_induction/` itself —
audit and delete in the same changeset as Phase 2 to avoid three
concept-graph-shaped code paths existing simultaneously.
