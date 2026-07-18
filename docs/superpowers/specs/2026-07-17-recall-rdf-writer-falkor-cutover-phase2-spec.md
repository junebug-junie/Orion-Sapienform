# Fuseki full retirement — phase 2: recall, remaining rdf-writer node kinds, Hub memory-graph, historical migration

**Date:** 2026-07-17
**Status:** Proposal — not yet reviewed by Juniper
**Mode:** Proposal (touches memory / recall / cognitive-graph persistence; AGENTS.md §0A proposal mode)
**Extends:** `docs/superpowers/specs/2026-07-16-falkordb-property-graph-routing-design.md` (the doctrine — routing layer, property-graph anti-cathedral rules, dual-run ladder A→D, graph-name-per-rail convention). This document does not redefine any of that; it applies it to the surface the doctrine explicitly deferred.
**Related:**
- `docs/architecture/rdf_store_v1_cutover.md` (Fuseki as active RDF sink — now partially superseded)
- `docs/superpowers/pr-reports/2026-07-17-rdf-writer-cognition-metacog-fuseki-kill-pr.md` (PR #1155, pushed not yet merged as of this writing — killed 2 of ~10 rdf-writer node kinds outright, no Falkor migration needed since Postgres already owned them)
- `services/orion-recall/app/storage/rdf_adapter.py`, `services/orion-recall/app/memory_graph_sparql.py` (the actual code Juniper's original ask points at)

---

## Why this document exists

Juniper's ask was "migrate recall profiles to Falkor, kill RDF writes, identify the stores, decide what to do with existing data." Investigation found:

1. **"Recall profiles" is a read-side problem, not a data-migration problem by itself.** `orion/recall/profiles/*.yaml` are query-shaping configs (`rdf_top_k`, weights); the actual work is rewriting `orion-recall`'s SPARQL read adapters to Cypher against whatever Falkor writer produces the equivalent graph.
2. **Fuseki has more writers than one.** At least three independent, currently-live producers push into it (`orion-rdf-writer`, Hub's memory-graph-approval flow, `orion-graph-compression`), plus two already-quarantined/legacy ones (`orion-gdb-client`, and now `cognition.trace`/`metacog.trace` as of PR #1155). "Kill Fuseki" means retiring all of them, and they don't share a schema today.
3. **A real doctrine for this already exists and is already adopted for one wedge** (`2026-07-16-falkordb-property-graph-routing-design.md`): substrate + Concept Atlas are live on Falkor, drive audits already left RDF for Postgres, and `cognition.trace`/`metacog.trace` are being killed outright (PR #1155, pushed). That doctrine's own "first wedge" explicitly listed **"no rdf-writer metacog/cognition/identity migration"** and **"no Fuseki decommission"** as non-goals — this document is the second wedge picking up exactly what was deferred, using the same routing/property vocabulary rather than inventing a competing one.
4. **Juniper's own framing, given directly in this session:** the property-graph schema needs to be designed before a migration plan for historical data makes sense, and this is explicitly **not a one-shot** — expect multiple agents and multiple PRs. This document is the schema-and-inventory step that unblocks that; it is not itself an implementation PR.

---

## Current state (verified this session, static/config-level — not a fresh live traffic audit; the doctrine's Appendix A traffic numbers are from 2026-07-16 and are now stale post-#1153/#1155)

### Writers into Fuseki

| Writer | Status | Node/edge kinds | Notes |
|---|---|---|---|
| `orion-rdf-writer` | **Live, canonical bus-driven writer.** Backend factory (`rdf_store.py::build_rdf_store_client()`) only knows `graphdb\|fuseki\|generic\|rdf4j` — **no Falkor option exists in code today.** | See full breakdown below | The actual target of "kill RDF writes" for chat/claim/social data |
| Hub memory-graph-approval | **Live.** `MEMORY_GRAPH_APPROVAL_BACKEND=auto` → `RDF_STORE_GRAPH_STORE_URL` + `RDF_STORE_UPDATE_URL` | `orionmem:AffectiveDisposition` + entities/situations/edges, ontology `orionmem-2026-05` | Separate human-approval workflow (`services/orion-hub` memory-graph suggest/approve/consolidation-draft routes), **not** driven by `orion-rdf-writer` or `orion:rdf:enqueue` |
| `orion-graph-compression` | **Live, best-effort.** `writer.py::_build_sparql_update()` | `CompressionRegionV1` region summaries | Postgres (`store.py`) is the real SoR; Fuseki write is explicitly degraded-tolerant (`fuseki_ok=%s` logged, not gated on). Also a **reader** — its Episodic/Substrate/SelfStudy federators pull from Fuseki as primary input for Leiden clustering |
| `cognition.trace` / `metacog.trace` via `orion-rdf-writer` | **Being killed outright, PR #1155 (pushed, not yet merged).** No Falkor migration — Postgres (`cognition_traces`, `orion_metacognitive_trace`) already owned this data; RDF copy was pure redundancy. | — | Precedent for "verify a real consumer exists before migrating; kill instead of migrate if none does" |
| `orion-gdb-client` (GraphDB, not Fuseki) | Quarantined, `GDB_CLIENT_ENABLED=false` by default | — | Out of scope, already dead |

### `orion-rdf-writer`'s remaining subscribe list (post-#1155, from `settings.py::get_all_subscribe_channels()`)

```text
orion:rdf:enqueue              # RdfWriteRequest from orion-cortex-exec, orion-vision-scribe, orion-spark-concept-induction
orion:rdf-collapse:enqueue
orion:collapse:*  (CHANNEL_EVENTS_COLLAPSE)
orion:tags:*      (CHANNEL_EVENTS_TAGGED, CHANNEL_EVENTS_TAGGED_CHAT)  -> Claim nodes (tags.enriched, emit_claims=True)
                  # STALE as of 2026-07-18 (this snapshot predates it): CHANNEL_EVENTS_TAGGED_CHAT
                  # (orion:tags:chat:enriched) removed -- FalkorDB-only now, see
                  # services/orion-meta-tags/README.md "Historical note". This whole block is a
                  # frozen point-in-time snapshot already out of date for other kills too
                  # (orion:worker:rdf / CORTEX_LOG_CHANNEL below were removed by PR #1167) --
                  # trust settings.py::get_all_subscribe_channels() over this list.
orion:chat:social:stored        -> SocialRoomTurn / SocialConceptEvidence
orion:core:events (CHANNEL_CORE_EVENTS)
orion:worker:rdf  (CHANNEL_WORKER_RDF)
CORTEX_LOG_CHANNEL              -> CognitiveStepExecution (rdf_builder.py:241)
orion:chat:history:turn / orion:chat:history:log  -> ChatSession/ChatTurn/ChatMessage
orion:memory:identity:snapshot  -> node kind not yet audited in this pass
orion:memory:goals:proposed     -> node kind not yet audited in this pass
orion:world:pulse:graph (CHANNEL_WORLD_PULSE_GRAPH) -> node kind not yet audited in this pass
```

Node types confirmed by reading `rdf_builder.py` this session: `ChatSession`/`ChatTurn`/`ChatMessage`, `Claim` (subject/predicate/obj/confidence/salience/extractorService/extractorVersion/node/timestamp), `SocialRoomTurn`/`SocialConceptEvidence`, `CognitiveStepExecution`. **Identity-snapshot, goals-proposed, and world-pulse-graph kinds were not read this session** — each needs the same per-channel live-consumer audit PR #1155 already did for cognition/metacog before anyone decides migrate-vs-kill for them. Do not assume Falkor-migrate is the right call by default — PR #1155's finding (Postgres already owns it, RDF was redundant) may repeat.

### Readers of Fuseki relevant to recall

| Reader | File | Notes |
|---|---|---|
| `orion-recall` RDF adapter | `app/storage/rdf_adapter.py` — 7 functions, raw `requests.post()` SPARQL against `RECALL_RDF_ENDPOINT_URL` | **This is what `rdf_top_k` in every recall profile (`reflect.v1`, `deep.graph.v1`, etc.) actually drives.** The literal target of "migrate recall profiles." |
| `orion-recall` memory-graph reader | `app/memory_graph_sparql.py` | Reads Hub's `orionmem:AffectiveDisposition` |
| `orion-cortex-exec` chat_stance | `app/chat_stance.py:311-330` | Runs an **independent, near-identical copy** of the same `orionmem:AffectiveDisposition` SPARQL query, not routed through orion-recall. Possible drift, not confirmed intentional — flag for Q3 below. |
| `orion-cortex-orch` concept profile | `app/concept_profile_config.py` | Reads via `RECALL_RDF_ENDPOINT_URL` alias — separate from Concept Atlas's already-Falkor-live substrate path |
| `orion-graph-compression` federators | Episodic/Substrate/SelfStudy federators | Already fail-open per graph (confirmed live behavior via PR #1155's consumer audit) |

---

## Property-graph schema proposal (per doctrine's anti-cathedral allowlist rules)

**Revision note:** the first version of this section carried the RDF triple shape forward almost unchanged — a generic `Claim` node with `predicate`/`object` string properties is a reified triple, not a property-graph redesign, and directly violates the doctrine's own rule 1 ("edges carry meaning; properties carry measurements"). Juniper caught this. The section below replaces it, grounded in a live query against the running `orion-athena-fuseki` container (2026-07-17) rather than the write/read code shape alone — reading `rdf_builder.py`/`rdf_adapter.py` tells you the *container* RDF uses, not what's actually *in* it, and the two turned out to differ in an important way.

### Ground truth (verified live, 2026-07-17, direct SPARQL against `orion-athena-fuseki`)

| Claim | Evidence |
|---|---|
| `Claim` nodes are structurally redundant (predicate/object could be an edge) but are the *complete* source, not a duplicate of one | `SELECT DISTINCT ?pred WHERE { ?claim a Claim ; predicate ?pred }` → exactly 2 values across all 1,451 Claims: `hasTag` (844), `mentionsEntity` (607). No open-vocabulary claim predicates exist in the live store. |
| Direct `hasTag`/`hasEntity` edges are a **proper subset** of `Claim`, not an independent copy — pair-level set comparison, not just counts | `hasTag`: 704 direct-edge `(turn, value)` pairs, all 704 also exist as a `Claim`; **138 additional pairs exist only as a `Claim`** (16% of the 842 real hasTag facts would be silently dropped by a backfill that reads direct edges only). `mentionsEntity`: 540 direct-edge pairs, all 540 also in `Claim`; **67 pairs exist only as a `Claim`** (11% of 607). Zero direct-only pairs found for either predicate. **Backfill must read from `Claim`, not the direct edges — `Claim` is not disposable data, it is the more complete source.** |
| `confidence`/`salience`/`extractor_service` on `Claim` are dead constants, not real per-fact signal | Checked the full distinct-value set across all 842 hasTag Claims and all 607 mentionsEntity Claims: `confidence` = `{0.0}`, `salience` = `{0.0}`, `extractor_service` = `{"meta-tags"}` — one value each, no variation, ever. `orion-meta-tags` writes these fields but never actually computes them; they are wiring, not evidence. Carrying `0.0` forward as if it were a real "the extractor was 0% confident" score would be a live confabulation risk — it looks like epistemic data and isn't. `timestamp` **is** real and varies per record and is only available via `Claim` (direct edges carry no per-fact timestamp) — worth preserving; the other three are not worth carrying as if meaningful. |
| Tag values are heterogeneous, mostly non-graph-shaped | Sample of 11 distinct `hasTag` values: `"Bruce"`, `"LinkedIn"`, `"dozen"`, `"18 years ago"`, `"about 13 years ago"`, `"about a month ago"`, `"last week"`, `"that day on"`, `"yesterday"`, `"sentiment:negative"`, `"sentiment:positive"`. At least 3 categories tangled into one predicate: real short labels, relative-time expressions, and a `sentiment:` string-prefix convention that is actually a scalar field pretending to be a tag. |
| Entity values include real recurring canonicalizable entities *and* noise | Top `hasEntity` values by frequency: `Juniper` (53), `Orion` (44), `today` (37 — not an entity), `one` (11 — not an entity), `Athena` (10), `2` (8 — not an entity), `Atlas` (8), `Circe` (8), `1` (7), `SMX2` (7), `3`/`4` (6 each), `a DL380 G10` (6), `v100` (6), `32GB`/`350 GB`/`384 GB` (5 each). The mesh's own node names (Juniper/Orion/Athena/Atlas/Circe) recur constantly — exactly the traversal-worthy case a property graph is for. Bare numbers and generic words are NER noise, not entities. |
| Scale is small | 922 distinct `ChatTurn`s total. `orion:enrichment` graph: 1,451 Claims, 704 `hasTag` + 540 `hasEntity` direct edges, 235 separate `Enrichment` provenance records (`collapseId`/`enrichmentType`/`processedBy`/`enriches` — describes the enrichment *process*, not memory content). Well under AGENTS.md §14's 100k-row backfill threshold even combined with chat/social. |
| Graph-name fragmentation already exists in the live store | The same logical graph is split across differently-normalized IRIs from different eras — e.g. `http://conjourney.net/graph/orion/cognition` (652,217 triples) vs `orion:cognition` (493,210 triples) are two *separate* physical Fuseki graphs holding what should be one logical stream, per `graph_iri_for_sparql()`'s normalization changing over time. A naive per-graph-name backfill would double-count or split identical logical data. Query must `UNION` all IRI variants of a logical graph, not trust one name. |
| Most of Fuseki's current content isn't migrating to Falkor at all | `orion:cognition` (~1.1M triples combined) + `orion:metacog` (~434K combined) are being deleted outright, not migrated (PR #1155) — Postgres already owns this data. That's the large majority of everything currently in the store. What's left for `recall.*` (chat/enrichment/social) is comparatively tiny, ~2-3K logical records. |

### Corrected design

1. **Stop *writing* the generic `Claim` node going forward; for backfill, `Claim` is the source to read, not the thing to discard.** These are two different decisions and the first draft conflated them. Going forward, two real typed edges replace it in the live writer:
   - `(:ChatTurn)-[:HAS_TAG {ts}]->(:Tag)`
   - `(:ChatTurn)-[:MENTIONS_ENTITY {ts}]->(:Entity)`

   `confidence`/`salience`/`extractor_service` are **dropped from the schema entirely**, not carried forward as edge properties — the ground-truth table above shows they've never held a real value (constant `0.0`/`"meta-tags"` across all 1,449 live Claims). Doctrine rule 3 ("no property without a consumer") cuts the same way here as for tag noise: a field that has never once varied is not evidence, and keeping it invites a future reader to mistake "wiring never finished" for "the extractor said zero confidence." If `orion-meta-tags` is later fixed to actually compute these, add the field back then, with a producer that populates it — matching PR #1155's own precedent of not carrying forward unused capability just because it's already there.

   For the **backfill**, source facts from `Claim` (not the direct `hasTag`/`hasEntity` edges) — the direct edges are a strict subset of `Claim`'s `(turn, value)` pairs (verified: 704/704 and 540/540 direct pairs also exist as a `Claim`), and `Claim` alone additionally covers 138 hasTag and 67 mentionsEntity facts that have **no** direct edge at all. Reading direct edges only would silently drop ~16%/11% of real historical facts. `ts` from `Claim` carries forward (real, varying); `confidence`/`salience`/`extractor_service` do not (per above — nothing to carry, they were never populated).

2. **`sentiment:*` values are not tags.** Split at ingest (write-time and backfill-time, same transform) into a `sentiment: "positive" | "negative"` property directly on `(:ChatTurn)`. This is real information (an LLM classification of the turn) that was being smuggled through a string-tag mechanism because RDF triples don't have a cheap "just add a field" option — Cypher property graphs do, so use it instead of carrying the workaround forward.

3. **`(:Entity)` is canonical and deduplicated, not one-node-per-mention.** Identity key: normalized name (case-fold, trim). `Juniper` mentioned in 53 different turns is **one** `(:Entity {name: "Juniper"})` node with 53 incoming `MENTIONS_ENTITY` edges — not 53 separate literal-valued nodes. This is the actual point of migrating off RDF-as-triples: real neighbor traversal ("every turn that mentions Circe") becomes a 1-hop query against a stable node instead of a `CONTAINS(LCASE(...))` string scan repeated per query, which is what `rdf_adapter.py` does today (see `_build_chatturn_keyword_filter`). `(:Tag)` gets the same treatment for the subset of tag values worth keeping (see next point).

4. **Filter noise at backfill/write time, don't graph it.** Bare numbers (`"1"`, `"2"`, `"511"`), generic words (`"one"`, `"today"`), and relative-time expressions (`"yesterday"`, `"18 years ago"`, `"about a month ago"`) are not entities or tags with traversal value — nothing in `orion-recall` queries on `tag="today"` today (confirmed: `rdf_adapter.py` never filters by exact tag value, only full-text CONTAINS scans). Doctrine rule 3 ("no property without a consumer") applies at the value level here, not just the field level. Proposed filter: reject `hasEntity`/`hasTag` values that are pure digits, on a small stopword list, or match a relative-time regex, logging `property_cathedral_rejected` per the doctrine's existing observability contract rather than silently dropping — so this decision is inspectable and reversible if a real consumer for temporal tags shows up later.

5. **`Enrichment` provenance records (235, `collapseId`/`enrichmentType`/`processedBy`/`enriches`) are audit/process metadata, not memory content** — per doctrine rule 4 ("scalars and short evidence refs on the graph; large/process data in Postgres"), these are a Postgres candidate, not a Falkor node type. Not designing this further here; flagged for its own live-consumer audit (same PR #1155 methodology) before deciding kill-vs-migrate, matching how `cognition.trace`/`metacog.trace` turned out to be pure redundancy once actually checked.

6. **`(:ChatTurn)`** — the one part of the original draft that was already reasonably grounded (checked against `rdf_builder.py`'s actual write fields, not guessed):
```text
turn_id, session_id, prompt, response, ts, verb?, model?, node?,
prompt_tokens?, completion_tokens?, total_tokens?, intent?, topic?, sentiment?
```
Edge: `(:ChatSession)-[:HAS_TURN]->(:ChatTurn)`.

7. **The direct-edge write is the one to drop going forward, not `Claim`'s content.** Once the live writer emits `HAS_TAG`/`MENTIONS_ENTITY` edges directly (point 1), `orion-rdf-writer` should stop writing *both* the plain `hasTag`/`hasEntity` triple and the `Claim` reification — one edge write replaces both. This is in scope for the Phase 2 writer patch (roadmap below), not deferred.

8. **A separate, real finding surfaced by this investigation, out of scope for this migration but worth its own ticket:** `orion-meta-tags` has apparently never actually computed confidence/salience for tag/entity extraction — every one of 1,449 live records has the exact same placeholder. That's a live instance of AGENTS.md §0A's "no empty-shell cognition" (schema-valid payload, meaningless content) in the current production pipeline, discovered as a side effect of grounding this schema, not something this migration should silently fix or silently carry forward as if it were real.

### New graph name: `orion_recall`

Per doctrine Appendix B ("two Falkor usages may share one instance with separate graph names... do not silently share labels/properties across rails"): chat/entity/tag data has a different vocabulary and write cadence than both `orion_substrate` (concept nodes/edges, tick-rate writes) and `graphiti_temporal` (uuid-keyed episodes/entities). Recommend a **third graph name, `orion_recall`**, same shared FalkorDB instance (`services/orion-falkordb`), not merged into either existing graph. Note the *name* collision risk this doctrine already warns about: recall's `(:Entity)` and Graphiti's `(:Entity)` are different schemas under the same label in different graphs — fine because graph names isolate them, but worth calling out explicitly so nobody later "unifies" them casually.

### Workload keys (extending the doctrine's route table)

```text
recall.chat_turn      {"primary": "falkor", "shadow": "sparql"}   # ladder stage A initially
recall.tag_entity     {"primary": "falkor", "shadow": "sparql"}   # replaces recall.claim — no Claim node in the new design
recall.social_turn    {"primary": "falkor", "shadow": "sparql"}
recall.step_execution {"primary": "?", "shadow": "?"}             # needs live-consumer audit first (PR #1155 pattern) before committing to migrate
memory.affective_disposition {"primary": "?", "shadow": "?"}       # Hub ontology — see Q2 below, do not assume same graph as recall.*
```

**`(:SocialRoomTurn)`** — mirrors `ChatTurn` shape plus `redaction_level`, `recall_safe`, `continuity_anchor`; `(:SocialConceptEvidence)` as a separate node with `(:SocialRoomTurn)-[:SUPPORTED_BY]->(:SocialConceptEvidence)`. **Not yet ground-truthed against live data** the way ChatTurn/Claim/Tag/Entity now are (545 combined triples across graph-name variants, per the scale table above, but the *value* distribution — is `SocialConceptEvidence` similarly redundant with something else? — has not been queried). Do the same live-query pass before finalizing, not before Phase 2 implementation starts.

**Explicitly not proposed here:** a schema for `identity_snapshot`/`goals_proposed`/`world_pulse_graph`/`CognitiveStepExecution`, the `Enrichment` provenance record type, or Hub's `orionmem` ontology — those need the same live-data-grounding pass this section just did for Claim/Tag/Entity, not a guess from reading the writer code. Do not let Phase 1's audits skip this step — reading `rdf_builder.py` alone would have shipped the wrong `Claim` schema.

---

## Historical data migration plan (the piece missing from the doctrine)

The doctrine's substrate wedge never needed this — Concept Atlas was in-memory before Falkor, so there was no "historical Fuseki data" to carry forward. Chat/Tag/Entity/SocialRoomTurn data is different: it has been accumulating in Fuseki for months and has real recall value (this is literally what `reflect.v1`/`deep.graph.v1` profiles query today). At real measured scale (922 ChatTurns, ~1,244 real tag/entity facts after dedup, ~545 social-turn triples) this is a small backfill, not a big-data problem — well under AGENTS.md §14's 100k-row/100MB stop-and-ask threshold. Confirm the same for `SocialRoomTurn`/`SocialConceptEvidence` once that schema is ground-truthed (still open per above).

Proposed shape, once a workload's schema is frozen and reviewed:

1. **Freeze the schema for that workload** (e.g. `recall.chat_turn`) via the allowlist review above — no backfill against a schema that's still moving.
2. **Batch ETL script** (`scripts/`, deterministic, not agent judgment — matches AGENTS.md §4): SPARQL `SELECT` per **logical** named graph — `UNION`ing all IRI variants of that graph (see the "graph-name fragmentation" ground-truth row above; querying only `orion:chat` and missing `http://conjourney.net/graph/orion/chat` would silently drop ~25% of the turns), paginated, mapped through the same allowlist model the new writer uses (shared Pydantic model — one schema, two producers: live writer and backfill script).
3. **Transform, not just copy**: this is where the corrected schema actually gets applied, not just a syntax port —
   - **Source from `Claim` nodes, not the direct `hasTag`/`hasEntity` edges** — verified live that direct edges are a strict subset (704/704 and 540/540 also present as a `Claim`); reading direct edges only would silently drop the 138 hasTag + 67 mentionsEntity facts that exist *only* as a `Claim`. One `HAS_TAG`/`MENTIONS_ENTITY` edge per `Claim`, carrying `ts` forward; `confidence`/`salience`/`extractor_service` are dropped, not carried (verified dead constants — see ground truth).
   - Split `sentiment:*` tag values into the `ChatTurn.sentiment` property; they never become `Tag` nodes.
   - Reject digit-only / stopword / relative-time-regex tag and entity values (logged `property_cathedral_rejected`, per doctrine's observability contract, not silently dropped — auditable and reversible).
   - Normalize remaining `Entity`/`Tag` values to a canonical identity key (case-fold + trim) so `MERGE` actually deduplicates `"Juniper"` across all 53 mentions into one node, rather than creating near-duplicate nodes per casing variant.
4. `MERGE`d into Falkor keyed on the doctrine's mandated stable identity field (`turn_id` for ChatTurn, `(name)` normalized for Entity/Tag — not a synthetic `claim_id`, since `Claim` no longer exists as a graph node after backfill, only as the historical source it was read from), so a partial/re-run backfill is idempotent.
5. **Snapshot before writing**, per AGENTS.md §14 (background jobs/backfills) — row counts and a sample to `/tmp/<job-name>/` before any Falkor write, `/tmp/<job-name>/progress.log` during, before/after counts plus the reject-log summary (how many tag/entity values were filtered, and which) in `/tmp/<job-name>/report.md` after.
6. **Verification**: row-count parity per graph (accounting for the intentional dedup/filter — the after-count for `recall.tag_entity` should be *lower* than raw triple count, and that's correct, not a bug), spot-check N real turns/entities resolvable in Falkor that only existed in Fuseki pre-backfill, confirm a real Falkor entity node (e.g. `Circe`) has all 8 expected incoming edges, confirm `orion-recall`'s adapter (once cut over) returns them.
7. **Cutover the live writer** (`orion-rdf-writer` → Falkor for that workload, with the `emit_claims` double-write removed per the Corrected design section above) using the doctrine's dual-run ladder A→D — do this *after* backfill, not before, so there's no gap between "backfill snapshot taken" and "live writer switched," which would silently drop turns written in between.
8. **Cutover the readers** (`orion-recall`'s adapter) once writer + backfill are both verified.
9. Fuseki stays up, read-only, for that workload until Juniper decides to decommission (matches the substrate-runtime precedent — cut-forward the write path, don't force a Fuseki teardown in the same patch).

This plan is intentionally generic across `recall.chat_turn`/`recall.tag_entity`/`recall.social_turn` — each gets its own backfill run since they're different named graphs, but the mechanism is the same script parameterized by workload.

---

## Phased roadmap (multiple PRs, as instructed — do not attempt as one patch)

```text
Phase 0 (this document): schema + inventory spec. Needs Juniper review before Phase 1 starts.

Phase 1: DONE (2026-07-17, this session) — per-channel live-consumer audits
  for the not-yet-read node kinds, using PR #1155's methodology. Findings:

  - `orion:cortex:telemetry` (CognitiveStepExecution, CORTEX_LOG_CHANNEL) —
    KILL. Zero producers anywhere in the repo, not even registered in
    channels.yaml. Dead subscription with zero possible live traffic —
    trivial: unsubscribe `orion-rdf-writer`, delete the handler. No Falkor
    work, no backfill (there is nothing to backfill).
  - `orion:world_pulse:graph:upsert` (GraphDeltaPlanV1) — KILL /
    deprioritize. Real producer (`orion-world-pulse`) and real dispatch code
    exist, but `WORLD_PULSE_GRAPH_ENABLED=false` in the live `.env` — fully
    inert today, and even enabled it defaults to dry-run. Not worth reviving
    a dormant SPARQL path; if world-pulse ever needs graph storage, design
    it fresh against Falkor rather than migrate dead code.
  - `orion:memory:identity:snapshot` / `orion:memory:goals:proposed`
    (IdentitySnapshotV1/GoalProposalV1) — KILL. Live Fuseki query found
    **zero** graphs matching autonomy/identity/goals — this path has never
    recorded a single triple in the running store, despite a real producer
    (`orion/spark/concept_induction`) existing. Meanwhile `identity_snapshots`
    already has a real, actively-pruned Postgres store
    (`services/orion-self-state-runtime/app/store.py::SelfStateRuntimeStore`,
    real SQLAlchemy engine + batched prune job) and `goal_context_listener.py`
    in `orion-substrate-runtime` already consumes goal proposals for live
    active-goal state. Same shape as the cognition/metacog kill (PR #1155):
    a real consumer already exists elsewhere; RDF was never actually load-bearing.
  - Hub `orionmem`/memory-graph-approval — MIGRATE, not kill (confirms the
    original Phase-2-spec framing). `memory_graph_routes.py::approve_memory_graph_draft`
    genuinely writes both Postgres (draft/workflow state — `asyncpg.PostgresError`
    handled explicitly) and Fuseki (the graph content itself — separate
    `requests.RequestException`/Fuseki-lock-exhaustion handling) — Postgres
    does not duplicate the graph content, only the approval workflow around
    it. Real downstream consumers exist (`orion-recall`'s `memory_graph_sparql.py`,
    `orion-cortex-exec`'s `chat_stance.py`). Stays its own phase (8) with its
    own schema design, not started here.

  **Net effect: 3 of 5 audited kinds are trivial kills with no Falkor work
  and nothing to backfill (never had live data, or already durably owned by
  Postgres). Only Hub's `orionmem` is real migration work.** This meaningfully
  shrinks total scope — most of what's left in Fuseki outside chat/tag/entity
  turns out to already be dead weight, not migration debt.

Phase 2: DONE (2026-07-18) — see
  docs/superpowers/plans/2026-07-18-recall-tag-entity-falkor-writer-plan.md
  for the full design. **Corrected mid-review: this does NOT live in
  orion-rdf-writer** as originally written above — that service's own
  doctrine role is "legacy RDF materialize" only. The Cypher writer lives in
  `orion-meta-tags` (`app/falkor_recall_writer.py`), the actual producer of
  tag/entity extraction, matching how every other real Falkor producer in
  the codebase works (substrate-runtime, graphiti-adapter, concept-induction
  all write from their own originating service, never bolted onto
  rdf-writer). Also corrected: `recall.chat_turn` turned out to not need a
  live writer at all — `chat.history` RDF was killed by a concurrent PR
  before this phase started, so `ChatTurn` is now a thin anchor node
  (turn_id/session_id/ts/correlation_id, no text) created on demand by the
  tag_entity write, not a separate dual-run workload. No `GraphPersistenceRouter`
  route table used — a plain `RECALL_FALKOR_TAG_ENTITY_ENABLED` flag instead,
  since orion-rdf-writer and orion-meta-tags are two independent services
  each reacting to the same event, not one service picking a primary/shadow
  backend for its own write (the router's actual shape). `emit_claims`
  double-write in orion-rdf-writer is untouched by this phase — that
  service's own Fuseki write continues exactly as before; only
  `orion-meta-tags` gained a new, additive, dark-by-default write.

Phase 3: historical backfill script for recall.chat_turn + recall.tag_entity,
  run against the shadow graph written in Phase 2, verified per the plan
  above — including the dedup/filter transform, not a raw copy.

Phase 4: orion-recall's rdf_adapter.py Cypher rewrite for the two functions
  that matter most (fetch_rdf_chatturn_fragments, fetch_rdf_graphtri_fragments),
  gated behind a new flag mirroring RECALL_GRAPHITI_IN_CHAT's pattern
  (e.g. RECALL_FALKOR_IN_CHAT), ships dark.

Phase 5: flip ladder to primary=falkor shadow=fuseki (stage B), verify parity,
  then shadow=none (stage C), flip the flag live (stage D retires the SPARQL
  producer path for this workload).

Phase 6: repeat Phase 2-5 for recall.social_turn.

Phase 7: SocialConceptEvidence, then whatever Phase 1's audits determined is
  real (not already-redundant like cognition/metacog turned out to be).

Phase 8: Hub orionmem/memory-graph-approval — separate design decision
  (Q2 below), own phase regardless of outcome.

Phase 9: chat_stance.py's duplicate orionmem query — collapse into the same
  read path as orion-recall's memory_graph_sparql.py, or confirm intentional
  (Q3 below).

Phase 10: Fuseki decommission checklist, once all above are verified in
  production and Juniper explicitly signs off — not bundled into any earlier
  phase.
```

---

## Non-goals (this document)

- Not writing any code — this is Phase 0 only.
- Not deciding Hub's `orionmem` ontology unification with `recall.chat_turn`'s schema (Q2).
- Not auditing `identity_snapshot`/`goals_proposed`/`world_pulse_graph`/`CognitiveStepExecution` node shapes in detail — flagged for Phase 1, not designed here.
- Not proposing Fuseki decommission timing — Phase 10, gated on everything else.
- Not re-litigating the doctrine's routing layer, property rules, or dual-run ladder — reused as-is.

## Open questions for Juniper

1. **Schema design (ChatTurn/Tag/Entity, "Corrected design" section)** — approve killing `Claim` and replacing with real `HAS_TAG`/`MENTIONS_ENTITY` edges plus a `sentiment` property, or object to any of the 7 points? The noise-filtering rule (point 4 — reject digit-only/stopword/relative-time values) is the one most worth a second look: it's a real, lossy decision, not just a rename. Alternative if full rejection feels too aggressive: keep filtered values but flag them `low_quality: true` instead of dropping, so they're excluded from default traversal/recall scoring but not destroyed.
2. **Hub's `orionmem` ontology**: fold into `orion_recall`'s graph/schema, or keep as its own graph name entirely (`orion_memory_graph`)? It's a materially different workflow (human-approval, not bus-driven) — recommend its own graph name and its own phase (Phase 8), not blocking Phases 1-7. Needs the same live-query grounding pass as ChatTurn/Claim before its schema is drafted — not done in this document.
3. **`chat_stance.py`'s duplicate `orionmem` SPARQL query** — intentional second consumer, or drift to collapse? Low cost to answer now, avoids carrying a stale duplicate through the whole migration.
4. **Backfill scope**: real measured scale is small (922 ChatTurns, ~1,244 tag/entity facts pre-dedup, ~545 social-turn triples) — well under the AGENTS.md §14 threshold, so recommend backfilling all of it rather than a bounded window, unless there's a reason (PII, staleness) to want a cutoff Juniper knows about that isn't visible from row count alone.
