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

The doctrine already settled the *rules* (closed allowlists, edges carry meaning, no property without a consumer, `metadata` is a capped quarantine bag, one canonical identity field, `graph_name`/workload label not IRIs). Applying them to the recall-relevant RDF shapes:

### New graph name: `orion_recall`

Per doctrine Appendix B ("two Falkor usages may share one instance with separate graph names... do not silently share labels/properties across rails"): chat/claim/social data has a different vocabulary and write cadence than both `orion_substrate` (concept nodes/edges, tick-rate writes) and `graphiti_temporal` (uuid-keyed episodes/entities). Recommend a **third graph name, `orion_recall`**, same shared FalkorDB instance (`services/orion-falkordb`), not merged into either existing graph.

### Workload keys (extending the doctrine's route table)

```text
recall.chat_turn      {"primary": "falkor", "shadow": "sparql"}   # ladder stage A initially
recall.claim          {"primary": "falkor", "shadow": "sparql"}
recall.social_turn    {"primary": "falkor", "shadow": "sparql"}
recall.step_execution {"primary": "?", "shadow": "?"}             # needs live-consumer audit first (PR #1155 pattern) before committing to migrate
memory.affective_disposition {"primary": "?", "shadow": "?"}       # Hub ontology — see Q2 below, do not assume same graph as recall.*
```

### Starting allowlists (draft — needs Juniper/review sign-off before implementation, per doctrine rule 3 "no property without a consumer")

**`(:ChatTurn)`** — replaces `ORION.ChatTurn` triples:
```text
turn_id, session_id, prompt, response, ts, verb?, model?, node?,
prompt_tokens?, completion_tokens?, total_tokens?, intent?, topic?
```
Edge: `(:ChatSession)-[:HAS_TURN]->(:ChatTurn)`, `(:ChatTurn)-[:HAS_TAG]->(:Tag)`, `(:ChatTurn)-[:HAS_ENTITY]->(:Entity)`.

**`(:Claim)`** — replaces `ORION.Claim` triples:
```text
claim_id, subject_turn_id, predicate, object, confidence?, salience?,
extractor_service, extractor_version?, node?, ts?
```
Edge: `(:ChatTurn)-[:HAS_CLAIM]->(:Claim)` (currently modeled as `subject` property in RDF; property-graph should make this a real edge per rule 1 — "edges carry meaning").

**`(:SocialRoomTurn)`** — mirrors `ChatTurn` shape plus `redaction_level`, `recall_safe`, `continuity_anchor`; `(:SocialConceptEvidence)` as a separate node with `(:SocialRoomTurn)-[:SUPPORTED_BY]->(:SocialConceptEvidence)`.

**Explicitly not proposed here:** a schema for `identity_snapshot`/`goals_proposed`/`world_pulse_graph`/`CognitiveStepExecution` or Hub's `orionmem` ontology — those need their own audit-then-schema pass (see roadmap).

---

## Historical data migration plan (the piece missing from the doctrine)

The doctrine's substrate wedge never needed this — Concept Atlas was in-memory before Falkor, so there was no "historical Fuseki data" to carry forward. Chat/Claim/SocialRoomTurn data is different: it has been accumulating in Fuseki for months and has real recall value (this is literally what `reflect.v1`/`deep.graph.v1` profiles query today).

Proposed shape, once a workload's schema is frozen and reviewed:

1. **Freeze the schema for that workload** (e.g. `recall.chat_turn`) via the allowlist review above — no backfill against a schema that's still moving.
2. **Batch ETL script** (`scripts/`, deterministic, not agent judgment — matches AGENTS.md §4): SPARQL `CONSTRUCT`/`SELECT` per named graph (`orion:chat`, `orion:enrichment`, `orion:chat:social`), paginated, mapped through the same allowlist model the new writer uses (shared Pydantic model — one schema, two producers: live writer and backfill script), `MERGE`d into Falkor keyed on the doctrine's mandated stable identity field (`turn_id`/`claim_id`), so a partial/re-run backfill is idempotent.
3. **Snapshot before writing**, per AGENTS.md §14 (background jobs/backfills) — row counts and a sample to `/tmp/<job-name>/` before any Falkor write, `/tmp/<job-name>/progress.log` during, before/after counts in `/tmp/<job-name>/report.md` after.
4. **Verification**: row-count parity per graph, spot-check N real turns/claims resolvable in Falkor that only existed in Fuseki pre-backfill, confirm `orion-recall`'s adapter (once cut over) returns them.
5. **Cutover the live writer** (`orion-rdf-writer` → Falkor for that workload) using the doctrine's dual-run ladder A→D — do this *after* backfill, not before, so there's no gap between "backfill snapshot taken" and "live writer switched," which would silently drop turns written in between.
6. **Cutover the readers** (`orion-recall`'s adapter) once writer + backfill are both verified.
7. Fuseki stays up, read-only, for that workload until Juniper decides to decommission (matches the substrate-runtime precedent — cut-forward the write path, don't force a Fuseki teardown in the same patch).

This plan is intentionally generic across `recall.chat_turn`/`recall.claim`/`recall.social_turn` — each gets its own backfill run since they're different named graphs, but the mechanism is the same script parameterized by workload.

---

## Phased roadmap (multiple PRs, as instructed — do not attempt as one patch)

```text
Phase 0 (this document): schema + inventory spec. Needs Juniper review before Phase 1 starts.

Phase 1: per-channel live-consumer audits for the not-yet-read node kinds
  (identity_snapshot, goals_proposed, world_pulse_graph, CognitiveStepExecution,
  Hub orionmem/memory-graph-approval), using PR #1155's methodology: check for
  a real Postgres/other consumer before assuming Falkor-migrate is the answer.
  Can run as independent, parallel PRs — no shared code touched.

Phase 2: FalkorRdfStoreClient (or equivalently-named Cypher writer) added to
  orion-rdf-writer for recall.chat_turn + recall.claim, behind the doctrine's
  route table, ladder stage A (primary=fuseki, shadow=falkor) — dark, no
  behavior change yet.

Phase 3: historical backfill script for recall.chat_turn + recall.claim,
  run against the shadow graph written in Phase 2, verified per the plan above.

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

1. **Schema allowlists above** (ChatTurn/Claim/SocialRoomTurn draft) — approve, or specific fields to cut/add before Phase 2 starts? Doctrine rule 3 requires a named consumer per property; the draft above only includes fields `rdf_adapter.py` demonstrably reads today.
2. **Hub's `orionmem` ontology**: fold into `orion_recall`'s graph/schema, or keep as its own graph name entirely (`orion_memory_graph`)? It's a materially different workflow (human-approval, not bus-driven) — recommend its own graph name and its own phase (Phase 8), not blocking Phases 1-7.
3. **`chat_stance.py`'s duplicate `orionmem` SPARQL query** — intentional second consumer, or drift to collapse? Low cost to answer now, avoids carrying a stale duplicate through the whole migration.
4. **Backfill scope**: all historical Fuseki data for `recall.chat_turn`/`recall.claim`, or a bounded window (e.g. last N months)? Affects Phase 3's runtime and snapshot size (AGENTS.md §14's 100k-row/100MB stop-and-ask threshold may bind here — real row count not yet measured this session).
