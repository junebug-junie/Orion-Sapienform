# FalkorDB property-graph doctrine + persistence router — design spec

**Date:** 2026-07-16
**Status:** WEDGE LIVE — guts implemented on `feat/falkordb-property-graph-routing` (intent, routes, property guard, Falkor/routed stores, `services/orion-falkordb/`), merged via #1099. **2026-07-16:** graphiti `--profile falkordb` removed; live Falkor cutover to shared stack documented in `services/orion-falkordb/README.md`. **2026-07-17:** Concept Atlas / Hub wiring done and live-verified (`SUBSTRATE_STORE_BACKEND=falkor`, PR #1105) — no longer deferred; see the acceptance-checks update below. **2026-07-17:** Spark concept-induction post-save materialization cuts over to Cypher-native `FalkorSubstrateStore` (`CONCEPT_PROFILE_GRAPH_BACKEND=falkor`; concept-only writes) — no longer emits `rdf.write.request` for profiles when Falkor path is live. `substrate-runtime`'s own cutover remains open (freshly re-checked 2026-07-17, not just carried over from the ground-truth table above: `services/orion-substrate-runtime/.env` still reads `SUBSTRATE_STORE_BACKEND=sparql`) and the `orion:kg:edge:ingest.v1` → rdf-writer deprecation remain open, tracked separately.
**Mode:** Proposal (touches memory / cognitive-graph persistence; AGENTS.md §0A proposal mode)
**Related:**
- `docs/superpowers/specs/2026-07-15-concept-atlas-graph-pipeline-design.md` (first consumer seam)
- `docs/superpowers/specs/2026-07-13-graphiti-core-backend-activation-spec.md` (existing Falkor use)
- `docs/architecture/rdf_store_v1_cutover.md` (Fuseki as active RDF sink today)
- `docs/superpowers/pr-reports/2026-07-15-drive-audit-graph-path-kill-pr.md` (wrong-tool precedent)

---

## Arsonist summary

Fuseki/TDB2 is the wrong tool for Orion's hot cognitive workloads: append-heavy SPARQL UPDATE loops, index amplification, compaction downtime, and disk we cannot afford. Recent program evaluation has already moved several workloads (notably drive audits) to SQL because they were never graphs. What remains graph-shaped is **properties and typed edges**, not strict RDF triples.

FalkorDB is already in the stack (Graphiti crystallizations). Concept Atlas already builds a real concept graph on `SubstrateGraphStore` and explicitly refused SPARQL/RDF. The missing piece is not another database shopping trip — it is a **doctrine + routing layer** so every new graph write has:

1. a typed intent (not N-Triples as the interchange format),
2. a workload-keyed route (rdf / falkor / sql / dual / disabled),
3. a property contract that forbids cathedral metadata bloat,
4. a first real producer/consumer pair that proves the rail.

**First producer/consumer:** substrate + concept-atlas / topic-foundry (the Concept Atlas pipeline). That is the narrow implementation wedge; this document is the broader contract that wedge must obey.

---

## Core question

How does Orion use FalkorDB as a first-class property-graph rail — with explicit property design, service contracts, and a dual-path router that can keep legacy RDF workloads alive during migration — without recreating RDF-shaped bloat as “property cathedrals”?

---

## Ground truth (verified 2026-07-16)

| Claim | Evidence |
|---|---|
| Fuseki is live and under write pressure | `orion-athena-fuseki` healthy; ~7.2k SPARQL UPDATEs / 30 min (~4/sec), mostly substrate-runtime |
| Substrate-runtime is on SPARQL | `services/orion-substrate-runtime/.env`: `SUBSTRATE_STORE_BACKEND=sparql` |
| Hub / cortex substrate defaults are in-memory | hub + cortex-exec: `SUBSTRATE_STORE_BACKEND=in_memory` — Concept Atlas and chat do not share the runtime store instance |
| Drive audits already left RDF | 2026-07-15 kill PR: Postgres `drive_audits` is sole sink; rdf-writer handler deleted |
| Concept Atlas pipeline landed on substrate, not RDF | adapter, reconcile embedding merge, relation classification, Hub Concept Atlas tab |
| Concept Atlas design forbids SPARQL wiring | 2026-07-15 spec non-goal: do not wire into `GraphDBSubstrateStore` / rdf-writer |
| FalkorDB already serves crystallizations | `orion-graphiti-adapter` + `FALKORDB_ENABLED=true`, `RELATES_TO` schema live |
| RDF writer abstraction is too late for Falkor | `RdfStoreClient.write_graph(content: str, …)` — N-Triples only |
| Substrate protocol already speaks nodes/edges | `orion/substrate/store.py::SubstrateGraphStore` |
| Topic-foundry still has a competing RDF ingest channel | `orion:kg:edge:ingest.v1` → rdf-writer / graphdb (Concept Atlas path is the keep) |
| Concept-induction profile materialization is Falkor (not RDF) when live | `CONCEPT_PROFILE_GRAPH_BACKEND=falkor` → `FalkorSubstrateStore` concept-only writes into `orion_substrate`; RDF path retained as `rdf` compat only |

---

## What FalkorDB is for (and is not)

### FalkorDB is for

- **Typed property graphs** where traversal / neighborhood / hybrid search is the job
- **Hot cognitive working graphs** that today thrash Fuseki (substrate nodes/edges at tick rates)
- **Relation-shaped memory** already proven on Graphiti (`Entity` + `RELATES_TO`)
- **Concept Atlas durability** when in-memory is no longer acceptable across restarts
- Workloads whose natural unit is `(node_kind, properties, edge_predicate, edge_properties)`

### FalkorDB is not for

- Append-only audit / time-series (drives, traces) → **Postgres**
- Blob / document SoR → **Postgres** (or object storage), project edges if needed
- Ontology / OWL / SHACL reasoning → not a goal; do not reintroduce via Cypher labels
- “Put everything in the graph so it feels connected” → property cathedral
- Replacing Graphiti’s Postgres neighborhood projection overnight — Graphiti keeps Postgres SoR for crystallizations; Falkor remains the hybrid-search / relation projection unless a later design says otherwise

### Other engines explicitly out of scope for this doctrine

| Engine | Why not the standard |
|---|---|
| LadybugDB | Embedded / in-process; already only for GitNexus FTS in harness-governor. Not a multi-service bus sink. |
| Memgraph | In-memory-first durability story we do not want to own while escaping Fuseki ops. |
| ArcadeDB | Multi-model + optional RDF invites never leaving triples. |
| New RDF store | Solves engine pain, not representation pain. |

One hot property-graph engine (FalkorDB) + Postgres SoR + temporary RDF compatibility. No third graph database without a new design review.

---

## Property design doctrine (anti-cathedral)

RDF bloat came from treating every fact as a triple and every namespace as a named graph. Property-graph bloat comes from stuffing arbitrary keys onto nodes “just in case.” Same failure mode, different syntax.

### Rules

1. **Edges carry meaning; properties carry measurements.**  
   Prefer `(:Concept)-[:CONTRADICTS]->(:Concept)` over `concept.flags.contradicts_ids = [...]`.

2. **Closed property sets per node/edge kind.**  
   Every writable kind declares an allowlist (Pydantic model with `extra="forbid"`). Open `metadata: dict` is a **quarantine bag**, not a schema — see rule 6.

3. **No property without a consumer.**  
   Adding a property requires naming: producer, consumer, and one test or Hub/debug surface in the same changeset (AGENTS.md §0A). Names without consumers are rejected.

4. **Scalars and short enums only on the graph node.**  
   Large text, embeddings, raw evidence payloads stay in Postgres / vector host; the graph stores **refs** (`evidence_ref`, `embedding_ref`, `crystallization_id`).

5. **Identity is explicit and stable.**  
   One canonical id field per kind (`node_id`, `uuid`, `crystallization_id`). Do not invent parallel `id` / `key` / `iri` / `uri` without a migration note. Graphiti lesson: wrong identity property → empty search.

6. **`metadata` is write-audited and size-capped.**  
   - Max keys / max bytes per node (exact caps in implementation plan).  
   - Unknown keys log `property_cathedral_rejected` and are dropped (or fail closed in dual-run).  
   - Promote a metadata key to a first-class field only when a second consumer needs it.

7. **Promotion / lifecycle is a field, not a parallel graph.**  
   Reuse `SubstratePromotionStateV1` / crystallization status. Do not create `ProposedConcept` vs `CanonicalConcept` label taxonomies unless a query requires it and a consumer exists.

8. **Named-graph habit → `graph_name` / workload label, not IRIs.**  
   Routing uses workload keys (`substrate.concept`, `memory.crystallization`, …). Compatibility RDF graph IRIs are adapter-only, never the interchange contract.

### Starting allowlists (normative for v1)

**Substrate concept node (Concept Atlas path)** — fields already in `ConceptNodeV1` + provenance/activation/temporal; do not expand without a consumer:

```text
node_id, node_kind, label, definition?,
anchor_scope, subject_ref?, promotion_state, risk_tier,
temporal, signals, provenance, metadata (quarantine)
```

**Substrate edge** — `SubstrateEdgeV1` predicates stay closed (`supports` / `contradicts` / `refines` / `co_occurs_with` / …). New predicates require schema + registry + consumer in one patch.

**Graphiti Entity (existing)** — keep current mapping (`uuid`, `name`, `group_id`, `crystallization_id`, `sensitivity`, embeddings as Vectorf32). Do not merge substrate concept properties into Entity without an explicit dual-model design.

### Anti-patterns (reject in review)

- `metadata.ontology_path`, `metadata.shacl_class`, free-form `tags: list[str]` with no scorer
- Per-emotion / per-symptom property dictionaries
- Storing full chat turns or N-Triples blobs as node properties
- “Temporary” keys that survive two releases without a consumer

---

## Services and contracts

### Logical components

```text
┌──────────────────────────────┐
│ Producers                    │
│  substrate-runtime           │
│  hub concept-atlas ingest    │
│  topic-foundry (via atlas)   │
│  graphiti-adapter (existing) │
│  rdf-writer (legacy)         │
└──────────────┬───────────────┘
               │ GraphWriteIntent / SubstrateGraphStore ops
               ▼
┌──────────────────────────────┐
│ Graph persistence router     │  (new — shared library, not necessarily a new service)
│  route table by workload     │
│  dual-run + parity hooks     │
└──────┬───────────┬───────────┘
       │           │
       ▼           ▼
┌────────────┐  ┌─────────────────┐
│ RDF adapter│  │ Falkor adapter  │
│ fuseki     │  │ Cypher / driver │
└────────────┘  └────────┬────────┘
                         │
                  ┌──────┴──────┐
                  │ Postgres    │  SoR for audits, crystallizations,
                  │ (existing)  │  optional substrate snapshot tables
                  └─────────────┘
```

### New shared contract: `GraphWriteIntent` (proposed)

Location (proposed): `orion/graph/write_intent.py` + schema registry entry.

Not N-Triples. Minimum shape:

```text
GraphWriteIntentV1
  workload: str          # e.g. "substrate.concept", "substrate.drive_state"
  operation: upsert_node | upsert_edge | delete_node | delete_edge | append_event
  identity_key: str      # stable upsert key
  node?: { kind, id, properties }   # closed model per kind
  edge?: { predicate, source_id, target_id, properties }
  provenance: { producer, source_refs[], observed_at }
  compatibility:
    rdf_graph_name?: str # only if route includes rdf
  routing_hint?: str     # optional override; default from route table
```

**Invariant:** adapters may *emit* RDF or Cypher; producers must not.

### Existing contracts to keep / extend

| Contract | Role |
|---|---|
| `SubstrateGraphStore` | First router-facing API for substrate workloads (wrap, do not replace) |
| `ConceptNodeV1` / `SubstrateEdgeV1` | Property allowlists for concept atlas |
| Graphiti episode / Entity / `RELATES_TO` | Crystallization rail; stays behind graphiti-adapter |
| `RdfWriteRequest` / `orion:rdf:enqueue` | Legacy only; new hot paths must not add kinds without routing review |
| `KgEdgeIngestV1` | Competing topic-foundry → RDF path; retire toward atlas adapter (Concept Atlas non-goal already) |

### Route table (config contract)

Workload-keyed, not global `GRAPH_BACKEND=…`.

Proposed env shape (exact keys in implementation plan):

```text
GRAPH_PERSISTENCE_ROUTES_JSON=
{
  "substrate.concept": {"primary": "falkor", "shadow": "none"},
  "substrate.drive_state": {"primary": "falkor", "shadow": "sparql"},
  "substrate.brain_frame": {"primary": "falkor", "shadow": "sparql"},
  "memory.crystallization": {"primary": "falkor", "shadow": "none"},
  "rdf.metacog_trace": {"primary": "rdf", "shadow": "none"},
  "rdf.cognition_trace": {"primary": "rdf", "shadow": "none"}
}
```

Allowed targets: `falkor` | `sparql` | `rdf` | `in_memory` | `postgres` | `disabled`  
Shadow modes: `none` | same set for dual-write / compare.

`SUBSTRATE_STORE_BACKEND=routed` (or `falkor`) becomes the substrate-runtime / Hub entry that consults this table for substrate.* workloads.

### Service ownership

| Concern | Owner service / package |
|---|---|
| Route table + adapters | `orion/graph/` (shared) + thin wiring in callers |
| Substrate tick producers | `orion-substrate-runtime` |
| Concept Atlas read/ingest | `orion-hub` (`concept_atlas_routes.py`) |
| Topic → concept nodes | `orion/substrate/adapters/topic_foundry.py` (already) |
| Crystallization → Falkor | `orion-graphiti-adapter` (already; remains separate rail initially) |
| Legacy RDF materialize | `orion-rdf-writer` |
| Fuseki ops | `orion-rdf-store` (shrinking role) |
| SQL SoR | `orion-sql-db` / `orion-sql-writer` |

**Do not** invent a new “graph-router” microservice in v1 unless dual-run and multi-writer contention force it. Prefer a shared library used by existing processes. Promote to a service only with evidence.

### Observability contract

Every routed write emits structured logs / traces:

```text
graph_route_selected workload=… primary=… shadow=…
graph_write_committed backend=… workload=… identity_key=… elapsed_ms=…
graph_parity_mismatch workload=… field=…   # dual-run only
property_cathedral_rejected workload=… keys=…
```

Acceptance for ops: Fuseki `/update` rate must be attributable per workload after routing ships.

---

## Routing layer behavior

1. Resolve `workload` → primary (+ optional shadow).
2. Validate properties against kind allowlist.
3. Write primary; on failure follow workload policy (`fail_open` vs `fail_closed` — substrate hot path defaults fail_open to in-memory/cache like today; dual-run shadow failures never fail the primary).
4. If shadow set, write async or sync-best-effort; compare snapshot hashes or node counts periodically (not every tick for brain-frame).
5. Never send N-Triples into the Falkor adapter. Never send Cypher into the RDF adapter. Adapters own serialization.

### Dual-run ladder (per workload)

```text
A. primary=sparql|rdf, shadow=falkor     # learn parity
B. primary=falkor, shadow=sparql|rdf     # prove consumers
C. primary=falkor, shadow=none           # cut shadow
D. disable legacy producer path          # stop paying twice
```

### Read path

Reads follow primary. Shadow is write/compare only unless an explicit `read_backend` override exists for debugging. Concept Atlas and substrate snapshot APIs must not silently fan out to Fuseki once primary is Falkor.

---

## First producer / consumer: substrate + concept induction seam

This section is the **narrow wedge**. It does not shrink the doctrine above; it is the first proof.

### Why this seam

- Concept Atlas already produces/consumes typed concept nodes and edges on `SubstrateGraphStore`.
- Topic-foundry → substrate adapter already exists; competing RDF `KgEdgeIngest` path is the one to starve.
- Substrate-runtime is the dominant Fuseki UPDATE source; moving `substrate.*` workloads drops ~88% of update traffic (2026-07-16 audit).
- Concept induction **profiles** and **DriveEngine** share a process but not a data dependency — do not route drive *audits* to Falkor (Postgres SoR stays). Drive *state projection into substrate* is a substrate workload, not a drives-RDF revival.

### Producer set (phase 1)

| Producer | Workload key | Today | Target |
|---|---|---|---|
| Hub Concept Atlas ingest (topic-foundry) | `substrate.concept` | Falkor (`SUBSTRATE_STORE_BACKEND=falkor`) | keep |
| Spark concept-induction profile post-save | `substrate.concept` | Cypher-native Falkor (`CONCEPT_PROFILE_GRAPH_BACKEND=falkor`; concept-only) | keep; RDF compat via `=rdf` |
| Substrate-runtime drive_state materialization | `substrate.drive_state` | SPARQL upserts | Falkor primary, sparql shadow then off |
| Substrate-runtime brain-frame / dynamics | `substrate.brain_frame` (and siblings) | SPARQL | same ladder |

### Consumer set (phase 1)

| Consumer | Need |
|---|---|
| Concept Atlas Hub APIs | `query_concept_region` / summary / network against durable store |
| Substrate-runtime snapshot / brain-frame | same `SubstrateGraphStore.snapshot` semantics |
| Recall `concept_region` collector | salience-gated read; must see same store Hub writes |

### Explicit non-goals for the first wedge

- No drive-audit graph revival
- No rdf-writer metacog/cognition/identity migration
- No merging Graphiti Entity schema with ConceptNodeV1
- No Ladybug/Memgraph/Arcade introduction
- No Fuseki decommission (freeze pressure first; decommission later)

### Success for the wedge (measurable)

1. Concept Atlas ingest → Falkor → Hub network view survives hub restart.
2. Substrate-runtime primary=falkor → Fuseki `/update` rate drops ≥80% from the 30-min baseline (~7.2k).
3. Property allowlist rejects a synthetic cathedral key in tests.
4. Dual-run parity report for `substrate.concept` shows zero critical mismatches before shadow off.
5. `KgEdgeIngest` → rdf-writer path documented as deprecated; no new topic-foundry features target it.

---

## Relationship to existing rails

```text
                    ┌── Postgres SoR (crystallizations, drive_audits, cards)
Memory beliefs ─────┼── Graphiti adapter ── Falkor (RELATES_TO search)
                    └── (unchanged by this doctrine initially)

Concept cognition ── SubstrateGraphStore ── Router ── Falkor (Concept Atlas)
                                      └── (temp) SPARQL shadow

Legacy RDF dumps ── rdf-writer ── Fuseki (traces, residual named graphs)
```

Two Falkor *usages* may share one FalkorDB instance with **separate graph names** (`graphiti_temporal` vs `orion_substrate`) until a unification design exists. Do not silently share labels/properties across rails.

---

## Risks

| Severity | Risk | Mitigation |
|---|---|---|
| High | Hub in-memory vs runtime SPARQL split means “Falkor primary” must be wired in **both** processes or Atlas and runtime diverge | Single `build_substrate_store_from_env()` + shared route table; integration test across hub ingest + runtime snapshot |
| High | Property cathedral via `metadata` | Allowlist + reject logs + size caps; review checklist |
| Medium | Graphiti + substrate collide in one Falkor graph | Separate graph names; documented ownership |
| Medium | Dual-run doubles write load during ladder A/B | Short windows; shadow async; never dual entire rdf-writer firehose |
| Medium | Treating router as excuse to keep Fuseki forever | Ladder requires shadow-off exit criteria per workload |
| Low | New microservice temptation | Library-first; service only with contention evidence |

---

## Non-goals (doctrine-level)

- Full Fuseki retirement in the first implementation plan
- SPARQL compatibility layer inside Falkor
- New ontology / SKOS / OWL layer
- Keyword trigger lists for conversational stance (unrelated; still banned)
- Replacing Postgres neighborhood BFS in Graphiti
- Migrating Ladybug GitNexus usage

---

## Acceptance checks (before calling doctrine “adopted”)

- [ ] Spec reviewed by Juniper
- [ ] Implementation plan exists for wedge only (substrate + concept atlas), linking back to this doctrine
- [ ] `GraphWriteIntentV1` (or equivalent) registered in schema registry with producer+consumer tests
- [x] Route table documented in `.env_example` for substrate-runtime (#1099) and hub (#1105)
- [ ] Property allowlist tests include a cathedral rejection case
- [ ] Live metric: Fuseki update rate attribution by workload after wedge deploy
- [x] Concept Atlas durability demo (restart hub, concepts remain) — 2026-07-17, live-verified against
      the real `orion-athena-falkordb` container, not just Hub's API response (which would look
      identical on the old `in_memory` backend): `redis-cli GRAPH.QUERY orion_substrate "MATCH (n)
      RETURN count(n)"` → 3 nodes, `MATCH ()-[e]->() RETURN type(e), count(e)` → `associated_with`:
      2, matching Hub's `/api/substrate/concepts/summary` exactly. Hub container restarted a second
      time with no config change; FalkorDB's node count stayed at 3 (not 6, same `node_id`s) —
      confirms idempotent upsert against a real persistent backend, not a store recreated fresh
      on every boot.
- [x] Concept-induction post-save no longer RDF-materializes when `CONCEPT_PROFILE_GRAPH_BACKEND=falkor`
      (concept-only Cypher-native writes into shared `orion_substrate`; unit-tested). Live outcome is
      concept **nodes** only — the profile mapper emits no concept↔concept edges yet, and induction
      writes bypass Hub's identity materializer (parallel identity islands; merge is a named follow-up).

---

## Recommended next patch (after spec approval)

1. **Implementation plan** for wedge: `RoutedSubstrateGraphStore` / `SUBSTRATE_STORE_BACKEND=falkor|routed`, Falkor graph name `orion_substrate`, dual-run for `substrate.drive_state` + `substrate.concept` only.
2. Wire Hub Concept Atlas + substrate-runtime to the same builder/env.
3. Leave graphiti-adapter alone except documenting graph-name coexistence.
4. Open a follow-up issue (not this patch): deprecate `orion:kg:edge:ingest.v1` → rdf-writer.

Do not start with rdf-writer kind migration or Fuseki compact theater.

---

## Open questions for Juniper

1. Prefer **one FalkorDB database, two graph names** (`graphiti_temporal`, `orion_substrate`) vs two containers? (Recommendation: one container, two graph names. Promote compose ownership to `services/orion-falkordb/` when substrate becomes a second dependent — same pattern as `orion-rdf-store` / `orion-sql-db`; do not leave the engine nested only under graphiti-adapter.)
2. For Concept Atlas durability, is **Falkor-only** enough for v1, or do we also want a Postgres substrate snapshot table as SoR with Falkor as projection? (Recommendation: Falkor primary for graph queries; optional Postgres snapshot later if backup/ops demand it — do not dual-SoR in wedge.)
3. Should `GraphWriteIntent` be a bus schema from day one, or an in-process library contract first? (Recommendation: in-process + SubstrateGraphStore wrap first; bus schema when a second process must enqueue intents without calling the store.)

---

## Appendix A — Live Fuseki audit snapshot (2026-07-16, 30 min)

| Endpoint | Count | Dominant source |
|---|---|---|
| `/orion/update` | ~7208 | substrate-runtime SPARQL |
| `/orion/query` | ~982 | recall / substrate reads / hub |
| `/orion/data` | ~144 | rdf-writer bulk (identity, metacog, cognition) |

Drive audits: not in this firehose (path killed). Concept Atlas: not writing Fuseki (in-memory).

## Appendix B — Vocabulary map (do not unify casually)

| Rail | Node identity | Edge type | SoR |
|---|---|---|---|
| Substrate / Concept Atlas | `node_id` | `SubstrateEdgePredicateV1` | Falkor (target; in-memory/SPARQL today) |
| Graphiti crystallizations | `uuid` (`gent_…`) | `RELATES_TO` | Postgres + Falkor projection |
| Legacy RDF | IRI / named graph | predicate IRI | Fuseki |
