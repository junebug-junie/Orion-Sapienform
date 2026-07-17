# Cypher-native substrate + Postgres-via-bus split — design spec

**Date:** 2026-07-16
**Status:** PROPOSAL — agreed direction after Falkor wedge (#1099 / #1105); not yet implemented
**Mode:** Proposal (touches cognitive-graph persistence + drive measurement SoR; AGENTS.md §0A)
**Related:**
- `docs/superpowers/specs/2026-07-16-falkordb-property-graph-routing-design.md` (router / Falkor wedge; this doc **supersedes** its “migrate drive_state into Falkor” implication)
- `docs/superpowers/pr-reports/2026-07-15-drive-audit-graph-path-kill-pr.md` (Postgres SoR precedent for drives)
- `docs/superpowers/specs/2026-07-15-concept-atlas-graph-pipeline-design.md` (Concept Atlas consumer)
- `orion/substrate/falkor_store.py` (current Falkor adapter — still `payload_json` SoR)

---

## Arsonist summary

The Falkor wedge proved the engine and Hub Concept Atlas durability. It did **not** redesign the data model. Live Falkor writes still store opaque `payload_json` — RDF `payloadJson` with Cypher syntax. That is a lift-and-shift, not a property graph.

Next work must:

1. **Redesign graph-shaped substrate for Cypher-native storage** (typed labels, relationship types, scalar properties; no JSON blob as SoR).
2. **Keep measurement / time-series / latest-row state in Postgres**, written via **bus → `orion-sql-writer`**, never HTTP into sql-writer.
3. **Split substrate-runtime workloads by shape**, not migrate “the whole service” onto Falkor.

Drive measurement (pressures, activations, dominant drive, summary, tension kinds) is Postgres SoR. Chat stance must move off the transitional graph snapshot node toward the same measurement rail Mind already uses (`drive_audits` / bus-fed SQL). Graph gets a drive hook only if a named neighborhood/activation consumer ships in the same patch — default: none.

---

## Core question

How does Orion finish the Falkor cutover without recreating SPARQL-shaped blobs in Cypher, while putting relational drive/measurement state on the bus→Postgres rail the mesh already trusts?

---

## Ground truth (verified 2026-07-16 / conversation)

| Claim | Evidence |
|---|---|
| Hub Concept Atlas is on Falkor | `services/orion-hub/.env_example`: `SUBSTRATE_STORE_BACKEND=falkor`; live-verified in Falkor routing spec |
| Falkor adapter SoR is still `payload_json` | `orion/substrate/falkor_store.py` MERGE/SET `n.payload_json` |
| SPARQL store uses the same blob pattern | `orion/substrate/graphdb_store.py` `orion:payloadJson` |
| Substrate-runtime still SPARQL | `services/orion-substrate-runtime/.env`: `SUBSTRATE_STORE_BACKEND=sparql` |
| Drive audits are Postgres-only via bus | kill PR; `orion:memory:drives:audit` → sql-writer → `drive_audits` |
| Mind already reads drive measurement from Postgres | `fetch_drive_state_facet_for_mind` / thought facet → `drive_audits` |
| Chat stance still reads graph snapshot nodes | `chat_stance.py` `_project_autonomy_from_beliefs` filters `snapshot_source == "drive_state"` |
| Runtime materializes drive into substrate graph | `BiometricsSubstrateWorker._materialize_drive_state_to_substrate` |
| Drive state already has a bus channel | `orion:memory:drives:state` / `DriveStateV1` |
| sql-writer is bus-native | README: durable consumer of subscribe channels; no producer HTTP contract |

---

## Hard constraints (agreed)

1. **No SPARQL smoosh into Falkor.** Retire `payload_json` / `payloadJson` as the durable schema for new and migrated graph writes.
2. **Cypher-native model** for graph-shaped work: labels, relationship types, closed scalar property allowlists, queryable without JSON parse.
3. **Postgres where sane** for measurement, history, receipts, projections, latest-row reads.
4. **Bus → `orion-sql-writer` for those Postgres writes.** No direct HTTPS/API writes to sql-writer from producers. Existing runtime reducer tables that already use the service’s own Postgres store stay as-is until a separate redesign; **new** measurement durability must not invent Hub→sql-writer HTTP.
5. **Split by workload shape**, not by service name.

---

## What is graph-shaped vs Postgres-sane

### Falkor / Cypher-native (keep / redesign)

| Workload | Why graph |
|---|---|
| Concept nodes + typed edges (`supports` / `contradicts` / `refines` / `co_occurs_with` / …) | Relation traversal, Concept Atlas network, contradiction/hotspot regions |
| Activation / salience neighborhood queries | Focal slice, dynamics seeding across related nodes |
| Prediction-error / surprise as **node scalars that participate in propagation** | Only if dynamics/attention consumers read them as graph signals |

### Postgres via bus → sql-writer (keep / move)

| Workload | Why Postgres |
|---|---|
| Drive audits (`memory.drives.audit.v1`) | Already done |
| Drive **measurement / latest state** for stance/Mind | Latest-row + history; Mind already on `drive_audits` |
| Grammar receipts / projection tables (existing) | Already reducer → Postgres in substrate-runtime (own DSN today) |
| Brain-frame **samples as telemetry/history** (if persisted for replay) | Time-series samples, not edge traversal |

### Explicit non-destinations

- Do **not** put drive snapshot SoR on Falkor.
- Do **not** migrate generic `orion-rdf-writer` kinds in this effort.
- Do **not** merge Graphiti Entity schema with ConceptNode.
- Do **not** dual-SoR drive state permanently (Postgres + graph snapshot).

---

## Cypher-native substrate model (normative)

### Rejected pattern

```text
(:SubstrateNode {node_id, payload_json})   # opaque SoR — FORBIDDEN for new design
```

### Target pattern

```text
(:Concept {
  node_id, identity_key?,
  label, promotion_state, risk_tier?,
  anchor_scope, subject_ref?,
  salience, activation?, recency?,
  embedding_ref?, evidence_ref?
})
  -[:CONTRADICTS {edge_id, identity_key?, salience}]-> (:Concept)
```

Rules:

1. **Closed property allowlists** per label / relationship type (Pydantic / schema; `extra="forbid"`). Reuse doctrine from Falkor routing property guard.
2. **Relationship type = predicate** (already validated closed `SubstrateEdgePredicateV1`).
3. **Scalars only** on the graph; large text / embeddings / raw evidence stay in Postgres or vector host behind refs.
4. **`metadata` quarantine** remains size-capped and audited; unknown keys rejected. Promote to first-class Cypher properties only with a second consumer.
5. **Optional `schema_version` + `updated_at`** on nodes for migration/hydration — not a blob dump.
6. **Reads that need full typed `ConceptNodeV1`** may reconstruct from Cypher properties + optional Postgres blob **by ref**; Cypher properties remain authoritative for graph queries.
7. **Hydration / cold start** loads typed properties into the in-process cache; do not require `payload_json` to exist.

### Adapter ownership

- `orion/substrate/falkor_store.py` becomes Cypher-native writer/reader.
- `SubstrateGraphStore` API stays the producer-facing contract (nodes/edges), not N-Triples and not raw Cypher in callers.
- SPARQL/`payloadJson` path remains legacy until runtime cutover; no new features target it.

---

## Drive measurement: Postgres SoR + consumer migration

### Decision (agreed): option A

**Postgres is SoR for drive measurement.** Graph does not carry a full drive snapshot node as SoR.

### Write path

```text
DriveEngine
  → bus: orion:memory:drives:state  (DriveStateV1)
  → bus: orion:memory:drives:audit  (DriveAuditV1)  [already → sql-writer]
        ↓
orion-sql-writer
        ↓
Postgres (extend drive_audits and/or add latest-drive projection table —
           exact table shape in implementation plan; prefer reuse of drive_audits
           if latest-row semantics already cover stance needs)
```

Constraints:

- Producers publish bus envelopes only.
- sql-writer remains the durable SQL consumer.
- No `httpx` / Hub proxy POST to sql-writer for this seam.

### Read path (target)

| Consumer | Today | Target |
|---|---|---|
| Mind / thought `drive_state_compact` | Postgres `drive_audits` | unchanged |
| Chat stance `chat_drive_state` | Substrate graph `state_snapshot` / `drive_state` | Postgres (same measurement rail as Mind), fail-open |
| Runtime `_materialize_drive_state_to_substrate` | Upserts graph nodes | **Stop** as SoR writer; delete or gate off once stance reads Postgres |

### Optional graph hook (default off)

A thin Falkor hook (e.g. `drive_ref` / dominant-drive scalar on an existing cognitive node, or a typed edge into a concept region) is allowed **only when**:

- a named consumer needs it for neighborhood/activation, and
- producer + consumer + test ship in the **same** changeset.

Otherwise: no drive nodes on Falkor.

### Dual-run

Short dual (graph snapshot + Postgres) is allowed only as a **migration ladder with exit criteria**, not as architecture. Prefer flip chat stance read to Postgres first, then delete graph materialization.

---

## Substrate-runtime cutover (after Cypher-native adapter)

Do **not** flip `SUBSTRATE_STORE_BACKEND=falkor` on runtime while SoR is still `payload_json`.

Order:

1. **Cypher-native Falkor adapter** + Hub Concept Atlas still green (restart durability, network query).
2. **Drive measurement:** stance → Postgres; stop graph drive snapshot SoR writes.
3. **Runtime graph-shaped writers** (dynamics, prediction-error scalars, attention-related substrate nodes that are truly graph) → Cypher-native Falkor, preferably via `routed` (primary falkor, shadow sparql) briefly, then shadow off.
4. **Measure Fuseki `/update` drop**; attribute remaining traffic.
5. Later (separate): deprecate `orion:kg:edge:ingest.v1` → rdf-writer.

Postgres reducer loops inside runtime (biometrics / execution / transport / chat / route projections) are **out of scope** for Falkor migration. They already use Postgres. Whether those should eventually publish via bus→sql-writer instead of the service-local DSN is a **separate** proposal — not required for this design.

---

## Relationship to Falkor routing doctrine

| Topic | 2026-07-16 routing spec | This spec |
|---|---|---|
| Engine | Falkor + Postgres + temp RDF | unchanged |
| Concept Atlas | Falkor primary | Falkor **Cypher-native** (fix blob SoR) |
| `substrate.drive_state` → Falkor | Listed as target | **Superseded:** Postgres via bus; not Falkor SoR |
| Router / dual-run ladder | Keep for graph workloads | Keep; do not use to justify drive dual-SoR forever |
| Property cathedral rules | Keep | Keep; apply to Cypher properties |

---

## Risks

| Severity | Risk | Mitigation |
|---|---|---|
| High | Chat stance regression when leaving graph snapshot | Mirror Mind’s bounded Postgres fetch; fail-open; regression tests on `_project_autonomy_from_beliefs` + live smoke |
| High | Cypher redesign breaks Concept Atlas mid-flight | Adapter feature flag / dual property write during Hub-only canary; restart hydration test |
| Medium | Someone “temporarily” keeps `payload_json` forever | Acceptance: graph queries filter on native properties without parsing JSON; tests assert no `payload_json` SoR writes |
| Medium | Direct SQL from a new producer, skipping bus | Review checklist + channel registry; sql-writer route map is the only durable write door for this seam |
| Low | Temptation to put brain-frame telemetry in Falkor | Brain-frame history → Postgres if persisted; live sample may read graph snapshot without copying history into Cypher |

---

## Non-goals

- Full Fuseki decommission in this patch series
- Generic rdf-writer → Falkor migration
- Graphiti + substrate schema unification
- Rewriting all substrate-runtime reducers onto bus→sql-writer
- SPARQL compatibility layer inside Falkor
- Drive-audit graph revival

---

## Acceptance checks

- [ ] Spec reviewed by Juniper
- [ ] Implementation plan exists (Cypher-native adapter first; drive SoR second; runtime graph writers third)
- [x] `FalkorSubstrateStore` writes typed Cypher properties; unit tests assert MERGE/SET shape **without** `payload_json` as SoR

**Adapter evidence:** `orion/substrate/tests/test_falkor_store.py` and
`orion/substrate/tests/test_falkor_codec.py` verify native Cypher properties,
native hydration, metadata quarantine behavior, and Hub Concept Atlas route
compatibility through a hydrated Falkor store test double.
- [ ] Concept Atlas Hub summary/network survives restart against real Falkor after redesign
- [ ] Chat stance drive projection reads Postgres measurement rail (bus-fed); graph drive snapshot materialization off or deleted
- [ ] No new producer path calls sql-writer over HTTP for drive measurement
- [ ] Runtime SPARQL cutover (if included in same plan series) uses Cypher-native adapter, not blob port
- [ ] Fuseki update-rate attribution after runtime graph cutover (graph-shaped writers only)

---

## Recommended next patch

1. Implementation plan: **Cypher-native `FalkorSubstrateStore`** (replace `payload_json` SoR; Hub Concept Atlas regression).
2. Implementation plan: **chat stance → Postgres drive measurement**; stop `_materialize_drive_state_to_substrate` as SoR (extend sql-writer / `drive_audits` only if latest-row gaps exist; bus only).
3. Implementation plan: **substrate-runtime graph writers → Falkor Cypher-native** (`routed` then primary-only).

Do not start with “flip runtime env to falkor” on the blob adapter.

---

## Open questions (narrowed)

1. **Latest-drive table shape:** Reuse `drive_audits` latest-row semantics for stance, or add an explicit `drive_state_latest` projection written by sql-writer from `orion:memory:drives:state`? *(Recommendation: prefer `drive_audits` if fields suffice; add a thin latest table only if state vs audit cadence/fields diverge.)*
2. **Blob retention during Cypher migration:** Keep `payload_json` as a deprecated shadow property for one canary window, or hard cut? *(Recommendation: hard cut in adapter tests; optional one-release dual-write only if Hub live risk demands it, with kill date.)*

---

## Appendix — philosophy checklist (why this split)

| Mesh rule | Application here |
|---|---|
| Event substrate first | Drive measurement via bus events, not service-local graph invent |
| Runtime truth / no empty shell | Stance must read a real Postgres row Mind already trusts |
| Edges meaning / properties measurements | Concepts+relations → Cypher; drive pressures → SQL columns |
| No property without consumer | No Falkor drive node without a neighborhood consumer |
| Wrong-tool precedent | Drive audit RDF kill → do not recreate on Falkor |
| Thin seams | Adapter redesign + consumer flip + runtime graph cutover as separate patches |
