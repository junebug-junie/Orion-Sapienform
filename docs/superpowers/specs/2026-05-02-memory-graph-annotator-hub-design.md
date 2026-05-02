# Memory graph annotator (Hub) + dual-write GraphDB — design

**Date:** 2026-05-02  
**Status:** Draft — awaiting operator review before implementation planning  
**Extends:** [Orion Memory Cards v1](./2026-05-01-orion-memory-cards-v1-design.md), [Memory Cards offboarding guide](../guides/2026-05-01-memory-cards-v1-offboarding.md)

---

## 1. Purpose

Operators need a **post-hoc graph annotator** in the Hub **Memory** area to turn natural-language turns into **ontology-backed relational structure** with **provenance** (which turns, which subjects, inferred vs human-asserted). Examples include multi-entity utterances (“cats” vs “Joey”, actions, time, emotional/trust impact).

This spec defines:

1. **RDF in GraphDB** as the detailed relational store (named graphs, PROV-aware lineage); ontology detail — classes, properties, mapping to cards — is **§4**.
2. **Explicit dual-write** into existing **Postgres memory cards** (`memory_cards`, `memory_card_edges`, `subschema`) so current rails keep working.
3. **Hybrid inference:** brain-lane LLM proposes a **draft**; **ontology + validation rules** normalize and flag conflicts; the operator **commits** from the UI.
4. **Three downstream consumers** treated as first-class in v1: **orion-recall**, **cortex-orch known-facts / memory inject**, **cortex-exec chat stance / autonomy GraphDB path** — each with **documented read contracts** so schemas stay aligned.

---

## 2. Goals and non-goals

### Goals

- Hub UX: select transcript turns → **suggest** (brain lane) → **validate** → edit → **approve** with inferred vs asserted distinction and overrides.
- **Versioned ontology package** (TTL) + **projector mapping** (RDF → Postgres) + **consumer projection contracts** (tests enforce drift prevention).
- **memory-graph** responsibility boundary: validation, GraphDB I/O, synchronous Postgres projection, SPARQL helpers — **not** “fat Hub” duplicating RDF logic.
- **Synchronous dual-write** on approve (see §7): operators do not wait for an async queue for cards to match the graph.

### Non-goals (v1)

- Replacing Memory Cards v1 **policies** (visibility, lane, trust) — annotator **must** respect existing card contracts when projecting.
- Live **during-chat** annotation as the primary workflow (post-hoc Hub only for v1).
- Full OWL reasoning in GraphDB beyond what validation explicitly uses (optional future).

---

## 3. Architecture (hybrid service + synchronous dual-write)

### 3.1 Recommended decomposition

| Layer | Responsibility |
|--------|----------------|
| **Hub** | Memory tab UI: turn picker, graph editor, suggest/validate/approve actions; session auth unchanged (`ensure_session`). |
| **Cortex (brain lane)** | LLM **draft** graph payload only — same routing/budget/observability as brain chat (`mode: brain`, gateway routes per executive policy: e.g. **`quick`** for a thin suggest step if split; **`chat`** for heavier extraction — exact verb/step names left to implementation plan). |
| **memory-graph service** | Parse draft → merge with edits → **validate** → **write GraphDB** → **project Postgres** in one **approve** RPC; exposes versioned ontology id. |

This combines **approach 1** (synchronous dual-write semantics) with **approach 3** (dedicated service boundary): correctness lives **inside** memory-graph, not split across Hub ad hoc calls.

### 3.2 Optional evolution

Implement memory-graph as an **`orion/` library + Hub embed** first, extract to standalone FastAPI **later**, only if operational needs require it — **without** changing the approve RPC contract.

### 3.3 GraphDB placement

- **Named graphs** segregate operator memory from substrate collapse defaults, e.g. `https://orion.example/ns/memory/ng/{workspace}/{session}` (exact IRI scheme to be fixed in ontology package).
- **Stance / recall SPARQL** must filter by **memory named graph IRIs** (or dedicated repository if ops mandates — preference: **one repo**, named-graph discipline, unless security isolation forces split).

---

## 4. Ontology vocabulary (normative sketch)

This section fixes **what kinds of things exist in the graph**, **which predicates link them**, and **how that relates** to Memory Cards v1. Exact IRIs and SHACL files ship in the ontology package; this spec constrains **shape**, not every final URI suffix.

### 4.1 Namespaces

| Prefix | Role |
|--------|------|
| `orionmem:` | **Canonical Orion memory extension** — domain classes and predicates not adequately covered by imports (stable base IRI versioned per ontology release). |
| `prov:` | **PROV-O** — activity, entity, derivation, attribution (provenance). |
| `schema:` | **Schema.org** — reuse `Person`, `Thing`, `Event`, `Organization`, `Place`, `CreativeWork`-adjacent patterns where they fit without distortion. |
| `rdfs:` / `rdf:` | RDFS/RDF built-ins for typing and labels. |

All resource IRIs for **entities introduced by annotation** should be **UUID-based** under `orionmem:` (e.g. `orionmem:entity/{uuid}`) to avoid accidental collisions with substrate collapse IRIs. **Turn / session / chat** identifiers reuse existing deployment URIs when the Hub already emits them; otherwise `prov:wasDerivedFrom` links to a **quoted utterance entity** `orionmem:utterance/{turn_id}` supplied by Hub.

### 4.2 Upper layers (reuse, not reinvent)

- **PROV-O:** Every **committed** assertion or Situation is linked with **`prov:wasDerivedFrom`** to the **utterance entity** (or directly to a PROV `Entity` representing the turn transcript snippet). Optional **`prov:wasAttributedTo`** when the speaker role matters (operator vs model vs external).
- **Schema.org:** Use **`schema:name`**, **`schema:description`**, **`schema:startDate` / `schema:endDate`** where literal facts attach cleanly. **`schema:subjectOf` / `schema:about`** may connect an animal/person node to a narrative Event when appropriate.
- **Do not** overload Schema.org past readability: if “trust toward a breed after an incident” does not fit `schema:Event`, use an **`orionmem:Situation`** or **`orionmem:AffectiveEpisode`** class with documented properties (below).

### 4.3 Orion domain classes (minimum set for v1)

These classes extend the graph beyond plain Schema.org; SHACL must enumerate allowed properties per class.

| Class | Intent |
|-------|--------|
| **`orionmem:UtteranceSlice`** | PROV-aligned **anchor** for “this graph fragment explains these turns.” Subjects of derivation edges. |
| **`orionmem:Situation`** | Bounded interpretation of what happened (may correspond to one sentence span): links **participants**, **time**, **affect**, **impact**. |
| **`orionmem:Participant`** | Any focal referent: person, pet, group, abstract category **in role** (not necessarily `schema:Person`). Use **`orionmem:participantKind`** literal or link to a typed entity (see below). |
| **`orionmem:TypedEntity`** | **`schema:Thing`** in RDF terms + optional **`orionmem:entityKind`** (`cat`, `breed`, `person`, …) when species/breed/taxon matters for stance. |
| **`orionmem:AffectiveDisposition`** | Durable attitude (e.g. trust/distrust toward class X): use for “lost trust in this breed” if not modeled as a single Event property. |
| **`orionmem:ImpactAssertion`** | Explicit claim about consequence (“impact is …”): keeps **impact** as first-class for recall/inject. |

v1 **does not** require a full emotion ontology; a small closed vocabulary on **`orionmem:affectLabel`** (`anger`, `annoyance`, `affection`, …) plus optional **`orionmem:intensity`** is enough for validation.

### 4.4 Object properties (relations between individuals)

Normative **meaning**; exact owl:inverseOf pairs are optional in v1.

| Property | Typical use |
|----------|-------------|
| **`orionmem:inSituation`** | Participant → Situation (who is involved in the episode). |
| **`orionmem:targetOf`** | Directed edge for blame/praise/action recipient (subject **pissed off** target). |
| **`orionmem:aboutEntity`** | Situation or assertion **about** a TypedEntity (Joey, breed, cats-in-general). |
| **`orionmem:generalizationOf`** | Link specific instance (Joey) to category (breed / cats). |
| **`orionmem:dispositionToward`** | Participant or Situation → AffectiveDisposition (trust stance). |
| **`prov:wasDerivedFrom`** | Any **memory assertion** → UtteranceSlice or quoted Entity. |
| **`schema:about`** | Optional parallel when Schema.org Story/Event modeling is used. |

**Multi-subject utterances:** encode as **one `orionmem:Situation`** with **multiple `orionmem:inSituation`** edges (many participants) **or** multiple Situations sharing the same **`prov:wasDerivedFrom`** if the operator splits readings; the annotator UI should prefer **one Situation** with multiple focal nodes unless the text forces a split.

### 4.5 Datatype properties (literals)

| Property | Range | Notes |
|----------|--------|------|
| **`orionmem:occurredAt`** / **`schema:startDate`** | `xsd:date` or `xsd:dateTime` | “Last week” normalized with optional **`orionmem:timeQualitative`** (`last_week`) when precise date unknown. |
| **`orionmem:affectLabel`** | string (enum in SHACL) | Coarse affect for stance/recall. |
| **`orionmem:impactSummary`** | string | Short human impact line for inject/recall. |
| **`rdfs:label`** | string | Display label for any entity (e.g. “Joey”). |

### 4.6 Inference vs operator assertion (metadata, not a second graph)

LLM-suggested triples are **not** a separate ontology; they are the **same predicates** with additional **provenance**:

- **`prov:wasGeneratedBy`** → `orionmem:Activity/inference/{run_id}` (PROV Activity), **or**
- **`orionmem:assertionMode`** literal on reified statements (`orionmem:Assertion` node with **rdf:subject** / **rdf:predicate** / **rdf:object** — **only if** you need triple-level qualification in v1; otherwise store **`orionmem:inferenceConfidence`** on **edges** projected into `memory_card_edges.metadata`).

**Operator-approved** commits strip “draft-only” flags; overridden edges are **replaced** in GraphDB and in Postgres projection.

### 4.7 Mapping to Memory Cards v1 (dual-write projection)

**Cards:** Typically **one primary card per Situation or per TypedEntity focal node** (implementation plan chooses merge vs split rules). **`memory_cards.types`** / **`anchor_class`** derive from **`rdf:type`** and **`orionmem:entityKind`** via **`projector_mapping`**.

**Edges (`memory_card_edges.edge_type`):** map RDF pairs onto the existing closed **`EDGE_TYPES`** set where possible:

| RDF pattern (lossy but bounded) | `EDGE_TYPES` |
|---------------------------------|--------------|
| `orionmem:generalizationOf`, `schema:memberOf`, `rdf:type` to category | `instance_of` / `associated_with` |
| `orionmem:inSituation`, `orionmem:aboutEntity` | `associated_with` / `relates_to` (metadata carries finer predicate IRI) |
| Contradiction edges (explicit in UI) | `contradicts` |
| Utterance → card | `derived_from` |
| Evidence nodes | `evidence_for` / `evidence_against` |

When RDF is **finer-grained** than `EDGE_TYPES`, **preserve the RDF predicate IRI** in **`memory_card_edges.metadata.predicate`** (or equivalent) so recall/stance can evolve without schema churn.

**`subschema`:** Nested structured literals from the Situation (participants, impact, time window, raw spans) **must** round-trip in **`projector_mapping`** so inject does not parse RDF in hot paths.

---

## 5. Schema alignment package (normative deliverables)

Past work did not fully align RDF, cards, and consumers. **v1 blocks implementation** until these artifacts exist:

1. **`ontology/` (repo path TBD in implementation plan)** — Versioned TTL implementing §4: imports **PROV-O**, **Schema.org** fragments, **`orionmem:`** definitions, **SHACL** for §4.3–§4.5.
2. **`projector_mapping.yaml` (or JSON)** — Deterministic mapping: RDF types → `memory_cards.types` / `anchor_class`; predicates → `memory_card_edges.edge_type` + `metadata`; nested literals → `subschema` keys (§4.7).
3. **Consumer contracts** — Short tables:
   - **Recall:** which fields/predicates participate in fusion scoring vs display-only.
   - **Inject:** which projected rows appear in `known_facts`-style blocks and max token budget interaction.
   - **Stance:** which classes/properties autonomy SPARQL queries must join on (aligned with existing `chat_stance.py` graph probe patterns).
4. **Golden exemplar** — One narrative test fixture (multi-subject text → RDF + Postgres snapshot).

Ontology **version** is carried on approve requests; memory-graph rejects mismatches.

---

## 6. Data flow

1. Operator selects **session + turn ids** (and optional existing card anchors).
2. **Suggest:** Hub → cortex brain lane → **draft graph DTO** (not committed).
3. **Validate:** Hub → memory-graph **validate-only** → violations + normalized preview.
4. **Edit:** Operator adjusts nodes/edges; inferred edges visually distinct from asserted.
5. **Approve:** memory-graph: validate → **GraphDB write** (named graph) → **Postgres projector** → success payload with `card_id`s / edge ids / graph revision id.
6. **Consumers** read per §8.

---

## 7. Dual-write and failure semantics

**Order (normative default):** validate → **GraphDB** → **Postgres**. Rationale: RDF is the detailed source; Postgres is projection for existing rails.

If **Postgres fails after GraphDB succeeds:** memory-graph **must** execute **compensating deletion** of the same approve batch triples (or equivalent tombstone in-graph + reconcile job — pick one implementation strategy; **no silent divergence**). Document the chosen strategy in the implementation plan with tests.

If **GraphDB fails:** no Postgres writes.

---

## 8. Downstream consumers (v1 scope)

### 8.1 Recall (`orion-recall`)

- Continue **cards** backend on Postgres projection.
- Add **optional** SPARQL expansion (memory-graph or recall-local client) **bounded** by named graph + query timeout — **only** when enabled via settings/profile to avoid latency surprises.

### 8.2 Known facts / memory inject (`orion-cortex-orch`)

- Extend materialization to consume **new subschema/edge projections** from projector mapping (exact prompt shaping in implementation plan).
- Preserve existing timeouts and visibility rules from Memory Cards v1.

### 8.3 Chat stance / autonomy (`orion-cortex-exec`)

- Ensure annotated entities appear in **same GraphDB endpoint/repo** stance uses, with **queries updated** to include memory named graphs **without** collapsing unrelated substrate noise.

---

## 9. Hub UI (high level)

- **Memory tab** gains a **graph annotator** panel: turn list, graph canvas or structured inspector (implementation plan chooses component depth).
- Actions: **Suggest**, **Validate**, **Approve**; toggles for showing **inferred** edges and **provenance** (turn ids).
- No standalone canvas requirement for v1 wireframe perfection — **correctness and contracts** outweigh polish.

---

## 10. Testing and acceptance

| Layer | Minimum |
|--------|---------|
| Unit | Projector mapping; validation rejects invalid drafts; ontology version gate. |
| Integration | Approve round-trip: GraphDB contains triples; Postgres rows match golden exemplar. |
| Cross-service (environment permitting) | Recall sees cards; inject shows new lines; stance query returns annotated entity binding. |

---

## 11. Open points for implementation plan only

- Exact **brain lane** verb/step names for suggest vs validate orchestration.
- **Compensation** implementation detail (delete vs tombstone).
- **Single deployable** memory-graph vs library-in-Hub phase A.

---

## 12. References (code)

- Contracts/DAL: `orion/core/contracts/memory_cards.py`, `orion/core/storage/memory_cards.py`
- Hub memory routes: `services/orion-hub/scripts/memory_routes.py`
- Stance GraphDB: `services/orion-cortex-exec/app/chat_stance.py`
- Recall cards: `services/orion-recall/app/cards_adapter.py`, `fusion.py`
