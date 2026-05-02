# Memory graph annotator (Hub) + dual-write GraphDB — design

**Date:** 2026-05-02  
**Status:** Draft — awaiting operator review before implementation planning  
**Extends:** [Orion Memory Cards v1](./2026-05-01-orion-memory-cards-v1-design.md), [Memory Cards offboarding guide](../guides/2026-05-01-memory-cards-v1-offboarding.md)

---

## 1. Purpose

Operators need a **post-hoc graph annotator** in the Hub **Memory** area to turn natural-language turns into **ontology-backed relational structure** with **provenance** (which turns, which subjects, inferred vs human-asserted). Examples include multi-entity utterances (“cats” vs “Joey”, actions, time, emotional/trust impact).

This spec defines:

1. **RDF in GraphDB** as the detailed relational store (named graphs, PROV-aware lineage).
2. **Explicit dual-write** into existing **Postgres memory cards** (`memory_cards`, `memory_card_edges`, `subschema`) so current rails keep working.
3. **Hybrid inference:** brain-lane LLM proposes a **draft**; **ontology + validation rules** normalize and flag conflicts; the operator **commits** from the UI.
4. **Three downstream consumers** treated as first-class in v1: **orion-recall**, **cortex-orch known-facts / memory inject**, **cortex-exec chat stance / autonomy GraphDB path** — each with **documented read contracts** so schemas stay aligned.

---

## 2. Goals and non-goals

### Goals

- Hub UX: select transcript turns → **suggest** (brain lane) → **validate** → edit → **approve** with inferred vs asserted distinction and overrides.
- **Versioned ontology package** (TTL) + **projector mapping** (RDF → Postgres) + **consumer projection contracts** (tests enforce drift prevention).
- **memory-graph** responsibility boundary: validation, GraphDB I/O, synchronous Postgres projection, SPARQL helpers — **not** “fat Hub” duplicating RDF logic.
- **Synchronous dual-write** on approve (see §6): operators do not wait for an async queue for cards to match the graph.

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

## 4. Schema alignment package (normative deliverables)

Past work did not fully align RDF, cards, and consumers. **v1 blocks implementation** until these artifacts exist:

1. **`ontology/` (repo path TBD in implementation plan)** — Versioned TTL: imports **PROV-O**, **Schema.org** fragments (Person, Thing, Event where applicable), **Orion extensions** (classes/predicates for trust, affect, commitments). **SHACL shapes** for shapes that must validate before commit.
2. **`projector_mapping.yaml` (or JSON)** — Deterministic mapping: RDF types → `memory_cards.types`, predicates → `memory_card_edges.edge_type` / `metadata`, nested literals → `subschema` keys.
3. **Consumer contracts** — Short tables:
   - **Recall:** which fields/predicates participate in fusion scoring vs display-only.
   - **Inject:** which projected rows appear in `known_facts`-style blocks and max token budget interaction.
   - **Stance:** which classes/properties autonomy SPARQL queries must join on (aligned with existing `chat_stance.py` graph probe patterns).
4. **Golden exemplar** — One narrative test fixture (multi-subject text → RDF + Postgres snapshot).

Ontology **version** is carried on approve requests; memory-graph rejects mismatches.

---

## 5. Data flow

1. Operator selects **session + turn ids** (and optional existing card anchors).
2. **Suggest:** Hub → cortex brain lane → **draft graph DTO** (not committed).
3. **Validate:** Hub → memory-graph **validate-only** → violations + normalized preview.
4. **Edit:** Operator adjusts nodes/edges; inferred edges visually distinct from asserted.
5. **Approve:** memory-graph: validate → **GraphDB write** (named graph) → **Postgres projector** → success payload with `card_id`s / edge ids / graph revision id.
6. **Consumers** read per §7.

---

## 6. Dual-write and failure semantics

**Order (normative default):** validate → **GraphDB** → **Postgres**. Rationale: RDF is the detailed source; Postgres is projection for existing rails.

If **Postgres fails after GraphDB succeeds:** memory-graph **must** execute **compensating deletion** of the same approve batch triples (or equivalent tombstone in-graph + reconcile job — pick one implementation strategy; **no silent divergence**). Document the chosen strategy in the implementation plan with tests.

If **GraphDB fails:** no Postgres writes.

---

## 7. Downstream consumers (v1 scope)

### 7.1 Recall (`orion-recall`)

- Continue **cards** backend on Postgres projection.
- Add **optional** SPARQL expansion (memory-graph or recall-local client) **bounded** by named graph + query timeout — **only** when enabled via settings/profile to avoid latency surprises.

### 7.2 Known facts / memory inject (`orion-cortex-orch`)

- Extend materialization to consume **new subschema/edge projections** from projector mapping (exact prompt shaping in implementation plan).
- Preserve existing timeouts and visibility rules from Memory Cards v1.

### 7.3 Chat stance / autonomy (`orion-cortex-exec`)

- Ensure annotated entities appear in **same GraphDB endpoint/repo** stance uses, with **queries updated** to include memory named graphs **without** collapsing unrelated substrate noise.

---

## 8. Hub UI (high level)

- **Memory tab** gains a **graph annotator** panel: turn list, graph canvas or structured inspector (implementation plan chooses component depth).
- Actions: **Suggest**, **Validate**, **Approve**; toggles for showing **inferred** edges and **provenance** (turn ids).
- No standalone canvas requirement for v1 wireframe perfection — **correctness and contracts** outweigh polish.

---

## 9. Testing and acceptance

| Layer | Minimum |
|--------|---------|
| Unit | Projector mapping; validation rejects invalid drafts; ontology version gate. |
| Integration | Approve round-trip: GraphDB contains triples; Postgres rows match golden exemplar. |
| Cross-service (environment permitting) | Recall sees cards; inject shows new lines; stance query returns annotated entity binding. |

---

## 10. Open points for implementation plan only

- Exact **brain lane** verb/step names for suggest vs validate orchestration.
- **Compensation** implementation detail (delete vs tombstone).
- **Single deployable** memory-graph vs library-in-Hub phase A.

---

## 11. References (code)

- Contracts/DAL: `orion/core/contracts/memory_cards.py`, `orion/core/storage/memory_cards.py`
- Hub memory routes: `services/orion-hub/scripts/memory_routes.py`
- Stance GraphDB: `services/orion-cortex-exec/app/chat_stance.py`
- Recall cards: `services/orion-recall/app/cards_adapter.py`, `fusion.py`
