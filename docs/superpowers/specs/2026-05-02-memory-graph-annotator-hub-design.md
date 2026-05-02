# Memory graph annotator (Hub) + dual-write GraphDB — design

**Date:** 2026-05-02  
**Status:** Draft — awaiting operator review before implementation planning  
**Extends:** [Orion Memory Cards v1](./2026-05-01-orion-memory-cards-v1-design.md), [Memory Cards offboarding guide](../guides/2026-05-01-memory-cards-v1-offboarding.md)

---

## 1. Purpose

Operators need a **post-hoc graph annotator** in the Hub **Memory** area to turn natural-language turns into **ontology-backed relational structure** with **provenance** (which turns, which subjects, inferred vs human-asserted). Examples include multi-entity utterances (“cats” vs “Joey”, actions, time, emotional/trust impact).

This spec defines:

1. **RDF in GraphDB** as the detailed relational store (named graphs, PROV-aware lineage); **ontology, predicates, cardinalities, projections**, and **worked examples** are **§4** and **Appendices A–D**.
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

## 4. Ontology vocabulary (normative)

This section defines **classes**, **properties**, **cardinalities**, **projection to Memory Cards v1**, and ties to **Appendices A–D**. Turtle IRIs ship in the ontology package; here is the **semantic contract** implementers and reviewers use.

### 4.1 Design principles

1. **`orionmem:`** holds anything Schema.org distorts (stance toward a **breed class**, fine-grained roles, narrative **Situation**).
2. **Schema.org** carries portable literals and **`schema:Thing` / `schema:Person`** when types fit without bending semantics.
3. **PROV-O** carries **derivation** from chat and optional **inference** activity — avoid storing turn text in RDF except on **`orionmem:UtteranceSlice`** via **`schema:text`** when audit requires it.
4. Every **approved** graph fragment **MUST** tie to the transcript through **`prov:wasDerivedFrom`** (UtteranceSlice or deployment turn URI).

### 4.2 Namespaces and IRIs

| Prefix | Role |
|--------|------|
| `orionmem:` | Orion memory extension — versioned ontology release (fragment `orionmem:v2026-05` **or** separate ontology IRI — pick one in TTL and never mix). |
| `prov:` | PROV-O |
| `schema:` | Schema.org |
| `rdfs:` / `rdf:` | Typing and labels |

**Entity nodes:** HTTPS UUID IRIs `https://…/mem/entity/{uuid}` **or** CURIEs resolved against one base — **one style per deployment**, documented in TTL.

**Turn linkage:** Prefer **existing** Hub/bus turn URIs. Else **`orionmem:utterance/{stable_turn_id}`** where `stable_turn_id` matches Postgres/bus correlation.

**Named graphs:** SPARQL filters **must** use the graph IRIs listed under **`subschema.memory_graph.named_graphs`** (Appendix D).

### 4.3 Class taxonomy (v1)

```
owl:Thing
  schema:Thing
  prov:Entity
    orionmem:UtteranceSlice       ← quoted turn text + anchor for derivation
    orionmem:TypedEntity          ← Joey, breed node, “cats”, Juniper, …
    orionmem:NamingLink           ← optional: surface string ↔ entity
    orionmem:Situation            ← interpreted episode (“what happened”)
    orionmem:AffectiveDisposition ← durable stance (trust toward breed)
    orionmem:ImpactAssertion      ← bullet consequence for inject/recall
  prov:Activity
    orionmem:InferenceActivity    ← one LLM suggest run (optional)
```

**`orionmem:TypedEntity`** **SHOULD** also carry **`rdf:type`** `schema:Thing` or `schema:Person` when appropriate for Schema.org tooling.

**`orionmem:entityKind` (literals)** — closed enum in SHACL; guides **`anchor_class`** projection:

| `entityKind` | Meaning | `anchor_class` hint |
|----------------|---------|---------------------|
| `person` | Human | `person` |
| `animal` / `pet` | Non-human animal | `concept` or mapper extension |
| `breed` / `taxon` | Breed, species class | `concept` |
| `collective` | Plural habit (“cats”) | `concept` |
| `abstract` | Generic topic | `concept` |

### 4.4 Naming and coreference

| Construct | Purpose |
|-----------|---------|
| **`rdfs:label`** | Canonical display string on **`TypedEntity`**. |
| **`orionmem:surfaceForm`** | Exact substring from turn when it differs from label. |
| **`orionmem:NamingLink`** | Optional: **`surfaceForm`** + **`orionmem:denotes`** → **`TypedEntity`**. |
| **`orionmem:coreferenceGroup`** | Shared UUID literal for “same referent” across turns (operator-maintained). |

**Minimum entities for the running example:** Joey (instance), breed node (class), cats collective, Juniper — **four** **`TypedEntity`** nodes unless operator merges (discouraged for stance).

### 4.5 Object properties — catalogue (v1)

Cardinality hints: **0..1** optional, **1** exactly one where marked **required** by SHACL shape.

| Property | Domain → Range | Cardinality | Role |
|----------|------------------|---------------|------|
| **`prov:wasDerivedFrom`** | Situation, TypedEntity, AffectiveDisposition, ImpactAssertion → UtteranceSlice or turn IRI | **1..*** per asserted artifact | provenance chain |
| **`prov:wasGeneratedBy`** | same → **`orionmem:InferenceActivity`** | 0..1 | draft inference |
| **`prov:wasAttributedTo`** | UtteranceSlice → TypedEntity (`person`) | 0..1 | speaker |
| **`orionmem:inSituation`** | TypedEntity → Situation | 0..* | enrollment as participant |
| **`orionmem:participantRole`** | literal on entity **when** `inSituation` used | 0..1 | `agent`, `patient`, `stimulus`, `topic`, `observer`, `mentioned` |
| **`orionmem:stimulusEntity`** | Situation → TypedEntity | 0..1 | salient cause (Joey angered…) |
| **`orionmem:aboutEntity`** | Situation → TypedEntity | 0..* | topical “about” (cats in general) |
| **`orionmem:targetOfNegativeAffect`** | Situation → TypedEntity | 0..* | who/what is blamed (breed) |
| **`orionmem:contradictsSituation`** | Situation → Situation | 0..* | explicit mutual exclusion / correction |
| **`orionmem:generalizationOf`** | TypedEntity → TypedEntity | 0..* | Joey → breed class |
| **`orionmem:specializationOf`** | inverse direction | | maps **`parent_of`** / **`child_of`** by arrow |
| **`orionmem:dispositionToward`** | TypedEntity (holder) → AffectiveDisposition | 0..* | Juniper → disp node |
| **`orionmem:dispositionTarget`** | AffectiveDisposition → TypedEntity | 1..1 | stance **about** breed |
| **`schema:about`** | Situation → TypedEntity | 0..* | interop duplicate of topical focus |
| **`schema:subjectOf`** | TypedEntity → Situation | 0..* | inverse framing |

**Multi-subject sentences:** Prefer **one `Situation`**, multiple **`inSituation`** edges with distinct **`participantRole`**, plus **`stimulusEntity`** vs **`aboutEntity`** to split “cause” vs “topic.”

### 4.6 Datatype properties — catalogue (v1)

| Property | Range | Notes |
|----------|--------|------|
| **`rdfs:label`** | `xsd:string` | required on entities surfaced in UI |
| **`schema:name` / `schema:description`** | string | schema interop + card **`summary`** |
| **`orionmem:affectLabel`** | enum | `anger`, `annoyance`, `affection`, `fear`, `trust`, `distrust`, `neutral`, … |
| **`orionmem:affectPolarity`** | `xsd:float` | −1..1 optional |
| **`orionmem:intensity`** | `xsd:float` | 0..1 optional |
| **`orionmem:impactSummary`** | string | **`ImpactAssertion`** body |
| **`orionmem:occurredAt`** | `xsd:date` / `xsd:dateTime` | Situation time |
| **`orionmem:timeQualitative`** | enum | `last_week`, `yesterday`, … when date fuzzy |
| **`orionmem:trustPolarity`** | enum | `trust`, `distrust`, `ambivalent` on **`AffectiveDisposition`** |
| **`orionmem:inferenceConfidence`** | `xsd:float` | on suggested edges → **`memory_card_edges.metadata`** |
| **`orionmem:assertionMode`** | enum | `inferred`, `operator_asserted`, `imported` |

### 4.7 Situation shape (SHACL target)

Every **`orionmem:Situation`** approved for dual-write **SHOULD** have:

1. **`prov:wasDerivedFrom`** ≥1 utterance or turn IRI.  
2. ≥1 link to a **`TypedEntity`** via **`aboutEntity`**, **`stimulusEntity`**, and/or **`inSituation`**.  
3. **`orionmem:occurredAt`** **or** **`orionmem:timeQualitative`**.  
4. Optional **`orionmem:affectLabel`** **or** linked **`AffectiveDisposition`**.

### 4.8 Inference vs operator assertion

- Draft: **`inferenceConfidence`**, **`prov:wasGeneratedBy`** → **`InferenceActivity`**.  
- Committed: operator edges **`assertionMode`**=`operator_asserted` **or** inference markers stripped.  
- Triple-level reification (**`rdf:Statement`**) **optional**; prefer **`memory_card_edges.metadata`** unless stance requires quoted triples.

### 4.9 Mapping — full `EDGE_TYPES` ladder

Contract literals (`memory_cards.py`):  
`relates_to`, `contradicts`, `supersedes`, `supports`, `parent_of`, `child_of`, `precedes`, `follows`, `co_occurs_with`, `derived_from`, `evidence_for`, `evidence_against`, `tagged_as`, `instance_of`, `example_of`, `analogy_of`, `associated_with`.

| `EDGE_TYPES` | RDF triggers (keep finer IRI in **`metadata.predicate`**) |
|--------------|------------------------------------------------------------|
| **`relates_to`** | catch-all; **`schema:relatedTo`**, weak topical links |
| **`contradicts`** | **`orionmem:contradictsSituation`** (object Situation) |
| **`supersedes`** | newer Situation replaces older (temporal meta-edge) |
| **`supports`** | **`orionmem:supportsClaim`**, pro-relation to **`ImpactAssertion`** |
| **`parent_of` / `child_of`** | direction of **`generalizationOf` / `specializationOf`** |
| **`precedes` / `follows`** | temporal Situation links |
| **`co_occurs_with`** | two **`TypedEntity`** in same **`Situation`** without directed predicate |
| **`derived_from`** | **`prov:wasDerivedFrom`** |
| **`evidence_for` / `evidence_against`** | stance toward **`ImpactAssertion`** |
| **`tagged_as`** | **`skos:related`**, loose labels |
| **`instance_of`** | **`rdf:type`**, **`generalizationOf`** to class |
| **`example_of`** | instance exemplifying category |
| **`analogy_of`** | operator-marked analogy |
| **`associated_with`** | **`aboutEntity`**, **`schema:about`**, default topical |

### 4.10 Mapping — `MemoryCardV1` fields

| Card field | RDF source |
|------------|------------|
| **`title`** | **`Situation`** `rdfs:label` **or** synthesized from entity labels |
| **`summary`** | **`schema:description`** / **`impactSummary`** |
| **`types`** / **`anchor_class`** | **`rdf:type`** + **`entityKind`** via **`projector_mapping`** |
| **`time_horizon`** | **`occurredAt`** + qualitative → **`TimeHorizonV1`** table |
| **`evidence`** | UtteranceSlice excerpt + turn id |
| **`subschema`** | **Appendix D** |
| **`provenance`** | Approvals → `operator_highlight` or new enum **`operator_graph`** if contract extended in impl |

### 4.11 Consumer projections (summary)

- **Recall:** unchanged token scoring on cards + optional SPARQL boost when **`AffectiveDisposition`** matches query entities (detail: consumer contract doc).  
- **Inject:** **`subschema.memory_graph.facts`** lines + **`ImpactAssertion`** ordering.  
- **Stance:** SPARQL over **`AffectiveDisposition`**, **`trustPolarity`**, **`dispositionTarget`**, labels — extend `chat_stance.py` bindings.

---

## 5. Schema alignment package (normative deliverables)

Past work did not fully align RDF, cards, and consumers. **v1 blocks implementation** until these artifacts exist:

1. **`ontology/`** — Versioned TTL implementing §4 and **Appendix B** SHACL.
2. **`projector_mapping.yaml` (or JSON)** — §4.9–§4.10 + **Appendix D** keys.
3. **Consumer contracts** — Recall / inject / stance field lists keyed to §4.11.
4. **Golden exemplar** — Fixture derived from **Appendix A** (+ Postgres rows).

Ontology **version** is carried on approve requests; memory-graph rejects mismatches.

---

## 6. Data flow

1. Operator selects **session + turn ids** (and optional existing card anchors).
2. **Suggest:** Hub → cortex brain lane → **draft JSON** (Appendix C).
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
- Optional **`MemoryProvenance`** enum extension (**`operator_graph`**) for ledger clarity.

---

---

## Appendix A — Worked example (informative Turtle sketch)

**Utterance:** loves cats; Joey the cat angered the speaker last week; uncertain trust in the breed.

```turtle
@prefix orionmem: <https://orion.example/ns/mem/> .
@prefix schema:  <https://schema.org/> .
@prefix prov:    <http://www.w3.org/ns/prov#> .
@prefix xsd:     <http://www.w3.org/2001/XMLSchema#> .

orionmem:utterance/t42 a orionmem:UtteranceSlice ;
  rdfs:label "turn t42" ;
  schema:text "…cats… Joey … pissed me off … last week … trust … breed …" .

orionmem:entity/joey a schema:Thing , orionmem:TypedEntity ;
  rdfs:label "Joey" ;
  orionmem:entityKind "animal" ;
  orionmem:generalizationOf orionmem:entity/breed-x .

orionmem:entity/breed-x a orionmem:TypedEntity ;
  rdfs:label "this breed" ;
  orionmem:entityKind "breed" .

orionmem:entity/cats a orionmem:TypedEntity ;
  rdfs:label "cats" ;
  orionmem:entityKind "collective" .

orionmem:entity/juniper a schema:Person , orionmem:TypedEntity ;
  rdfs:label "Juniper" ;
  orionmem:entityKind "person" .

orionmem:sit/s1 a orionmem:Situation ;
  rdfs:label "Joey angered Juniper; cats and breed trust uncertain" ;
  prov:wasDerivedFrom orionmem:utterance/t42 ;
  orionmem:stimulusEntity orionmem:entity/joey ;
  orionmem:aboutEntity orionmem:entity/cats ;
  orionmem:targetOfNegativeAffect orionmem:entity/breed-x ;
  orionmem:occurredAt "2026-04-25"^^xsd:date ;
  orionmem:timeQualitative "last_week" ;
  orionmem:affectLabel "annoyance" .

orionmem:entity/joey orionmem:inSituation orionmem:sit/s1 ;
  orionmem:participantRole "stimulus" .

orionmem:entity/juniper orionmem:inSituation orionmem:sit/s1 ;
  orionmem:participantRole "patient" .

orionmem:entity/cats orionmem:inSituation orionmem:sit/s1 ;
  orionmem:participantRole "topic" .

orionmem:disp/d1 a orionmem:AffectiveDisposition ;
  prov:wasDerivedFrom orionmem:utterance/t42 ;
  orionmem:trustPolarity "ambivalent" ;
  orionmem:dispositionTarget orionmem:entity/breed-x ;
  schema:description "Uncertain whether to trust this breed after Joey incident." .

orionmem:entity/juniper orionmem:dispositionToward orionmem:disp/d1 .
```

---

## Appendix B — SHACL delivery checklist (normative)

| Shape target | Constraints |
|--------------|-------------|
| **`UtteranceSlice`** | `schema:text` **xor** external turn URI reference |
| **`Situation`** | §4.7 |
| **`TypedEntity`** | `rdfs:label`; `entityKind` required |
| **`AffectiveDisposition`** | `trustPolarity`; `dispositionTarget`; `prov:wasDerivedFrom` |
| **`ImpactAssertion`** | `impactSummary`; derivation |
| **`InferenceActivity`** | `prov:startedAtTime`; software agent string |

---

## Appendix C — Brain-lane draft JSON (`Suggest`, normative shape)

Brain lane returns **JSON**, not Turtle. **memory-graph** maps JSON → RDF before SHACL.

```json
{
  "ontology_version": "orionmem-2026-05",
  "utterance_ids": ["t42"],
  "entities": [
    { "id": "urn:uuid:…", "label": "Joey", "entityKind": "animal", "generalizes_to": "urn:uuid:…" }
  ],
  "situations": [
    {
      "id": "urn:uuid:…",
      "utterance_ids": ["t42"],
      "stimulus_entity_id": "urn:uuid:…",
      "about_entity_ids": ["urn:uuid:…"],
      "affectLabel": "annoyance",
      "timeQualitative": "last_week"
    }
  ],
  "edges": [
    { "s": "urn:uuid:…", "p": "orionmem:generalizationOf", "o": "urn:uuid:…", "confidence": 0.72 }
  ],
  "dispositions": [
    { "holder_id": "urn:uuid:…", "target_id": "urn:uuid:…", "trustPolarity": "ambivalent" }
  ]
}
```

**Rule:** **`p`** uses **CURIEs** declared in §4; unknown predicates → validation error unless the ontology release adds them.

---

## Appendix D — `subschema.memory_graph` (normative JSON)

Nested under **`memory_cards.subschema`** for inject/recall fast path. Regenerate from authoritative RDF on every approve.

```json
{
  "memory_graph": {
    "ontology_version": "orionmem-2026-05",
    "named_graphs": ["https://orion.example/ns/memory/ng/session/abc"],
    "situation_id": "urn:uuid:…",
    "utterance_ids": ["t42"],
    "facts": [
      {
        "subject": "Joey",
        "predicate": "angered",
        "object": "Juniper",
        "time": "last_week",
        "source": "situation"
      },
      {
        "subject": "Juniper",
        "predicate": "trust_toward",
        "object": "this breed",
        "polarity": "ambivalent"
      }
    ],
    "entity_refs": {
      "joey": "https://orion.example/ns/mem/entity/{uuid}",
      "breed": "https://orion.example/ns/mem/entity/{uuid}"
    }
  }
}
```

---

---

## 12. References (code)

- Contracts/DAL: `orion/core/contracts/memory_cards.py`, `orion/core/storage/memory_cards.py`
- Hub memory routes: `services/orion-hub/scripts/memory_routes.py`
- Stance GraphDB: `services/orion-cortex-exec/app/chat_stance.py`
- Recall cards: `services/orion-recall/app/cards_adapter.py`, `fusion.py`
