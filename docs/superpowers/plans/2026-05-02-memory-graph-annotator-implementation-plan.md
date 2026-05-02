# Memory graph annotator + dual-write GraphDB — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship post-hoc Hub memory graph annotation with ontology-backed RDF in GraphDB, synchronous Postgres projection onto Memory Cards v1, brain-lane JSON suggest, and consumption hooks for recall, memory inject, and chat stance — per [design spec](../specs/2026-05-02-memory-graph-annotator-hub-design.md).

**Architecture:** Implement **`orion.memory_graph`** as a shared Python package (validate JSON → RDF, SHACL, projector to `MemoryCardCreateV1` / edges / `subschema.memory_graph`). Hub calls this library from FastAPI routes (Phase D); optionally extract to a standalone service later without changing RPC payloads. GraphDB writes use named graphs; compensation on PG failure deletes the batch triples written in the same approve operation.

**Tech Stack:** Python 3.11+, FastAPI, pydantic v2, RDFLib (or existing repo RDF stack — align with `services/orion-rdf-writer/app/rdf_builder.py`), pySHACL or GraphDB’s validation if configured, asyncpg for Postgres (existing DAL patterns), GraphDB HTTP REST for RDF.

**Spec approval:** Operator acknowledged completion (“ok”) after ontology expansion; treat §4 and Appendices A–D as normative unless revised.

---

## Scope note (multi-subsystem)

This plan is **one document** with **phases** A–H. Phases **F–H** (recall / inject / stance) can proceed in parallel **after Phase C** lands the RDF + dual-write core; Phase D (Hub) depends on B–C for meaningful E2E.

---

## File structure (planned)

| Path | Responsibility |
|------|----------------|
| `ontology/memory/orionmem-v2026-05.ttl` | Classes, properties, `owl:imports` PROV + schema fragments; version IRI |
| `ontology/memory/shapes-orionmem-v2026-05.ttl` | SHACL shapes for §4.7 / Appendix B |
| `orion/memory_graph/__init__.py` | Package exports |
| `orion/memory_graph/dto.py` | Pydantic models for Appendix C JSON + approve payloads |
| `orion/memory_graph/json_to_rdf.py` | Draft JSON → `rdflib.Graph` with CURIE expansion |
| `orion/memory_graph/validate.py` | SHACL + closed-world checks |
| `orion/memory_graph/project.py` | RDF → `MemoryCardCreateV1`, edges, `subschema` per §4.9–4.10 / Appendix D |
| `orion/memory_graph/graphdb.py` | Named-graph INSERT; compensation DELETE by batch id |
| `config/memory_graph/projector_mapping.yaml` | RDF → card field mapping (editable without code) |
| `tests/test_memory_graph_*.py` | Contract + golden exemplar |
| `services/orion-hub/scripts/memory_graph_routes.py` | New router: validate / approve / suggest proxy |
| `services/orion-hub/static/js/app.js` | Memory tab annotator UI section |
| `services/orion-cortex-exec/app/chat_stance.py` | Named-graph UNION for memory graphs |
| `services/orion-recall/app/` | Optional SPARQL expansion module |
| `services/orion-cortex-orch/app/memory_inject.py` | Read `subschema.memory_graph.facts` |

---

### Phase A — Ontology artifacts + golden fixture

### Task A1: Ontology TTL skeleton + imports

**Files:**
- Create: `ontology/memory/orionmem-v2026-05.ttl`
- Create: `ontology/memory/README.md` (one paragraph: how to load in GraphDB workbench)

- [ ] **Step 1:** Create TTL declaring **`@prefix orionmem:`** base IRI (HTTPS, version suffix), **`owl:Ontology`** with `owl:versionIRI`, **`owl:imports`** for PROV-O and a minimal Schema.org subset (or use `schema:` prefix without full import if repo avoids large imports — document choice).

Example fragment:

```turtle
@prefix owl:   <http://www.w3.org/2002/07/owl#> .
@prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .
@prefix orionmem: <https://orion.local/ns/mem/v2026-05#> .

<https://orion.local/ns/mem/v2026-05/> a owl:Ontology ;
  owl:versionIRI <https://orion.local/ns/mem/v2026-05/> ;
  rdfs:label "Orion operator memory extension v2026-05" .

orionmem:TypedEntity a owl:Class ;
  rdfs:subClassOf <https://schema.org/Thing> ;
  rdfs:label "Typed entity" .

orionmem:Situation a owl:Class ;
  rdfs:label "Annotated situation" .
```

- [ ] **Step 2:** Add object/datatype properties from spec §4.5–4.6 with `rdfs:domain` / `rdfs:range` where stable.

- [ ] **Step 3:** Commit `ontology/memory/orionmem-v2026-05.ttl` + README.

```bash
git add ontology/memory/
git commit -m "feat(ontology): Orion memory extension TTL v2026-05 skeleton"
```

---

### Task A2: SHACL shapes (Appendix B)

**Files:**
- Create: `ontology/memory/shapes-orionmem-v2026-05.ttl`

- [ ] **Step 1:** Define `NodeShape` for `Situation` requiring `prov:wasDerivedFrom` minCount 1; `TypedEntity` requiring `rdfs:label` and `orionmem:entityKind`.

- [ ] **Step 2:** Commit.

---

### Task A3: Golden JSON + expected projection snapshot

**Files:**
- Create: `tests/fixtures/memory_graph/joey_cats_draft.json` (Appendix C shape)
- Create: `tests/fixtures/memory_graph/joey_cats_subschema_snapshot.json` (Appendix D shape)

- [ ] **Step 1:** Copy entity UUIDs from fixtures into stable deterministic UUIDs in JSON files (no random each run).

- [ ] **Step 2:** Commit fixtures.

---

### Phase B — `orion.memory_graph` core (no network)

### Task B1: DTO models (Appendix C)

**Files:**
- Create: `orion/memory_graph/dto.py`
- Create: `tests/test_memory_graph_dto.py`

- [ ] **Step 1:** Write failing test loading `joey_cats_draft.json` into pydantic models (`SuggestDraftV1` with nested entities, situations, edges, dispositions).

```python
# tests/test_memory_graph_dto.py
import json
from pathlib import Path

from orion.memory_graph.dto import SuggestDraftV1

def test_suggest_draft_parses_fixture():
    raw = json.loads(Path("tests/fixtures/memory_graph/joey_cats_draft.json").read_text())
    m = SuggestDraftV1.model_validate(raw)
    assert m.ontology_version == "orionmem-2026-05"
    assert m.utterance_ids == ["t42"]
```

- [ ] **Step 2:** Run `python3 -m pytest tests/test_memory_graph_dto.py -v` → expect import error until package exists.

- [ ] **Step 3:** Implement `orion/memory_graph/dto.py` with `model_config = ConfigDict(extra="forbid")` on top-level draft.

- [ ] **Step 4:** Run pytest → PASS.

- [ ] **Step 5:** Commit.

---

### Task B2: JSON → RDF (`json_to_rdf.py`)

**Files:**
- Create: `orion/memory_graph/json_to_rdf.py`
- Create: `tests/test_memory_graph_json_to_rdf.py`

- [ ] **Step 1:** Failing test: parse fixture → graph → assert triple count ≥ N and `orionmem:Situation` subject exists.

- [ ] **Step 2:** Implement CURIE resolution using a **fixed map** from §4 prefixes; unknown `p` in edges raises `ValueError`.

- [ ] **Step 3:** Commit.

---

### Task B3: Projector (`project.py`) + YAML mapping

**Files:**
- Create: `config/memory_graph/projector_mapping.yaml`
- Create: `orion/memory_graph/project.py`
- Create: `tests/test_memory_graph_project.py`

- [ ] **Step 1:** YAML lists `rdf_type_to_card_types`, `entity_kind_to_anchor_class`, `predicate_to_edge_type` with defaults from §4.9.

- [ ] **Step 2:** Test: project fixture graph → list of `MemoryCardCreateV1`-compatible dicts (use contracts import) with non-empty `title` for Situation node.

```python
from orion.core.contracts.memory_cards import MemoryCardCreateV1

def test_project_yields_valid_create_payload():
    # load graph from json_to_rdf + fixture; cards = project.to_creates(graph)
    card = MemoryCardCreateV1.model_validate(cards[0])
    assert card.title
```

- [ ] **Step 3:** Implement projector writing **`subschema.memory_graph`** per Appendix D.

- [ ] **Step 4:** Commit.

---

### Task B4: SHACL validation (`validate.py`)

**Files:**
- Create: `orion/memory_graph/validate.py`
- Modify: `orion/memory_graph` package `__init__.py` exports
- Create: `tests/test_memory_graph_validate.py`

- [ ] **Step 1:** Install/run pySHACL in tests against `shapes-orionmem-v2026-05.ttl` + converted graph; invalid graph (missing `wasDerivedFrom`) must fail.

- [ ] **Step 2:** Implement `validate_graph(g: Graph) -> list[str]` returning violation messages.

- [ ] **Step 3:** Commit.

---

### Phase C — GraphDB + Postgres dual-write

### Task C1: GraphDB client with named graph + batch id

**Files:**
- Create: `orion/memory_graph/graphdb.py`
- Create: `tests/test_memory_graph_graphdb_mocked.py`

- [ ] **Step 1:** Implement `insert_batch(graph: Graph, named_graph: URIRef, batch_id: str)` serializing N-Triples or Turtle to GraphDB REST **statements** API (document exact endpoint from GraphDB docs used elsewhere — grep repo for `GRAPHDB` HTTP patterns).

- [ ] **Step 2:** Implement `compensate_batch(batch_id: str)` deleting triples tagged with `orionmem:revisionBatch batch_id` **or** storing triple hashes — **pick one**: easiest v1 is **metadata triple** `orionmem:Revision orionmem:batchId "uuid"` linked to each top-level entity in batch; DELETE WHERE pattern documented in code comment.

- [ ] **Step 3:** Mock HTTP in tests; assert POST body contains named graph.

- [ ] **Step 4:** Commit.

---

### Task C2: Approve orchestration (validate → GraphDB → DAL)

**Files:**
- Create: `orion/memory_graph/approve.py`
- Modify: `orion/core/storage/memory_cards.py` (only if new helper needed) — prefer calling existing `create_card`, `add_edge` from planner in approve module
- Create: `tests/test_memory_graph_approve_integration.py` (requires `RECALL_PG_DSN` or testcontainers — mark `@pytest.mark.integration`)

- [ ] **Step 1:** Unit test with mocked GraphDB + in-memory asyncpg mock: on PG failure, `compensate_batch` called once.

- [ ] **Step 2:** Implement `approve_sync(draft_json, pool)` ordering: validate → insert GraphDB → insert Postgres via DAL → return ids; on PG exception run compensate.

- [ ] **Step 3:** Commit.

---

### Phase D — Hub API + UI

### Task D1: Hub routes

**Files:**
- Create: `services/orion-hub/scripts/memory_graph_routes.py`
- Modify: `services/orion-hub/scripts/main.py` — include router
- Modify: `services/orion-hub/app/settings.py` — `GRAPHDB_URL`, memory graph flags if missing
- Create: `services/orion-hub/tests/test_memory_graph_routes.py`

- [ ] **Step 1:** `POST /api/memory/graph/validate` body = Appendix C JSON → returns `{ "ok": bool, "violations": [...], "preview": {...} }`.

- [ ] **Step 2:** `POST /api/memory/graph/approve` → calls `approve_sync`, returns card ids.

- [ ] **Step 3:** Session gate: reuse `_pool` / `ensure_session` pattern from `memory_routes.py`.

- [ ] **Step 4:** Commit.

---

### Task D2: Minimal Hub Memory tab UI

**Files:**
- Modify: `services/orion-hub/templates/index.html` — annotator panel shell if needed
- Modify: `services/orion-hub/static/js/app.js` — fetch validate/approve; display violations

- [ ] **Step 1:** Add “Graph annotator” subsection under Memory tab: textarea for JSON draft (paste from devtools initially), buttons Validate / Approve.

- [ ] **Step 2:** Manual smoke: load Hub, paste fixture JSON, validate returns 200.

- [ ] **Step 3:** Commit.

---

### Phase E — Brain lane suggest

### Task E1: Cortex verb + prompt template

**Files:**
- Create: `orion/cognition/prompts/memory_graph_suggest_prompt.j2` (or under `services/orion-cortex-orch` — follow existing verb prompt layout)
- Modify: executor / orch plan definitions — **locate** where `chat_general` steps register; add **`memory_graph_suggest`** step producing JSON matching Appendix C

- [ ] **Step 1:** Grep codebase for `chat_quick` or verb registration; document exact file paths in commit message.

- [ ] **Step 2:** Prompt instructs: output **only** JSON, keys per Appendix C, `ontology_version` literal.

- [ ] **Step 3:** Hub button “Suggest” calls orch/exec with selected turn text in envelope metadata.

- [ ] **Step 4:** Commit.

---

### Phase F — Recall optional SPARQL

### Task F1: Bounded subgraph fetch

**Files:**
- Modify: `services/orion-recall/app/settings.py` — `RECALL_MEMORY_GRAPH_SPARQL_ENABLED`, timeout ms
- Create: `services/orion-recall/app/memory_graph_sparql.py`
- Modify: `services/orion-recall/app/fusion.py` — optional merge when enabled

- [ ] **Step 1:** SPARQL SELECT labels + disposition triples filtered by `named_graphs` from card `subschema.memory_graph.named_graphs`.

- [ ] **Step 2:** Integration behind flag default **false**.

- [ ] **Step 3:** Commit.

---

### Phase G — Memory inject

### Task G1: Materialize facts lines

**Files:**
- Modify: `services/orion-cortex-orch/app/memory_inject.py`

- [ ] **Step 1:** When assembling known facts, append bullets from `subschema.memory_graph.facts` with budget cap (reuse existing character budget patterns in file).

- [ ] **Step 2:** Unit test with fixture card row dict.

- [ ] **Step 3:** Commit.

---

### Phase H — Chat stance

### Task H1: Named graph UNION

**Files:**
- Modify: `services/orion-cortex-exec/app/chat_stance.py`

- [ ] **Step 1:** Load configurable **memory named graph IRIs** list from settings (env).

- [ ] **Step 2:** Extend SPARQL templates to `GRAPH ?g { ... }` FILTER `?g IN (...)` **or** `VALUES ?g` — match GraphDB query patterns already in file.

- [ ] **Step 3:** Commit.

---

## Verification commands (service-local)

| Phase | Command |
|-------|---------|
| B | `python3 -m pytest orion/memory_graph/tests tests/test_memory_graph_*.py -v` |
| D | `python3 -m pytest services/orion-hub/tests/test_memory_graph_routes.py -v` |
| G | `python3 -m pytest services/orion-cortex-orch/tests/ -q -k memory_inject` (add test file if missing) |

---

## Plan self-review (spec coverage)

| Spec section | Tasks |
|--------------|-------|
| §3 Hybrid service + sync dual-write | B–C, D routes |
| §4 Ontology | A1–A2, B2–B4 |
| Appendix C JSON suggest | B1, E1 |
| Appendix D subschema | B3, G1 |
| §7 Compensation | C1–C2 |
| §8 Recall | F1 |
| §8 Inject | G1 |
| §8 Stance | H1 |
| Hub post-hoc UX | D2, E1 |

**Placeholder scan:** No TBD steps; brain verb file path uses “locate via grep” as an explicit discovery step.

**Type consistency:** Draft uses `SuggestDraftV1` throughout B/E; `subschema.memory_graph` keys match Appendix D.

---

## Execution handoff

Plan saved to `docs/superpowers/plans/2026-05-02-memory-graph-annotator-implementation-plan.md`.

**Two execution options:**

1. **Subagent-driven (recommended)** — Fresh subagent per task, review between tasks. Use **subagent-driven-development** skill.

2. **Inline execution** — Same session with **executing-plans** checkpoints.

Which approach do you want for implementation?
