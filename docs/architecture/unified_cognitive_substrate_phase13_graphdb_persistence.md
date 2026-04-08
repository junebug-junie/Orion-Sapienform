# Unified Cognitive Substrate — Phase 13: GraphDB-Backed Persistence and Materialization

## Why this phase

Phases 1–12 established ontology, adapters, identity-aware materialization, dynamics, graph cognition, frontier landing, and review-loop controls. But substrate state was still effectively process-local in-memory.

Phase 13 introduces a durable GraphDB-backed substrate store while preserving all canonical substrate contracts and reconciliation semantics.

## Ownership split (explicit)

- **GraphDB (semantic/cognitive substrate):** canonical substrate nodes, edges, identity mappings, provenance-backed semantic state.
- **SQL (operational/control-plane):** review queue, runtime review executions, telemetry summaries, calibration and scheduling records.

This phase does **not** migrate operational review history into GraphDB.

## Design

### Store boundary preserved

`SubstrateGraphMaterializer` still owns deterministic reconciliation and merge semantics, but now targets a generic `SubstrateGraphStore` interface.

Backends:
- `InMemorySubstrateGraphStore` (fallback/dev/test)
- `GraphDBSubstrateStore` (durable semantic backend)

### Backend selection

`SubstrateGraphMaterializer` selects store backend from env when no explicit store is passed:

- `SUBSTRATE_STORE_BACKEND=in_memory|graphdb`
- `SUBSTRATE_GRAPHDB_ENDPOINT` (preferred full SPARQL endpoint)
- fallback endpoint composition: `GRAPHDB_URL` + `GRAPHDB_REPO`
- optional: `SUBSTRATE_GRAPHDB_GRAPH_URI`, `SUBSTRATE_GRAPHDB_TIMEOUT_SEC`, `SUBSTRATE_GRAPHDB_USER`, `SUBSTRATE_GRAPHDB_PASS`

### Graph mapping (canonical contracts preserved)

GraphDB writes persist the canonical substrate payloads plus indexed fields:

- node fields: `node_id`, `node_kind`, `anchor_scope`, `subject_ref`, promotion/risk, salience/confidence, observed time
- edge fields: `edge_id`, `source`, `target`, `predicate`, salience/confidence, observed time
- provenance lineage: serialized canonical provenance + evidence refs
- full canonical payload JSON for deterministic re-hydration
- identity map records (`identity_key -> canonical_id`) for deterministic reconciliation continuity

No parallel GraphDB-only ontology family is introduced.

## Read-back support (bounded)

Store interface now supports bounded region reads used by follow-on graph-query phases:

- focal slice (`read_focal_slice`)
- hotspot region (`read_hotspot_region`)
- contradiction region (`read_contradiction_region`)
- concept region (`read_concept_region`)
- provenance neighborhood (`read_provenance_neighborhood`)

## Non-goals

- no autonomy/runtime widening
- no SQL operational history migration
- no large UI expansion
