# Orion Autonomy Phase 3 Materialization

## Executive Summary

Phase 3 materializes the hardened autonomy artifacts into RDF/GraphDB-ready triples inside the existing `orion-rdf-writer` flow.

This patch selectively materializes:
- `memory.identity.snapshot.v1`
- `memory.drives.audit.v1`
- `memory.goals.proposed.v1`

It does **not** add autonomous execution, planner triggering, or a new autonomy runtime.

Overall readiness verdict:
- **Graph materialization for the hardened autonomy artifacts is implemented successfully.**
- **Production trust is improved, but still depends on broader RDF writer operational maturity and GraphDB ingestion reliability.**

Key tradeoffs:
- Materialization is selective and conservative.
- `debug.turn.dossier.v1` remains debug-only and is not promoted into canonical graph memory.
- No unstable `related_nodes` blob materialization was added; only semantically meaningful edges were promoted.
- Proposal artifacts are explicitly encoded as **proposal-only**, not executable plans.

## Input Artifacts

### Materialized
1. `memory.identity.snapshot.v1`
2. `memory.drives.audit.v1`
3. `memory.goals.proposed.v1`

### Deferred
1. `debug.turn.dossier.v1`
   - kept debug-only to avoid accidentally turning a join/debug helper into canonical cognition memory
2. generic `related_nodes`
   - too noisy and underspecified to deserve stable graph semantics yet
3. inferred cross-artifact links that are not explicit in payloads
   - especially proposal → snapshot links unless an explicit ref is later added

## Node Mapping

### Canonical node categories

#### Model-layer nodes
- `orion:ModelLayer`
- one URI per layer:
  - `.../modelLayer/self-model`
  - `.../modelLayer/user-model`
  - `.../modelLayer/world-model`
  - `.../modelLayer/relationship-model`

#### Entity anchor nodes
- `orion:ModelEntity`
- plus a layer-specific type:
  - `orion:SelfModelEntity`
  - `orion:UserModelEntity`
  - `orion:WorldModelEntity`
  - `orion:RelationshipModelEntity`

Entity node identity is built from `entity_id`, not from generic subject bucketing.
Examples:
- `self:orion`
- `user:juniper`
- `world:service:auth-api`
- `relationship:orion|juniper`

#### Artifact nodes
Stable artifact URIs are built from `artifact_id`:
- identity snapshots → `.../autonomy/identitySnapshot/{artifact_id}`
- drive audits → `.../autonomy/driveAudit/{artifact_id}`
- proposed goals → `.../autonomy/proposedGoal/{artifact_id}`

#### Provenance and lineage nodes
- `orion:ArtifactProvenance`
- `orion:SourceEventRef`
- `orion:EvidenceItem`
- `orion:TensionReference`
- `orion:CorrelationThread`
- `orion:TraceSpan`
- `orion:TurnContext`

#### Drive nodes
- `orion:DriveDimension`
- `orion:DriveAssessment`

### Node identity strategy

#### Identity snapshots
- primary key: `artifact_id`
- URI: `.../autonomy/identitySnapshot/{artifact_id}`

#### Drive audits
- primary key: `artifact_id`
- URI: `.../autonomy/driveAudit/{artifact_id}`

#### Proposed goals
- primary key: `artifact_id`
- URI: `.../autonomy/proposedGoal/{artifact_id}`

#### World-model entities
- primary key: `entity_id`
- URI: `.../entity/{sanitize(entity_id)}`
- canonical anchors prefer concrete identities such as `world:service:auth-api`, `world:telemetry`, `world:infra`
- explicit fallback remains allowed but non-canonical

#### Relationship-model entities
- primary key: `entity_id`
- URI: `.../entity/{sanitize(entity_id)}`
- currently uses `relationship:orion|juniper`
- remains distinct from self/user entity anchors

## Edge Mapping

### Common artifact edges
- artifact `orion:aboutEntity` entity
- artifact `orion:belongsToModelLayer` layer
- artifact `orion:hasProvenance` provenance node
- artifact `orion:hasCorrelation` correlation node
- artifact `orion:hasTrace` trace node
- artifact `orion:hasTurnContext` turn node
- artifact `orion:referencesSourceEvent` source event ref node
- artifact `orion:supportedByEvidence` evidence node
- artifact `orion:derivedFromTension` tension ref node

### Identity snapshot edges
- snapshot `orion:hasDriveAssessment` drive assessment node
- snapshot `orion:referencesDriveDimension` drive dimension node

### Drive audit edges
- audit `orion:hasDriveAssessment` drive assessment node
- audit `orion:referencesDriveDimension` drive dimension node
- audit `orion:highlightsActiveDrive` drive dimension node

### Proposed goal edges
- goal `orion:influencedByDrive` drive dimension node
- goal carries explicit literals for:
  - `orion:proposalStatus = "proposed"`
  - `orion:executionMode = "proposal-only"`

### Concrete examples
- `memory.drives.audit.v1` → `orion:derivedFromTension` → tension ref nodes
- `memory.identity.snapshot.v1` → `orion:aboutEntity` → world/service entity anchor
- `memory.goals.proposed.v1` → `orion:influencedByDrive` → `coherence`
- any artifact → `orion:referencesSourceEvent` → source event ref node with event/channel lineage

## Provenance Strategy

The graph preserves the hardening work from Phase 2.5 rather than flattening it.

### Preserved directly
- `correlation_id`
- `trace_id`
- `turn_id`
- `source_event_refs`
- `evidence_items`
- `evidence_summary`
- `evidence_text`
- `tension_refs`

### Representation strategy
- `correlation_id` becomes a `CorrelationThread` node
- `trace_id` becomes a `TraceSpan` node
- `turn_id` becomes a `TurnContext` node
- each source event ref becomes a `SourceEventRef` node
- each evidence item becomes an `EvidenceItem` node
- each tension ref becomes a `TensionReference` node
- artifacts link to these via explicit edges and `prov:wasDerivedFrom`

This supports the key questions:
- what caused this artifact?
- what evidence supported it?
- what tension(s) did it derive from?
- how does it join back to the originating cognitive flow?

## Model-Layer Integrity

The four model layers remain distinct and are materialized as separate layer nodes:
1. self-model
2. user-model
3. world-model
4. relationship-model

Layer-specific entity typing prevents collapse:
- self-model entities become `SelfModelEntity`
- user-model entities become `UserModelEntity`
- world-model entities become `WorldModelEntity`
- relationship-model entities become `RelationshipModelEntity`

World-model identities continue to prefer concrete anchors rather than generic `world`.

## Debug vs Canonical Boundary

### Canonical in graph
- identity snapshots
- drive audits
- proposed goals
- explicit provenance/evidence/tension/join nodes attached to those artifacts

### Debug-only
- `debug.turn.dossier.v1`

Why:
- it is a useful join/debug helper
- but it is not a stable cognitive memory primitive
- promoting it would risk enshrining operational scaffolding as canonical ontology

## Files Changed

### Added
- `services/orion-rdf-writer/app/autonomy.py`
- `services/orion-rdf-writer/tests/test_autonomy_materialization.py`
- `docs/reports/orion_autonomy_phase3_materialization.md`

### Modified
- `services/orion-rdf-writer/app/rdf_builder.py`
- `services/orion-rdf-writer/app/settings.py`
- `services/orion-rdf-writer/.env_example`
- `services/orion-rdf-writer/docker-compose.yml`
- `.env_example`

## Verification

### Tests run
- `pytest -q services/orion-rdf-writer/tests/test_autonomy_materialization.py`
- `pytest -q orion/spark/concept_induction/tests/test_concept_induction.py`
- `python -m compileall services/orion-rdf-writer/app services/orion-rdf-writer/tests`

### Smoke commands
- `python -m compileall services/orion-rdf-writer/app services/orion-rdf-writer/tests`
- `pytest -q services/orion-rdf-writer/tests/test_autonomy_materialization.py`
- `pytest -q orion/spark/concept_induction/tests/test_concept_induction.py`

### How to inspect resulting graph/triples
- publish or replay a `memory.identity.snapshot.v1` envelope into the RDF writer subscription path
- inspect GraphDB context `http://conjourney.net/graph/autonomy/identity`
- inspect GraphDB context `http://conjourney.net/graph/autonomy/drives`
- inspect GraphDB context `http://conjourney.net/graph/autonomy/goals`
- query for:
  - `orion:IdentitySnapshot`
  - `orion:DriveAudit`
  - `orion:ProposedGoal`
  - `orion:referencesSourceEvent`
  - `orion:supportedByEvidence`
  - `orion:derivedFromTension`

## Risks / Follow-ups

### Remaining weak spots
- RDF writer operational behavior is still only as strong as the existing GraphDB push/retry path
- there is not yet a dedicated recall/query layer for these autonomy graphs
- some graph predicates are Orion-local and not yet formalized into a broader ontology document

### What should be Phase 4
- add recall/query surfaces for autonomy graph neighborhoods
- formalize ontology docs for autonomy predicates/classes
- add richer materialization for explicit cross-artifact links when payloads carry direct refs

### What should NOT be done yet
- do not add goal execution
- do not infer commitments from proposals
- do not treat debug dossiers as canonical graph memory
- do not auto-link artifacts using brittle heuristics when the payload lacks an explicit reference

## If I were ruthless

1. Add explicit snapshot/audit/goal cross-refs in the payload contracts so graph links stop depending on indirect shared lineage.
2. Add GraphDB/SPARQL smoke tests that validate stored contexts and predicates end-to-end, not just local triple construction.
3. Publish a minimal autonomy ontology reference so downstream consumers know which predicates are stable and which are provisional.
