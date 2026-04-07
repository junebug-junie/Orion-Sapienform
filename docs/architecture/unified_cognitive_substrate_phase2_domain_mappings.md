# Unified Cognitive Substrate — Phase 2 (Domain-to-substrate adapter mappings)

## Why this phase exists

Phase 1 defined a canonical cognitive substrate ontology. Phase 2 makes it operationally useful by introducing deterministic adapters from major existing domain artifacts into that substrate.

## What is added in this phase

- A bounded adapter layer for:
  - concept induction artifacts
  - autonomy artifacts
  - spark snapshots
- Adapter outputs are canonical `SubstrateGraphRecordV1` payloads with typed nodes/edges from the Phase 1 substrate schema family.
- Mapping preserves provenance, anchor scope, and subject references where available.

## Mapping strategy (conservative by default)

- **Concept induction**
  - maps concepts → `ConceptNodeV1`
  - maps evidence refs → `EvidenceNodeV1`
  - maps cluster summaries conservatively → `HypothesisNodeV1` (not ontology-branch hardening)
  - emits support/co-occurrence relations and contradiction edges only when delta semantics justify it
- **Autonomy**
  - maps drive state/audit → `DriveNodeV1` and `StateSnapshotNodeV1`
  - maps goal proposals → `GoalNodeV1`
  - maps tension events → `TensionNodeV1`
  - emits bounded relation edges (`seeks`, `activates`/`suppresses`, `blocks`) only from explicit source semantics
- **Spark**
  - maps spark snapshots primarily → `StateSnapshotNodeV1`
  - maps explicit spark tensions to `TensionNodeV1`
  - emits `EventNodeV1` only when explicit transition metadata exists
  - avoids hardening spark snapshots into durable identity claims

## Adapter-based now vs source-native later

- Current systems remain source-native in their own schemas.
- Adapters provide a deterministic bridge into substrate contracts.
- Later phases can incrementally move producers/consumers toward substrate-native flows without broad cutover now.

## Boundary and safety posture

- No runtime workflow expansion in this phase.
- No dynamics/activation engine behavior introduced.
- No GraphDB/SQL responsibility collapse.
- No forced migration of existing persistence/read paths.

## What later phases can build on

- substrate-backed graph materialization/storage paths
- stronger cross-domain contradiction linking
- policy/promotion layers over shared substrate state
- progressive replacement of isolated ontology islands with source-native substrate mutation
