# Unified Cognitive Substrate — Phase 1 (Shared ontology + canonical contracts)

## Why this phase exists

Orion has grown strong control/audit/runtime planes (reasoning artifacts, promotion, triggers, runtime durability, SQL operational history), but core cognitive domains are still represented through partially separate schema families.

Phase 1 introduces a single graph-native substrate vocabulary so future concept induction, autonomy, spark, contradiction handling, and frontier mutation sources converge on one canonical semantic layer.

## What this phase solves

- Defines shared canonical substrate node/edge contracts.
- Defines shared cross-cutting primitives for confidence, salience, activation support, temporal validity, provenance, promotion state, anchor scope, risk tier, and subject references.
- Establishes one typed ontology family that future adapters can target without requiring immediate subsystem rewrites.

## Substrate vs control-plane vs operational SQL

- **Cognitive substrate (graph-native):** semantic nodes/edges and cognitive relations.
- **Control/audit/runtime planes:** orchestration, policy, lifecycle decisions, and operator controls.
- **Operational SQL:** durable runtime/audit history and read-side operations.

This phase **does not** collapse these layers. It creates a canonical semantic target while preserving existing control-plane and SQL responsibilities.

## Mapping path from existing domains (convergence intent)

- **Concept induction** → `ConceptNodeV1`, `EvidenceNodeV1`, `HypothesisNodeV1`, `ContradictionNodeV1`, plus `SubstrateEdgeV1` relations.
- **Autonomy** → `DriveNodeV1`, `TensionNodeV1`, `GoalNodeV1`, `StateSnapshotNodeV1`, plus relation edges.
- **Spark** → `StateSnapshotNodeV1` and `EventNodeV1` with tension/activation links.
- **Graph cognition views** → read views over substrate primitives (not permanent parallel ontology islands).
- **Mentor/teacher/frontier** (future) → mutation sources against substrate, not separate ontology families.
- **Runtime operational records** → remain SQL operational history (not substrate primitives).

## Phase boundary (explicit non-goals)

- No teacher/frontier integration in this phase.
- No activation/decay dynamics engine implementation (schema support only).
- No broad runtime workflow expansion.
- No hard-cut migration of all existing schemas/services.
- No GraphDB/SQL responsibility collapse.

## What later phases build on

Later phases can add:
- adapter mappings from existing domain schemas into substrate contracts,
- zone-aware promotion/policy over substrate state,
- controlled mutation pipelines (including mentor/frontier sources),
- dynamic activation/decay/pressure mechanics over this common ontology.
