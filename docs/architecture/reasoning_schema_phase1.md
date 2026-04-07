# Reasoning Schema Phase 1

Phase 1 introduces a canonical epistemic schema family in `orion.core.schemas.reasoning`.

## Added artifacts

- `ClaimV1`
- `RelationV1`
- `ContradictionV1`
- `MentorProposalV1`
- `PromotionDecisionV1`
- `VerbEvaluationV1`
- `SparkStateSnapshotV1` (reasoning-layer variant)

## Shared envelope guarantees

All reasoning artifacts inherit `ReasoningArtifactBaseV1` with:

- protected `anchor_scope` (`orion|juniper|relationship|world|session`)
- dynamic `subject_ref` for emergent entities/domains
- explicit status lifecycle (`proposed|provisional|canonical|deprecated|rejected`)
- explicit authority lineage (`sensed|user_asserted|local_inferred|mentor_inferred|human_verified`)
- provenance envelope (`ReasoningProvenanceV1`) with channel/kind/producer/evidence/correlation hooks
- explicit graph edge list (`ReasoningEdgeV1`)

Models use strict validation (`extra="forbid"`) to prevent drift.

## Important guardrails encoded

- `MentorProposalV1` is proposal-only (`status="proposed"`) and mentor-authored only (`authority="mentor_inferred"`).
- `ContradictionV1` is first-class and requires at least two involved artifact ids.
- `PromotionDecisionV1` records governance decisions as explicit artifacts.
- `SparkStateSnapshotV1` is first-class but distinct from identity facets.

## Registry surface

The schema registry now includes these reasoning contracts for typed loading and downstream channel binding.
The reasoning spark model is registered as `ReasoningSparkStateSnapshotV1` to avoid conflict with existing telemetry `SparkStateSnapshotV1`.

## Next phase seam

Phase 2 should map concept induction and autonomy outputs into these canonical artifacts through adapter functions,
then materialize through a typed reasoning write contract/repository seam.
