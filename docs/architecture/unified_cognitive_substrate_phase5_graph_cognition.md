# Unified Cognitive Substrate — Phase 5: Graph Cognition V1

## Why substrate-native graph cognition

Graph cognition is now anchored directly on the unified dynamic substrate (ontology + adapters + materialized state + deterministic dynamics), rather than aggregating fragmented graph islands. This keeps perception consistent with canonical cognitive state.

## Layering

Phase 5 introduces a deterministic perception stack:

1. bounded substrate-backed graph views
2. deterministic feature extraction
3. deterministic interpreters producing typed assessments
4. compact metacognitive perception brief

No learned models are used for core scores or contradiction judgments in V1.

## View builders over substrate

`build_graph_views(...)` builds bounded, typed views from materialized substrate state:

- SemanticGraphViewV1
- EpisodicGraphViewV1
- SelfGraphViewV1
- SocialGraphViewV1
- ExecutiveGraphViewV1
- ConceptGraphViewV1
- ContradictionGraphViewV1
- TemporalDeltaGraphViewV1

Each view is scope/time bounded, includes truncation markers, and only references canonical substrate node/edge ids.

## Feature families (deterministic)

`extract_graph_features(...)` computes bounded feature families:

- structural: counts, degree/connectivity, fragmentation indicators
- temporal: change density, inactivity, resurfacing, churn, persistence
- semantic: support/conflict counts, contradiction density, provenance diversity
- dynamic/self-regulatory: activation/pressure hotspots, dormancy, tension accumulation, coherence trend
- social/executive: unresolved commitments, stalled goals, retries, blockage, goal competition

These features explicitly consume Phase 4 dynamic state (`signals.activation`, `metadata.dynamic_pressure`, dormancy markers).

## Typed cognition outputs

`interpret_graph_cognition(...)` emits typed outputs:

- CoherenceAssessmentV1
- IdentityConflictSignalV1
- GoalPressureStateV1
- SocialContinuityAssessmentV1
- ConceptDriftSignalV1
- ContradictionCandidateSetV1

All outputs are deterministic, confidence-bounded, and evidence-bearing via `SignalEvidenceBundleV1` / `EvidenceSpanV1`.

## Unified perception brief

`build_metacog_perception_brief(...)` compacts the assessments into router-facing summary:

- top tensions
- top stabilizers
- overall priority
- deterministic recommended verbs
- confidence/degraded markers
- compact supporting evidence

## Boundaries / non-goals

- This is a perception layer; it does not add teacher/frontier mutation.
- No broad runtime orchestration rewrite.
- No LLM-dependent scoring logic.
- Graph cognition remains read-oriented over substrate state; SQL remains authoritative for operational runtime/audit history.

## Forward path

Later phases can build on this by:

- integrating perception brief into executive routing gates
- adding frontier expansion policies informed by contradiction/pressure trends
- introducing learned graph cognition after deterministic baseline validation
