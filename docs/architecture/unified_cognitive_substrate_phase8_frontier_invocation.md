# Unified Cognitive Substrate — Phase 8: Curiosity and Gap-Driven Frontier Invocation

## Why Phase 8 exists

Phases 6 and 7 made frontier generation and governed landing possible, but Orion still lacked a deterministic substrate-native policy for deciding **when** to invoke frontier and **what bounded region/task** to use.

## Invocation vs generation vs landing

- **Invocation (Phase 8):** derive curiosity/gap signals and decide invoke/defer/noop/operator-only.
- **Generation (Phase 6):** frontier returns typed graph-delta bundles.
- **Landing (Phase 7):** deterministic zone-aware governance decides proposal/provisional/materialize/HITL/block.

This decomposition keeps each policy layer inspectable.

## Curiosity signal philosophy

Invocation signals are deterministic and substrate-native, derived from:

- graph cognition outputs (contradictions, drift, pressure, identity conflict)
- dynamic substrate state (activation/pressure/hypothesis markers)
- structural patterns (ontology sparsity)
- operator-approved request mode

Signals remain bounded and include focal refs, task candidates, zone, strength, and confidence.

## Deterministic task and region selection

Task selection is rule-bound by signal type:

- contradiction hotspot → `contradiction_discovery`
- ontology sparsity → `ontology_expand`
- concept instability → `relation_discovery`
- evidence-gap cluster → `evidence_gap_scan`
- unresolved pressure → `autonomy_hypothesis`

Region selection is bounded (top-k nodes/edges) and zone-safe.

## Zone-aware safety at invocation time

Invocation stage itself enforces conservatism:

- world/concept zones can invoke with lower thresholds
- autonomy zone requires stronger signal threshold (defer otherwise)
- self/relationship zone yields operator-only posture by default

This ensures strict-zone safety before generation and landing.

## Integration path

Phase 8 adds bounded orchestration:

signal evaluation
→ invocation plan
→ Phase 6 expansion request
→ typed delta bundle
→ Phase 7 landing/materialization

No broad always-on runtime rollout is introduced.

## Forward path

Later phases can add:

- runtime adoption policy gates
- invocation-rate budgets and calibration feedback
- curiosity cycle scheduling and replay audits
