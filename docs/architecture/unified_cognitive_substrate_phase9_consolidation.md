# Unified Cognitive Substrate — Phase 9: Reflective Graph Consolidation

## Why Phase 9 exists

Phases 6–8 enabled frontier generation, governed landing, and bounded invocation. Phase 9 adds iterative consolidation so graph changes are revisited over time rather than treated as one-off mutations.

## One-off mutation vs iterative consolidation

- **Mutation path (Phases 6–7):** propose/land graph deltas.
- **Invocation path (Phase 8):** decide when to ask frontier.
- **Consolidation path (Phase 9):** compare prior-vs-current region state and decide reinforce/keep/requeue/damp/retire/priority/operator-only.

## Bounded review-cycle philosophy

Consolidation is explicit and bounded:

- request defines focal node/edge region
- evaluator caps region size
- prior cycle snapshot is optional but used when available
- outputs typed decisions + cycle record (inspectable)

No uncontrolled recursion or always-on background self-modification.

## Comparison logic

Region comparison includes:

- node/edge persistence ratios
- activation/pressure deltas
- contradiction persistence deltas
- evidence-gap marker persistence deltas
- isolated frontier-structure deltas

This creates deterministic state-delta digests before outcome selection.

## Outcome semantics

- `reinforce`: stable, integrated region
- `keep_provisional`: plausible but not stable enough
- `requeue_review`: unresolved contradiction/gap remains
- `damp`: weak/low-activation or isolated structures cooling down
- `retire`: stale low-value gap/hypothesis structures
- `maintain_priority`: unresolved high-pressure contradictions/tensions
- `operator_only`: strict self/relationship zone conservatism

## Zone-aware conservatism

- world/concept zones can reinforce/retire more freely
- autonomy remains conservative and pressure-sensitive
- self/relationship remains operator-mediated

Repeated cycles cannot silently escalate strict zones into canon.

## Integration points

Consolidation sits on top of existing layers:

- materialized substrate state
- dynamic activation/pressure state
- graph cognition outputs
- frontier landing summaries
- curiosity/invocation loop hints via requeue outcomes

No broad runtime rollout is introduced in this phase.

## Forward path

Later phases can add:

- calibrated cycle scheduling
- bounded requeue-to-invocation policies
- operator-facing review dashboards for consolidation lineage
