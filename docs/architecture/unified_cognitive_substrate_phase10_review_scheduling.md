# Unified Cognitive Substrate — Phase 10: Review Scheduling and Revisit Cadence

## Why Phase 10 exists

Phase 9 produced consolidation outcomes (including `requeue_review`) but did not operationalize when/how those reviews should recur. Phase 10 adds bounded revisit scheduling with queue semantics, cadence, budgets, and termination controls.

## Consolidation vs scheduling

- **Phase 9:** decide what to do with a reviewed region.
- **Phase 10:** decide if/when to revisit that region again.

This separation keeps review decisions and scheduling policy inspectable.

## Deterministic cadence rules

Scheduler maps consolidation outcomes to schedule outcomes:

- `maintain_priority` / `requeue_review` → enqueue soon (urgent/normal cadence)
- `keep_provisional` / `reinforce` → schedule later (monitoring cadence)
- `damp` → slower revisit cadence
- `retire` / `noop` → terminate scheduling
- strict self/relationship zone → operator-only

## Queue semantics

`GraphReviewQueue` provides bounded deterministic behavior:

- region-based upsert (deduplicate by zone+focal refs)
- bounded queue capacity with deterministic eviction
- eligible-next selection by priority/time/state
- typed queue snapshots for introspection

## Cycle budget and termination controls

- per-zone max cycle budgets via policy
- eligibility checks enforce cycle ceilings
- feedback path supports suppression after repeated no-change cycles
- explicit termination/suppression states in queue items

This prevents unbounded revisit churn.

## Calibration hooks

`GraphReviewCyclePolicyV1` exposes tuneable deterministic knobs:

- cadence durations (urgent/normal/slow)
- per-zone max cycle counts
- queue max size
- low-value suppression threshold

These are bounded and intended for later calibration, not learned policy.

## Safety posture

- strict zones remain operator-mediated at scheduling stage
- no hidden escalation through repeated review loops
- no broad runtime rollout is introduced

## Forward path

Later phases can add:

- richer no-change detection from cycle deltas
- operator-facing queue dashboards
- narrow runtime adoption gates for scheduled review execution
