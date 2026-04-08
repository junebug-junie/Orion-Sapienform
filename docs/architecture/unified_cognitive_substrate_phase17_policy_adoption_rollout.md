# Unified Cognitive Substrate — Phase 17: Operator-Controlled Policy Adoption, Staged Rollout, and Rollback

## Why this phase exists

By Phase 16, review-loop telemetry and advisory calibration could recommend changes, but operators still lacked a first-class operational seam to stage, activate narrowly, compare outcomes, and roll back quickly.

Phase 17 introduces that seam as explicit, manual policy adoption plumbing.

## Core stance

Manual operator control is required.

- No automatic policy mutation.
- Baseline defaults remain safe and valid.
- Rollback is first-class.
- Existing strict-zone, queue budget, suppression, and deterministic execution protections remain intact.

## Policy profile model

Phase 17 adds typed policy adoption contracts for substrate review-loop and semantic-read tuning:

- profile contract (`SubstratePolicyProfileV1`),
- adoption request/result,
- rollback request/result,
- audit event,
- resolution/introspection/comparison models.

Rollout scope is bounded by invocation surface and target zone, with optional operator-only gating.

Overrides are bounded to review-loop and safe query-path knobs (cadence/cycles/suppression/follow-up/query limits/cache toggle).

## Manual profile store and runtime resolution

A deterministic profile store (`SubstratePolicyProfileStore`) now supports:

- stage without activation,
- explicit activation,
- rollback to previous/baseline/explicit profile,
- active/staged/rolled-back inspection,
- audit event history,
- deterministic runtime policy resolution by surface + zone + operator mode,
- baseline-vs-active comparison hook.

## Integration points

### Review scheduling

Scheduler now resolves effective policy per invocation-surface/zone before cadence and cycle-budget derivation.

### Review runtime

Runtime resolves active policy for each selected queue item and:

- applies bounded query-limit overrides to consolidation semantic region reads,
- gates frontier follow-up with explicit policy override,
- exposes policy mode/profile in runtime audit metadata.

## Source honesty and boundedness

- Policy adoption state is explicit (`staged`, `active`, `inactive`, `rolled_back`).
- Rollback transitions are audited.
- Baseline fallback remains explicit when no profile matches scope.
- Query/read overrides remain bounded by schema-validated limits.

## Follow-on work

- optional durable backing store for policy profile history,
- richer operator-facing before/after telemetry diff tooling,
- bounded canary percentage rollout if operationally required.
