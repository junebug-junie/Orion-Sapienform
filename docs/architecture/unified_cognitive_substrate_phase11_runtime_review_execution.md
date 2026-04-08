# Unified Cognitive Substrate — Phase 11: Narrow Runtime Review Execution

## Why Phase 11 exists

Phase 10 made reviews schedulable and bounded, but queued items were still passive. Phase 11 operationalizes exactly **one bounded review cycle per runtime invocation** on narrow surfaces.

## Runtime surfaces (narrow adoption)

Phase 11 only supports:

- `operator_review`
- `chat_reflective_lane`

No ambient daemon, no broad always-on loop.

## Queue authority remains primary

Runtime execution reads queue state and only selects items that are due, unsuppressed, unterminated, and within cycle budget. Selection is deterministic (priority + due time + created ordering).

## Single-cycle execution flow

1. Select one eligible queue item.
2. Increment review cycle budget (`mark_reviewed`).
3. Run one deterministic consolidation cycle for that item.
4. Apply feedback (`no_change` updates suppression semantics).
5. Re-apply scheduling from consolidation output (for next cadence).
6. Return typed runtime result with budget before/after and audit notes.

No recursion, no multi-item chain in one call.

## Strict-zone conservatism

Self/relationship zone items are blocked on non-operator surfaces and return `operator_only` outcomes.

## Optional frontier follow-up

A follow-up hook exists but is:

- default-off,
- gated to `operator_review`,
- invoked only when enabled and consolidation still indicates unresolved follow-up pressure (`requeue_review` / `maintain_priority`).

## Observability

`GraphReviewRuntimeResultV1` captures:

- invocation surface
- selected queue item
- outcome (`executed`, `noop`, `suppressed`, `terminated`, `operator_only`, `failed`)
- cycle budget before/after
- queue state summary
- follow-up invocation flag
- audit/debug notes

## Forward path

Future phases can add:

- tighter runtime integration points in operator tooling,
- calibrated selection tie-breakers,
- richer follow-up tracing across Phase 8→6→7,

while preserving single-cycle bounded semantics.
