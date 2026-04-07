# Phase 9: Runtime Signal Fidelity, Result Persistence, and Operator Introspection

## What changed from Phase 8

Phase 8 validated bounded runtime adoption with conservative heuristic synthesis.

Phase 9 keeps the same two live invocation surfaces (`chat_reflective_lane`, `operator_review`) but improves:

- trigger signal fidelity using richer reasoning-backed inputs
- structured typed persistence for runtime decisions/results
- bounded operator introspection for recent runtime records
- replay-friendly traceability (record ids, request ids, artifact refs)

## Runtime signal fidelity

`EndogenousRuntimeSignalBuilder` now assembles trigger requests with deterministic bounded inputs from:

- reasoning summary fields
- reasoning repository contradiction queries
- lifecycle evaluation (`evaluate_entity_lifecycle`)
- bounded selected artifact refs
- autonomy/reflective summaries
- recent endogenous trigger history

The builder also emits `EndogenousRuntimeSignalDigestV1` to preserve which sources were used.

## Structured runtime persistence

Phase 9 introduces typed persistence contracts in `orion/core/schemas/endogenous_runtime.py`:

- `EndogenousRuntimeSignalDigestV1`
- `EndogenousRuntimeExecutionRecordV1`
- `EndogenousRuntimeAuditV1`
- `EndogenousRuntimeResultV1`

Runtime outcomes are persisted through `InMemoryEndogenousRuntimeRecordStore` as structured execution records (not logs-only / ctx-only).

Persisted records include:

- invocation surface
- correlation/session/subject refs
- full trigger request + decision + plan
- signal digest
- mentor invocation + materialized artifact refs
- audit events, timestamps, success/failure metadata

## Operator introspection surface

A bounded introspection helper is provided:

- `inspect_endogenous_runtime_records(...)`

Supported filters:

- invocation surface
- workflow type
- decision outcome
- subject ref
- bounded limit

`build_chat_stance_inputs` now optionally attaches recent records for operator review contexts (`chat_endogenous_runtime_recent`) while keeping failures non-blocking.

## Safety posture

Phase 9 preserves Phase 8 safety controls:

- no new runtime invocation surfaces
- master/per-surface gating and sampling
- runtime workflow allowlist and mentor runtime gate
- non-blocking failure handling for execution/persistence/introspection
- cooldown/debounce/anti-loop behavior remains in Phase 7 evaluator/orchestrator path

## Traceability

Execution records now preserve replay/debug linkage via:

- stable `runtime_record_id`
- trigger `request_id`
- correlation/session ids
- selected artifact refs
- materialized artifact refs
- mentor linkage fields (request id when invoked)

This enables future offline analysis and optional bounded feedback loops without broad runtime rewiring.
