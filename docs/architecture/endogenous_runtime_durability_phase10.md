# Phase 10: Durable Endogenous Runtime Persistence, Replayability, and Bounded Consumption

## Scope

Phase 10 preserves the same two live runtime surfaces (`chat_reflective_lane`, `operator_review`) and focuses only on durability + replay utility.

No new ambient runtime surfaces are introduced.

## Durable store seam

Runtime persistence now uses a store abstraction:

- `EndogenousRuntimeRecordStore` protocol
- `InMemoryEndogenousRuntimeRecordStore` (tests/dev fallback)
- `JsonlEndogenousRuntimeRecordStore` (durable append-only JSONL)

The runtime service selects backend via config:

- `ENDOGENOUS_RUNTIME_STORE_BACKEND=memory|jsonl`
- `ENDOGENOUS_RUNTIME_STORE_PATH`
- `ENDOGENOUS_RUNTIME_STORE_MAX_RECORDS`

## Durability posture

`JsonlEndogenousRuntimeRecordStore` appends typed execution records and trims by bounded max-record count.

Durable write failures are non-blocking:

- main chat stance path continues
- audit status is explicit (`persist_failed`)
- warnings remain grep-friendly

## Replay/retrieval support

Record retrieval supports deterministic bounded filters:

- invocation surface
- workflow type
- outcome
- subject ref
- mentor invocation flag
- created-after timestamp
- limit

This enables operator inspection across process restarts when using durable JSONL storage.

## Bounded downstream consumption seam

Phase 10 adds a read-oriented downstream seam:

- `consume_for_reflective_review(query: EndogenousRuntimeQueryV1)`
- module helper: `consume_endogenous_runtime_for_reflective_review(...)`

Output is typed (`EndogenousRuntimeConsumptionItemV1`) and explicitly read-only; it does not mutate canonical reasoning state.

## Typed contract continuity

Phase 9 runtime models remain canonical and were extended (not replaced) for durability/query/consumption use:

- `EndogenousRuntimeExecutionRecordV1`
- `EndogenousRuntimeAuditV1`
- `EndogenousRuntimeResultV1`
- `EndogenousRuntimeQueryV1`
- `EndogenousRuntimeConsumptionItemV1`

## Safety and rollback

- Runtime execution remains non-blocking.
- Mentor branch remains explicitly gated.
- Existing allowlist/cooldown/debounce behavior remains in the orchestration path.
- Backend rollback is one env change (`jsonl` -> `memory`) with no runtime-flow rewrite.
