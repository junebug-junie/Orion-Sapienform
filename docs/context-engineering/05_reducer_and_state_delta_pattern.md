# Reducer and State Delta Pattern

Reducers are Layer 3.

They consume substrate trace histories and produce committed substrate state changes.

## Primary artifacts

```text
StateDeltaV1
ReductionReceiptV1
```

A reducer should answer:

```text
Given this trace history, what durable substrate state changed?
```

## Reducer responsibilities

A reducer should:

- fetch eligible trace events;
- group events by trace id or target id;
- validate expected semantic roles;
- extract bounded state facts;
- produce stable delta ids;
- produce reduction receipts;
- persist projection state if needed;
- advance a cursor idempotently;
- fail isolated from unrelated reducers.

## Reducer non-goals

Reducers should not:

- call LLMs;
- emit proposals;
- execute actions;
- mutate policy;
- inspect raw private payloads;
- read service-private internals;
- produce untyped dict swamp.

## Suggested files

For shared reducer logic:

```text
orion/substrate/<domain>/
  __init__.py
  extract.py
  reducer.py
  pipeline.py
```

For runtime integration:

```text
services/orion-substrate-runtime/app/store.py
services/orion-substrate-runtime/app/worker.py
services/orion-substrate-runtime/app/settings.py
```

For persistence:

```text
services/orion-sql-db/manual_migration_<domain>_substrate_loop.sql
```

## Reducer inputs

Reducers should read from substrate trace persistence, usually:

```text
grammar_events
```

The current schema uses `GrammarEventV1`; design docs should call these substrate trace events.

## Reducer outputs

A reducer emits receipts/deltas, usually persisted to:

```text
substrate_reduction_receipts
```

A reducer may also maintain a projection table, for example:

```text
substrate_execution_trajectory_projection
```

## Stable ids

Delta ids should be deterministic when possible.

Recommended preimage:

```text
<target_kind>:<target_id>:<operation>:<source_trace_id>:<source_event_ids_hash>
```

Receipt ids should also be deterministic when rerunning the same source window would otherwise duplicate work.

## Pressure hints

When a reducer output should affect field digestion, include bounded pressure hints.

Example:

```json
{
  "execution_load": 0.375,
  "reasoning_load": 0.05,
  "failure_pressure": 0.0,
  "egress_confidence": 1.0,
  "execution_friction": 0.0
}
```

Pressure hints should be:

- bounded where possible;
- semantically named;
- documented in field lattice/channel mapping;
- stable enough for field digestion;
- not raw service payloads.

## Reducer tests

Required tests:

- schema roundtrip;
- trace grouping;
- extraction from valid trace roles;
- ignored unknown/irrelevant roles;
- stable delta ids;
- idempotent pipeline;
- cursor advancement;
- projection persistence;
- malformed trace handling;
- no raw payload leakage.

## Runtime tests

If integrated into `orion-substrate-runtime`, test:

- reducer tick is isolated from other ticks;
- reducer disabled flag works;
- missing tables produce clear errors;
- reducer failure does not kill unrelated loops;
- receipts are persisted;
- cursor does not skip unprocessed events.

## Field handoff

If a reducer produces deltas for Layer 4, document:

```text
target_kind
pressure_hints
node_id/capability_id mapping
field channel mapping
expected perturbations
```

## Acceptance criteria

A reducer is acceptable when:

- it consumes substrate traces, not private service internals;
- it emits `StateDeltaV1` / `ReductionReceiptV1`;
- it is idempotent;
- it preserves evidence refs;
- it has isolated runtime failure behavior;
- it documents whether field digestion is in scope;
- it has live SQL proof queries.