# Substrate Trace Doctrine

## Definition

A substrate trace is the shared causal record of meaningful transitions in the Orion mesh.

A service may keep native local payloads for execution, storage, transport, UI, model calls, or external APIs. Those native payloads are local. But any transition that matters to Orion's future cognition must leave a substrate trace.

The current implementation schema is `GrammarEventV1`.

The design concept is **Substrate Trace**.

## Core rule

```text
Native payloads are local.
Substrate traces are shared.
Typed frames are compiled artifacts.
Every meaningful transition must be causally legible.
```

## What a substrate trace records

A substrate trace records:

- what happened;
- where it happened;
- when it happened;
- which service and node observed it;
- what source event, artifact, frame, receipt, or delta it derived from;
- what consequence it produced;
- what evidence supports it;
- what confidence, risk, uncertainty, or scope bound applies;
- how it connects to other events in a causal episode.

## What a substrate trace is not

A substrate trace is not:

- a raw payload dump;
- a mirror of every internal dict;
- a packet log;
- a full prompt/completion store;
- a stack trace archive;
- an unbounded debug blob;
- a replacement for native service contracts.

## Useful distinction

```text
Native operational contract:
  The thing a service needs to do its local job.

Shared substrate trace contract:
  The causal shadow of meaningful transitions that later layers can reason over.
```

For example:

- `orion-hub` may use HTTP/WebSocket payloads locally; operator decisions should emit substrate traces.
- `orion-sql-writer` may use SQL inserts locally; write commits/failures should emit substrate traces.
- `orion-bus` may use Redis messages locally; channel violations/backpressure should emit substrate traces.
- `orion-cortex-exec` may execute plans locally; plan/step/result lifecycle should emit substrate traces.

## Relationship to `GrammarEventV1`

Do not rename code casually.

For now:

```text
Substrate Trace = design concept
GrammarEventV1 = current implementation schema
```

Docs should say:

```text
substrate trace event, implemented today as GrammarEventV1
```

## Relationship to typed frames

Typed frames are compiled artifacts derived from traces, state deltas, field state, attention, and prior frames.

Examples:

- `FieldAttentionFrameV1`
- `SelfStateV1`
- `ProposalFrameV1`
- `PolicyDecisionFrameV1`
- `ExecutionDispatchFrameV1`
- `FeedbackFrameV1`
- `ConsolidationFrameV1`

Typed frames do not eliminate substrate traces. The frame is the artifact; the trace records the meaningful lifecycle and causal lineage of creating that artifact.

## End-state

The mesh becomes legible to itself.

That means no important service behavior remains an opaque box forever. Local internals may stay private, but substrate-relevant transitions must be traceable, reducible, digestible, and eventually consolidatable.