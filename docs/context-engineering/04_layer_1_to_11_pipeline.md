# Layer 1–11 Pipeline Guide

This document explains how a service transition moves through the Orion cognition substrate.

Not every service needs every layer. The point of the context pack is to make the decision explicit.

## Layer 1: Organs / ingress

Local service transitions become substrate traces.

Examples:

- biometrics sample observed;
- cortex-exec step completed;
- operator approval submitted;
- SQL write failed;
- bus channel violation observed;
- dispatch candidate blocked;
- feedback outcome captured.

Primary implementation today:

```text
GrammarEventV1
orion:grammar:event
grammar_events
```

## Layer 2: Trace substrate

Substrate trace events are persisted and made queryable.

A useful trace includes:

- stable trace id;
- source service;
- source node;
- semantic role;
- atom/edge;
- event kind;
- evidence refs;
- temporal/causal links.

## Layer 3: Reducers

Reducers compile trace histories into durable state deltas and receipts.

Primary artifacts:

```text
StateDeltaV1
ReductionReceiptV1
```

Reducer questions:

- Which trace roles matter?
- What target kind is updated?
- What target id is affected?
- Is the operation create/update/noop?
- What pressure hints or state changes are produced?
- What evidence ids prove the reduction?

## Layer 4: Field digestion

State deltas perturb the field/lattice/tensor state.

Primary artifact:

```text
FieldStateV1
```

Field digestion maps pressure hints onto nodes, capabilities, channels, and edges.

Examples:

```text
execution_load -> execution_pressure
failure_pressure -> reliability_pressure
cpu_pressure -> resource pressure
```

## Layer 5: Attention

Field state selects what matters now.

Primary artifact:

```text
FieldAttentionFrameV1
```

Attention should preserve:

- source field tick;
- salient targets;
- scores;
- dominant channels;
- evidence refs;
- suggested observation mode.

## Layer 6: Self-state

Attention and field synthesize Orion's current operating condition.

Primary artifact:

```text
SelfStateV1
```

Self-state should preserve:

- source field tick;
- source attention frame;
- dimensions;
- stabilizers;
- unresolved pressures;
- summary labels;
- dominant targets.

## Layer 7: Proposal

Operating condition becomes possible actions.

Primary artifact:

```text
ProposalFrameV1
```

Proposal is not action. It is action pressure made inspectable.

## Layer 8: Policy

Proposals are gated by risk, consent, reversibility, and scope.

Primary artifact:

```text
PolicyDecisionFrameV1
```

Policy is not execution. It decides whether a proposal is approved read-only, requires operator review, is deferred, rejected, or approved for execution.

## Layer 9: Dispatch

Approved decisions become dry-run/prepared/dispatch envelopes.

Primary artifact:

```text
ExecutionDispatchFrameV1
```

Default mode should be dry-run. Actual dispatch should route to cortex-orch first, not directly to cortex-exec, unless explicitly smoke-testing.

## Layer 10: Feedback

Consequences, failures, blocks, absences, and unchanged outcomes are captured.

Primary artifact:

```text
FeedbackFrameV1
```

Feedback is not learning yet. It is consequence made observable.

## Layer 11: Consolidation

Repeated consequences become durable pattern structures.

Primary artifacts may include:

```text
ConsolidationFrameV1
MotifObservationV1
ExpectationV1
PriorCandidateV1
SparseTensorSliceV1
SchemaCandidateV1
```

Consolidation is repeated consequence becoming structure.

## Decision checklist

For a service transition, ask:

1. Does it need a substrate trace?
2. Does it need a reducer?
3. Does it produce pressure/state deltas?
4. Should it affect field state?
5. Could attention care about it?
6. Could self-state reflect it?
7. Could it suggest proposals?
8. Does it require policy?
9. Could it dispatch?
10. Does it have consequences or absence to capture?
11. Could repeated occurrences form motifs or expectations?

## Minimal path patterns

### Telemetry organ

```text
Layer 1 trace -> Layer 3 reducer -> Layer 4 field -> Layers 5/6 -> maybe 11
```

### Runtime engine trace

```text
Layer 1 trace -> Layer 3 reducer -> Layer 4 field -> Layers 5/6/7/8/9/10 -> 11
```

### Operator approval

```text
Layer 1 trace -> Layer 8 policy/control evidence -> Layer 10 feedback -> Layer 11 consolidation
```

### Infrastructure anomaly

```text
Layer 1 trace -> Layer 3 reducer -> Layer 4 field pressure -> Layer 5 attention -> Layer 10/11
```

### Read-only UI endpoint

```text
No trace required by default.
Optional trace only if inspection itself becomes substrate-relevant.
```

## Anti-patterns

Avoid:

- service emits raw payloads and calls it trace;
- service emits traces but no reducer when field impact is expected;
- typed frames without source ids;
- policy/dispatch bypassing proposal and policy layers;
- feedback that retries or mutates policy;
- consolidation that silently changes behavior.