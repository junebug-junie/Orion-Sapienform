# Evidence and Epistemic Classes

Substrate traces and derived artifacts must preserve how we know what we claim.

This guide borrows the evidence discipline from the architecture graph design.

## Evidence classes

### Observed

Measured from runtime or captured traffic.

Examples:

- persisted trace event;
- bus message observed;
- SQL write succeeded;
- policy frame row exists;
- feedback frame captured dry-run outcome;
- health check observed during a time window.

Observed claims should include time window and source ids.

### Declared

Machine-parseable repository or config facts.

Examples:

- `docker-compose.yml` service definitions;
- channel catalog entries;
- `.env_example` keys;
- schema definitions;
- import/client references;
- service port manifests.

Declared claims are not proof of runtime behavior.

### Documented

Human-authored text in repo docs.

Examples:

- README descriptions;
- design specs;
- PR reports;
- ARCH.md files.

Documented claims may be stale and should carry source path and commit/ref.

### Inferred

Synthesized claims from interpretation.

Examples:

- “this service appears to be a control plane”;
- “this route likely affects policy review”;
- “these services are causally coupled.”

Inferred claims must not be treated as Observed unless telemetry or persistence proves them.

## Claim discipline

Every substrate-relevant claim should be able to answer:

```text
What is claimed?
How do we know?
What evidence ids support it?
What class of evidence is it?
What time window applies?
What uncertainty remains?
```

## Evidence refs

Prefer stable refs:

```text
trace_id
event_id
frame_id
receipt_id
delta_id
source_event_id
correlation_id
session_id
artifact_id
policy_id
proposal_id
decision_id
dispatch_id
feedback_frame_id
git_path@git_ref
otel_trace_id
```

## Substrate traces

Trace atoms should include evidence refs rather than raw blobs.

Example:

```json
{
  "summary": "Policy decision requires operator review",
  "proposal_id": "proposal:...",
  "decision_id": "decision:...",
  "evidence_refs": ["proposal.frame:...", "policy.frame:..."]
}
```

## Typed frames

Typed frames should carry source ids explicitly.

Examples:

```text
source_self_state_id
source_attention_frame_id
source_field_tick_id
source_proposal_frame_id
source_policy_frame_id
source_execution_dispatch_frame_id
```

## Reducers

Reducers should preserve the trace ids/event ids that caused a delta.

Reduction receipts should make it possible to audit:

```text
trace events -> extraction -> state delta -> receipt -> field perturbation
```

## Architecture claims

If a context-engineering or meta-services agent writes architecture claims, it must classify them:

```text
Observed
Declared
Documented
Inferred
```

Inferred claims should be quarantined or clearly marked when they lack observed/declared support.

## Anti-patterns

Avoid:

- treating README prose as runtime truth;
- treating compose dependencies as proof of traffic;
- treating model interpretation as observed behavior;
- emitting trace atoms with no evidence refs;
- frames with no source ids;
- pressure hints with no reduction receipt lineage;
- policy decisions with no proposal source;
- feedback with no dispatch source.

## Acceptance criteria

A service is evidence-disciplined when:

- substrate traces include source/evidence refs;
- typed frames include source ids;
- reducers preserve trace-to-delta lineage;
- docs distinguish observed, declared, documented, and inferred claims;
- live proof commands produce observed evidence before declaring a layer live.