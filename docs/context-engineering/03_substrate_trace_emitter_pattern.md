# Substrate Trace Emitter Pattern

Do not create a central emitter service.

Each service owns local trace emission because each service owns the semantics of its meaningful transitions.

## Standard file layout

For a Python service, prefer:

```text
services/<service>/app/grammar_emit.py
services/<service>/app/grammar_publish.py
services/<service>/tests/test_<service>_grammar_emit.py
services/<service>/tests/test_<service>_grammar_publish_fail_open.py
```

The design concept is substrate trace. The current implementation file names may still use `grammar_*` until the codebase is renamed deliberately.

## Split responsibilities

`grammar_emit.py`:

- pure builders;
- no Redis;
- no SQL;
- no network calls;
- no side effects;
- validates event shape;
- redacts or omits sensitive material;
- builds stable ids where possible.

`grammar_publish.py`:

- fail-open publisher;
- env-flag controlled;
- catches/logs exceptions;
- never breaks the service's native runtime path.

## Env flags

Use service-specific flags:

```text
PUBLISH_<SERVICE>_GRAMMAR=false
GRAMMAR_EVENT_CHANNEL=orion:grammar:event
```

Default should usually be `false` until the service is live-smoked.

## Trace shape

A useful trace is a bounded causal episode:

```text
trace_started
  atom_emitted
  edge_emitted
  atom_emitted
  edge_emitted
trace_ended
```

Trace id format:

```text
<service-short>:<node-id>:<correlation-or-artifact-id>
```

Examples:

```text
cortex.exec:athena:<correlation_id>
policy.runtime:athena:<frame_id>
hub.operator:athena:<session_id>
sql.writer:athena:<event_id>
bus.transport:athena:<message_id>
```

## Semantic role naming

Good semantic roles name meaningful transitions:

```text
request_received
request_invalid
artifact_built
artifact_persisted
decision_approved_read_only
decision_requires_operator_review
dispatch_candidate_blocked
feedback_outcome_captured
write_failed
schema_validation_failed
```

Bad roles encode implementation trivia:

```text
function_called
loop_entered
dict_created
line_47_ran
```

## Atom rules

Atoms should be compact semantic facts.

Examples:

```text
Policy approved read-only proposal
Dispatch candidate blocked because operator review required
Feedback captured dry-run-only outcome
SQL writer failed to persist event
Hub received operator approval decision
```

Atom metadata should be safe and bounded:

```json
{
  "proposal_id": "...",
  "decision": "approved_read_only",
  "risk_score": 0.05,
  "required_policy_gate": "read_only"
}
```

Prefer references over payloads:

```text
payload_ref
artifact_id
frame_id
event_id
receipt_id
delta_id
correlation_id
session_id
```

## Edge rules

Edges connect source and consequence.

Use existing `RelationType` values unless the schema has been intentionally extended.

Preferred relationships:

```text
contains
derived_from
temporal_successor
rendered_as
```

If a needed relation such as `blocked_by` or `caused_by` is not available in the closed enum, use the closest existing relation and document the semantic gap. Do not invent enum literals casually.

## Redaction rules

Never mirror raw or high-risk payloads into trace events.

Avoid:

```text
raw model prompts
raw model completions
credential material
large blobs
full audio/image/video payloads
full private user text unless explicitly approved
raw database connection material
unbounded stack traces
```

Use:

```text
safe summary
artifact id
payload ref
redacted error class
bounded status fields
evidence refs
```

## Channel catalog

If the service publishes substrate trace events, update:

```text
orion/bus/channels.yaml
```

Add the service as a producer for:

```text
orion:grammar:event
```

## README requirements

Each service README should document:

- env flags;
- trace roles;
- trace id format;
- what is emitted;
- what is never emitted;
- tests;
- live smoke command;
- whether a downstream reducer exists or is deferred.

## Builder tests

Tests should assert:

- valid `GrammarEventV1`;
- closed `GrammarEventKind` values only;
- closed `RelationType` values only;
- stable ids where expected;
- trace start/end lifecycle;
- expected semantic roles;
- no unsafe payload leakage;
- source/evidence refs present.

## Publisher tests

Tests should assert:

- publisher exceptions do not raise;
- service runtime path continues;
- warning/log is emitted;
- publishing disabled is a no-op.

## Acceptance criteria

A service emitter is acceptable when:

- semantic mapping is local to the service;
- shared helpers are reused but not turned into a god mapper;
- trace emission is behind an env flag;
- publisher is fail-open;
- tests prove schema validity and redaction;
- existing service behavior is unchanged when publishing is disabled.