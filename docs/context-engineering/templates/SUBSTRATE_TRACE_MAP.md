# Substrate Trace Map: <service>

## Trace id format

```text
<service-short>:<node-id>:<correlation-or-artifact-id>
```

## Semantic roles

| Role | When emitted | Atom type | Layer | Evidence refs |
|---|---|---|---|---|
| <role> | <condition> | observation | 1/2 | correlation_id |

## Trace lifecycle

```text
trace_started
  atom_emitted
  edge_emitted
  atom_emitted
trace_ended
```

## Atoms

Describe bounded atom summaries and safe metadata.

| Role | Summary pattern | Metadata |
|---|---|---|
| <role> | <summary> | <safe keys> |

## Edges

Describe lineage/provenance edges.

| From | Relation | To |
|---|---|---|
| <atom id> | derived_from | <source artifact id> |

Use closed `RelationType` values only unless the enum is deliberately extended.

## Redaction rules

Never emit:

- raw model prompts
- raw model completions
- credential material
- full payload blobs
- full private text unless explicitly allowed
- database connection material
- unbounded debug dumps

Emit references instead:

- payload_ref
- artifact_id
- event_id
- trace_id
- frame_id
- receipt_id
- delta_id
- correlation_id
- session_id

## Publishing

Env flags:

```text
PUBLISH_<SERVICE>_GRAMMAR=false
GRAMMAR_EVENT_CHANNEL=orion:grammar:event
```

Publishing must be fail-open.

## Tests

Expected tests:

```text
services/<service>/tests/test_<service>_grammar_emit.py
services/<service>/tests/test_<service>_grammar_publish_fail_open.py
```

## Live proof query

```sql
select
    created_at
  , source_service
  , trace_id
  , event_json::jsonb #>> '{atom,semantic_role}' as semantic_role
  , event_json::jsonb #>> '{atom,summary}' as summary
from grammar_events
where source_service = '<service>'
order by created_at desc
limit 20;
```
