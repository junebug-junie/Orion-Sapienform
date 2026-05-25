# Agent Handoff Template Guide

Use this when handing any service to an agent for substrate trace adoption.

## Standard handoff

```markdown
# Service substrate trace adoption handoff

## Target service

`services/<service>`

## Goal

Implement service-owned substrate trace emission for substrate-relevant transitions in `<service>`, then document whether the service needs reducers, field digestion, typed frames, feedback, or consolidation.

Do not create a central god emitter service.

Use shared schema/helpers, but keep service-specific semantic mapping local.

## Required reading

Read:

```text
docs/context-engineering/README.md
docs/context-engineering/00_substrate_trace_doctrine.md
docs/context-engineering/01_service_taxonomy.md
docs/context-engineering/03_substrate_trace_emitter_pattern.md
docs/context-engineering/04_layer_1_to_11_pipeline.md
services/<service>/AGENT_CONTEXT.md
services/<service>/SERVICE_PORTS.yaml
services/<service>/SUBSTRATE_TRACE_MAP.md
services/<service>/LAYER_PIPELINE_PLAN.md
```

If service-local files are missing, create them from templates first.

## Implementation requirements

Add or update:

```text
services/<service>/app/grammar_emit.py
services/<service>/app/grammar_publish.py
services/<service>/tests/test_<service>_grammar_emit.py
services/<service>/tests/test_<service>_grammar_publish_fail_open.py
```

Use `Substrate Trace` in docs and `GrammarEventV1` for the current implementation schema.

## Semantic mapping

Define local semantic roles based on service transitions, not function names.

Required trace properties:

```text
trace_id
event_id
source_service
source_node
semantic_role
source/evidence refs
bounded summary
no unsafe payloads
```

## Redaction

Do not emit raw model prompts, raw completions, credential material, full blobs, full private payloads, or unbounded debug dumps. Use ids, refs, summaries, and redacted error classes.

## Tests

Run:

```bash
PYTHONPATH=. pytest services/<service>/tests/test_<service>_grammar_emit.py -q
PYTHONPATH=. pytest services/<service>/tests/test_<service>_grammar_publish_fail_open.py -q
```

If a reducer/frame layer is added, run its tests too.

## Live proof

Provide at least one SQL query showing recent trace events from this service.

If reducer/field/frame layers are implemented, provide proof at each layer.

## PR report

Include:

- service ports touched;
- role classification;
- semantic roles emitted;
- redaction guarantees;
- tests run;
- live smoke status;
- downstream reducer/digester/frame follow-ups;
- known gaps.
```

## Review checklist

A reviewer should reject the PR if:

- semantic roles are generic function names;
- raw sensitive payloads are mirrored into trace events;
- publisher failure can break service behavior;
- service semantics are placed in a global god mapper;
- trace events lack evidence refs;
- README/AGENT_CONTEXT files are not updated;
- downstream layer claims are made without proof.