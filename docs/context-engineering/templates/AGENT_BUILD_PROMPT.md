# Agent Build Prompt: <service>

## Target

`services/<service>`

## Goal

Implement service-owned substrate trace emission for substrate-relevant transitions in `<service>`, then document the service's path through Layers 1–11.

Do not create a central emitter service.

Use shared schema/helpers, but keep service-specific semantic mapping local.

## Required reading

Read:

```text
docs/context-engineering/README.md
docs/context-engineering/00_substrate_trace_doctrine.md
docs/context-engineering/01_service_taxonomy.md
docs/context-engineering/02_service_port_manifest_spec.md
docs/context-engineering/03_substrate_trace_emitter_pattern.md
docs/context-engineering/04_layer_1_to_11_pipeline.md
docs/context-engineering/07_evidence_and_epistemic_classes.md
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

## Rules

- Builder must be pure.
- Publisher must be fail-open.
- Publishing must be env-flag controlled.
- No raw model prompts or completions.
- No credential material.
- No large blobs.
- No full private payload mirroring.
- Use evidence refs and source ids.
- Update channel catalog only if publishing is enabled.
- Existing behavior must be unchanged when publishing is disabled.

## Tests

Run:

```bash
PYTHONPATH=. pytest services/<service>/tests/test_<service>_grammar_emit.py -q
PYTHONPATH=. pytest services/<service>/tests/test_<service>_grammar_publish_fail_open.py -q
```

Add reducer/frame tests only if downstream layers are in this PR.

## Live proof

Provide SQL proof from `grammar_events`.

If reducer/field/frame stages are implemented, provide proof at each layer.

## PR report must include

- service ports touched;
- roles classified;
- semantic roles emitted;
- redaction guarantees;
- tests run;
- live smoke status;
- downstream follow-ups;
- known gaps.
