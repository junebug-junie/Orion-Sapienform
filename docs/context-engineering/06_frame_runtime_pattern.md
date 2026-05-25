# Frame Runtime Pattern

Frame runtimes are substrate cognition layers.

They consume prior substrate artifacts and emit typed compiled frames.

Examples:

```text
FieldStateV1
FieldAttentionFrameV1
SelfStateV1
ProposalFrameV1
PolicyDecisionFrameV1
ExecutionDispatchFrameV1
FeedbackFrameV1
ConsolidationFrameV1
```

## Core rule

Typed frames are compiled artifacts. They are not a replacement for substrate traces.

A frame runtime should persist the frame and, where useful, emit substrate traces describing the frame lifecycle and its causal lineage.

## Frame runtime responsibilities

A frame runtime should:

- load latest eligible source artifacts;
- avoid reprocessing the same source id/window;
- build a deterministic typed frame;
- preserve source ids and evidence refs;
- persist the frame idempotently;
- expose read-only debug/latest endpoint when useful;
- include a smoke script and SQL proof query.

## Frame runtime non-goals

A frame runtime should not:

- read private service internals;
- call LLMs unless explicitly designed and schema-bounded;
- execute actions unless it is the dispatch layer and policy allows it;
- mutate policy/proposal weights;
- hide source lineage;
- emit opaque dict blobs.

## Standard service layout

```text
services/orion-<frame>-runtime/
  app/
    __init__.py
    main.py
    settings.py
    store.py
    worker.py
  Dockerfile
  docker-compose.yml
  requirements.txt
  .env_example
  README.md
```

## Standard shared package layout

```text
orion/<frame_domain>/
  __init__.py
  builder.py
  policy.py
  scoring.py
  extractors.py
```

Not every domain needs every file.

## Standard config

```text
config/<frame_domain>/<policy_or_config>.v1.yaml
```

The config should include:

- schema version;
- policy/config id;
- thresholds;
- limits;
- scoring weights;
- channel mappings;
- redaction or exclusion rules where applicable.

## Standard frame fields

A typed frame should usually include:

```text
schema_version
frame_id / self_state_id / artifact_id
generated_at
source_*_id fields
policy_id / config_id
scores or dimensions
candidates / observations / decisions where applicable
evidence_refs
warnings
```

## Lifecycle traces

If a frame runtime emits substrate traces, useful roles include:

```text
frame_build_started
source_artifact_loaded
source_artifact_missing
candidate_emitted
candidate_suppressed
decision_emitted
observation_emitted
frame_persisted
frame_build_failed
```

Do not emit full frame JSON into trace atoms. Emit frame ids and bounded summaries.

## Idempotency

Frame runtimes should be idempotent per source id/window.

Examples:

```text
ProposalFrameV1: one frame per source SelfStateV1 and proposal policy id
PolicyDecisionFrameV1: one frame per source ProposalFrameV1 and policy id
ExecutionDispatchFrameV1: one frame per source PolicyDecisionFrameV1 and dispatch policy id
FeedbackFrameV1: one frame per source ExecutionDispatchFrameV1 and feedback policy id
ConsolidationFrameV1: one frame per time window and consolidation policy id
```

## Live proof shape

Each frame runtime should have a smoke script that proves:

1. source artifact exists;
2. runtime container is running;
3. destination table exists;
4. latest frame row exists;
5. critical fields have expected values;
6. no disallowed side effects happened.

## Acceptance criteria

A frame runtime is acceptable when:

- typed schema exists and validates;
- config/policy loads;
- builder is deterministic;
- runtime persists idempotently;
- source ids and evidence refs are present;
- live smoke proves the chain;
- no behavior mutation happens outside the layer's explicit responsibility.