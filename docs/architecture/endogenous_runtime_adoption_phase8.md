# Phase 8: Controlled Runtime Adoption for Endogenous Workflows

## Scope of live adoption

Phase 8 adopts endogenous evaluation into **two narrow runtime surfaces** in `orion-cortex-exec`:

1. `chat_reflective_lane` (message contains reflective markers such as contradiction/revisit/self-review language)
2. `operator_review` (explicit `ctx["endogenous_runtime_operator_review"] == True`)

No other runtime surfaces invoke endogenous workflows in this phase.

## Runtime gating controls

Runtime behavior is controlled via service settings/env flags:

- `ENDOGENOUS_RUNTIME_ENABLED`
- `ENDOGENOUS_RUNTIME_SURFACE_CHAT_REFLECTIVE_ENABLED`
- `ENDOGENOUS_RUNTIME_SURFACE_OPERATOR_ENABLED`
- `ENDOGENOUS_RUNTIME_ALLOWED_WORKFLOW_TYPES`
- `ENDOGENOUS_RUNTIME_ALLOW_MENTOR_BRANCH`
- `ENDOGENOUS_RUNTIME_SAMPLE_RATE`
- `ENDOGENOUS_RUNTIME_MAX_ACTIONS`

This allows immediate rollback at global, per-surface, mentor-branch, or workflow-subset granularity.

## Safety posture

`EndogenousRuntimeAdoptionService` enforces:

- master + per-surface gating
- deterministic sampling gate
- bounded plan max-actions override
- allowed-workflow filtering in runtime mode
- mentor branch default disabled and separately gated
- exception isolation (`failed` audit result, no primary path exception)

## Runtime execution policy

Runtime uses Phase 7 evaluator/planner/orchestrator and keeps behavior bounded:

- cooldown/debounce/lifecycle suppressions remain active
- disallowed runtime workflows are downgraded to suppression
- mentor workflow is suppressed when mentor runtime execution is disabled
- primary chat stance assembly continues regardless of endogenous runtime result

## Observability

Each runtime invocation emits grep-friendly structured logs:

- invocation surface
- enable flags
- allowed workflow list
- decision outcome
- selected/suppressed workflow type
- cooldown/debounce flags
- actions and materialized artifact IDs
- timing and error status

`build_chat_stance_inputs` stores the runtime audit result in `ctx["chat_endogenous_runtime"]` for downstream inspection.

## Rollback and operator ergonomics

Fast disable options:

1. set `ENDOGENOUS_RUNTIME_ENABLED=false` (global hard stop)
2. disable one surface flag while keeping the other enabled
3. remove workflow types from `ENDOGENOUS_RUNTIME_ALLOWED_WORKFLOW_TYPES`
4. force `ENDOGENOUS_RUNTIME_ALLOW_MENTOR_BRANCH=false`
5. reduce to partial rollout using `ENDOGENOUS_RUNTIME_SAMPLE_RATE`

This preserves conservative, low-blast-radius runtime adoption while keeping the feature operationally inspectable and reversible.
