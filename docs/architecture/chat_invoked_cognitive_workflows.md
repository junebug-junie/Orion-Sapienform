# Chat-Invoked Cognitive Workflows

## Why workflows are distinct from Actions skills

Orion already has three different layers with different responsibilities:

1. **Cognitive primitives / verbs** such as `dream_cycle`, `journal.compose`, `self_concept_reflect`, and `concept_induction`.
2. **Chat-invoked cognitive workflows** such as `dream_cycle`, `journal_pass`, `journal_discussion_window_pass`, `self_review`, and `concept_induction_pass`.
3. **Operational capabilities / Actions skills** such as notification, scheduling, runtime inspection, or other bounded side effects.

The workflow lane keeps **workflow identity on the cognition side**. A workflow can reuse an actions-side capability as a helper, but the named workflow itself is not modeled as an `orion-actions` skill. This preserves the distinction between “run a bounded reflective routine” and “perform an operational side effect.”

## How named workflow invocation works

Phase 1 is deterministic and registry-driven:

`Hub chat -> workflow alias resolver -> explicit workflow request metadata -> Cortex Orch workflow runtime`

### Resolution rules

- The registry lives in `orion/cognition/workflows/registry.py`.
- Each workflow declares:
  - `workflow_id`
  - `display_name`
  - `description`
  - aliases / invocation phrases
  - user/autonomous invocability
  - execution mode
  - action-call allowance
  - persistence policy
  - result surface policy
  - ordered adapter steps
  - planner hints
- Hub checks the prompt against the registered aliases **before** ordinary planner fallback when no explicit verb override is already in play.
- On a match, Hub stamps `context.metadata.workflow_request` with the resolved workflow id and alias match.
- Orch detects `workflow_request` and routes into the explicit workflow runtime instead of defaulting to `chat_general` or planner inference.

## Initial workflow adapters

### `dream_cycle`
- Reuses the existing `dream_cycle` cognition verb.
- Preserves the existing dream persistence path (`dream.result.v1` and downstream consumers).
- Returns a concise completion summary in chat metadata and `final_text`.

### `journal_pass`
- Reuses `orion.journaler.worker.build_manual_trigger` and `build_compose_request`.
- Reuses `journal.compose` for prose generation.
- Persists only through the existing append-only `journal.entry.write.v1` boundary on `orion:journal:write`.

### `journal_discussion_window_pass`
- Parses an explicit relative window (minutes / hours / day(s)) from the user text, or defaults to 24 hours when the user asks to journal **our chat / discussion** without a narrower window (see `orion.discussion_window.timeframe.parse_journal_discussion_lookback_seconds`).
- Invokes the bounded read-only skill `skills.chat.discussion_window.v1` (Cortex Exec) to load a **contiguous, time-bounded** slice of rows from SQL `chat_history_log` (`created_at` boundary by default). Optional `user_id` / `source` SQL filters are **not** taken from Hub chat routing metadata (those labels often differ from `chat_history_log` columns); set `context.metadata.discussion_window_user_id` and/or `discussion_window_source` when you need explicit scoping. This path does **not** use `session_id` semantics or semantic recall.
- Builds a manual journal trigger via `build_discussion_window_journal_trigger` (transcript in `prompt_seed`, `source_ref` encoding window bounds and turn count).
- Reuses `journal.compose` and persists only through `journal.entry.write.v1` on `orion:journal:write`, same as `journal_pass`.
- Registry aliases cover phrases such as “journal the last hour”, “journal our chat discussion for the last day”, and “do a journal discussion pass”; additional minute/hour/day phrasing is matched when the same journal-command intent is detected (see `resolve_user_workflow_invocation` in `orion/cognition/workflows/registry.py`).

### `self_review`
- Reuses `self_concept_reflect` and existing self-study reflection writebacks.
- Surfaces a structured self-review summary without redefining self-study storage.

### `concept_induction_pass`
- Reuses the concept induction profile store (`orion.spark.concept_induction.store.LocalProfileStore`).
- Reads existing profile state only and synthesizes a bounded review.
- Does **not** mutate concept state as part of chat invocation.

## Observability and auditability

The workflow runtime emits bounded lifecycle logs:

- `workflow_requested`
- `workflow_started`
- `workflow_completed`
- `workflow_failed`

It also propagates workflow metadata into `CortexClientResult.metadata.workflow`, alongside the machine-readable workflow registry payload for future inspection.

## Adding a future workflow

1. Add a new `WorkflowDefinition` to `orion/cognition/workflows/registry.py`.
2. Register explicit aliases and planner hints.
3. Implement a thin adapter in `services/orion-cortex-orch/app/workflow_runtime.py`.
4. Reuse an existing cognition verb / journal / graph / state pathway where possible.
5. Add focused tests for alias resolution, Orch routing, and adapter behavior.

## Planner-facing surfacing

The registry is intentionally machine-readable. Hub and Orch attach the serialized workflow registry in metadata so planner-style systems can later inspect:

- available workflow ids
- descriptions and aliases
- user-invocable status
- persistence/result policies
- ordered adapter steps

That planner-facing visibility is additive. The Phase 1 user experience does **not** depend on planner inference; it depends on explicit registry matching.
