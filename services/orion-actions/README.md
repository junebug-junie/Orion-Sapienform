# orion-actions

## Executive overview

`orion-actions` is Orion's **operational workflow bridge**. It is not the cognition runtime. It provides bounded operational support around workflows, especially:

- durable schedule persistence,
- due-run claiming / wakeup,
- dispatching scheduled workflow runs back into the normal Orch workflow lane,
- schedule lifecycle management operations,
- operational notifications (including recurring-schedule attention notifications),
- bounded operational side effects (daily pulse/metacog, journaling triggers, collapse-response dispatch).

**Recall profile for dispatch:** `ACTIONS_RECALL_PROFILE` (default `collapse_mirror.v1`) is passed into Cortex `recall.profile` and metadata `recall_profile` for collapse-mirror response and journaling paths that use `settings.actions_recall_profile`. See `services/orion-actions/.env_example`.

This service sits between chat/operator intent and execution infrastructure:

- **Hub** collects operator intent (chat or UI) and shows schedule state.
- **Orch** owns workflow execution.
- **Actions** owns schedule operations and dispatch bridging.
- **Notify** handles notification fan-out, dedupe, and delivery channels.

---

## Core framing (must stay true)

- **Workflow execution is cognition-owned** (Orch / workflow runtime).
- **Schedule entries are durable operational pointers** to workflow runs.
- **`orion-actions` is an operational substrate**, not a second cognition engine.
- **`orion-actions` does not own workflow identity**.
- **Hub is an operator window onto backend truth**, not a second scheduler.
- **Analytics/health summarize durable backend truth**, they do not invent a parallel monitoring system.

Important clarification:

- `dream_cycle`, `journal_pass`, `self_review`, and `concept_induction_pass` are **named cognition workflows**, not “Actions skills”.
- Scheduled runs are re-dispatched through the same Orch workflow lane as immediate runs.
- Actions does not execute a shadow copy of those workflows.

---

## What Actions owns vs does not own

### What Actions owns

- Durable schedule persistence (`ACTIONS_WORKFLOW_SCHEDULE_STORE_PATH`).
- Schedule lifecycle state (`scheduled`, `paused`, `cancelled`, `completed`, etc.).
- Due claiming and scheduler wakeup loop.
- Bounded management operations (`list`, `cancel`, `pause`, `resume`, `update`, `history`).
- Schedule run/event history capture (bounded windows).
- Derived schedule analytics/health from stored schedule + run history.
- Attention notifications for degraded/failing/overdue recurring schedules.
- Operational bridging for scheduled workflow dispatch.

### What Actions does **not** own

- Workflow identity and workflow logic (`dream_cycle`, `journal_pass`, `self_review`, `concept_induction_pass`).
- Generic planner/autonomous cognition behavior.
- Arbitrary autonomous scheduling/planning outside explicit policies.
- Client/UI source of truth (Hub reflects backend state; it does not define it).
- A standalone monitoring platform separate from schedule truth.

---

## Workflows, verbs, and actions: relationship model

- **Workflow**: a named cognition-level execution contract (for example `self_review`).
- **Verb**: runtime verb(s) used by Orch to execute steps inside a workflow.
- **Action (operational)**: bounded operational behavior in `orion-actions` (schedule persistence, dispatch, management, attention notify, etc.).

Examples:

- `self_review` workflow executes through Orch and internally uses `self_concept_reflect`.
- `journal_pass` workflow executes through Orch and journal compose/write boundaries.
- Actions can schedule those workflows, but does not define them.

---

## End-to-end architecture flows

### 1) Immediate workflow invocation (chat)

1. User asks in Hub chat (for example: “Run a self review”).
2. Hub request builder resolves workflow alias and attaches `workflow_request` metadata with execution policy.
3. Orch workflow runtime executes the workflow immediately.
4. Orch may send completion/failure notification (based on `notify_on`) via notify persistence channel.
5. Hub shows response from Orch.

**Roles**
- Hub: intent capture + display.
- Orch: execution owner.
- Actions: not in hot path for immediate run.
- Notify: emits notification if requested.

### 2) Scheduled workflow invocation (one-shot)

1. User asks for scheduled run (for example: “Run your dream cycle tonight at 2”).
2. Hub/Orch derive `execution_policy.schedule`.
3. Orch publishes `WorkflowDispatchRequestV1` to Actions workflow trigger channel.
4. Actions persists schedule entry (durable pointer).
5. Scheduler loop claims when due.
6. Actions redispatches through normal Orch request lane with `scheduled_dispatch` metadata and `invocation_mode=immediate`.
7. Orch executes workflow normally.

### 3) Recurring scheduled workflow invocation

Same as one-shot, but schedule spec is recurring (`daily`/`weekly`).
On claim, Actions advances `next_run_at` to next recurrence; if dispatch fails, Actions requeues toward claimed slot.

### 4) Workflow completion notification

- Notify-on behavior (`none|success|failure|completion`) is part of workflow execution policy.
- Orch emits notification requests; Notify handles delivery semantics.
- For scheduled workflows, this still occurs through Orch after Actions redispatch.

### 5) Schedule management via chat

1. User says a management phrase (list/cancel/pause/resume/update).
2. Hub parser emits `workflow_schedule_management` metadata.
3. Orch calls Actions management channel by RPC.
4. Actions returns structured manage response (including error codes).
5. Orch returns response text + payload to Hub.

### 6) Schedule management via Hub UI

1. Hub calls schedule APIs (`/api/workflow/schedules`, `/history`, action routes).
2. Hub API forwards management request through Cortex/Orch lane.
3. Actions executes operation and returns structured payload.
4. Hub renders inventory/details/edit outcomes directly from payload.

### 7) Attention notifications for degraded/failing schedules

1. Actions derives recurring schedule analytics from durable schedule + run history.
2. Actions evaluates transitions (`entered`, `reminder`, `recovered`) with cooldown.
3. On signal, Actions sends `workflow.schedule.attention.v1` notification request.
4. Notify delivers attention message with dedupe window.
5. Hub can surface resulting notifications and schedule health state.

---

## Supported named user-invocable workflows (cognition side)

These are defined in workflow registry/runtime, not in Actions.

### `dream_cycle`
- **Purpose**: run bounded dream cycle context/routine/interpret path.
- **Example phrases**: “Run your dream cycle”, “dream now”, “run a dream pass”.
- **Schedulable**: yes (through execution policy + Actions schedule bridge).
- **Notify on success/failure/completion**: yes (`notify_on`).
- **Typical result**: concise dream outcome summary.
- **Persisted meaning**: downstream dream artifact/result pathways (`dream.result.v1`) when applicable.

### `journal_pass`
- **Purpose**: run bounded journal compose flow and append-only write path.
- **Example phrases**: “Do a journal pass”, “write a journal entry”.
- **Schedulable**: yes.
- **Notify**: yes.
- **Typical result**: drafted title/body summary plus write acknowledgement.
- **Persisted meaning**: append-only journal write (`journal.entry.write.v1`).

### `self_review`
- **Purpose**: bounded metacognitive reflection synthesis.
- **Example phrases**: “Run a self review”, “do a metacognitive review”.
- **Schedulable**: yes.
- **Notify**: yes.
- **Typical result**: findings-oriented self-review summary.
- **Persisted meaning**: may include graph/journal writebacks if underlying adapter emits them.

### `concept_induction_pass`
- **Purpose**: bounded review of concept induction profiles.
- **Example phrases**: “Run through your concept induction graphs”, “run concept induction”.
- **Schedulable**: yes.
- **Notify**: yes.
- **Typical result**: profile summary (concept/cluster counts, freshness).
- **Persisted meaning**: primarily reads existing profile state; no broad uncontrolled mutation in this pass.

---

## Supported operational action types (as implemented today)

### Workflow schedule operations

- Schedule workflow dispatch (`workflow.schedule.v1` handling).
- List schedules.
- Cancel schedule.
- Pause schedule.
- Resume schedule.
- Update schedule fields (`run_at_utc`, cadence/day/time/timezone, `notify_on`, optional revision check).
- Fetch bounded history/events per schedule.

### Dispatch bridge behavior

- Claim due schedules in batch.
- Mark claimed run as dispatched.
- Redispatch into Orch workflow lane (not local shadow execution).
- Requeue recurring schedules on dispatch failure.

### Notification behaviors

- Workflow attention notifications (`workflow.schedule.attention.v1`) for recurring schedules.
- Existing notify integration for daily ops / other bounded operational actions.

### Bounded mesh ops capability bridge surfaces

The capability bridge now includes bounded operational skill surfaces for mesh/runtime rounds, while preserving the verb-vs-skill split:

- `assess_mesh_presence` → family `mesh_presence` → `skills.mesh.tailscale_mesh_status.v1` (read-only).
- `assess_storage_health` → family `storage_health` → `skills.storage.disk_health_snapshot.v1` (read-only).
- `summarize_recent_changes` → family `repo_change_intel` → `skills.repo.github_recent_prs.v1` (read-only).
- `housekeep_runtime` → family `runtime_housekeeping` → `skills.runtime.docker_prune_stopped_containers.v1` (mutating risk class; dry-run default; explicit execute opt-in).

Additionally, a bounded orchestration skill is available:

- `skills.mesh.mesh_ops_round.v1`
  - collects mesh presence,
  - derives active nodes,
  - snapshots disk health for active nodes,
  - optionally retrieves recent PR digest/changelog grouping,
  - optionally performs docker stopped-container housekeeping (dry-run default),
  - returns structured summary suitable for trace/memory/downstream summarization,
  - optionally writes exactly one explicit journal entry (`ops.mesh_round.v1`) when `write_journal=true`.

### Mesh ops env/config knobs (cortex-exec skill runtime)

These are bounded operational settings used by the mesh ops skill surfaces:

- `SKILLS_MESH_OPS_TIMEOUT_SEC`
- `ORION_ACTIONS_TAILSCALE_PATH`
- `ORION_ACTIONS_SMARTCTL_PATH`
- `ORION_ACTIONS_NVME_PATH`
- `ORION_ACTIONS_GITHUB_API_URL`
- `GITHUB_TOKEN`
- `ORION_ACTIONS_GITHUB_OWNER`
- `ORION_ACTIONS_GITHUB_REPO`
- `ORION_ACTIONS_MESH_DEFAULT_LOOKBACK_DAYS`
- `ORION_ACTIONS_DOCKER_PRUNE_DEFAULT_UNTIL`
- `ORION_ACTIONS_DOCKER_PROTECTED_LABELS`
- `SKILLS_ALLOW_MUTATING_RUNTIME_HOUSEKEEPING` (must be true for execute mode)

### Analytics/health behaviors

Derived per schedule from durable schedule+run records:

- health state,
- overdue status/duration,
- missed-run estimate,
- recent outcomes,
- last success/failure,
- recent success/failure counts,
- `needs_attention`.

---

## Verbal / natural-language invocation examples

### 1) Workflow execution phrases

Expected to resolve:
- “Run your dream cycle”
- “Do a journal pass”
- “Run a self review”
- “Run through your concept induction graphs”

### 2) Workflow scheduling phrases

Patterns currently supported by policy parser include one-shot + recurring forms such as:
- “Run your dream cycle tonight at 2”
- “Do a self review every Friday”
- “Run concept induction and notify me when done”
- “Run journal pass tomorrow morning”
- “Run a dream cycle every night at 23”

### 3) Workflow management phrases

Examples that map to management intent resolver:
- “What workflow runs do I have scheduled?”
- “Cancel my Friday self review”
- “Move my nightly journal pass to 10pm”
- “Pause my Sunday concept induction”
- “Resume my nightly journal pass”

### 4) Notification/attention phrases

`notify_on` phrases are parsed in bounded form (for example: “notify me when done”, “only on failure”).
Attention notifications are system-generated from schedule health transitions.

### Parser expectations (bounded)

- Explicit schedule phrasing works best.
- Absolute/specific times are safer than vague relative phrasing.
- If schedule intent is not parsed into policy, workflow may run immediately.

---

## Schedule lifecycle and management semantics

### Creation

- Orch emits `WorkflowDispatchRequestV1` with scheduled policy.
- Actions validates schedule and persists `WorkflowScheduleRecordV1`.
- Schedule identity is durable (`schedule_id`) and separate from workflow identity.

### One-shot vs recurring

- One-shot: claims once, then transitions to completed.
- Recurring: each claim advances `next_run_at`; state remains scheduled while future run exists.

### Durable identity + revision

- `schedule_id` is stable durable identifier.
- `revision` increments on state/policy changes.
- Update supports `expected_revision` for stale edit protection.

### Conflict/stale edit behavior

- Revision mismatch returns `schedule_revision_conflict` with expected/current revisions.
- Ambiguous workflow-only selection returns `ambiguous_selection` and candidate list.

### Management operations

- `list`: returns schedules (optionally bounded global history/events).
- `cancel`: terminal cancellation (cannot resume afterwards).
- `pause`: allowed for active schedules.
- `resume`: only valid from paused.
- `update`: patch schedule + optionally notify policy.
- `history`: bounded per-schedule run/event history.

### Restart/reload behavior

- Schedule/run/event state is file-backed and reloads on process restart.
- Due claims are restart-safe against duplicate one-shot dispatch in normal restart flow.

### Recurring failure/requeue behavior

- If dispatch fails after claim, run is marked failed and recurring schedule is moved back to a scheduled state.
- Claimed slot metadata is used to restore earlier next-run when appropriate.

---

## Schedule analytics and health model

Analytics are computed from durable backend schedule + run history (not client-side inference):

- `health`: `healthy | degraded | failing | paused | idle | cancelled`
- `is_overdue` + `overdue_seconds`
- `missed_run_count` estimate (daily/weekly recurring)
- `recent_outcomes` / `most_recent_result_status`
- `last_success_at` / `last_failure_at`
- `recent_success_count` / `recent_failure_count`
- `needs_attention`

Health behavior is intentionally bounded:

- no runs yet -> `idle`
- repeated failures or overdue-without-success -> `failing`
- mixed failures/overdue -> `degraded`
- otherwise -> `healthy`

---

## Attention notifications

Attention signals are evaluated for **recurring** schedules only and emitted when conditions transition or reminders are due:

- conditions: `failing`, `overdue` (beyond threshold), `degraded` (bounded failure threshold)
- transitions: `entered`, `reminder`, `recovered`
- bounded by cooldown + dedupe window

Goals:

- proactive operator awareness,
- minimal spam via transition-awareness and reminder cooldown,
- backend-truth-derived summaries rather than synthetic monitors.

---

## Error handling and response contracts

Schedule management returns structured responses with:

- `ok` + `message`
- `error_code` (machine-readable)
- `error_details` (structured context)
- optional schedule/schedules/history/events payloads

Current structured error codes include:

- `invalid_management_payload`
- `ambiguous_selection`
- `schedule_not_found`
- `already_cancelled`
- `already_paused`
- `unsupported_transition`
- `missing_patch`
- `invalid_patch`
- `schedule_policy_missing`
- `schedule_revision_conflict`

UI/operator clients should branch on `error_code`, not brittle message-string matching.

---

## Operator surfaces

### Chat (Hub → Orch → Actions)

- Natural-language workflow invocation.
- Natural-language schedule management.

### Hub schedule UI / APIs

- Schedule inventory list.
- Filtered views (active/paused/cancelled).
- Details modal (state, cadence, analytics).
- Edit/manage flows (pause/resume/cancel/update).
- History/events view per schedule.

The Hub UI is a presentation surface over backend responses, not a parallel schedule authority.

---

## Current v1 constraints / known boundaries

- **Storage substrate**: local JSON file persistence (`workflow_schedules.json`) with atomic temp-file replace; acceptable for v1 durability and simplicity.
- **Process model**: in-process lock (`RLock`) and local file imply practical single-writer assumptions; not a distributed scheduler.
- **Dispatch cadence**: scheduler loop ticks every ~45s; precision is bounded by loop interval and downstream availability.
- **History windows**: bounded retention (`runs` history limit; event truncation).
- **Recurring analytics scope**: missed-run estimation currently daily/weekly only.
- **Attention scope**: recurring schedules only.
- **Browser test status**: schedule-panel browser smoke test exists but is dependency-gated (`pytest.importorskip("playwright.sync_api")`), so it may be skipped in CI environments lacking Playwright.
- **Intentionally deferred**: distributed locking/leader election, generalized autonomous scheduler behavior, and full monitoring-platform features.

---

## How to extend safely

### Add a new named workflow

1. Add workflow definition + aliases in cognition workflow registry.
2. Add Orch runtime execution path for that workflow.
3. Keep workflow logic in cognition/Orch boundaries.
4. Allow scheduling by reusing existing execution policy + Actions bridge.

### Add a new schedule management capability

1. Extend management request/response schema.
2. Implement operation in `WorkflowScheduleStore.apply_management`.
3. Preserve structured error codes.
4. Surface via Orch management bridge and Hub API/UI.

### Add a new analytics field

1. Add field to `WorkflowScheduleAnalyticsV1`.
2. Derive from durable schedule/run truth in store.
3. Keep UI as renderer, not source of analytics truth.

### Add a new attention signal

1. Encode bounded condition logic in attention evaluation.
2. Emit transition-based signals with cooldown/dedupe.
3. Keep focus on actionable schedule health, not generalized monitoring.

### Guardrails

- Do **not** move workflow identity/logic into Actions.
- Do **not** create client-only schedule truth.
- Do **not** add generic autonomous scheduler/planner behavior.
- Do **not** turn this into a full monitoring platform.

---

## Practical references in repo

- Actions service runtime: `services/orion-actions/app/main.py`
- Durable schedule store + analytics + attention: `services/orion-actions/app/workflow_schedule_store.py`
- Workflow schemas/contracts: `orion/schemas/workflow_execution.py`
- Workflow registry and alias matching: `orion/cognition/workflows/registry.py`
- Workflow execution policy parsing: `orion/cognition/workflows/execution_policy.py`
- Workflow management intent parsing: `orion/cognition/workflows/management.py`
- Orch workflow runtime and schedule bridge: `services/orion-cortex-orch/app/workflow_runtime.py`
- Hub request builder and schedule APIs/UI:
  - `services/orion-hub/scripts/cortex_request_builder.py`
  - `services/orion-hub/scripts/api_routes.py`
  - `services/orion-hub/static/js/workflow-schedule-ui.js`
  - `services/orion-hub/templates/index.html`
