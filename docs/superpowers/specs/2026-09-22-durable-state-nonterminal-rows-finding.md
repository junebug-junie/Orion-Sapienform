# Finding: why `substrate_durable_run_state` only receives `completed` rows since 2026-09-14

Date: 2026-09-22. Status: investigation only, no code change in this PR.

## Plain-English verdict

The lifecycle rows did not stop because something broke. On 2026-09-14 curiosity
runs were moved onto a new "resource admission" path inside the durable-runs
service, and that path keeps its own diary of what a run is doing (a different
Postgres table, a different bus channel, a different event shape). It only
copies one entry back into the old diary that the Hub, sql-writer and the
runtime-activity page still read: the final "completed" entry. Every other step
(waiting for a GPU lane, admitted, running, retrying, failed, cancelled) is
recorded in the new diary and never reaches the old one.

The three hypotheses were: (a) the runner no longer emits non-terminal
transitions; (b) they are emitted but sql-writer drops or rejects them; (c) they
are emitted on a different channel, kind or shape that the writer's route map
does not match. The answer is **(c) with an intentional narrowing**:
the transitions are still emitted, on a different channel and kind
(`orion:durable:resource:event` / `durable.resource.event.v1`), and the only bridge
back to `orion:durable:run:state` / `DurableRunStateV1` was deliberately written to
forward `run.completed` alone. Nothing is being dropped or rejected by sql-writer,
because nothing else is ever published on the channel it listens to.

The runner's per-node emitter (`_emit_state`, which also wrote the
`substrate_attention_schema process='durable_run'` rows) is simply no longer on
the code path for admitted runs, which is why those rows stop on the same day.

## Evidence

### The table itself

`substrate_durable_run_state`, rows per day by status (live query 2026-09-22):

```
2026-09-13|completed|2   failed|27   resumed|39   running|6
2026-09-14|completed|2
2026-09-15|completed|9
...
2026-09-21|completed|17
```

Last non-`completed` row: `2026-09-13 13:02:30 UTC` (run `58a6cc15ceb4`, entry_id
a plain uuid hex, the runner's default). First row after the boundary:
`2026-09-14 02:48:42 UTC`, run `gpu2-live-20260914T022620`, entry_id
`gpu2-live-20260914T022620:terminal:completed:state`. Every row since has the
`<run_id>:terminal:completed:state` entry_id shape — a different producer.

### The producer that writes that shape

`services/orion-durable-runs/app/admission_runtime.py:344-355` (outbox drain in
`reconcile()`):

```python
for raw in await self.store.pending_outbox():
    event = ResourceEventV1.model_validate(raw)
    if event.event == "run.completed" and event.entry_id == f"{event.run_id}:terminal:completed":
        completion = DurableRunStateV1(entry_id=event.entry_id+":state", ..., node="finish", status="completed", ...)
        if not await self.runner._publish(self.settings.state_channel, DURABLE_RUN_STATE_KIND, completion, ...):
            continue
    if await self.runner._publish(RESOURCE_EVENT_CHANNEL, RESOURCE_EVENT_KIND, event, ...):
        await self.store.ack_outbox(event.entry_id)
```

That is the only place the admitted path publishes on `orion:durable:run:state`,
and it is gated to `run.completed`. `run.failed`, `run.cancelled`, `run.running`,
`run.resumed`, `run.retrying`, `run.waiting_resource`, `run.admitted` all go out
only as `durable.resource.event.v1` on `orion:durable:resource:event` (and into
`durable_resource_events`, written by `orion/durable_admission/store.py:_event`).

### The runner emitter is not reached for admitted runs

`services/orion-durable-runs/app/main.py:_handle_request` returns after
`admission.submit(request)` whenever `request.admission is not None`; the Hub
always sets it while `durable_admission_enabled` is on
(`services/orion-hub/scripts/curiosity_investigation.py:2950-2954`). The
`DurableRunner.start_run` -> `_emit_state` path (`runner.py:450-507`) is therefore
never entered.

Runtime proof: `docker logs --since 48h orion-athena-durable-runs | grep -c durable_run_state`
returns `0` (that log line is emitted at the end of every `_emit_state` call,
`runner.py:507`). `durable_run_publish_failed` also 0.

The container is running HEAD code (`md5sum` of `admission_runtime.py`, `store.py`,
`admitted_graph.py` inside `orion-athena-durable-runs` matches the checkout).

### The new diary is alive and does hold the missing transitions

`durable_resource_events` for `date(generated_at)='2026-09-21'` (query:
`SELECT event, count(*) FROM durable_resource_events WHERE date(generated_at)='2026-09-21' GROUP BY 1`):
`run.running` 96, `run.resumed` 31, `run.waiting_resource` 51, `run.admitted` 28,
`run.started` 28, `run.accepted` 20, `run.retrying` 4, `run.completed` 17,
`run.failed` 0, plus lane bookkeeping (`run.lane_assigned`, `run.resource_granted`,
`run.lane_swap_suppressed`, `run.resource_eligibility_expanded`). The four
`run.retrying` rows all carry `{"node": "harness_turn"}` —
those are the turn deaths Orion reported; they never became `failed` rows anywhere
because the admitted graph retries up to `DURABLE_RUNS_RETRY_MAX_ATTEMPTS=3` before
projecting `failed`, and the runs then completed on a later attempt.

### sql-writer is not rejecting anything

`docker logs --since 72h orion-athena-sql-writer | grep -i "durable|unknown kind|validation|reject"`
shows only retention-loop lines for `substrate_durable_run_state`. The route map
still has `"durable.run.state.v1": "DurableRunStateSQL"` and the channel is
force-appended (`services/orion-sql-writer/app/settings.py:59,636`). There is no
`durable.resource.event.v1` route in sql-writer at all, so the resource events are
not persisted by sql-writer either (they land in Postgres through the admission
store's own connection, not via the bus).

### When it changed

- `34e389a4c` 2026-09-13 02:26 UTC "feat: add durable resource admission for Curiosity runs" — adds `admission_runtime.py` with the `run.completed`-only bridge.
- `c664ea7df` 2026-09-14 00:38 UTC "chore(durable-runs): enable admitted Curiosity defaults" — turns the path on by default.
- First `:terminal:completed:state` row 2026-09-14 02:48 UTC; last runner-shaped row 2026-09-13 13:02 UTC. The boundary matches the deploy of those two commits.

The same PR widened `DurableRunStatusV1` to include `waiting_resource`, `admitted`,
`retrying`, `cancelled`, `paused`, `accepted`, `queued` (`orion/schemas/durable_run.py:79-82`),
which suggests a fuller bridge was anticipated and then not wired. The
architecture doc (`docs/architecture/durable-resource-admission.md`) and the PR
report do not state a reason for forwarding only `run.completed`.

## Why this is not fixed in this PR

- It is a contract decision, not a one-line bug: which of the two diaries is the
  source of truth for a run's timeline. Two live consumers already read the
  narrow bridge and treat any `DurableRunStateV1` as runner-shaped:
  `orion/hub/runtime_activity.py` (`ACTIVE_RUN_STATUSES = {"running","resumed"}`,
  `TERMINAL_RUN_STATUSES = {"completed","failed","abandoned"}` — `cancelled`,
  `retrying`, `waiting_resource` would be silently mis-bucketed) and
  `services/orion-hub/scripts/hub_surface_routes.py` (counts by status). The Hub's
  `_handle_run_state` (`curiosity_investigation.py:3228-3241`) feeds every
  transition to the runtime-activity page and returns for anything but
  `curiosity.investigate`+`completed`, so forwarding more statuses changes only
  that page's view -- but that page is exactly what needs the wider status set
  first.
- Three sibling patches (run-story page, outreach recording, runner timing
  fields) are in flight against these exact files.
- The admitted graph's node vocabulary (`resource_request`, `resource_wait`,
  `run_started`, `retry_wait`, `failed`) is not the runner's `CURIOSITY_NODES`
  that `DurableRunStateV1.node` documents, so a naive mirror would violate the
  schema comment at `durable_run.py:54-55`.

## Recommended patch (one service, small, but needs the sibling agents' agreement)

Extend the outbox bridge in `admission_runtime.py:reconcile()` so a defined set of
lifecycle events is forwarded as `DurableRunStateV1` on `orion:durable:run:state`,
keeping `entry_id = f"{event.entry_id}:state"` for idempotency (sql-writer's PK
dedup already handles replays). The mirrored set must be explicit, because not
every `run.*` event is a lifecycle fact:

- Progress facts, mirrored as-is: `run.waiting_resource`, `run.admitted`,
  `run.running`, `run.resumed`, `run.retrying` (status = event name minus `run.`,
  `node = detail["node"]` when present, else `"admission"`).
- Terminal facts, mirrored ONLY from `finish_projection` rows whose entry_id is
  `f"{run_id}:terminal:{status}"`: `completed`, `failed`, `cancelled`.
- Not mirrored: the operator-request events written by `control()`
  (`admission_runtime.py:393` records `run.paused`/`run.cancelled`/`run.resumed`
  when the button is pressed, before the graph has actually stopped -- mirroring
  those would mark a run finished while it is still running); `run.accepted`,
  `run.started` (no status in `DurableRunStatusV1`; `run.started` is the
  "began under a lease" fact and would need a status added first); and the lane
  bookkeeping events.

`workflow` must come from the admission row, not a literal: the outbox loop only
has `ResourceEventV1` payloads in scope, so this needs either one
`self.store.get_run(event.run_id)` per outbox event or `pending_outbox()` joining
`durable_admission_runs.request->>'workflow'` (the join is cheaper and keeps the
loop one query). Sketch:

```python
PROGRESS = {"waiting_resource", "admitted", "running", "resumed", "retrying"}
TERMINAL_MIRROR = {"completed", "failed", "cancelled"}
status = event.event.removeprefix("run.")
is_terminal = event.entry_id == f"{event.run_id}:terminal:{status}"
if (status in PROGRESS) or (status in TERMINAL_MIRROR and is_terminal):
    mirror = DurableRunStateV1(entry_id=event.entry_id+":state", run_id=event.run_id,
        workflow=workflow_from_row, thread_id=event.thread_id,
        node=str(event.detail.get("node") or ("finish" if is_terminal else "admission")),
        status=status, correlation_id=event.correlation_id,
        generated_at=event.generated_at, detail=event.detail)
    if not await self.runner._publish(self.settings.state_channel, DURABLE_RUN_STATE_KIND, mirror, ...):
        continue
```

Consumer-first order (CLAUDE.md §6): (1) `orion/hub/runtime_activity.py` learns the
wider status set (`retrying`/`waiting_resource`/`admitted` as active,
`cancelled` as terminal); (2) `hub_surface_routes.py` status buckets; (3) then the
producer above; (4) regression test in
`services/orion-durable-runs/tests/test_admission_review_regressions.py` asserting
one `DurableRunStateV1` per mirrored outbox event and none for a `control()`
request event; (5) a live check that `substrate_durable_run_state` again shows
`running`/`failed` rows for an admitted run.

Alternative that avoids touching the old contract: have the run-story page read
`durable_resource_events` directly (it already has every transition with
node names, and `durable_admission_runs.created_at` gives the true start time).
That is the cheaper path for the redesign doc's "start time and failure timeline"
need and does not change any bus behavior.

The `substrate_attention_schema process='durable_run'` rows (one per transition,
`runner.py:492-505`) are a separate decision: they were the surface lane's view
of the runner and nothing on the admitted path writes them. Either accept that
the surface lane no longer sees durable transitions, or add the same
`AttentionSchemaV1` publish to the bridge above. Not recommended without a
consumer that needs it.

## Side finding 1 (live): the existing bridge mislabels self-sense-eval runs as curiosity runs

The `run.completed` bridge hardcodes `workflow="curiosity.investigate"`
(`admission_runtime.py:348`). Since 2026-09-21 Hub also submits `self_sense_eval`
runs through admission (`curiosity_investigation.py:2994`) and cortex-exec submits
`self_study.reflect` (`services/orion-cortex-exec/app/self_study.py:1492`). Live
(2026-09-22 07:01 UTC):

```sql
SELECT s.run_id, s.workflow, r.request->>'workflow'
FROM substrate_durable_run_state s JOIN durable_admission_runs r USING(run_id)
WHERE s.generated_at>='2026-09-21' AND r.request->>'workflow' <> s.workflow;
-- 20260921T205245Z-2822b8 | curiosity.investigate | self_sense_eval
-- 20260921T235425Z-f129a0 | curiosity.investigate | self_sense_eval
-- 20260922T025855Z-b88ac9 | curiosity.investigate | self_sense_eval
```

Because Hub's `_handle_run_state` gate passes `curiosity.investigate`+`completed`,
those three self-sense-eval completions reached `_enqueue_help_requests_after_run`
and the self-inquiry mirror branch as if they were curiosity runs. This is a real
defect in the current bridge, independent of whether it is widened; the
`workflow_from_row` change above fixes it. Whether those three mis-routed
completions did anything harmful downstream is UNVERIFIED.

## Side finding 2 (live, not the question asked): one run is in a hot failure loop

While pulling evidence, `orion-athena-durable-runs` was logging
`durable_checkpoint_resume_failed run=54537b5b5ccc` with
`SubmissionConflict: run demand is immutable` from
`orion/durable_admission/store.py:153` (`register_demand`) 425 times in 30 minutes
(06:23-06:53 UTC), and writing a `run.checkpoint_resume_failed` event row on each:
3,345 such rows for this one run in `durable_resource_events` between the 02:47 UTC
container start and 07:01 UTC (one every ~4.6 s; still running at 07:01, 139
tracebacks in the preceding 10 minutes).
Cause: the run's LangGraph checkpoint holds
`admission.alternatives == ["agent-burst"]` (decoded from
`SELECT encode(blob,'hex') FROM checkpoint_blobs WHERE thread_id='54537b5b5ccc' AND channel='admission'`,
msgpack: a one-element array), while its `durable_admission_runs.request` and
`durable_resource_demands.requirement` rows hold `["agent-burst", "chat-burst"]`.
`register_demand` compares the two and refuses. Earlier today six 09-21 runs were
in the same state (`durable_driver_failed ... run demand is immutable`, 77 each);
five have since reached `completed` (03:35-05:26 UTC) after PR #2284
(`235f104b5`, additive `widen_alternatives`) was deployed at 02:47 UTC. Run
`54537b5b5ccc` (created 2026-09-21 13:34 UTC) still has the mismatch.

**UNVERIFIED:** who rewrote the stored rows. No code path updates
`durable_admission_runs.request` or `durable_resource_demands.requirement`
(`grep` across the repo finds only `INSERT ... ON CONFLICT DO NOTHING` and
`updated_at`/`control`/`terminal` updates), Postgres `log_statement=none`
(`docker exec orion-athena-sql-db psql -U postgres -Atc 'show log_statement'`), and
the PR #2284 message says the stored row is not rewritten. Runs created after
06:00 UTC today (`3abe26f3c7a2`, `20260922T060916Z-a3b2d8`, `42d62cce268a`,
`c5ea1fe58c0c`) have matching checkpoint and row (`["agent-burst","chat-burst"]`
both, same decode as above), so new runs are not affected. Recommended: compare only caller-supplied
fields in `register_demand` (drop `alternatives` from the equality, the same
carve-out `submit()`'s `comparable()` already makes for
`allow_elastic_activation`), plus a backoff on `run.checkpoint_resume_failed`
so one poisoned run cannot write an event row every 4 seconds. Not done here:
it is inside the admission loop and sibling work is active there.

## What is still UNVERIFIED

- Whether forwarding only `run.completed` was a conscious scope cut or an
  oversight — no doc or commit message says. Ask before widening.
- Who rewrote the stored `alternatives` on the 09-21 runs (above).
- Whether the three self-sense-eval completions mis-routed through Hub's
  curiosity completion hook had any downstream effect.
- Live subscribe to `orion:durable:run:state` during an admitted run was not
  performed (no run was in the bridge window while I watched; the log evidence
  from both producer and consumer was sufficient to place the gap).
