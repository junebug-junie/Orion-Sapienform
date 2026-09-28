# Visual reverie on the durable graph path

Date: 2026-09-28. Status: approved direction (Juniper, same day), implementing.

## Arsonist summary

Orion's image-making action (`render_scene`, the visual reverie) is decided by the
real autonomy pipeline -- baseline scheduler + attention-ranked proposals -> policy ->
allocator -> execution-dispatch -> cortex-exec -- but it *executes* as one blocking
HTTP call into `orion-thought` (`POST /visual-chain/run-once`). A restart mid-render
loses the run, a failure is scored as Orion failing, dispatch measures GPU cost with a
170 s RPC stopwatch, and the diffusion GPU is arbitrated by the old durable `/capacity`
broker instead of the GPU pool.

This moves execution onto an admitted `orion-durable-runs` LangGraph workflow,
`reverie.visual`, with a checkpoint after every stage so a crash never re-renders a
finished image, retries that never count as failure, and dispatch settling the real
outcome from the run's terminal event instead of blocking.

## Current architecture (verified 2026-09-28)

- Decision: `orion/reverie/baseline.py::schedule` (one image per 5400 s, enabled in
  `config/proposals/visual_baseline.v1.yaml`) + `render_scene` template in the proposal
  arena -> policy -> allocator (baseline gets first claim on allowance) -> dispatch.
- Execution: `RenderSceneVerb` (`services/orion-cortex-exec/app/verb_adapters.py`)
  -> thought `/visual-chain/run-once` -> `_run_visual_chain_body`:
  context reads -> continuity reset -> slot select -> **interpret** (metacog LLM,
  *before* generation) -> prompt -> thermal gate -> `/capacity` permit -> diffusion
  -> disk + percept upload -> caption (vision-host) -> chain row -> production receipt.
  Everything between stages is in memory.
- Idempotency: `reverie_visual_attempt` keyed by `dispatch_id`; `chain_id == attempt_id`.
- Outcome: dispatch reads the verb's synchronous `outcome`, `latency_ms` = RPC time.
  `deferred_resource` is missing from dispatch's outcome allowlist (recorded `unknown`).
- Durable runs: admitted graphs take one GPU pool hold (`resource_request ->
  resource_wait -> work`). Only `run.completed` reaches `orion:durable:run:state`;
  admitted failed/cancelled runs are never published there.

## Design

### Graph: `reverie.visual`

```text
prepare -> resource_request -> resource_wait -> generate -> caption -> finish
   ^            ^                                  |            |
   |            +------ retry_wait <---------------+------------+
   +-- (prepare failure) retry_wait
```

- **prepare** (no hold): thought claims the attempt for the dispatch, reads context,
  resolves continuity, selects the slot, runs interpret, builds the prompt and freezes
  it on the attempt row (`stage_json.stage = "prepared"`). Replaying returns the frozen
  plan -- rotation never advances twice, interpretation never differs.
- **generate** (under a pool hold on the `diffusion` class): thought validates the hold
  ref, runs the thermal gate, the `/capacity` permit (world-model still shares gpu2
  through it until GPU pool stage 5), diffusion, and the content-addressed disk write,
  then records `stage = "generated"` + sha/path/mime/size. Replaying with an artifact
  already recorded and on disk returns it with no GPU call. `stage = "generating"` with
  no sha blocks a second diffusion call (an abandoned thread may still be running).
- The hold is released right after generate. Interpret and caption must never carry
  the diffusion hold (the pool attaches child calls to the hold's own role).
- **caption** (no hold): reload bytes by sha, percept upload, vision caption, chain
  row, production receipt, execution receipt, `finish_visual_attempt`. Every write here
  is already idempotent.
- Deferrals (thermal refused, capacity busy, resource deferred, generation failure,
  transport error) are **retries**, not failures: release the hold, back off
  (`retry_base_sec * 2^n`, capped), resume from the last recorded stage. Retries never
  spend the run's attempt budget.
- The only terminal failure is the run deadline: `admission.deadline_at` = submit +
  the baseline interval. After it the run ends `failed` with `last_error =
  retry_window_expired`; the next scheduled need takes over.

### Routing the diffusion hold

`config/gpu_pool.yaml` gains `hold_routes:` (read by durable-runs `hold_placement`
only, never by the gateway) with `diffusion: {class: diffusion, priority: background}`
(holds always take `ResourceRequirementV1.priority`, which is `background`; the diffusion
class has one role and one slot, so priority only orders holds on that seat).
`ResourceRequirementV1` accepts `resource = "service.route.<lane>"` in addition to
`llm.route.<lane>`. The gateway's LLM route catalog is unchanged.

### Kickoff (async accept)

`RenderSceneVerb` submits `DurableRunRequestV1(workflow="reverie.visual",
run_id="reverie-visual-" + uuid5(dispatch_id), brief=ReverieVisualRunBriefV1(...))`
through the existing cortex-orch durable ingress and returns as soon as the receipt
confirms acceptance: `outcome="unknown"` plus `settlement={state: "pending",
durable_run_id, submitted_at}`. The `VisualRunOutcome` type stays closed -- no
`accepted` value.

### Settlement (execution-dispatch)

- durable-runs publishes **every** admitted terminal state (completed, failed,
  cancelled) on `orion:durable:run:state`; sql-writer already stores them in
  `substrate_durable_run_state`. (This also fixes reflect's waiter hanging on
  failures.)
- The dispatch worker's tick reconciles pending render results: join
  `substrate_durable_run_state` on `run_id`, terminal only, and upsert the **same**
  `result:{dispatch_id}` row with the real `visual_outcome`, execution receipt,
  `settlement.state = "settled"`, and `latency_ms` = the run's own measured GPU seconds
  (`detail.visual_elapsed_sec`). Then re-emit `ActionOutcomeEmitV1` with the same
  `action_id` so `action_outcomes` / chat stance see what really happened.
- **Failures never count against Orion:** a run that ends without an image settles
  `visual_outcome = "unknown"` with `settlement.reason`, `success = False` but no
  failure feedback kind, and `latency_ms` stays NULL (no motor spend, no cost sample).
- Orphans (pending past deadline + margin with no terminal row) settle `unknown`,
  reason `settlement_timeout`.

### Feedback robustness

feedback-runtime defers a frame while any of its render results is `pending`, bounded
by `FEEDBACK_VISUAL_SETTLE_MAX_SEC`; past the bound it scores with the visual
observation `unknown` (never blocks the queue indefinitely).

### Also fixed

- `deferred_resource` added to dispatch's outcome allowlist.
- Deferral chain rows carry `continuity_streak`, `context_slot_rotation` and
  `prior_description`, so a thermal/resource deferral no longer resets continuity.

## Proposal-mode fields

- **Capability change:** image runs survive restarts and resume from the last stage;
  failures retry instead of scoring as Orion failing; GPU cost measured honestly.
- **Data touched:** `reverie_visual_attempt` (+`stage_json`), `reverie_visual_chain`
  (deferral rows gain continuity keys), `substrate_dispatch_results` (settlement
  upsert), `action_outcomes` (re-emit), `substrate_durable_run_state` (new workflow).
- **Privacy boundary:** unchanged. The brief carries only dispatch ids; context text
  (incl. memory crystallizations) stays inside thought and its own tables.
- **Proof it worked:** a `reverie.visual` run in `substrate_durable_run_state` with
  `completed` + `detail.chain_id`; the matching chain row with a production receipt;
  the dispatch result settled with real `latency_ms`; a pool `granted`/`released` pair
  for holder `durable-runs:reverie-visual-*`.
- **Dangerous failure mode:** a run that loops forever holding gpu2. Bounded by the
  run deadline, the pool's `max_hold_sec`/TTL, and releasing the hold after generate.
- **Disable / roll back:** `CORTEX_EXEC_RENDER_SCENE_DURABLE_ENABLED=false` returns
  `RenderSceneVerb` to the direct `/visual-chain/run-once` call (kept intact). The legacy
  claim releases attempts a dead durable run left behind (abandoned-in-flight window and
  `ORION_VISUAL_CHAIN_ATTEMPT_MAX_AGE_SEC`), so rollback is never blocked by leftovers.
- **Stranded-attempt bounds:** abandon by `dispatch_id` (no `attempt_id` needed), retried
  from the durable-runs outbox until acked (given up after 3 h); thought's claim expires
  `active`/`unknown` attempts older than `ORION_VISUAL_CHAIN_ATTEMPT_MAX_AGE_SEC` (7200 s),
  and cortex-exec clamps the run window to `REVERIE_VISUAL_MAX_RETRY_WINDOW_SEC` (6600 s)
  so no live run outlives that.

## Deploy order

1. `services/orion-sql-db/manual_migration_reverie_visual_attempt_stage.sql` (after
   `manual_migration_reverie_visual_attempt.sql`); optional
   `manual_migration_durable_resource_abandon_pending_v1.sql`.
2. orion-sql-writer, then orion-gpu-pool / LLM gateway images carrying the new
   `config/gpu_pool.yaml` `hold_routes`, then orion-durable-runs.
3. orion-thought (step channel consumer).
4. orion-execution-dispatch-runtime, orion-feedback-runtime.
5. orion-hub, orion-cortex-orch (widened `DurableWorkflowV1`), then orion-cortex-exec last
   (it starts submitting runs).

## Non-goals

- Moving diffusion off `/capacity` permits (GPU pool stage 5).
- Changing what decides whether an image happens (baseline, proposals, allocator).
- Reordering interpret after caption.

## Acceptance checks

- Graph test: crash after generate resumes at caption with no second diffusion call.
- Graph test: thermal/resource deferral retries without spending attempts; deadline
  ends the run `failed` / `retry_window_expired`.
- Thought tests: prepare replay returns the frozen plan; generate replay with a
  recorded artifact makes no GPU call; hold-ref mismatch refuses.
- Dispatch test: a terminal state replayed through the reconcile step settles the row,
  re-emits the outcome, and sets `latency_ms` from `visual_elapsed_sec`; a failed run
  leaves `latency_ms` NULL.
- Feedback test: pending render defers, bound expiry scores `unknown`.
- Schema/channel/registry gates green; env synced.
