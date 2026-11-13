# orion-durable-runs

Durable, resumable cognition runs, kicked off from cortex. Step 3 of the
attention-schema-surface ordering; design and evidence in
`docs/superpowers/specs/2026-09-06-durable-cognition-runs-from-cortex-design.md`.

**What it does, in one sentence:** a curiosity investigation (Orion's longest act
of cognition, 10-40 minutes) used to live entirely in Hub's process memory and died
with every Hub redeploy; now its state machine lives here under a Postgres
checkpointer, a restart of Hub or of this service resumes the run at its next
node with the same `run_id`, and cortex sees every kickoff.

## Flow

```
Hub tick (scheduling, material, worldview, prompt)
  -> orion:cortex:request  (context.metadata.durable_run = DurableRunRequestV1)
  -> orion-cortex-orch dispatches -> orion:durable:run:request   (this service, single consumer)
  -> graph: harness_turn -> read_turn_result -> publish_attention_row -> journal -> finish
       harness_turn = RPC back to Hub on orion:curiosity:turn:request (only Hub can run the saga)
  -> every node transition: orion:durable:run:state -> sql-writer (substrate_durable_run_state)
                                                    -> Hub (completed + reach_out -> outreach)
                            + one attention-surface row (process=durable_run)
```

## Resume semantics

- A thread is unfinished when the compiled graph's state snapshot still has a
  `next` node. On boot and every `DURABLE_RUNS_RESUME_SWEEP_SEC` the runner re-invokes
  every such thread with `None` input; LangGraph continues from the next node with
  every earlier node's result intact.
- `harness_turn` is the one node whose *work* is not resumable: Hub runs an FCC
  subprocess. A restart mid-turn re-issues the turn (same `run_id`, `attempt+1`).
  Resume saves the daily-cap slot, the continuation, and everything after the turn --
  not the FCC minutes.
- A thread older than `DURABLE_RUNS_MAX_AGE_HOURS` is abandoned (one `abandoned` state
  event), never resumed into a different day's material.

### Checkpoint pruning (2026-11-12)

The Postgres saver keeps every checkpoint, pending write and channel version forever, and every
saver call (including the resume sweep's `alist(None)` every `DURABLE_RUNS_RESUME_SWEEP_SEC`)
runs behind ONE lock per saver. Measured 2026-11-12: the sweep's listing read 53.8k checkpoints in
16 s warm (~4M buffer hits), 9 s of it the three `temporal_self` day threads, whose blob join
hashes every blob the thread ever wrote (5,321 rows to return 10). While it runs, temporal_self's
own checkpoint read waits on that lock.

`app/checkpoint_prune.py` keeps, per `(thread_id, checkpoint_ns)`: the newest checkpoint, its
pending writes, and the blob each channel's newest version points at. It deletes older checkpoints,
their writes, and superseded channel versions. Nothing at or above the newest is touched, so a
writer racing it is safe. Durable-runs never reads history (no `aget_state_history`, no
`checkpoint_id` in a config; the sweep keeps only the newest per thread).

- In-service: `DURABLE_RUNS_CHECKPOINT_PRUNE_MODE` = `off` | `dry_run` (shipped) | `apply`, every
  `DURABLE_RUNS_CHECKPOINT_PRUNE_INTERVAL_SEC` (3600), first pass 5 min after boot, on its own
  connection (never the saver's lock). Log line `checkpoint_prune mode=... ckpt_del=...`;
  `/health` -> `checkpoint_prune`. Skips a pass while a `pg_dump` runs.
- By hand: `docker exec orion-athena-durable-runs python /app/scripts/prune_checkpoints.py`
  (dry run; `--apply` deletes; `--thread-prefix temporal_self:` limits it; `--json`).
- Only the `public` checkpoint tables (durable-runs). The GPU pool's `gpu_pool` schema is not covered.
- It never deletes whole threads: a finished run keeps one checkpoint, so the sweep still grows with
  the number of threads (follow-up: drop terminal threads older than `DURABLE_RUNS_MAX_AGE_HOURS`).
- Repo-wide readers of older history: `orion/memory/referents/backfill.py` (now falls back to the
  newest checkpoint's `answer_text`). A `DeltaChannel` (`add_messages`) in any durable-runs graph
  would be truncated; `tests/test_checkpoint_prune.py` fails if one appears.

### One failed-attempt policy for admitted work nodes (2026-10-23)

`app/admitted_graph.py:failed_attempt` decides what an exception from a work node does, so a
crash or timeout resumes from the checkpoint instead of ending the run. Used by
`curiosity.investigate`, `self_sense_eval`, `self_study.reflect` and `curiosity.compaction`.

- **Transient** (any other exception: the node timer's `TimeoutError`, a failed turn's
  `RuntimeError`, a transport error that raises): the hold is handed back (a re-queued hold
  keeps its place), `attempt` +1, and the run waits `DURABLE_RUNS_RETRY_BASE_SEC * 2^(n-1)`
  (capped at `DURABLE_RUNS_RETRY_MAX_SEC`) in `retry_wait`, or re-queues at once in graphs
  without that node (reflect, compaction: `backoff=False`). Note: the runner's turn RPC turns
  a bus timeout or bad reply into an `ok=False` result, not an exception, so for self-sense
  that is a recorded empty answer (the per-question contract), not a retry.
- **Deterministic** (`ValueError` incl. pydantic `ValidationError`, `TypeError`,
  `LookupError`, `AttributeError`, ...; `is_deterministic`, by type, never by error text):
  fails at once with `last_error = "deterministic: ..."`: a code or input bug a retry cannot fix.
- **Rejected model answer** (`orion.cognition.model_output.ModelOutputRejected`, a `ValueError`
  subclass raised only where an LLM answer is parsed/validated, e.g. `record_answer` in
  `orion/curiosity/compaction.py`): treated as **transient** -- the next sample differs. Failing
  these fast (#2899) stopped curiosity.compaction: 0 batches saved 2026-10-24..27 over 9 runs.
- **Budget spent** (`DURABLE_RUNS_RETRY_MAX_ATTEMPTS`; urgent runs 2) or a retry that cannot
  get a whole attempt before the deadline: `failed`.

Not on this policy, on purpose: `reading.turn` (Hub's seed queue owns bounded retries; a
second loop would nest), `journal.compose` and `memory.episode_distill` (an unparseable
draft does recover on a fresh sample: their own minimum-attempt retry), `compactor.digest`
(falls back to the deterministic merge), `orion_day.letter` (per-node attempts),
`reverie.visual` / `dream.carry` (deferrals that never spend an attempt). Each already
retries with backoff.

### A step's RPC published to nobody waits out the deploy (2026-11-03)

Every RPC a work step sends asks the bus for a live listener (`require_listener=True`). When the
answering service is restarting, the bus raises `RpcPublishedToNobody` (a subclass of
`RpcNoListener`): PUBSUB NUMSUB read 0 workers in the same transaction as the PUBLISH, so nobody
received the request. Live 2026-10-31: 16 reading runs failed this way in four bursts, each a
redeploy.

`app/admitted_graph.py:listener_wait` handles it before any graph's normal failure path: the
hold is handed back, the run waits `DURABLE_RUNS_LISTENER_WAIT_BASE_SEC * 2^(n-1)` (capped at
`DURABLE_RUNS_LISTENER_WAIT_MAX_SEC`; defaults 60 s doubling to 300 s), then takes a fresh hold
and replays the step. **No attempt is spent.** `listener_waits` counts these per run; past
`DURABLE_RUNS_LISTENER_WAIT_MAX_TRIES` (default 6, ~22 min in total; 0 = off), or when the wait
would leave no whole attempt before the deadline, the error takes the graph's normal path as
before. Urgent runs (`brief.urgent`) wait within the urgent budget instead: at most 2 waits, 10 s
then 20 s. Each wait is a `run.listener_wait` event (`waits`, `case`, `retry_at`, `error`; `case` is
`published_to_nobody` or `lost_in_flight`). Graphs that go
straight back to `resource_request` (reading, compactor digest, curiosity compaction) are parked
by `hold = {"retry_at", "listener_wait"}`, which `AdmissionRuntime.register` waits out; graphs
with a `retry_wait` node also get `retry_at` on the state.

Wired into: `curiosity.investigate`, `self_sense_eval`, `harness.echo_smoke`, `pr.field_note`,
`reminiscence.episode`,
`curiosity.compaction` (via `failed_attempt`), `reading.turn`, `compactor.digest`,
`journal.compose`, `memory.episode_distill`, `orion_day.letter`. `reverie.visual` and
`dream.carry` already treat any transport error as a backed-off deferral that spends no attempt.

**The in-flight case is replayed only for `reading.turn`.** `RpcNoListener` "while awaiting its
reply" means the request WAS delivered and its listener went away mid-step: it may have done work
(a curiosity turn has tools), and a request that itself crashes its worker would loop. Every other
workflow keeps the normal attempt path for it. `reading.turn` opts in with
`app/reading_graph.py:IN_FLIGHT_RESENDS` (2): the Hub reading turn is read-only (`no_write`, no
chat history; the motor gets WebFetch/WebSearch only) and nothing lands until Hub consumes the
run's result, so a duplicate costs compute plus its own trace rows under a fresh turn correlation
id. Same backoff, urgent and deadline rules; its own counter `listener_in_flight_resends`; past 2
the run fails as before. `DURABLE_RUNS_LISTENER_WAIT_MAX_TRIES=0` turns this off too. Live
2026-11-03: 3 readings failed this way during Hub redeploys at 18:12 and 18:26 UTC.

Eval: `evals/deploy_window_listener_wait.py` (16 readings unheard for a deploy window: all 16 fail
with the wait off, all 16 complete with it on, no attempt spent; plus a Hub that vanishes
mid-turn: those readings are re-sent and complete, at most 2 re-sends each). Mutation eval:
`evals/listener_wait_mutations.py`.

## Workflow registry (2026-09-21)

### Admitted reading turns (2026-09-26)

`reading.turn` exists only in the admission registry, not the legacy unleased
runner. Its typed brief names a seed, stage, immutable prompt, session and turn
budget. Resource waiting uses the same checkpoint/hold/heartbeat/recovery machinery
as the existing admitted graphs. The work node calls Hub over
`orion:reading:turn:request`, with a required hold, and checkpoints its result and
source-fetch receipts. Hub owns parsing, evidence validation and journal landing.
There is no second model retry loop: Hub's seed queue retains bounded retries.
Hold recall/loss requeues without spending a reading attempt.

The status API includes `reading_result`, `error` and `work_started` for this
workflow. The last field comes from the existing persisted `run.started` event,
not config or a guessed elapsed-time threshold. Failed/completed events carry
the actual hold-fenced turn correlation. Queue cancellation is not completion.

### Memory episode distiller (`memory.episode_distill`, 2026-10-02, SHADOW)

One closed conversation episode distilled into Orion-voiced memories (`app/episode_distill_graph.py`,
contracts `orion/schemas/memory_episode.py`, spec `docs/superpowers/specs/2026-09-30-memory-episode-redesign-design.md`):
`load_episode -> resource_request -> resource_wait -> distill -> persist -> finish`.

- **Trigger:** this service subscribes `orion:memory:episode:closed` (orion-memory-consolidation's
  Rule 3 shadow tracker) and submits the run itself through admission. `run_id = memdistill-<episode_id>`,
  so a re-delivered event is a duplicate the store refuses. Command-only and non-direct episodes get no run.
- **Hold:** route `memory_distill` (`config/gpu_pool.yaml`: its own class `memory_distill`, roles `[agent]`,
  i.e. the 27B on gpu1 only -- never the gpu2 seat or the lent chat card) at `priority: system`. System runs have their own driver slot (`MAX_CONCURRENT_SYSTEM_DRIVERS=1`), so a
  distill run is not stuck behind four background turns, and the pool ranks it above background holds.
- **distill:** ONE direct LLM gateway call (`orion:exec:request:LLMGatewayService`) with `options.gpu_lease`;
  not through cortex-orch (its recall step would mix retrieved memories into the evidence). JSON-object output,
  thinking off. An unparseable answer is a bounded, backed-off attempt.
- **persist:** releases the hold, validates deterministically (`orion/memory/episode/validate.py`: every quote
  checked against the turn's FULL text; voice downgraded when its evidence does not support it; internal channels
  never labelled as Juniper's words), then writes `episode_memory*`, `memory_tension_shadow` and
  `episode_distill_run` in one idempotent transaction. Rejected candidates go only to `episode_memory_event`.
- **Reconciler** (`app/episode_distill_reconcile.py`, every `MEMORY_EPISODE_RECONCILE_INTERVAL_SEC`): closed
  episodes with no `episode_distill_run` are resubmitted as ordinary durable runs -- the base id for a lost close
  event, `memdistill-<episode>-a<N>` for a failed one older than 1 h, at most `MEMORY_EPISODE_DISTILL_MAX_ATTEMPTS`.
- **Nothing live reads these tables.** The daily old-vs-new report (orion-memory-consolidation) does.
- **Env:** `MEMORY_EPISODE_WRITER_ENABLED` (kill switch), `CHANNEL_MEMORY_EPISODE_CLOSED`,
  `MEMORY_EPISODE_DISTILL_ROUTE`, `MEMORY_EPISODE_DISTILL_TIMEOUT_SEC` (<= 900), `MEMORY_EPISODE_DISTILL_MAX_TOKENS` (8192; an answer cut
  there fails as `distill_truncated_at_max_tokens`, not as invalid JSON),
  `MEMORY_EPISODE_DISTILL_DEADLINE_HOURS` (< 24), `CHANNEL_LLM_INTAKE`.
- **Deploy order:** apply `manual_migration_episode_memory_v1.sql`; deploy orion-sql-writer (parses
  `DurableRunStateV1` with the new workflow), orion-llm-gateway and orion-gpu-pool (they read the new
  `memory_distill` route from `config/gpu_pool.yaml`), then this service. orion-memory-consolidation (the
  producer of the close event) can go any time after.

#### Referents after each distill (memory Stage 2, 2026-10-06)

After the memories are written, the same transaction (in a savepoint, so a fault never loses
them) resolves each memory's referent keys to graph nodes and keeps Juniper's own names for them
(`orion/memory/referents`). Kill switches, all shipped on:

| Concept | What it means in plain English | Producer | Consumer | Test |
|---|---|---|---|---|
| `MEMORY_REFERENTS_ENABLED` | Run the referent step at all. | settings | `AdmissionRuntime._referent_policy` | `tests/test_episode_distill_referent_policy.py` |
| `MEMORY_ALIAS_GROUNDING_AUTO_ACCEPT` (`alias_grounding_v1`) | A name Juniper actually said is usable at once. Off: every name waits for review. | settings | `orion/memory/referents/resolve.py` | same + `orion/memory/referents/tests` |
| `MEMORY_COOCCURRENCE_AUTO_ACCEPT` (`source_cooccurrence_v1`) | Things named together in one of Juniper's sentences are linked at once. Off: the links are only proposed. | settings | `orion/memory/referents/cooccurrence.py` | same |

### Admitted journal compose (`journal.compose`, 2026-09-30)

One journal entry, composed under a GPU pool hold (`app/journal_compose_graph.py`, contract
`orion/schemas/journal_compose_run.py`): `resource_request -> resource_wait -> compose -> publish -> finish`.
Admission-only (the request validator refuses it without `admission`). `compose` sends the
`journal.compose` cortex verb (`orion.journaler.build_compose_request`) with `options.gpu_lease`, so
the gateway attaches the call to the hold; `llm_route` stays the brief's route (the hold's role is
never a route). Waiting for the hold is never an attempt -- a busy pool at 06:00 local is a
checkpointed wait bounded by `admission.deadline_at`. A failed compose (non-ok, empty or unparseable
draft, transport) is an attempt: the hold is handed back and the run sleeps in `retry_wait`
(`DURABLE_RUNS_RETRY_BASE_SEC` * 2^n, capped at `DURABLE_RUNS_RETRY_MAX_SEC`) before asking again,
at least `JOURNAL_COMPOSE_MIN_ATTEMPTS` (6) times. `publish` releases the hold, then publishes
`journal.entry.write.v1` with the brief's fixed `entry_id` and the checkpointed `created_at`; a
publish failure raises and the driver's bounded checkpoint resume retries publish without
recomposing. sql-writer's journal table is insert-only, so a replayed write with the same entry_id
is dropped and `journal.created` is not re-emitted (no second email). The brief carries the
curiosity section as pre-rendered text (`body_appendix` + `body_appendix_markers`), never another
service's schema. First producer: orion-actions'
world-pulse journal (`trigger_kind=world_pulse_digest`, run_id `world-pulse-journal:<world-pulse run_id>`,
deadline = next local midnight). Finish detail: `line=journal`, `entry_id`, `trigger_kind`,
`published`, `attempts`.

Deploy order (additive `Literal`/brief on `extra="forbid"` models): orion-durable-runs, then
orion-cortex-orch (it validates `DurableRunRequestV1`) and orion-sql-writer (it validates
`DurableRunStateV1` rows), then the producer.

### Admitted visual reverie (`reverie.visual`, 2026-09-28)

One image per run, checkpointed after every stage
(`app/reverie_visual_graph.py`; spec
`docs/superpowers/specs/2026-09-28-visual-reverie-durable-graph-design.md`).
orion-thought does every stage's work over `orion:reverie:visual:step:request`
(reply `orion:reverie:visual:step:reply:<correlation>`, one correlation per RPC,
derived from run id + step + call count + restart fence so replies never cross):

```text
prepare -> resource_request -> resource_wait -> generate -> caption -> finish
   ^            ^                                  |            |
   |            +------ retry_wait <---------------+------------+
   +-- retry_wait                                     (any node) -> failed
```

- **prepare** (no hold): thought claims the attempt, runs interpret, freezes the
  prompt. `done` saves `attempt_id`; `terminal` (e.g. `already_satisfied`) goes
  straight to `finish` without ever asking the pool for a hold.
- **generate** runs under the run's pool hold: `resource = service.route.diffusion`
  is placed by `config/gpu_pool.yaml` `hold_routes.diffusion` (class `diffusion`).
  The hold is **released as soon as generate answers**, so caption (and interpret,
  inside prepare) never carry it.
- **caption** (no hold). `retry` with reason `needs_generate` (the image is gone from
  disk) goes back through the hold to generate, after a backoff of
  `DURABLE_RUNS_RETRY_BASE_SEC * 2^(n-1)` (capped at `DURABLE_RUNS_RETRY_MAX_SEC`) on the
  run's regeneration count `n` (state `regenerations`), so a storage fault that keeps
  losing the image cannot re-render back to back. Still bounded only by the deadline.
- **Retries never spend the run's attempt budget.** A step `retry`, an RPC timeout or
  a transport error releases any hold and waits `retry_after_sec` (thought's hint) or
  `DURABLE_RUNS_RETRY_BASE_SEC * 2^n` capped at `DURABLE_RUNS_RETRY_MAX_SEC` (never
  below 1 s), then resumes at the same stage. A recalled/lost hold re-queues. A generate
  or caption `retry` whose reason says thought has no prepared attempt (`not_prepared`,
  `plan_not_frozen`, `attempt_missing`) releases any hold and resumes at **prepare**
  instead, since retrying the same stage could never fix it.
- **The run deadline is the only failure bound**: `admission.deadline_at` (submit +
  baseline interval; a resubmit of the same run with a different `deadline_at` is the
  same run -- the first submission's deadline wins). Whichever node, wait or driver
  notices it ends the run `failed` with `last_error = retry_window_expired`.
- **Abandon is reliable.** Whenever a run ends without `completed` (deadline, operator
  cancel, resume-failure bound, any graph failure), the driver records
  `run.abandon_pending` in the run's event log *before* the terminal projection, then
  sends thought `step = abandon` in the background (30 s bound) *after* it -- the
  terminal never waits on thought. `attempt_id` is sent when known; otherwise thought
  resolves the attempt by `visual_request.dispatch_id` (a lost prepare reply cannot
  strand a claimed attempt). Only `done`, or `terminal` with reason `attempt_mismatch`
  (nothing of ours to close), counts: it is recorded as `run.abandon_acked` and clears
  the pending record. A timeout, transport error, `retry`, or any other `terminal`
  (e.g. an older thought rejecting the request) stays pending and the reconcile loop
  re-sends it after `DURABLE_RUNS_RETRY_BASE_SEC * 2^(n-1)` (capped at
  `DURABLE_RUNS_RETRY_MAX_SEC`). Pending records live in `durable_resource_events` (no
  new table; optional index `manual_migration_durable_resource_abandon_pending_v1.sql`),
  are re-read newest-first every `DURABLE_RUNS_HOLD_STATUS_POLL_SEC`, and so survive a
  durable-runs restart. After `ABANDON_GIVE_UP_SEC` (3 h, `orion/durable_runs/registry_store.py`)
  retries stop: thought's own max-age sweep has released the attempt by then. Paused or
  completed runs are never abandoned.
- Restart: `generate` is the work node. A restart mid-generate fences it and replays it
  under the same hold (thought's generate is idempotent: the recorded artifact, or a
  `generate_in_flight` retry). A restart after generate resumes at caption, with no
  hold and no second generate.
- Stage RPC wait: `DURABLE_RUNS_REVERIE_VISUAL_STEP_TIMEOUT_SEC` (generate waits
  `max(that, brief.timeout_sec)`; the held generate is also bounded by the brief's
  budget and the deadline).

`run.completed` detail: `dispatch_id`, `proposal_id`, `decision_id`, `attempt_id`,
`chain_id`, `outcome`, `reason`, `artifact_sha256`, `execution_receipt`, `retries`,
`generate_elapsed_sec`, `visual_elapsed_sec` (thought's real-work seconds summed over
done steps -- dispatch's cost; never queue or hold-wait time), `started_at`,
`finished_at`. `run.failed` / `run.cancelled` carry the same ids plus `retries`,
`error` / `last_error` and the last stage `reason`. The status API adds a
`reverie_visual` block (attempt, outcome, retries, retry_at, elapsed, deadline),
`error` and `work_started`.

A **dream image hop** is a `reverie.visual` run whose brief carries `dream_hop`
(`DreamHopImageV1`, submitted by a `dream.carry` run below). Every step request it sends
(prepare/generate/caption/abandon) carries the same `dream_hop`, and its `run.completed`
detail adds `caption` (what thought saw). A waking painting is unchanged: its step payloads
omit `dream_hop` entirely (an older orion-thought keeps accepting them during a rolling
deploy) and its detail has no `caption` key.

### Admitted dream carry-through (`dream.carry`, 2026-10-10)

Spec: `docs/superpowers/specs/2026-10-10-dream-carry-through-design.md`. Contract:
`orion/schemas/dream_carry.py`. Graph: `app/dream_carry_graph.py`.

```text
next_hop -> resource_request -> resource_wait -> text_hop -> next_hop      (even hop: text)
next_hop -> image_submit -> image_wait (polls itself) -> next_hop          (odd hop: image)
next_hop -> finish_dream -> finish                                         (all hops made)
hop stage: retry_wait -> same stage;  deadline / terminal -> finish_dream (partial)
```

- **Text hop:** one `DreamCarryStepRequestV1(step="text")` RPC to orion-dream on
  `orion:dream:carry:step:request`, under the run's own hold (admission
  `llm.route.metacog_background`, class metacog, background); the hold is released as soon as
  the hop is checkpointed. `retry` (or transport trouble) hands the hold back and backs off: never
  an attempt. `terminal` stops the carry.
- **Image hop:** a child `reverie.visual` run submitted inside durable-runs
  (`AdmissionRuntime.submit`), run id `reverie_visual_run_id(dream_hop_dispatch_id(carry, hop))`,
  painting the previous text hop's `image_prompt`, diffusion hold, deadline
  `min(carry deadline, now + DREAM_CARRY_CHILD_WINDOW_SEC)` (default 2400, never above 6600), so a
  hot cabinet cannot keep thought's single painting slot on a dream picture for hours. No child
  (first or replacement) is submitted with less than `DREAM_CARRY_CHILD_MIN_WINDOW_SEC` of the carry
  left: the carry finishes partial (`image hop N: only Ns left`). The carry holds nothing:
  `image_wait` re-reads the child's terminal fact (`DurableRunRegistryStore.terminal_detail`) every
  30 s (one checkpoint per poll: LangGraph has no checkpoint-free wait in this runtime).
  - A child that ends without a picture for a **retryable** reason (failed with
    `retry_window_expired` / `workflow_deadline`, or completed `deferred_thermal` /
    `deferred_busy` / `deferred_resource` / `unknown`) is replaced, after
    `max(admission backoff, DREAM_CARRY_CHILD_RETRY_GAP_SEC)` (default 900: thought's 600 s cooldown
    plus a baseline tick, so a waking painting can claim the slot in between), by a fresh child for the same hop with dispatch
    `dream-carry:<run>:<hop>:r<n>`, at most `DREAM_CARRY_CHILD_MAX_ATTEMPTS` (3) children per hop and
    only while `DREAM_CARRY_CHILD_MIN_WINDOW_SEC` (900) of the carry is left. Live reason: most
    waking paintings end `retry_window_expired` after `thermal_refused`.
  - Anything else (cancelled, a terminal failure, a non-deferral outcome), or attempts / window
    exhausted, stops the carry (`image hop N: <why>`, e.g. `image hop 3: thermal_refused x3`).
  - Every child id is kept in `child_run_ids`; a carry that ends (any terminal, completed-partial
    included) cancels the current attempt's child if it is still running.
- **Finish:** `step="finish"` with every hop made and `stopped_reason`; orion-dream writes the
  dream. It keeps retrying for `DREAM_CARRY_FINISH_GRACE_SEC` (default 1800) past the deadline so
  a partial carry is still written; after that the run fails with the last finish error.
- **Deadline** (`admission.deadline_at`, required: `AdmissionRuntime.submit` refuses a carry
  without one, since text/finish retries never spend attempts): any hop stage past it finishes **partial** with
  `stopped_reason="deadline at hop N: <last reason>"`. A carry that made no hop at all still calls
  finish with `hops=[]`: orion-dream answers done with `dream_id="story-fallback:<trigger_id>"` (the
  sleep's one-paragraph story) or terminal for a hand-started carry (run failed). The driver does not fail a pending carry at its deadline (the graph
  owns it); it only backstops it after deadline + grace + `DURABLE_RUNS_RETRY_MAX_SEC`.
- Hops in the checkpoint are never redone (the next hop index is `len(hops)`); a replayed image
  submit dedupes on the deterministic child run id.

`run.completed` detail: `trigger_id`, `hops_planned`, `hops_made`, `stopped_reason`, `dream_id`,
`child_run_ids`, `retries`. `run.failed` / `run.cancelled` add `error` / `last_error` and the
in-flight `child_run_id`. The status API adds a `dream_carry` block.

### Admitted compactor digest (`compactor.digest`, 2026-09-30)

The LLM half of cortex-orch's daily `github_compactor_pass` / `chat_history_compactor_pass`
(`app/compactor_digest_graph.py`; contract `orion/schemas/compactor_digest_run.py`). Before this
the compactors made several digest calls in-process inside one synchronous workflow RPC, each on a
one-inference gateway lease; at 06:00 the pool was busy and those failed with
`gpu_pool_unavailable:deadline`, papered over by an in-process retry and a 3600 s scheduler wait.

    resource_request -> resource_wait -> digest (loops) -> finalize -> finish

- cortex-orch fetches the day (GitHub PRs / chat turns), builds the chunk inputs, and submits the
  run on `llm.route.agent` background admission with `deadline_at` = window end + 24 h. The
  `run_id` is `compactor:<workflow>:<window>[:<repo>]:<input sha256[:12]>` (plus `:g<N>` for a
  re-submission after a failed/cancelled run), so a re-dispatch of the same window finds the
  existing run instead of starting a second one.
- `digest` makes ONE call per node run under `AdmissionRuntime.execute` -- a chunk digest, or the
  merge -- with `options.gpu_lease` so the gateway attaches it to the hold, and checkpoints the
  result before the next call. Which call is next is the pure step machine in
  `orion/cognition/compactor/map_reduce.py`. A restart resumes at the first undigested chunk.
  Waiting for the hold is never an attempt; a failed call (verb failure, empty / rejected / invalid
  JSON) is one of `DURABLE_RUNS_RETRY_MAX_ATTEMPTS` for that call (reset after each success).
  A chunk that exhausts them fails the run; a merge that exhausts them, drops a chunk's refs, or
  whose input is over `DIGEST_INPUT_CHAR_BUDGET` falls back to the deterministic join
  (`merge_mode=concatenated`, `merge_skipped_reason` says why).
- `finalize` releases the hold (no GPU needed), assembles the digest (card prose fitted,
  `journal_body` untrimmed) and sends `CompactorDigestResultV1` back to cortex-orch as
  `workflow_request.durable_digest`; orch writes the memory card + journal entry (stable ids, so a
  replay upserts) and notifies per the schedule's policy. A failed finalize raises and is retried by
  the driver's bounded checkpoint-resume without re-running any LLM call.
- `WORK_NODES["compactor.digest"] = {"digest"}`. The terminal state row settles the orion-actions
  schedule run that submitted it.

### Harness steps: `app/harness_step.py` + smoke `harness.echo_smoke` (2026-10-29)

A durable workflow can run one step as one harness turn (the same `claude -p` motor chat and
curiosity use) and checkpoint the turn's final text as the step's result. This is the seam the
planned `reminiscence.episode` hops sit on.

- **Transport: nothing new.** The step sends the existing `CuriosityTurnRequestV1` on
  `orion:curiosity:turn:request` through `Runner._run_turn`, the RPC curiosity.investigate and
  self_sense_eval already use. Hub runs it with `execute_unified_turn` (`no_write`) under the run's
  GPU pool hold; Hub refuses a stale hold before the turn starts, and the governor dispatches it
  exactly as it dispatches a self_sense_eval question (a held turn skips Hub's own `_run_lock`). Explicit `session_id` and `source_tag`, so a step never
  borrows curiosity's investigation session.
- **Result:** `run_harness_step` returns `{step_key, text, final_len, turn_correlation_id, debug
  (bounded), harness_elapsed_sec, harness_started_at, harness_finished_at}`; the graph stores it at
  `state["steps"][step_key]`. `turn_correlation_id` is the harness turn's correlation id: the turn's
  stages are in `cockpit_turn_sighting` under it.
- **Failure:** `ok=False` (refusal, Hub timeout, `no_lookup`, `error_shaped_text`), empty text, or a
  reply for another turn raises `HarnessStepFailed` (a `RuntimeError`, so retryable). The graph sends
  it through the shared `failed_attempt` policy: backoff in `retry_wait`, a fresh hold, a fresh turn
  id (`harness_step_correlation_id`: uuid5 over run, step, attempt, fence, hold seq, take-backs).
  Past `DURABLE_RUNS_RETRY_MAX_ATTEMPTS` the run fails with the turn's error and names the turn
  (`detail.turn_correlation_id`). A step never completes with `final_len=0`.
- **Resume:** a checkpointed step is never re-run. A worker that dies mid-step: recovery cancels the
  pending step's turn by its checkpoint-derived id, then replays under the same hold.
- **Hub's lookup gate applies (hard constraint):** Hub's `_generate` requires at least
  `MIN_HARNESS_STEPS` (3) harness stream steps by default, and `_turn_result_for` does not override
  it. A step that uses no tool comes back `no_lookup` (a step failure) on every attempt. The smoke's
  prompt runs one Bash command for that reason. A pure writing step needs an optional
  `require_lookup` on `CuriosityTurnRequestV1` first (not built).
- **Session:** Hub resumes one claude session per `(run_id, source_tag)`. A retry continues the
  failed attempt's session; steps that need separate sessions need distinct `source_tag`s.

**Smoke `harness.echo_smoke`** (`app/harness_smoke_graph.py`, contract
`orion/schemas/harness_step_run.py`): `resource_request -> resource_wait -> harness_step -> finish`,
one step `echo`, agent lane, nothing persisted beyond the checkpoint, the admission row and its state
rows. Hand-triggered only:

```bash
python3 services/orion-durable-runs/scripts/harness_echo_smoke.py
```

It POSTs the run to `http://localhost:8124/runs`, polls `GET /runs/<id>` (which shows
`harness_steps`, `error`, `attempt`, `retry_at`, `harness_turn_meta` for this workflow), prints PASS
only when the run completed and the step text carries `harness echo smoke ok`, and prints the
`cockpit_turn_sighting` query for the turn. `--print` shows the request without submitting it.
Exit codes: 0 PASS, 1 FAIL, 2 submit refused or unreachable, 3 still running when polling gave up.
Brief `timeout_sec` must be 60..3600 s. Finish detail: `line=harness_smoke`, `turn_correlation_id`, `steps.<key>` (text capped at 2000
chars, `text_truncated`).

**Deploy order:** `harness.echo_smoke` is an additive `DurableWorkflowV1` value on
`extra="forbid"` models. Deploy orion-sql-writer, orion-hub, orion-actions and orion-cortex-orch
before triggering the first smoke, or an old validator drops its state rows.

### PR field notes: GitHub poll + `pr.field_note` (2026-11-04, spec Patch 2)

After every merged PR, Orion writes a short field note on what changed in their body (spec
`docs/superpowers/specs/2026-11-04-pr-field-notes-design.md`; the dig half is #3100). This service
runs the automatic half.

- **Poll** (`app/pr_field_note_poll.py`, every `PR_FIELD_NOTES_POLL_SEC`=900 s). One unauthenticated
  GitHub search for PRs merged since `max(now - PR_FIELD_NOTES_LOOKBACK_H, PR_FIELD_NOTES_NOT_BEFORE)`.
  No token, no gh, no git: the repo is public. The 72 h lookback repairs a missed tick. A PR with a
  ledger record is done. A new PR costs one `/pulls/<n>/files` call to classify it:
  - dependency-only or tiny docs-only (Juniper's decision 2): the poll writes its **code record** at
    once (two more calls: merge sha, commit authors). No run, no model turn.
  - anything else: one `pr.field_note` run, id `pr-field-note-<n>`, agent lane, background priority,
    deadline 20 h.
  A PR whose runs all failed gets one more run (`-a2`) an hour later; then it stops and shows as a
  miss in `dig field-notes`. At most one note run is in flight (the spec's "one at a time"), at most
  10 GitHub detail calls per tick (the public limit is 60/h). A rate limit, a 5xx or a network error
  ends the tick. One PR's own trouble (a 404 on its files, a refused code record) is logged in
  `pr_errors` and that PR alone waits for the next tick. A PR whose first files page is full (100)
  always gets a note run, never a code record.
  Nothing merged before `PR_FIELD_NOTES_NOT_BEFORE` is touched (decision 3: no backfill).
- **Run** (`app/pr_field_note_graph.py`, contract `orion/schemas/pr_field_note_run.py`):
  `precheck -> resource_request -> resource_wait -> field_note -> finish`, and a retry goes
  `retry_wait -> precheck`. `precheck` finishes a PR that already has a record (`already_recorded`)
  without a turn: at the start without asking the pool, and before a retry so an attempt that saved
  the note but replied badly is not repeated. `field_note` is one harness turn on the #2958 seam
  (`app/harness_step.py`; source tag `pr_field_note`, session `pr-field-notes`). Its fixed prompt
  spells out the two dig commands, says code checks are only for a sentence that claims something the
  packet does not state, and asks for about 3 tool calls (never more than 6). The hard bound is the
  turn time, `PR_FIELD_NOTES_TURN_TIMEOUT_SEC`=900 s. Hub keeps its finalize reserve
  (`HUB_CURIOSITY_HELD_TURN_FINALIZE_RESERVE_SEC`=330) out of that, so the motor gets 540 s; at 420 it got
  60 s and the first live run timed out. `tests/test_pr_field_note_budget.py` gates the arithmetic.
- **Done only with a receipt.** `verify_receipt` (code, inside the step, before the checkpoint) needs
  `field-note #<n> saved fn:<n>:<hex>` for THIS PR in the reply AND that id as a note for this PR in
  the ledger. Anything else is a failed attempt (the shared retry policy), never a completed run.
- **Ledger access.** compose mounts the host's `/mnt/orion-fcc/self` (`PR_FIELD_NOTES_SELF_HOST`) at
  `PR_FIELD_NOTES_LEDGER_DIR`. Only `app/pr_field_note_ledger.py` touches it. It is the one named
  exception in `tests/test_organ_names_partition.py`, and it hands back ids and booleans, never note
  text (a test checks every `.note` use is an `is not None`). A code record's `body_map` is empty:
  this container has neither the repo nor the live atlas.
- **Cost.** Each run is an act `make:durable.pr_field_note` (`app/run_acts.py`). Its family is tier 1
  in `config/physiology/families.yaml` with a 24 h deferrable window (Juniper's decision 1). Bids are
  shadow. Measured cost per note (wall time, whole-machine Wh, GPU slot seconds from the act; tokens of
  every attempt from `llm_call_usage`; Hub harness steps):
  `python3 services/orion-durable-runs/scripts/pr_field_note_cost.py`. An act the ledger could not
  attribute (`wh_by_host: null`) reads as UNREAD for that run; it does not fail the report.
- **Surfaces.** `/health` -> `field_note_poll` (the last tick). `POST /pr-field-notes/poll` runs one tick
  now. `?since=<iso>` re-polls a longer window after an outage over the lookback. When nothing has run
  for longer than the lookback, a tick searches the stretch the lookback lost and logs
  `field_note_poll_gap_possible` naming the merges with no record and no run (it never backfills them
  itself). A `since` before `PR_FIELD_NOTES_NOT_BEFORE` is refused.
  `GET /runs/<id>` shows `pr_field_note` (pr, outcome, note id) and `harness_steps`.
- **Deploy order:** `pr.field_note` is an additive `DurableWorkflowV1` value. Deploy orion-sql-writer,
  orion-hub, orion-actions and orion-cortex-orch before orion-durable-runs.
- **Off:** `PR_FIELD_NOTES_ENABLED=false` and restart: no poll, no runs, no code records.

### Daily reminiscence: overnight tick + `reminiscence.episode` (2026-11-07, design Patch 2)

Once a day Orion visits one week of their own past and writes five short hops about it (design
`docs/superpowers/specs/2026-11-07-daily-reminiscence-design.md`; the dig half is #3163). This service
runs the daily half.

- **Tick** (`app/reminiscence_poll.py`, every `REMINISCENCE_POLL_SEC`=600 s), only inside the Denver window
  `[REMINISCENCE_HOUR_LOCAL, REMINISCENCE_END_HOUR_LOCAL)` = 03:00-09:00. Agent-lane time is the real cost
  and the agent lanes are contended by day. Today's episode is the Denver date. Five hops already live
  today and no failed run today (Juniper asked in chat, or the run completed): nothing. Five hops live
  but today's run failed (its cross-read ran out of attempts): the retry runs only the cross-read. Else the week is the one today's first hop fixed, or the oldest
  week that ended before today with no complete episode (the same rule `dig reminisce` uses). A week with
  no commit on main is never visited alone: it joins every following quiet week and the next week with a
  commit into one period (design section 1a). The tick counts commits in the dig clone, mounted read-only
  at `REMINISCENCE_REPO_DIR` (`/mnt/orion-fcc/repo`; the image installs `git`). Unreadable history:
  `history_unreadable`, no run. Only quiet weeks open: `waiting_for_commit`, no run. One run
  `reminiscence-<episode>` (agent lane, background priority, deadline = the window's end). A day whose run
  failed gets one more (`-a2`) 30 min later, only while two step attempts still fit before 09:00. At most
  one reminiscence run is unfinished at a time. A week left unfinished is picked again the next day.
- **Run** (`app/reminiscence_graph.py`, contract `orion/schemas/reminiscence_run.py`):
  `precheck -> resource_request -> resource_wait -> step (x6) -> finish`, a retry goes
  `retry_wait -> precheck`. `step` runs the next of six steps under the run's one hold: the five hops
  (`what_it_was`, `what_it_was_like`, `where_i_am_now`, `where_im_going`, `otherwise`), then one
  `cross_read` (read all five, rewrite at most two, `dig close <week end>`). Each step is one harness turn
  on the #2958 seam with its own source tag (`reminiscence_<step>`: its own claude session; session
  `reminiscence`), each attempt bounded by `REMINISCENCE_TURN_TIMEOUT_SEC`=900 s (540 s of motor), each
  step its own `DURABLE_RUNS_RETRY_MAX_ATTEMPTS`. The ledger is the checkpoint: `precheck` records every hop
  already live today as `already_written` before it asks for a hold, so such a hop takes no hold and no
  turn. A step record keeps only the receipts it proved, never the rest of the reply (a reply may repeat
  hop words; the ledger stays the only place they live); a failed attempt's error names the reply length.
- **Done only with receipts.** A hop step needs `reminisce <hop> saved rm:<episode>:<hop>:<hex>` for THIS
  hop and today in the reply AND that id in the ledger for that hop. The cross-read needs the line
  `reminisce cross-read done`, and every rewrite receipt in it must be a rewrite in the ledger. `finish`
  completes the run only when all five hops are live (`outcome: complete`). A step whose day has passed
  fails (its hops would land in another episode).
- **Ledger access.** The same `/mnt/orion-fcc/self` mount as field notes (`REMINISCENCE_LEDGER_DIR`).
  Only `app/reminiscence_ledger.py` touches the reminiscence ledger: read-only, the second named exception
  in `tests/test_organ_names_partition.py`, and it hands back ids, dates, counts and booleans, never a
  hop's text, refs or claims (a test checks no `.note`/`.because`/`.claims` use, no write call, and every
  public return type). The gate to recall stays closed: that test's reader allowlist is unchanged.
- **Cost.** Each run is an act `make:durable.reminiscence` (`app/run_acts.py`). Its family is tier 2 in
  `config/physiology/families.yaml` (the design's default), agent lanes, a 6 h window, shadow bids. The
  design's estimate is ~30 min of agent-lane GPU slot a day (median), ~100 (p90). Measure it:
  `python3 services/orion-durable-runs/scripts/pr_field_note_cost.py --workflow reminiscence.episode`
  (wall time, whole-machine Wh, GPU slot seconds, tokens of all six turns and every attempt; it says how
  many of the 7 episodes the design asks for are in).
- **Surfaces.** `/health` -> `reminiscence_tick` (the last real tick's decision). `GET /reminiscence/plan`
  (`?any_hour=true` decides as if inside the window) shows what the tick would do now and the first step's
  prompt, and submits nothing. `GET /runs/<id>` shows `reminiscence` (episode, week, outcome, hop ids, next
  step) and `harness_steps`.
- **Deploy order:** `reminiscence.episode` is an additive `DurableWorkflowV1` value. Deploy
  orion-sql-writer, orion-hub, orion-actions and orion-cortex-orch before orion-durable-runs.
- **Off:** `REMINISCENCE_ENABLED=false` and restart: no tick, no runs. This is not the privacy gate.

### Every admitted terminal reaches `orion:durable:run:state` (2026-09-28)

`completed`, `failed` and `cancelled` admitted runs are all published as a
`DurableRunStateV1` (node `finish`, `failed`, `finish`) from the same outbox row as
their `run.<status>` lifecycle event (`entry_id = <run>:terminal:<status>:state`,
acked only after both publishes, deduped by sql-writer). Before this only
`completed` was, so a waiter such as cortex-exec's self-study reflect never saw an
admitted failure end the run.

### Admitted Orion's Day letter (`orion_day.letter`): Prime's end-of-day turn (2026-11-12)

Orion's letter about yesterday (America/Denver). Since living Prime Part 3 (design #3224) the letter
is Prime's own act: orion-mind sees the day closed with no letter, opens a `make` act (tier 0: no bid,
since 2026-11-13) and orders the run from Hub (`POST /api/orion-day/letter-run`), again as the next attempt
(`orion-day-<date>-<n>`) when a run writes nothing. Hub gathers the day with `orion/orion_day/`
(read-only SQL, the honest input of `day_input.py`) and submits the admitted request under Prime's
run id and correlation id (`llm.route.agent`, priority `system`, `deadline_at` from Prime's order).
Contract: `orion/schemas/orion_day.py`. Graph (`app/orion_day_graph.py`):

    resource_request -> resource_wait -> write_letter -> persist -> finish

- `write_letter`: ONE unified turn on the durable turn seam (`app/harness_step.py`) under the run's
  hold: Hub session `orion_day_letter` (Prime's hand-back reads it as a "letter" turn), source tag
  `orion_day_letter` (not framed as a curiosity run), `require_lookup=False` (Orion may dig and
  recall; a pure writing turn must not fail the lookup gate). The user message is
  `orion/orion_day/letter_turn.py`'s: the writing instructions lifted from the retired note verb,
  the days that got no letter, the whole record (closed-day block, temporal core, Juniper, curiosity,
  self-sense, reveries, the trimmed tail) and a closing `## Threads I am carrying` section. The turn's
  own prefix adds the Prime block (Part 1). The reply is split: the letter is `note_md`, the threads
  `carry_forward_md`. Same failure policy as `harness_smoke_graph`: a lost/recalled hold replays the
  node; a refusal, timeout, empty reply, transport error or a letter under 400 chars is one attempt
  (`DURABLE_RUNS_RETRY_MAX_ATTEMPTS`, backoff in `retry_wait`); past the budget the run fails with
  no row. A checkpointed letter is never regenerated. A worker that dies mid-turn has that turn
  cancelled by its checkpoint-derived id.
- The hold asks for `requirements.minimum_context_tokens` = record + frame + harness prefix + the
  letter's completion, so a heavy day is never placed on the 65,536-token chat card.
- The hold is released as soon as the letter is checkpointed.
- `persist` (no GPU): `INSERT INTO orion_day_letter ... ON CONFLICT (letter_date) DO NOTHING`
  (`app/orion_day_store.py`), then publishes the STORED row's letter as `journal.entry.write`
  (id `uuid5(letter_date)`, the row's `created_at`, `trigger_kind=orion_day_letter`; journal email
  disabled: Hub sends the styled email). `sources` carries the per-source input shares, the turn's
  `turn_correlation_id`, Prime's `prime_act_id` and the `missed_letters` the turn was told about.
  `carry_forward_expires_at` is NULL: the curiosity carry-forward offer is retired.
- The checkpoint keeps only a slim brief; the full brief is read from `durable_admission_runs.request`.
- Finish detail and the status API's `orion_day` block: `persisted`, `persist_outcome` (`written` /
  `already_written` -- key on this, not on `completed`), `turn_correlation_id`, `prime_act_id`, and
  `threads` (at most 5), which Prime reads to open concerns.
- Deploy order: orion-hub, then orion-durable-runs, then orion-mind (cortex-exec any time after
  durable-runs: nothing calls the retired verbs). An in-flight run checkpointed at the old
  `write_note` / `write_carry_forward` nodes cannot resume on this graph: check `GET /runs/unfinished`
  for `orion_day.letter` first.

`DurableRunner` can drive more than one compiled graph, keyed by
`DurableRunRequestV1.workflow`. Today only `"curiosity.investigate"` is registered --
this is plumbing for a second and third workflow (self-sense-eval's own graph, reflect's
own graph) landing in follow-on PRs, not a behavior change on its own. Full rationale
(the shared-checkpointer safety argument, `_peek_workflow`'s ordering, the
backward-compat default for pre-registry checkpoints) lives once, next to the code, in
`app/runner.py`'s module docstring and `WorkflowSpec`/`_peek_workflow`'s own docstrings --
read those rather than a second copy here.

Adding a real second workflow needs, at minimum: a new `WorkflowSpec` (its own graph,
node list, `finish_detail`), a new entry in `DurableWorkflowV1`
(`orion/schemas/durable_run.py`), and its own request/brief shape if it doesn't
fit `CuriosityRunBriefV1`.

### `self_study.reflect` (2026-09-21) -- the third registered workflow

Its own graph (`app/reflect_graph.py`): `llm_call -> finish`. Two nodes, not three --
unlike `self_sense_eval`, finding validation and the journal/`self_concept_history`
writes stay in `orion-cortex-exec`, unmoved. That logic (`_finding_from_llm_item`'s
evidence-chain construction, `publish_self_reflection_artifacts`,
`publish_self_concept_history_from_reflection`) needs the FULL `SelfSnapshotV1`/induced
concepts for evidence grounding, not just the small `self_study_reflect_input` summary
this graph's brief carries -- moving it here would have meant either re-deriving that
context inside this service (duplicating cortex-exec's repo-scan logic) or serializing
the whole snapshot through the brief on every dispatch. Neither was worth it for a graph
whose only real benefit is running the ONE LLM call on a durable run's GPU pool hold.

Admitted (stage 4.5, `app/admitted_reflect_graph.py`): `resource_request -> resource_wait ->
llm_call -> finish`. Before 4.5 an admitted reflect run fell back to the CURIOSITY graph (the
registry had no reflect entry). `llm_call` runs under the hold and sends its ref as
`options.gpu_lease` (cortex-exec forwards it; the gateway attaches), `llm_route` unchanged.

`llm_call` sends the exact same `CortexClientRequest` shape
(`verb="self_study.reflect"`, `options={"policy_dispatch_only": True, ...}`)
`orion-cortex-exec`'s own `_call_self_study_reflect_llm` always built directly -- this
patch only moves WHERE it's sent from (`DurableRunner._call_reflect_llm`, this
service's first-ever caller of cortex-orch's verb-dispatch channel; every other graph
here only talks to Hub and the state channel). `cortex-orch`'s existing
`self_study.reflect` verb dispatch is completely unchanged.

`orion-cortex-exec` dispatches this run then waits SYNCHRONOUSLY for its completion
event (subscribes to the state channel, filters on `run_id`, bounded by the same
`SELF_STUDY_REFLECT_TIMEOUT_SEC` the direct RPC always used as its deadline) --
`_call_self_study_reflect_llm`'s external contract (`list[dict] | None`) is completely
unchanged either way, so nothing downstream of it needed to change. A
failed/unconfirmed dispatch falls back to the direct RPC unchanged; an accepted
dispatch whose run genuinely fails does NOT also fall back (no double-spending the
reflect timeout budget on two separate attempts for one logical reflection).

### `self_sense_eval` (2026-09-21) -- the second registered workflow

Its own graph (`app/self_sense_graph.py`): `ask_questions -> publish -> finish`.

**Durable since 2026-10-23.** The admitted graph asks ONE question per `ask_questions` run,
so each answer is checkpointed when it lands; a timeout, crash or take-back resumes at the
next unanswered question. Each question's node timer is its share of the brief budget
(`timeout_sec / len(questions)`, passed to `execute` as `node_timeout_sec`); Hub's turn
limit is that share minus 30 s, so Hub answers (empty, recorded) before the timer cancels
the turn. Each question's turn id is a uuid5 of the run, the question and the replay
counters (`question_turn_id`), so it derives from the checkpoint: a restarted worker cancels
the turn its predecessor left running in Hub. It is also on the `run.started` event
(`step_correlation_id`) before the turn runs, and `last_turn` (`question_key`,
`turn_correlation_id`, `timeout_sec`) is checkpointed with each outcome. A failed attempt
goes through the shared `failed_attempt` policy above (`retrying` -> `retry_wait` ->
`resource_request`), with the deadline check reserving one share per unanswered question.
`attempt` counts the current question's failures. A question that spends its budget or fails
deterministically is recorded as an empty answer naming the error (each answer records its
own `attempts`), and the run goes on; it fails only when every answer is empty, on
`workflow_deadline`, or when not one more question fits before the deadline. Live
cause: run `20261023T180730Z-0842e5` held agent-gpu2 for its whole 8840 s budget on one
hung question, then failed with every answer lost.
Deliberately NOT a branch inside `graph.py`'s curiosity pipeline -- self-sense-eval has
no material/worldview read, no journal, no outreach, and asks four fixed questions
instead of one open prompt, so sharing `journal`/`read_turn_result` would have meant
threading a `line`-conditional through nodes real production investigation runs already
depend on.

Reuses curiosity's own turn-execution RPC to Hub as-is (`Deps.run_turn` ==
`DurableRunner._run_turn`, unchanged) -- Hub's `_handle_turn_request` was already
content-agnostic, so the only wire change needed was additive:
`CuriosityTurnRequestV1.session_id`, so each question runs under the shared clean
session `orion.evals.self_sense_runner.SESSION_ID` instead of curiosity's own
investigation session. `CuriosityRunBriefV1` gained three more additive fields for this
workflow: `questions` (the four fixed `(key, text)` pairs, Hub's copy of
`orion.schemas.self_sense.SELF_SENSE_QUESTIONS`), `self_definition_version` and
`lived_answers` (read by Hub before dispatch, same "Hub does the DB reads, the runner
just executes" split `material` already uses for investigation).

Hub's dispatch (`curiosity_investigation.py:_dispatch_self_sense_eval_durable_run`)
uses the exact same generic ingress every verb already shares
(`cortex-orch/app/durable_runs.py`) -- not a new RPC path, one more
`CortexClientRequest` carrying a `durable_run` metadata blob, same as
investigation/self-inquiry's own `_dispatch_durable_run`.

**Caught before merge, not after:** `self_sense_graph.py` transitively imports
`orion.evals.self_sense_runner` -> `orion.evals.self_sense`, and that module reads
`config/field/orion_field_topology.v1.yaml` at IMPORT time. Since this module is
imported eagerly at `DurableRunner.__init__` (building both graphs), this service's
Dockerfile needed the same `COPY config/field /app/config/field` line that fixed the
exact same bug in `orion-hub` on 2026-09-21 -- without it, this container would have
failed to boot entirely the moment this workflow was registered, not just failed one
tick. Verified with a real `docker build` + import + `DurableRunner()` construction
inside the built image, not just a scratch-container copy.

## Finish detail fields (`orion:durable:run:state`, `detail`)

`DurableRunStateV1.detail` is a free dict (no schema change when a key is added). What
each workflow's `completed` event carries, and what every `failed` event carries:

**`curiosity.investigate`** (`app/graph.py::finish_detail`): `line`, `self_definition`,
`lived_answer`, `self_question_family`, `reach_out`, `reach_out_why`, `continue_line`,
`finding_text` (<= 8000 chars), `journal_entry_id`, `attempts`, plus -- when the runner
recorded them (2026-09-22) -- `turn_correlation_id`, `harness_elapsed_sec`,
`harness_started_at`, `harness_finished_at`.

- `turn_correlation_id`: the correlation the harness turn actually ran under -- the
  per-lease derived id when admitted (`turn_correlation_id()`), else the run's own.
  This is the PK of `harness_turn_trace`, so a run's harness row is now joinable
  structurally instead of by text-searching `final_text`.
- `harness_elapsed_sec`: wall seconds the runner itself measured around the Hub RPC
  (monotonic clock; includes bus transit and reply decode). Hub's own
  `debug.elapsed_sec` is a different, in-Hub measurement and still goes to the journal.
- `harness_started_at` / `harness_finished_at`: ISO-8601 UTC stamps from the same
  measurement.

All four are additive and optional. They come from the `harness_turn_meta` state key
`harness_turn` writes; a thread checkpointed before that key existed (turn already
done, resumed after deploy) finishes with them absent -- not null, not an error. A
leased pre-key checkpoint still yields `turn_correlation_id` from the older
`debug.turn_correlation_id` breadcrumb.

Urgent runs (`brief.urgent` set, Plan 3 of the urgent-curiosity spec) add three keys,
absent on every ordinary run:

- `urgent`: `{incident_id, trigger, subject, question, requested_at}` from the seed (the
  evidence bundle stays in the brief).
- `incident_report`: the run's newest valid `:IncidentReport`
  (`orion/curiosity/incident_report.py`) as `{incident_id, is_real, likely_cause,
  evidence: [...], severity, operator_action, confidence}`, or `null`.
- `report_flag`: `null` with a report; `"no_structured_verdict"` whenever
  `incident_report` is `null` (missing/malformed node, unreadable graph).

The seed also rides the Hub turn request (`CuriosityTurnRequestV1.urgent`), and an urgent
admitted run gets at most `min(DURABLE_RUNS_RETRY_MAX_ATTEMPTS, 2)` attempts per node with
`10 s·2^n` backoff (capped by `DURABLE_RUNS_RETRY_MAX_SEC`) instead of the service budget.
An urgent admitted run's terminal `failed` detail also carries `urgent`, and its `cancelled`
detail is `{"error": "cancelled", "urgent": {...}}` (ordinary cancels stay `{}`), including
when a cancel wins the race against completion. Hub gives an urgent admission `deadline_at` (its overall urgent timeout); a run still
queued or mid-turn at that moment fails with `workflow_deadline` and the `urgent` detail.
An urgent run never keeps the outreach hold, even when the turn asks to reach out: its
report replaces reach-out.

**`self_sense_eval`** (`app/self_sense_graph.py::finish_detail`): `line`, `published`,
`failed`, `empty`, `attempts`, `turns` -- a `{question_key: {turn_correlation_id,
harness_elapsed_sec, harness_started_at, harness_finished_at}}` map, one entry per
answered question, same field meanings as above (one self-sense run is several turns).

**`self_study.reflect`** (`app/reflect_graph.py::finish_detail`): unchanged.

**`failed`** (any workflow, `DurableRunner._failed_detail`): `error`, `node`, and
`turn_correlation_id` when the workflow can name it (curiosity only: the recorded value
if the turn had completed, else the identity the raising `harness_turn` derived from the
same state). The admitted path's terminal projection (`AdmissionRuntime._terminal`,
which serves `failed` and `cancelled`) carries `error` plus `turn_correlation_id` only
when the graph recorded it -- never re-derived there, because the lease is already
cleared by then and a fresh derivation would name the run's lineage instead of the turn
that failed. On that path the recorded id is the one the last attempt was issued, or
would have been issued, under: if admission raised before the RPC left (lease lost), the
id names an attempt no turn ran for, so the join returns no row -- never a wrong row.
The worker-recovery fence stashes the same id before clearing an in-flight attempt's
lease, so a later deadline/cancel terminal names the fenced generation, not an older one.

**`compactor.digest`** (`app/compactor_digest_graph.py::finish_detail`): `line="compactor"`,
`workflow_id`, `window_label`, `chunk_count`, `merge_mode`, `merge_skipped_reason`,
`journal_entry_id`, `card_id`, `attempts` (call attempts, pool waits excluded), `gpu_roles`.

## RPC-health publish (mesh transport coverage, 2026-09-24)

The long-lived `rpc_bus` (opened in `app/main.py`'s lifespan) publishes an
`RpcHealthSnapshotV1` to `orion:rpc_health:snapshot` every
`RPC_HEALTH_PUBLISH_INTERVAL_SEC` (default 30) with `instance="main"`
(`RpcHealthPublisher`, `orion/core/bus/rpc_health_publish.py`). The signal gateway maps it
to organ `rpc_health_durable_runs` (pass-through, unregistered/exogenous).

Hops in `channel_latency` (when `RPC_HEALTH_CHANNEL_LATENCY_ENABLED=true`):

- the runner's `rpc_request` channels (harness turn to Hub, verb dispatch to cortex-orch);
- the GPU pool lease RPCs (`gpu_pool_lease`: acquire / status / heartbeat / release of the
  run's hold). Waiting in the pool's line is recorded by the pool under `gpu_pool_wait`, which
  equilibrium excludes from transport.

Stage 4.5 deleted every outbound HTTP hop this service had (gateway `/routes`, lane `/slots`,
cabinet, thought visual activity, the gpu2 controller) with `app/http_hops.py`; no reader keys
on those hop names.

Keys: `RPC_HEALTH_PUBLISH_ENABLED` (true), `RPC_HEALTH_PUBLISH_INTERVAL_SEC` (30),
`RPC_HEALTH_CHANNEL_LATENCY_ENABLED` (true). Consumer-first: rebuild
`orion-signal-gateway` and `orion-equilibrium-service` on PR #2312's build before this
service ships `channel_latency` (the schema is `extra="forbid"`).

## Situation graph: `situation.update` (2026-10-07, shadow)

Orion's running situation, per the spec `docs/superpowers/specs/2026-10-07-situation-graph-design.md` (step 2). One self-driven LangGraph thread per UTC day, `situation:juniper:<date>`, written by `app/situation_driver.py`. It is never submitted, never admitted, and needs no GPU lease.

- **Inputs:**
  - `orion:chat:history:turn`: a finished turn. Its prompt cues recall.
  - `orion:durable:run:state` rows for a completed `memory.episode_distill`: new facts landed.
  - A clock tick every `SITUATION_TICK_SEC`.

  Events that arrive during a step are coalesced into the next one.
- **Graph** (`app/situation_graph.py`): `ingest -> reduce -> expire -> cue -> prime_recall -> project -> done`. Facts are re-derived from `episode_memory` on every step (`app/situation_store.py`), so a step is idempotent. Checkpoints are written once per step (`durability="exit"`), not per node.
  - **Current state** (`whereabouts`, `doing`): only memories whose end date Juniper stated.
  - **`recent`:** `happened` memories told within `SITUATION_DEFAULT_TTL_HOURS`.
  - **`waiting_on`:** open follow-ups.
  - **Priming:** memories that share a cue referent, ranked by strength decayed over their half-life.
- **Output:** on a changed revision, Redis `orion:situation:latest` and bus `orion:situation:state` (`SituationStateV1`). A step that changed the situation, or failed to prime, also writes one `DurableRunStateV1` row (workflow `situation.update`). Quiet steps write nothing. Nothing reads the state until step 3.
- **Retention:** a new day's thread is seeded from the latest earlier day within `SITUATION_RETENTION_DAYS`. Older threads are removed with `adelete_thread`, so the resume sweep (which walks every checkpoint) stays bounded. The sweep skips these threads (`SELF_DRIVEN_WORKFLOWS` in `app/runner.py`).
- **Health:** `/health` -> `situation: {steps, revision, queued, last_error}`.
- **Kill switch:** `SITUATION_GRAPH_ENABLED=false`.

## Regulation: `temporal_self.update` / `regulate` (2026-10-10, no reader yet)

Temporal Self rev 4 (PR #2369) order 3. A second self-driven thread, `temporal_self:orion:<local
date>` (`ORION_SITUATION_TIMEZONE`), steps every `TEMPORAL_SELF_TICK_SEC` (120 s) and immediately
on a Juniper chat turn (`orion:chat:history:turn`; Orion's own outreach does not wake it):

    ingest -> regulate -> done

`regulate` reads three inputs and folds them through the pure reducer
`orion.regulation.arousal.classify_arousal` (hysteresis carried by the checkpoint):

- E1, minutes since Juniper's last turn (`orion.regulation.juniper_turns`: a row with a prompt
  that is not `client_meta.unsolicited`; the same rule the dream's idle gate now uses);
- S1, the cabinet reflex (`orion.autonomy.cabinet_heat`, `cabinet_hot` only, immediate);
- S2, GPU pool `queue_depth` summed, `>= ORION_REGULATION_STRAINED_GPU_QUEUE_MIN` (2) for
  `ORION_REGULATION_STRAINED_GPU_QUEUE_SEC` (300 s), from `gpu_pool_state_history`.

Result: `engaged` (turn within `DREAM_IDLE_MINUTES`), `idle`, `strained` (leaves only after
`ORION_REGULATION_STRAINED_CLEAR_SEC` of fresh, clear inputs), or `unknown` (any stale input;
never idle). Written to Redis `orion:regulation:latest` (TTL `ORION_REGULATION_REDIS_TTL_SEC`,
360 s) and `GET /regulation/state`; each new level episode (a level change, or a restart after a
gap longer than three ticks) is one `arousal_transition` row in
`temporal_self_event` (`services/orion-sql-db/manual_migration_temporal_self_event_v1.sql`,
hand-applied). The latest rest-drive reading (`orion:drive:rest:latest`) is embedded verbatim
for the trace; arousal never reads it. No dial reads arousal yet (spec order 6).

Strain loop part 1 (2026-10-20, `docs/superpowers/specs/2026-10-20-strain-loop-design.md`): a
level row is written only when the level changes, so a cause that joins an episode already
strained (the cabinet going critical mid-episode) was never recorded. While strained, the node
now also writes one row per strain cause new to the episode (`cause_row`, event id
`...:strained:at:<observed_at>`, `payload.joined`), one seed row (`payload.seed = true`) when
the episode has nothing recorded yet (the first step after deploy, a lost onset row), and a
keepalive (`joined = []`) every `STRAIN_KEEPALIVE_SEC` (1,800 s) so the chronology can close an
arc whose closing row was lost, and a row when the causes present now differ from the last row's
(a cause left, e.g. the cabinet cooled; at most every `STRAIN_CAUSE_CHANGE_MIN_SEC`, 600 s, so a
flapping cause costs at most one row per 10 min). Every row's reading carries the causes present
at that moment. A failed write is retried on the next step. Same table, kind and label, so readers of levels still
read `strained` (`scripts/analysis/measure_arousal_reader_replay.py` filters them out, since they
would restart a level's age). The causes already recorded and the last row's time ride in the
checkpoint as `strain_causes_seen` / `strain_row_at` / `strain_causes_now`.
Kill switch: `TEMPORAL_SELF_STRAIN_ARCS_ENABLED=false` (no cause rows, no strain arcs).

**Binding arcs (2026-11-03, body economy design Organ 5 "Felt state").** Strain only fires at the 34 °C
reflex; between the soft line and that, the heat price binds and the hot gate refuses GPU work, but
nothing in Orion's day said so. After pricing, the `physiology` node now writes `body_binding` rows
into `temporal_self_event` (`source_table='physiology'`, subject `orion:body:<constraint>`;
`app/body_binding.py`, insert `regulation_store.record_binding`). Bound means the price's
`bind_state` is `binding`, `exogenous_bind` or `stuck_at_ceiling` (what the body report calls
BINDING). Rows: `onset` once bound for 240 s, `change` when the bound state changes (at most every
600 s), `keepalive` every 30 min while bound, `release` once slack for 240 s (the arc ends at the first
slack reading). `no_reading` writes nothing; the reducer's 4,500 s stale timeout closes the arc at its
last row. An open episode that went unread (no_reading, or a reading gap over 3 x
`TEMPORAL_SELF_TICK_SEC`) marks its next row `after_gap`, so the chronicle never counts the unread
stretch as binding time. Event id `body_binding:<constraint>:<episode start>:<row>:<written at>`; the payload carries
the level, line, price, ceiling, the price's `why` and `self_spent`. The per-constraint episode rides
in the checkpoint as `binding_episodes`; a failed write keeps it, so the next tick retries. The
chronicle folds the rows into one `binding` arc per constraint (several open at once). Awareness only:
no price or gate reads it. Kill switch: `TEMPORAL_SELF_BINDING_ARCS_ENABLED=false` (no rows, no arcs).
Replay: `python -P services/orion-durable-runs/evals/body_binding_replay.py [--live --days 7]`.
Deploy order: orion-thought and orion-hub before this service (`binding` is new to `extra="forbid"`
frame readers).

**Self-monitor node (2026-11-03, self-monitor organ patch 2: host and publish).** Spec:
`docs/superpowers/specs/2026-10-20-self-monitor-organ-design.md`. Orion keeps a track record of their
own faculties (acts, forecasts, turn self-bets, vision expectations). The `self_monitor` node runs
after `immune`, at most once per `SELF_MONITOR_INTERVAL_SEC` (3,600 s):

1. It reads every bet source for the last 7 days (`app/self_monitor_store.py`; the SQL and adapters are
   `orion/self_monitor/collect.py`, the same path `python -P -m orion.self_monitor.report` uses). Each
   source reads in its own read-only transaction with a 15 s statement timeout. A failed source is
   named in `source_errors`, never skipped silently.
2. It reduces the bets per faculty and applies hysteresis: a new status must hold 3 evaluations
   (`orion/self_monitor/host.py`). The memory for that is a small per-faculty carry in this thread's
   checkpoint (`self_monitor.carry`), carried over midnight.
3. It publishes the held `SelfMonitorStateV1` to Redis `orion:self_monitor:latest` (TTL 3 h) and
   appends one row to `substrate_self_monitor_history` (migration
   `services/orion-sql-db/manual_migration_self_monitor_history_v1.sql`; rows older than
   `SELF_MONITOR_HISTORY_RETENTION_DAYS`, 30, are pruned by the node).
4. Each held status change writes one `temporal_self_event` row (`source_kind =
   faculty_status_changed`, subject `faculty:<name>`). While a notable status (broken, unreliable,
   miscalibrated) holds, a keepalive row follows every 12 h. The chronicle folds these rows into one
   `faculty` arc per notable stretch (see `orion/temporal_self/README.md`, "Faculty arcs"). A failed
   write is retried at the next evaluation.

`GET /self_monitor/state` returns the Redis state verbatim plus `view` (fresh or stale); 404 when no
evaluation landed in the last 3 h. `/health` shows the node's last summary under
`temporal_self.self_monitor` (counts by status, source errors, rows written). Own timeout
(`SELF_MONITOR_TIMEOUT_SEC`, 120 s); a failure is the warning `self_monitor_failed` and never touches the
regulation reading. Nothing obeys the state yet: effects (act gate, self-question, telling Juniper) are
later patches. Kill switch: `SELF_MONITOR_ENABLED=false` (no state, no rows, faculty arcs not read).

**Retro play source (2026-10-22, Orion plays Zelda section F).** The chronicle reads orion-hub's
`retro_play_sessions` (migration `services/orion-sql-db/manual_migration_retro_play_sessions_v1.sql`,
REQUIRED-BY this service and orion-hub) as source `retro_play`: one `play` arc per finished Retro
session, shown in the chat stance cue. The query is skipped while the table does not exist
(`to_regclass`), so the source can never fail the whole read. Kill switch:
`TEMPORAL_SELF_RETRO_PLAY_ENABLED` (ON).

**Street episodes (2026-10-23, walkway follow-ups spec PR #2763 item 1).** The chronicle reads notable
outside events as source `street_event`, three reads of one kind: published clean motion stories whose
rules version passed Juniper's spot-check gate (`vision_motion_story` + `vision_motion_story_verdict`;
labels carry no people count), doorbell rings (`vision_events`), and graded
handoffs with p < 0.05 (`vision_handoff_street_outbox`). One `street` arc each, labelled "Outside
(<camera>): ...", at most `TEMPORAL_SELF_STREET_EPISODES_PER_DAY` (12) a local day. Each read is skipped
while one of its tables is missing. Log lines: `temporal_self_street_episode arc_id=... source=...` per
new arc, `temporal_self_street_capped count=...` when the cap kept events out,
`temporal_self_street_visit_merged arc_id=... added=...` when a row joined an earlier arc of the same
visit (handoff link, or a ring and a story within 84 s; it spends no slot). A street row is available
when it was written (`created_at`, never before its end), so a story written after the 300 s read lag
is read in the window it was written in and folds (before 2026-10-29 it was stored late, never folded).
The story read adds the handoff loop's links (`vision_handoff`) and Juniper's word on a name
(`vision_motion_story_verdict`, `orion_ask`); each is skipped while its table is missing. Read-only
replay: `python -P services/orion-durable-runs/evals/run_street_episode_replay.py --days 7` (add
`--repo <old checkout>` for a before run). Kill switch: `TEMPORAL_SELF_STREET_EPISODES_ENABLED` (ON); off, the
source is not read and episodes are room-only again. Deploy order: orion-thought and orion-hub before
this service (the `street` arc kind is new to `extra="forbid"` frame readers).

**Office sessions (2026-11-05, office camera design PR #3127 step B).** The chronicle reads orion-sql-writer's
`vision_office_session` (migration `manual_migration_vision_office_session_v1.sql`, REQUIRED-BY sql-writer)
as source `office_session`: FINAL rows only (`status = 'closed'`), available at `closed_at` (a row is final
660 s after the visit ended). It adds no arc: it relabels today's closed office `company` arc that overlaps
it most, e.g. "Juniper (probably) in the office: sitting at a desk, opening a box; reaching into a cabinet",
and records `vision_office_session:<id>` in the arc's `related_refs`. The read is skipped while the table is
missing. Known limit: a visit that becomes final after local midnight finds no arc (yesterday's arcs are
already drained) and labels nothing. Kill switch: `TEMPORAL_SELF_OFFICE_SESSIONS_ENABLED` (ON). No new arc kind
or schema field reaches frame readers (only a label), so there is no deploy order.

Kill switches: `TEMPORAL_SELF_ENABLED=false` (no thread), `ORION_REGULATION_AROUSAL_ENABLED=false`
(reads `unknown`). Health: `/health` -> `temporal_self`.

## Temporal Self chronology: the `chronicle` node (2026-10-10, patch 3, no reader yet)

The same thread now runs `ingest -> regulate -> chronicle -> done`. `chronicle`
(`app/temporal_self_chronicle.py`) runs the pure reducer `orion/temporal_self/` live: it turns
rows Orion already writes into **arcs** (a stretch of the day Orion kept coming back to one
subject, or one bounded process) and one frame of "where am I in my day".

Each step reads forward from the stored watermark to `now - TEMPORAL_SELF_READ_LAG_SEC` (300 s), in
windows of at most an hour. Per window:

1. `app/temporal_self_sources.py` reads every bound source (the reference for every query is
   `orion/temporal_self/evals/export_fixture_day.sql`): broadcast ticks by `generated_at`, and every
   event whose *available* time (a process at its end) is in `[lo, hi)`;
2. a late-row probe re-reads the previous 30 min (memory episodes: 3 days; visual deferrals: 3 h,
   because `abandoned` replaces `active` ~90 min after `started_at`) for rows that were not there
   yet; they are folded as late (the reducer counts them in
   `frame.skipped_at_or_before_watermark`, never folds them) and stored with
   `temporal_self_event.late_unfolded = true`, so each is counted once;
3. ONE `fold` (ticks and events together), `advance_clock(hi)`, `build_frame`, `drain_closed_days`,
   in a worker thread (pure, CPU-bound; it shares the event loop with admission and `/health`
   otherwise). An in-window row that is already stored with an EARLIER available time (an upsert
   moved it, e.g. a rewritten `completed_at`) is dropped and counted (`moved_total`), never folded
   twice;
4. one body read (`orion_biometrics_cluster`, athena's cabinet temperature, `cabinet_ambient_spike`,
   stored visual deferrals) per newly closed non-reverie arc, into `arc.body`;
5. ONE transaction (`app/temporal_self_store.py`): new events, changed arcs, closed days, the
   `current_day` frame, cursors, and the reducer's own state (gzipped, `temporal_self_state`). The
   in-memory state advances only after the commit, so a crash or a failed commit re-reads the same
   window: no double fold, no dropped row. The commit refuses if the stored watermark moved (a
   second writer).

Bounded: a step stops starting new windows after 30 s, every chronicle statement has a 30 s
`statement_timeout`, and the node gives up after 120 s (`chronicle_failed`, retried next step).

Reducer state lives in its own table, never in the LangGraph checkpoint (it reaches ~1.2 MB at a
busy midday). A chronicle failure is a `chronicle_failed` warning on the step; `regulate` is
unaffected. First boot (no stored state) starts at local midnight `TEMPORAL_SELF_BACKFILL_DAYS` ago
(1: yesterday is closed on the first step). A stored state that no longer validates is reported
and that whole local day is re-folded from its midnight (arc ids are deterministic).

Routes (all read Postgres): `GET /temporal-self/frame` (current-day `TemporalSelfFrameV1`),
`/temporal-self/day/{day_id}` (closed day, or 404), `/temporal-self/arcs?day_id=&kind=`,
`/temporal-self/arcs/{arc_id}` (one arc by id from any retained day, or 404; the resolver for
reference kind `arc` in `orion/refs.py`), and outside the chronicle `GET /acts/{act_id}` (one
`grammar_acts` row, or 404; reference kind `act`),
`/temporal-self/threads` (concern loops raised in conversation with no verdict yet),
`/temporal-self/cursors` (per-source last folded row, the read watermark and its lag, late rows in
the last 24 h; a watermark more than `READ_LAG + 10 ticks` behind is named in `warnings`).

Retention every 6 h: `temporal_self_event` 30 days (this includes the regulate node's
`arousal_transition` rows), closed arcs 90 days, days 365 days
(`TEMPORAL_SELF_{EVENT,ARC,DAY}_RETENTION_DAYS`).

Tables: `services/orion-sql-db/manual_migration_temporal_self_v1.sql` (hand-applied BEFORE
deploy). Kill switch: `TEMPORAL_SELF_CHRONICLE_ENABLED=false`. Health: `/health` ->
`temporal_self_chronicle`. Eval: `evals/temporal_self_live_replay.py` runs this code path over the
10-09 fixture on a disposable Postgres (`--source-dsn` reads real tables read-only instead).

## Body: the `physiology` node (2026-10-23, body economy Patch A3, no reader yet; prices Patch B2)

Organ 1 of the body economy (`docs/superpowers/specs/2026-10-23-organism-body-economy-design.md`).
After `regulate`, at most once per `ORION_PHYSIOLOGY_MIN_INTERVAL_SEC` (100 s; chat-turn steps in
between are skipped), the node reads every constraint Orion lives under and writes one
`PhysiologyStateV1` (`orion/schemas/physiology.py`):

| constraint | level (unit) | line | source |
|---|---|---|---|
| `heat` | cabinet C | soft line `ORION_PHYSIOLOGY_HEAT_SOFT_LINE_C` (31) | this step's own regulate inputs (no second query) |
| `energy` | Orion month-to-date USD (local month) | `ORION_PHYSIOLOGY_ENERGY_MONTHLY_ENVELOPE_USD` (unset: no line) | `orion_energy_hourly` x `energy_cost_accrued` next-kWh rate (newest `computed_at`; unpriced recent hours carry the newest rate) |
| `lane:<role>` | GPU slot-seconds occupied in the last `ORION_PHYSIOLOGY_LANE_WINDOW_SEC` (at most `slots` leases count at once; the rest is `detail.lent_overlap_s`) | discovered slots x window (pool state RPC) | `gpu_pool_events` closes (`held_ms`, each on its own seat) + open grants |
| `attention:juniper` | outreach messages sent today, every door except the strain report and forced sends | `JUNIPER_DAILY_CAP` in `orion/physiology/attention_pacing.py` (6; read by Hub too); priced by the pacing threshold when `ORION_PHYSIOLOGY_ATTENTION_PACING=on` | `endogenous_outreach_decisions` |
| `claude` | limit events, plus `state` | not limited | `orion:substrate:claude_limit` (`ClaudeLimitObservationV1`), latest per kind in memory |

A failed, stale or never-observed source is `level=None` with a `no_reading` reason, never 0. Pure
logic: `orion/physiology/body.py`; reads: `app/physiology_store.py`. Transport: Redis
`orion:physiology:latest` (TTL `ORION_PHYSIOLOGY_REDIS_TTL_SEC`) and bus `orion:physiology:state` ->
sql-writer `physiology_state` (one row per tick). `/health` -> `temporal_self.physiology` shows the
last tick id and every `no_reading`. Own 30 s timeout; a failure is the step warning
`physiology_failed` and never touches the regulation reading. Off: `ORION_PHYSIOLOGY_ENABLED=false`.

Lanes (Patch B1, 2026-10-27):
- **Each held span is charged to the seat its own event names.** A lease the pool re-placed (aborted
  on `agent-gpu2`, re-granted on `agent`) used to be charged in full to its last seat. That is how
  `lane:agent` read 5,985.9 slot-s in a 3,600 s window on one slot (10-23 10:14:12; 747 of 2,778
  ticks over capacity by 10-27). Warning: `lane_leases_held_on_two_seats:<n>`.
- **Gap lending is real overlap, reported apart.** The pool lets a higher-priority one-off call run
  in an idle background hold's gap (scheduler H3). Both leases are live; the slot is occupied once.
  `level` caps concurrent leases at `slots`; `detail.lease_s` is the raw sum and
  `detail.lent_overlap_s` the difference. A seat with 0 live slots that still holds a lease is left
  uncapped and flagged `held_without_live_slot`.
- **Waiting self-work (hunger) sits beside each lane**, from `gpu_pool_state_history.deferrable_depth`
  (#2873): `deferrable_queued` (queued background durable-run holds, summed over fresh pool hosts,
  of every work class the pool's own class map lets run on this seat) and `deferrable_waiting_sec`
  (how long that queue has been non-empty as far as the history saw it; a hole in the history
  restarts the count; `deferrable_waiting_capped` = the whole 6 h lookback). Queued work has no
  seat yet, so one item sits beside every seat it may use: **do not sum lanes**. Unknown is
  `deferrable_no_reading` (backlog unread, stale, unrecorded, or the class map unread), never 0. A
  stale pool host is named in `deferrable_hosts_stale`. It is not strain and nothing prices it; it
  seeds Appetite. The class map comes from the pool state RPC (`include_config`).

Known limits: `lent_overlap_s` uses the seat's slots now, not as they were earlier in the window (a
seat that is down now is uncapped for the whole window);
`claude` reads `no_observation` while orion-cocreation-signals' `COCREATION_SIGNALS_CLAUDE_LIMIT_ENABLED`
is false.

### Budget: shadow prices and the strain dial (2026-10-27, body economy Patch B2)

Organ 2 Budget and Organ 5's `strain` dial, **shadow only**. Every physiology tick, after the Body
readings, `orion/physiology/budget.py` puts a price on each reading (`PhysiologyConstraintReadingV1.price`:
`value`, `ceiling`, `ceiling_source`, `bind_state`, `use`, `budget`, `excess`, `horizon_sec`, `why`) and
the dial on the state (`PhysiologyStateV1.strain`). Same Redis key, same bus channel, same
`physiology_state` row (prices live inside `constraints`; `strain` is recomputable from them with
`strain_dial`). **Nothing that decides reads them.** Debug readout for humans: `GET /physiology/state`.

| constraint | what the price charges (use vs budget) | `exogenous_bind` (clamped, listed apart from strain) |
|---|---|---|
| `heat` | Orion's own share in C: short-run sensitivity x watts above baseline, per host, against the room's headroom (soft line - (raw cabinet reading - Orion's share)); forecast = the newest reading | room over the line even with Orion's share at 0 (0.5 C hysteresis) |
| `energy` | month-end Orion cost forecast vs the $ envelope | the always-on floor alone reaches the envelope (5% hysteresis) |
| `lane:<role>` | lease-seconds asked of the seat (`detail.lease_s`) vs slots x window; waiting work is NOT an input | never (Patch C1). A seat with 0 live slots now whose window use already ran (`served_while_loaded`: a swap seat, diffusion / agent-gpu2) is slack; one with a lease still open (`held_without_live_slot`) is `no_reading` |
| `attention:juniper` | today's cap-counted sends paced to the end of the local day vs the cap | never. Self-spent (Patch C1): Orion's own outreach fills it, so it is listed in `strain.self_spent` and does not move the dial; its price still charges outreach wants |
| `claude` | `clear` = 0 use | `limited` |

Rules (`orion/physiology/prices.py`, Patch B): bounded ramp into bind (never a jump to the ceiling),
bind hysteresis, the point sensitivity (not the upper bound), ceilings `provisional` until Appetite has
graded wants to size them, and the stuck check: at the ceiling, unflagged, past one horizon reads
`stuck_at_ceiling` plus the warning `price_stuck_at_ceiling:<constraint>`. Rates are per hour, scaled
to the real tick. Unknown is `no_reading` and the price is held at its last value, never set to 0.

Heat sensitivities are data: `config/physiology/heat_sensitivity.yaml`, written only by
`scripts/analysis/fit_heat_sensitivity.py --write-config` and baked into the image
(`ORION_PHYSIOLOGY_HEAT_SENSITIVITY_PATH`). A host entry older than 30 days, a missing file, a
fitted host with no watts, or an awake host with no fit makes the heat price `no_reading`. Refit and
rebuild to refresh it. Off: `ORION_PHYSIOLOGY_PRICES_ENABLED=false` (Body readings only).

Known limits (review, 10-27):
- Orion's heat share uses the 2 h step-response sensitivity, so it is a **short-run lower bound**
  for load held for hours. `exogenous_bind` ("not my load") is therefore biased toward excusing
  Orion. No reader may decide on it before the sensitivity is calibrated (design Organ 6).
- The cabinet reading is compared raw. There is no sensor offset (removed 10-29: the sensor never
  moved; the 10-17 rise is real heat from the vision bring-up). A yaml that still carries
  `measurement_offsets` is refused, and heat reads `no_reading`.
- hecate counts as asleep below a 150 W 1 h mean, so a short burst can be missed.
- The carry (price, bind flag, ceiling run) crosses the daily graph thread (`CARRIED_KEYS`) and is
  kept for a day for a constraint missing from a tick. Warnings `heat_sensitivity_stale_in_days:<host>:<d>`
  (7 days ahead) and `heat_sensitivity_stale:<host>` say when to refit.
- `attention:juniper` paces the day only after 6 h of it; before that the use is the count itself.

### Appetite + Choice: `POST /physiology/bid` (2026-10-27, body economy Patch C1, SHADOW)

A producer sends a `WantV1` (`orion/schemas/physiology_want.py`) and gets a `WantDecisionV1` back:
funded or deferred, the lane, the prices it saw, its cost as a share of each budget, the net, the
binding constraint, what it displaced, `retry_after_sec`, and `shadow=true`. **Nobody obeys the
decision yet** (design Patch D cuts over one reader per PR). Every decision is published on
`orion:physiology:want:decision` and sql-writer stores it in `want_ledger`.

```text
net       = family_weight x value_rank - sum_k price_k x cost_k      (cost_k = share of constraint k's budget)
funded    = net > 0, or a hungry Tier 2 family under 5% of Tier 2's funded wants over the rolling 7 days (never past a capacity line)
placement = the cheapest acceptable lane with room; a seat with 0 live slots is a last resort
```

- **Families** (`config/physiology/families.yaml`, one per act-ledger actor): tier 0 never bids
  (`hub.chat_reply`, `vision.council`, `perception`), tier 1 upkeep (`actions.journal`, `dream.cycle`,
  `durable.journal_compose`), tier 2 the rest. Unknown family: deferred `unknown_family`.
- **Appetite** (`orion/physiology/appetite.py`): intensity `1 - exp(-t/tau)`, `t` = time since the
  family's last act closed (floored by B1's waiting deferrable work beside its lanes); `tau` = the
  family's median close-to-close gap (>= 5 gaps), else its `deferrable_window_s`. `value_rank` =
  percentile of intensity x outcome within the family's 30 days; `graded` = >= 30 graded closes
  (expired closes are not grades); ungraded families rank on intensity alone.
- **Weights** (`orion/physiology/weights.py`): shrinkage (kappa 30) from the prior toward the mean of
  the evidence rates that exist: downstream use (`act_downstream_use`, linked + window closed) and
  Juniper replies (resolver closes of `juniper_replied` acts). A missing source is skipped, never 0;
  under 10 evidenced acts stays at prior; 0.1/day drift cap (replayed day by day, stateless).
  Personality endorsement is excluded.
- **Reservations**: a funded bid reserves its cost on capacity lines (lanes, Juniper's cap) until
  `POST /physiology/bid/{want_id}/release` (the act closed) or expiry. Before the act opens, expiry is
  the earlier of one act-length (its expected seconds, 120 s .. 1 h) and the end of the want's window
  (Patch C3a: no act by then, no reservation); once the producer reports the open (durable runs: the
  first grant) it is one act-length from the open. Later
  bids must fit `forecast use + reservations + cost <= line`. In memory: a restart releases them.
  Tier 1 pays no lane price inside its reserved share (p75 of its 14-day hourly use, cap 50%).
- **Unpriced, said so**: heat, energy and claude are listed `unpriced:<constraint>` on every
  decision: per-act watt-hours and limit events are not measured yet (`ActCostV2.wh_by_host` null).
- **Never fail open**: no fresh body state (> `ORION_PHYSIOLOGY_BID_STALE_AFTER_SEC`), no act history
  or no families config is `physiology_unavailable`. The client (`orion/physiology/client.py`,
  0.5 s timeout) returns `None` on any failure and the caller keeps its own gate.

Reads: Redis `orion:physiology:latest`, and `grammar_acts` + `act_downstream_use` (30 days, cached
`ORION_PHYSIOLOGY_BID_HISTORY_TTL_SEC`). Debug: `GET /physiology/reservations`; `/health` ->
`physiology_market`. Off: `ORION_PHYSIOLOGY_BID_ENABLED=false` (the route answers 503). Read-only
sample against live prices: `scripts/analysis/sample_body_economy_bids.py`.

**The consent gate (2026-11-11, attention patch 3d).** Some acts change things that are hard to put back.
A want for such an act carries a consent request (`WantV1.consent_request`: family, verb, the act's
reversibility class, a payload digest, and the preconditions the producer attests). The market answers it
FIRST, before it reads the body state or the act history, with one pure function
(`orion/physiology/consent.py`) over `config/physiology/consent.yaml` (`ORION_PHYSIOLOGY_CONSENT_PATH`):

- `allow`: the want is priced as any want, and the verdict rides on the decision (`WantDecisionV1.consent`);
- `refuse` (irreversible, unknown and not reversible, a missing attest, no rules read) or `needs_consent`
  (the rule needs a standing consent Juniper has not merged): deferred, reason `consent`. The binding rule
  never admits it, not even at the window's end.

Standing consents are entries in `consent.yaml` that Juniper merges by PR (git history is the record). It
replaces only the retired policy rung's consent check; heat, arousal, caps, quiet hours, the Juniper-first
yield and the self-monitor gate stay where they are. A want with no consent request never reaches the gate.
Every verdict logs `physiology_consent verdict=... rule=... want=...`; `/health` -> `physiology_market`
shows `consent_rules`, `consent_standing`, `consent_error` and per-verdict counts. First consumer: dream
compaction apply (orion-dream `app/compaction_consent.py`). Eval: `evals/consent_gate_fixture.py`
(`--live` compares staged REM deltas to consent asks in `want_ledger`). A (family, verb) that has a rule must ask: a
want for it without a consent request is refused (`request_missing`). Deploy order: a decision that carries
`consent` needs rebuilt `extra=forbid` readers, so rebuild orion-sql-writer first, then this service, then
orion-dream (the only sender of consent requests).

**Exploration floor, weekly quota (Patch C3a).** Funded wants are counted from `want_ledger` (rolling 7
days, re-read with the history) plus this process's own funded decisions, deduplicated by
`decision_id`. A Tier 2 family under 5% of Tier 2's funded wants is funded when only the price said
defer; each floor fund raises its share, so the floor stops by itself. No ledger read yet: no floor.

**Prime's slow / quicken charges (2026-10-30, Orion Prime step 3).** Prime (orion-mind) can ask one
organ to slow down or quicken. Only a family that lists the signal under `receptors` in
`config/physiology/families.yaml` takes it; today that is curiosity investigate (the one family whose
bid binds), and it declares `slow` only. A quicken credit is never applied to a want the market values
at zero, and curiosity's only holds over the last week were exactly those, so a credit could not change
anything there yet. Prime writes the signal to Redis `orion:prime:regulate:<family>` (same Redis as the prices).
On every bid of that family the market reads it and, when it is in force, adds a temporary charge
(slow) or credit (quicken) of `magnitude x family_weight`, magnitude at most 0.5. So a want ranked
above the middle of its own family's history still beats it, and its hunger keeps rising while it
waits. The decision's `charge` and `net` include Prime's part (`cost_frac` and `prices_seen` do not). The exploration floor works as before, and a held run still goes at the end of its 6 h window.
The decision's `why` names the charge ("value 0.31 - lane charge 0.00 did not beat Prime's slow charge
0.54"); a deferral it causes has `binding_constraint = prime:slow`. After each bid the market writes
back what the charge did to that want (`held` / `outbid` / `released` / `unchanged` / `floor`) into the
same hash, tagged with the signal's id; Prime grades its signal from that record. To classify a bid the
market decides it both ways (the second time on a throwaway copy of its reservation book), so the
exploration floor and the capacity line count the same in both arms. A signal Prime held back
(`arm = control`) is read but never applied: the market records what it would have done. Code: `orion/physiology/prime_charge.py`,
`orion/physiology/choice.py` (`decide(..., prime=)`), `app/physiology_market.py`. Off:
`ORION_PHYSIOLOGY_PRIME_REGULATION=false` (the hash is never read). Replay of what a 0.5 slow charge
would have done to the last 7 days of real curiosity runs:
`services/orion-durable-runs/evals/prime_charge_replay.py --live --days 7`.

Since Orion Prime step 4 (2026-10-30) the dream cycle (`dream.cycle`) also declares `slow`. Its bids
come over HTTP from orion-dream, and this market reads Prime's signal for them the same way. The rule
that turns a decision into "go now" or "hold" is shared by both bid sites: `orion/physiology/binding.py`
(`app/run_bids.py` re-exports it). Dream's own bounds and replay: `services/orion-dream/app/physiology_bind.py`,
`services/orion-dream/evals/dream_binding_replay.py`.

### Personality values in the market (2026-11-13, personality rev 4 Move 2, BINDING, no shadow)

Orion's choices set what Orion values. `app/personality_values.py` (reads) + `orion/physiology/personality_values.py`
(care, holdback, runaway guard) + `orion/physiology/weights.py` (source (c)). Flag `PHYSIOLOGY_PERSONALITY_VALUES=on`
(`off` = kill switch: no read, no term, decided exactly as before).

- **Source (c), endorsement.** Family weights gain a fourth evidence rate: the share of REAL choice points
  (`personality_choice`; a "none" option was on offer, or the options spanned families) where Orion picked an
  option of the family. Lanes map to families: outreach -> `hub.outreach`, reading -> `hub.world_pulse_read`,
  curiosity -> `durable.curiosity_investigate`, prime -> `prime`; an option key `<family>::<id>`
  (`weights.endorsement_option_key`) overrides it. A curiosity prior menu (one family, no "none") is never a
  point. No point: no (c), never 0. Kappa 30 shrinkage as the other sources. Per family, (c) is correlated with
  (b) Juniper engagement on daily rates; r > 0.8 on >= 7 paired days drops (c) for that family. Logged each
  weight refresh: `personality_values_endorsement family=... rate=... corr_with_engagement=... excluded=...`.
- **Care.** `bid(..., object_text=...)` (run_bids passes the run's `brief.retrieval_query` or `brief.title`;
  `POST /physiology/bid?object_text=...` for HTTP producers, `orion.physiology.client.bid(object_text=...)`).
  care = clamp(max cos(text, tendency direction) x w/(w+10) x cap, 0, 1) over active `origin='choice'` tendencies;
  value = intensity x outcome x (1 + 0.5 x care) before the percentile. No text / no tendency / no match /
  embed failure: care exactly 0, reason on the decision.
- **Holdback.** 20% of want ids (sha256, deterministic) bid with care 0 and the weights WITHOUT (c). Binding.
- **Runaway guard.** A tendency matched by > 60% of shown options on each of the last 7 days has its care scaled
  by 0.6/share; logged `personality_values_tendency_capped`. Producers set `care_promoted` with
  `personality_values.care_promoted_keys(with, without, k)`, reading the "without" score off a decision with
  `personality_values.net_without_care(decision.warnings, decision.net)`.
- **Trace.** `want_ledger.decision.warnings`: `personality:arm=care|holdback`, `personality:care=<c>:tendency=<id>:cos=<x>`
  or `personality:care=0:reason=<why>`, `personality:endorsement=<rate>:n=<n>|none|excluded:<why>`,
  `personality:rank_without_care=`, `personality:net_without_care=`. `GET /physiology/reservations` -> `market.personality_values`.
- Replay (read-only): `python3 services/orion-durable-runs/evals/personality_values_replay.py --dsn ... --hours 72`.

### Admission bids (2026-10-27, body economy Patch C3a, SHADOW)

`app/run_bids.py`: a run in a want family (curiosity investigate, self-inquiry, self-sense eval,
journal compose -- the set `app/run_acts.py` opens acts for) bids in-process when `submit` accepts
it: `want_id` = run id, `object_ref` = `durable_run:<run_id>`, `correlation_id` = the run's
correlation id (the act's), window `latest` = the admission deadline. The bid is a background task:
`submit` returns its usual receipt whatever the market says. The bid reserves PENDING (runs wait
~1.5 h in the pool's queue): the first grant calls `Market.mark_opened`, which starts the
reservation, and the terminal calls `Market.release`. A grant or terminal that beats the background
bid is applied when the bid lands. A duplicate submit bids once. Log line:
`physiology_shadow_bid run=... corr=... decision_id=... obeyed=false`. Off:
`ORION_PHYSIOLOGY_SHADOW_BIDS_ENABLED=false`.

Disagreement report (read-only): `scripts/analysis/body_economy_shadow_vs_real.py --days 7` (the
four-cell shadow-vs-real table per family; `--source replay` re-decides the real admissions).

Live smoke (read-only, replays the node over stored ticks): `ORION_ANALYSIS_DSN=... python
scripts/analysis/smoke_body_economy_prices.py --hours 24`.

### Event wants and Prime's menu: `GET /physiology/event-wants` (2026-11-04, patch 2, read-only)

Attention as the want competition (docs/superpowers/specs/2026-11-04-attention-as-want-competition-design.md). Two producers post SHADOW **event wants** (`WantV1.event_ref`
+ `salience`): Hub reading (one per finished read) and sql-writer camera asks (one per open ask).

- **Event rank.** An event want ranks against the family's own past event wants (their recorded
  intensities, `want_ledger`, 30 days, first decision per want, read with the act history:
  `app/physiology_market.py` `EVENT_HISTORY_SQL`), not against the hunger-only act history. Under 10
  past events it is ranked as before (`value_basis` `event_cold:<basis>`); otherwise `event:<basis>`.
- **The menu.** `GET /physiology/event-wants?hours=24&limit=8` returns `FundedEventWantsV1`
  (`orion/schemas/physiology_want.py`): every event want whose LATEST decision is funded and whose event
  has not expired (window end, or an answered camera ask), highest net first, each labelled from its
  source row. Code: `app/event_wants_menu.py`. Reader: Orion Prime (`PRIME_CONCERNS_FROM_EVENT_WANTS`).
- **Replay.** `evals/event_wants_replay.py [--live --days 14]`: salience per source, the event rank vs
  the old rank, and how often Prime's frame would have had a funded event.

### The cooling world act: the market decides, the pool acts (2026-11-05, ladder retirement PR 5, BINDING)

Orion's one world act -- "hold back new background GPU work while the cabinet warms" (`shed_background_gpu`)
-- has its own loop here (`app/cooling_act.py`; pure rules `orion/autonomy/shed_act.py`). Until this PR its
only trigger was execution dispatch (an attention-workspace winner on the cabinet node); spec:
docs/superpowers/specs/2026-11-04-retire-proposal-policy-dispatch-rungs.md, "PR 5".

Every `ORION_COOLING_ACT_TICK_SEC` (60 s):

1. **Settle.** A treated shed still `shed_pending` settles from the pool's `gpu_pool_orion_shed` row, or as
   `settlement_timeout` at t0 + TTL + 300 s (`orion/autonomy/shed_settlement.py`). One `ActionOutcomeEmitV1`
   goes out, then the terminal is recorded on the episode. orion-sql-writer's scorer then scores it, unchanged.
2. **Eligible?** `orion/autonomy/self_shed.py` (cabinet elevated/hot/unknown and not critical, hardware-watch
   idle, background work present, nothing in flight), then the pool's mirrored caps (in flight, 15-min gap,
   1 h/day, a pool that refuses).
3. **The event.** The cabinet is *warming*: elevated or hot and rising >= `ORION_SHED_RISE_THRESHOLD_C` (0.5 C)
   in 15 minutes -- the same rise that made the cabinet node win attention. It posts an event want: family
   `self_regulate.shed` (Tier 1, a proposal), `event_ref` `cabinet_reading:athena:<UTC second>`, `salience` =
   the rise's strict-below percentile among the cabinet's own warming minutes of the last 30 days.
4. **Obey.** Funded: the 0.5 treated/control draw (hash of the episode id), the precommitted
   `substrate_world_action_episodes` row (eligibility + the want + the expected effect), then (treated) the one
   `orion:gpu_pool:shed:request`. Deferred: the same want bids again at the market's retry time, until its
   900 s event window ends. No answer (timeout, error, stale prices, no market): **no shed**.

Health: `/health` -> `cooling_act` (counts, the last tick's result, the open want). Kill:
`ORION_COOLING_ACT_ENABLED=false` + restart. The pool's own reflex and `GPU_POOL_ORION_SHED_ENABLED` are separate.
Replay: `evals/cooling_act_replay.py [--live --days 14]` runs this loop on real cabinet history with the real
market rule and the real scorer.

### Body-economy report: `GET /physiology/report` (2026-10-29, read-only)

One `PhysiologyReportV1` (`orion/schemas/physiology_report.py`, `schema_version: physiology.report.v1`)
over a window. Readers: the Hub `#body-economy` page (Phase 3) and the skill `skills.physiology.state.v1`
(Orion reads it as text: `python -P -m orion.dig body-economy` in chat and curiosity harness turns, or the
cortex-exec adapter). Code: `app/physiology_report.py`. No writes.

Query: `window=24h|7d|30d` (default 24h), `sections=` (comma list; default all), `family=`,
`outcome=funded|deferred`, `limit=` (decisions, 1..500, default 50). A bad value answers 422; the flag
off answers 503.

| key | source |
|---|---|
| `limits[]` (now + hourly level/price/bind state) | `physiology_state` (prices from 2026-10-27) |
| `wants{families, hourly}` | `want_ledger`, hand-run probes left out |
| `rhythm[]` (local day x hour) | `gpu_pool_events` lease closes, each lease spread over the hours it was held: background vs all lease-seconds |
| `money` | newest `energy` reading; `floor_usd` from its `why`; `orion_energy_hourly` per local day |
| `unit_costs[]` | tariff rate, `config/physiology/heat_sensitivity.yaml`, `energy_run_cost`, `grammar_acts.cost_json` (`wh_by_host`/`usd`/`heat_c_by_host` when filled; `basis` text alone is not measured; whole machine above the pinned rest from 2026-10-30, GPU above idle only before; one act is labelled an estimate, the family total is the honest number). Per host per UTC day (newest 3): "work not yet metered on <host>" from `orion_energy_attribution_hourly` (sql-writer): unattributed W·h above rest, with sponsored, charged-to-acts, coverage, and an OVER-ATTRIBUTED flag when an hour reads below 0. Per 1k tokens: per-act energy / that act's tokens in `llm_call_usage` joined on `correlation_id` (every recorder); totals count `recorder='gateway'` only. Table absent or empty -> `not_measured` with the reason |
| `waste` | lease closes whose reason is not `ok`/`completed`/`generated`/`text_done`/`operator:*`; the `released` twin of an expiry/abort is counted once |
| `queue_wait[]` | `waited_ms` on granted leases; family from `want_ledger` for durable runs, else the holder's service |
| `decisions[]` | `want_ledger`, newest first |
| `claude` | `dev_economics_ledger_log` (Claude Code sessions on athena, not Orion's harness) + the `claude` constraint |
| `vram[]` | `orion_biometrics_summary.measurements` `gpuN_mem_used_mb`/`gpuN_mem_total_mb`/`gpuN_watts`; card -> roles from `config/gpu_pool.yaml` |
| `processes[]` | the same newest biometrics row: `container_cpu_watts.<name>`, `container_cpu_watts_small`, `cpu_watts_host_other` (CPU watts above idle, PR #2977); top 12; `cores` is None (not published per container); `watts_at_wall_est` + `overhead_factor`/`overhead_basis` (see "Overhead" below) |
| `hosts[]` | what is spinning each machine (2026-10-30): the newest fresh biometrics row per host (`chassis_watts` from BMC or wall PDU, `gpu_watts_total`, `cpu_watts_total`, `fan_pct_max`, `gpuN_*`, `container_cpu_watts.*`) against Orion's body at rest. Rest is **pinned and dated**, never rolling (a rolling low would hide new always-on work such as the Zelda dojo): `config/physiology/rest_body.yaml` (`approved_rest_w`, else `rest_w`; path = sibling of `ORION_PHYSIOLOGY_HEAT_SENSITIVITY_PATH`, no new key), else defaults circe 541 W / athena 340 W / hecate 96 W (p05 of 2026-10-01..10-13). `other_w` = chassis − GPU − CPU (fans, PSU loss, memory, disks); each part and card is compared with its own p05 over the same dated window (cards: the trailing 7 days, because per-card watts began 2026-10-29; said in `missing`). `recent_floor_w` = trailing-7-day p05: above the pinned rest means new always-on draw. `usd_per_month_at_this_rate` = above-rest W x 730 h x the newest rate. Per card: `top_holders` = lease-seconds held per holder in the window (`gpu_pool_events.cards`; a durable run is named by its `durable_run_workflow.workflow`; time held, not time computing). `top_containers`: CPU watts above idle, sponsored work (rest_body `sponsored`, today `zelda-dojo-live-*` on athena) shown apart in `sponsored[]`. Container ids (circe's biometrics cannot read its docker root) become names from circe's container census on the `gpu-lane-controller@circe` heartbeat in Redis `equilibrium:state` (read-only). `unattributed_above_rest_w` (and `leased_unmetered_above_rest_w`, the part on GPU leases whose turn has no act) is the average watts of the newest complete hour per host in `orion_energy_attribution_hourly`, labelled by `attribution_hour_start` and `attribution_basis`. That hour lags now by design: the sql-writer writes an hour once it has ended plus 3 min, on a 10-min tick, so the newest `hour_start` is 1 h 3 min to about 2 h 13 min old (live 05:09Z: 03:00, written 05:00:54). Older than 3 h is said as stale, not shown. Asleep hosts are a `missing` reason. The pinned-window part rests are cached in-process for 6 h |
| `value_per_cost[]` | the market's own family-weight evidence (`Market.family_weights`) |
| `services` | mesh-guardian `stopped_container` cards in `notify_requests` (no recovery event: a listed service may be up again) |

A section is never silently empty: `missing[{section, reason}]` says why. One failing read empties only its
own section (`read failed: ...`). `timing_ms` has each read's time. Postgres does the hourly grouping, so
Python sees aggregated rows; reads run at most `ORION_PHYSIOLOGY_REPORT_CONCURRENCY` at once on the
shared pool; the same query is cached `ORION_PHYSIOLOGY_REPORT_CACHE_SEC`. Measured on the live DB
2026-10-29 (read-only, concurrency 3): 24h ~0.35 s, 7d ~0.8 s, 30d ~1.0 s (H5: under 2 s; the
slowest read is rhythm at 30 d, ~0.9 s).

**Overhead: one unit for every cost.** Every cost number means the whole machine at the wall ABOVE
REST, fans and PSU loss included: the unit the per-act meter charges (PR #2992). A card's watts, a
container's CPU watts and `energy_run_cost` (GPU above baseline) are components. Each is shown beside
`watts_at_wall_est` = its watts above idle x the host's `overhead_factor`
(`orion/energy/overhead.py`): a least-squares slope through zero of chassis above the pinned rest vs
GPU above its pinned p05 + CPU above `cpu_idle_watts`, over the trailing 14 days of hourly means of the
same biometrics readings (`HOSTS_OVERHEAD_SQL`, ~0.55 s, cached 6 h). Live 2026-10-30: circe 2.34
(R^2 0.90, 337 h), athena 1.27 (R^2 0.98). Fewer than 48 loaded hours, R^2 under 0.5 or k under 1:
#2992's circe fit (2.69, R^2 0.79), said in `overhead_basis` with `overhead_fitted=false`. Fields:
hosts `overhead_factor`/`overhead_basis`/`overhead_fitted`/`overhead_r2`/`overhead_hours`; cards
`above_idle_w`/`watts_at_wall_est`; containers and sponsored `watts_at_wall_est`; the run-cost unit
row (`one reverie_diffusion run (painting) on <node>`) has `wh`/`usd` at the wall (status `derived`)
and the GPU-only figure in `wh_component`. Painting acts carry no whole-machine cost yet (their act
and diffusion lease share no correlation id, PR #2977), so this row is the painting cost for now.
These wall figures are ESTIMATES: derived views of the one chassis reading, not new measurements.
k is an hours-average: fans and PSU loss follow load over minutes, so at one instant the parts'
estimates do not split the surplus (live 2026-10-30, two samples two minutes apart: this second's
ratio 1.23, then 6.11). Each host therefore also carries `load_above_idle_w`, `wall_est_parts_w`
(= load x k) and `overhead_now_ratio` (measured above rest / load now, the meter's per-second ratio),
and the dig text says so when the parts' sum and the measured surplus differ by more than 25%.
Through zero on purpose: like the meter, k charges all above-rest draw to load, including drift above
the pinned rest; the basis also gives the slope and intercept of a fit with an intercept (circe 2.05
x load + 76 W) so the drift is visible. Run costs are priced at the newest marginal rate, like acts. `hosts` adds four reads run side by side (2026-10-30, 30 d, first call: pinned parts ~1.2 s then cached, daily ~0.8 s, holders ~0.9 s, floor ~0.4 s; total report ~1.4 s). The newest tick older than 10 min is reported `stale`, not "now".
With `family=`, a `*` entry names the sections it does not narrow. A report with a failed read is not cached.
The concurrency limit is one semaphore shared by all requests; concurrent identical queries share one build.
`gpu_pool_events` has no persisted rollup; if 30 d grows past 2 s, that is the table to roll up.

Off: `ORION_PHYSIOLOGY_REPORT_ENABLED=false` (503; the skill then tells Orion the report is off).

## Runtime immune sweep: the `immune` node (2026-10-13, Temporal Self R4)

The thread runs `ingest -> regulate -> physiology -> immune -> chronicle -> done`. `immune` asks whether any of
Orion's own gauges has stopped telling the truth. It runs only while arousal reads `idle`, at most
once per `TEMPORAL_SELF_IMMUNE_INTERVAL_SEC` (6 h; the last run time is in the checkpoint and
carries over midnight). It reads 24 h of every field channel whose glossary entry carries
`semantics:` (`config/field/field_channel_glossary.v1.yaml`) from `substrate_field_state`, one
sample per minute (one read-only query, ~3 s live, 45 s statement timeout), and judges each with the
pure `orion/regulation/immune.py`: **constant**, **floor-locked**, **replayed** (same non-rest value
freshly re-stamped 30 times running), **decayed-to-zero** (exact powers of one ratio). Checks that
need a rest value read `semantics.rest_range` and say "cannot judge" without it. **As shipped,
the glossary has no `rest_range` yet** (a parallel patch adds it), so only decayed-to-zero is
live until it lands; `/health` -> `temporal_self.immune.coverage.rest_unjudged` counts the channels
still waiting on one. Field-channel reads carry no source row, so a replay frozen at exactly 0.0
or 1.0 is not judged (it cannot be told from a pinned real reading).

Each finding is one `immune_finding` row in `temporal_self_event` (id
`immune_finding:<metric>:<kind>:<day>`, so a later sweep the same day refreshes it);
`RegulationStateV1.immune_findings_open` counts the latest sweep's findings (`GET
/regulation/state`; left out of the Redis body until every `extra="forbid"` reader is rebuilt, see
`REDIS_PROJECTION_EXCLUDE`). The Orion's Day closed-day block renders the day's findings. Nothing
is ever fixed. A failure or timeout (90 s) is an `immune_failed` warning on the re-projected state;
a failed row write is `immune_write_failed` (the count still says what the sweep saw, the day block
shows only written rows); regulation and chronicle are unaffected. With arousal not `idle`
(including `ORION_REGULATION_AROUSAL_ENABLED=false`, which reads `unknown`) the sweep never runs and
`temporal_self.immune.skipped` says why. Kill switch: `TEMPORAL_SELF_IMMUNE_ENABLED=false`. Health: `/health` ->
`temporal_self.immune`. Live dry run: `docs/superpowers/evidence/2026-10-13-immune-sweep/`.

## Personality graph loop (rev 4, 2026-11-13: choices only)

What it does: every 10 minutes (and once at start) a reducer reads the choices Orion made since its
last read (what Orion picked at a choice point, and the options Orion turned down), turns each into a
direction ("this over that"), and folds the directions into a small weighted graph of tendencies
("dispositions"). A tendency backed by enough weighted choices becomes `active`; one that stops
firing fades and is retired. Rev 2 grouped embeddings of Orion's own prose; the result was job
boilerplate (live 2026-11-13: 59% reverie), so every prose source is deleted (kill means kill).

Spec: `docs/superpowers/specs/2026-10-16-orion-dynamic-personality-graph-design.md` (rev 4). Pure
reducer: `orion/personality/graph.py`. Code here: `app/personality_sources.py` (reads),
`app/personality_store.py` (writes), `app/personality_driver.py` (the tick and the bus intake).
This service is the single writer of `personality_choice`, `personality_disposition`,
`personality_evidence`, `personality_weight_history` and `personality_cursor`
(`services/orion-sql-db/manual_migration_personality_graph_v1.sql`, `..._naming_v1.sql` and
`..._choice_v1.sql`; apply them before deploy).

**Where choices come from.**

| Lane | Producer | How it arrives |
|---|---|---|
| `curiosity` | this service's curiosity sync: `curiosity_offer_decisions` JOIN `curiosity_run_outcomes` (read-only), one row per tested prior, alternatives = the shown priors not tested | inserted by the tick |
| `prime`, `outreach`, `reading`, `chat` | the act that made the pick (orion-mind Prime, Hub outreach, ...) | bus `orion:personality:choice` (`PersonalityChoiceV1`) |

**Publishing a choice (producers).** Build `orion.schemas.personality.PersonalityChoiceV1` and
publish it on `PERSONALITY_CHOICE_CHANNEL` (`orion:personality:choice`) with kind
`PERSONALITY_CHOICE_KIND` (`personality.choice.v1`). `choice_id` must be deterministic per decision
(e.g. `prime:<thought_id>`), so a re-publish is a no-op. Give every shown option (`key`, `text` as
Orion saw it, `market_rank`); set `declined_all=true` and leave `chosen` empty for "no move" / "say
nothing". A pick among one option, or a rule/cap/formula, is not a choice: the model refuses an
empty `alternatives`. The channel is pub/sub: a choice published while this service is down is lost.

Rules:

- **One choice, one evidence row** (`choice:<choice_id>`). Direction = unit(embed(chosen) -
  mean(embed(declined))); a `declined_all` choice uses "no move" / "say nothing" / "none of these"
  as the chosen side. Weight = 0.25 when Orion took the market's rank-1 option, else 1.0; halved when
  `care_promoted`; divided by `picks_in_decision`; a backlog choice is faded by its own age.
- **Labels are computed**: "<chosen gist> over <declined gist>, n of m" (m adds choices that went the
  other way). Never a prose excerpt, no model.
- **Backfill once, idempotently.** The curiosity sync starts at the epoch when it has no cursor:
  every stored curiosity choice lands in `personality_choice` on the first ticks (200 runs a tick).
  `choice_id` is deterministic, inserts are `ON CONFLICT DO NOTHING`, and a re-upserted outcome
  folds nothing twice.
- **Exact resume.** `personality_choice` is read by `(created_at, choice_id)` keyset after the
  `choice` cursor, up to now minus 120 s (and the Temporal Self watermark), 200 rows a tick.
  Evidence, nodes, history and cursors commit in one transaction.
- **A failed embed holds the choice cursor** at the previous row; after 3 failed ticks, or at once on
  an HTTP 4xx, the choice is consumed as skipped evidence. Option vectors are cached (4000 texts).
- **Only `origin='choice'` nodes** are loaded, folded, counted or shown. The rev 2 nodes stay in the
  table until `scripts/personality_rev3_retire_prose_nodes.py --apply` retires them (dry run by
  default; a production write Juniper approves separately).

Embeddings: `PERSONALITY_EMBED_URL` (vector-host `/embedding`, bge-large 1024-d, compute only;
both containers are on `app-net`), concurrency 4, timeout 15 s.

Routes: `GET /personality/frame` (`PersonalityFrameV1`), `GET /personality/dispositions?status=
active|candidate|retired|all&limit=50` (each with its last 60 weight-history points),
`GET /personality/evidence?disposition_id=&limit=50` (newest first). Each answers 503 when
`PERSONALITY_ENABLED=false` or the tables are missing. `/health` has a `personality` block (ticks,
last tick, last error, evidence last tick, choices read, curiosity rows synced, bus choices
inserted/invalid, held, nodes by status). Log lines: `personality_choice_received choice_id=...`,
`personality_tick {...}`.

Failure: a bad tick logs `personality_tick_failed error_type=...` and writes nothing; the service
keeps running. Kill switch: `PERSONALITY_ENABLED=false` (tables stay; bus choices are dropped with
`personality_choice_dropped`).

Tests: `tests/test_personality_loop.py` (fakes), `tests/test_personality_postgres.py` (real
Postgres, needs `ORION_ADMISSION_TEST_DSN`). Eval: `evals/personality_choice_replay.py` (read-only
replay of the live curiosity choices through the reducer; exits 1 if the data is degenerate).

### Orion names their own dispositions (2026-10-20)

Orion is shown one disposition at a time and says what it is: a name, "not me", or "the same as
this neighbor". This service picks what to show, and applies Orion's answer to the graph.

Spec: `docs/superpowers/specs/2026-10-17-personality-orion-names-dispositions-design.md` (decisions
2026-10-20). Needs `services/orion-sql-db/manual_migration_personality_naming_v1.sql` applied before
deploy (new `personality_disposition` columns and the `orion_disposition_naming` table).

- `GET /personality/naming/next` returns the next `DispositionNamingPacketV1` or 204. Due order
  (`orion/personality/naming.py`, `naming_due`): an active node never shown, strongest first; a node
  whose last answer was empty or refused, again after 3 days, at most 3 tries; a named or rejected node
  only when its evidence doubled or a new source now counts toward promotion, never sooner than 14
  days. Reading the packet marks nothing; the node counts as shown when Orion's answer row is applied.
- `GET /personality/naming?disposition_id=&limit=50` is the naming log, newest first.
- `/personality/dispositions` rows add `name`, `name_sentence`, `orion_verdict`, `merged_into`,
  `name_grounding_rank`, `name_grounding_margin`, `named_at`, `last_shown_at`, `shown_count`.
- Each tick reads `orion_disposition_naming` after the `naming` cursor (keyset `created_at,
  naming_id`, same read lag as the sources) and applies it after the fold, in the same transaction
  (`graph.apply_namings`). NAME sets the name only. REJECT marks the node; it keeps matching and
  fading, and leaves the frame. MERGE pools B into A only at centroid cosine >= 0.85
  (`merge_min_cosine`, set from P0 live cosines) and retires B with `merged_into`; otherwise
  `apply_status=merge_too_far`. No verdict adds a hit. Each applied row logs
  `personality_naming_applied naming=<id> verdict=<v> grounding_rank=<n> margin=<x> apply_status=<s>`
  after commit.
- Grounding: `name + " " + sentence` is embedded and ranked against every non-retired centroid.
  Recorded only, never a refusal. A failed embed holds the naming cursor for up to 3 ticks, then the
  name is applied without grounding.
- Own-name echo: evidence whose text contains the matched node's own current name still counts
  (`lane_counts`, `evidence_count`) but adds no weight, and is stored with `meta.cites_own_name=true`.
- Kill switch: `PERSONALITY_NAMING_APPLY_ENABLED=false` (answers wait in the table, the route answers 204).
- Roll back one merge: `scripts/personality_unmerge.py --disposition <B>` (dry run; `--apply` writes;
  approximate for A because the merge clamp is not reversible; stop the loop first).

Tests: `orion/personality/tests/test_naming.py`, `tests/test_personality_naming.py`, and the naming test
in `tests/test_personality_postgres.py`.

### Situation on every piece of evidence (2026-10-16)

Each new evidence row's `meta.situation` records the situation the row happened in, at the row's own time. It has three fields:

- `arousal`: the level from the latest Temporal Self `arousal_transition` at or before that time. It reads `unknown` when there is none, never a guessed `idle`.
- `day_phase`: the phase of Orion's local day.
- `company`: whether a Temporal Self `company` arc (someone at the room camera) was open.

This is the "when" half of the CAPS if-then signature, recorded so that situation subnodes can later be built from data. Nothing Orion reads uses it yet; only the Hub panel shows it (below). Attention is left out because it nearly always crowns a prediction-error node.

The lookup fails open. If the Temporal Self tables cannot be read, the evidence still folds without a situation, and the tick warns `situation_unavailable:<type>`.

Code: `orion/personality/situation.py` (pure) and `PersonalityStore.load_situation_inputs`.

`GET /personality/dispositions` gives each node a `situation_mix`: `{"n", "arousal": {level: count}, "day_phase": {phase: count}, "company": {"true": n, "false": n}}`. It counts that node's evidence rows by situation, in one grouped query (`SITUATION_MIX_SQL`) bounded to the returned node ids. Rows without a situation (all rows before 2026-10-16 23:36 UTC) are left out, and `n` counts only the rows that have one. A node with no stamped rows gets `n=0` and empty counts. It is observability only: weights, matching and the frame do not read it.

## Acts: want-family runs write to `grammar_acts` (2026-10-23, body economy Patch A5)

Every durable run Orion starts for itself leaves an efference copy in the act ledger, so the body
economy can grade each family and price it (`app/run_acts.py`). Flag:
`ORION_DURABLE_ACT_EMIT_ENABLED` (default `true`).

| family | workflow / line | actor | verb |
|---|---|---|---|
| curiosity investigate | `curiosity.investigate` / `investigate` | `durable.curiosity_investigate` | `infer` |
| self-inquiry | `curiosity.investigate` / `self_inquiry` | `durable.self_inquiry` | `ask` |
| self-sense eval | `self_sense_eval` | `durable.self_sense_eval` | `sense` |
| journal compose | `journal.compose` | `durable.journal_compose` | `make` |

- **Open:** when the run's FIRST pool grant is recorded (`run.resource_granted`). The queue wait is
  not the act. A run refused before any grant opens nothing. A restart or a later re-grant does not
  re-open (only the call that inserted the first grant row publishes).
- **Close:** in `_terminal`, with the actual terminal status: `completed` = 1, `failed` /
  `cancelled` = 0. Cost = ActCostV1 `seconds` (first grant -> terminal). The ledger fills GPU
  slot-seconds itself by joining `correlation_id` (= the hold lease's `turn_correlation_id`).
- **Never stuck open:** each act expects `artifact_produced` by a measured horizon (24 h curiosity
  lines, 12 h self-sense, 1 h journal or the run's own deadline + 600 s); the ledger expires an act
  past it at 0.
- **Excluded:** urgent curiosity incident runs (a sensor alarm forced them), reading.turn and
  reverie.visual (Hub / thought already emit their acts), dream.carry (dream producer), and upkeep
  workflows (episode distill, compaction, digest, orion_day letter, self_study.reflect).
- **Live replay (read-only):** `python evals/run_acts_live_replay.py --dsn <live dsn> --days 7`.

## Deploy order

Stage 4.5 is a cutover: follow `docs/runbooks/2026-09-25-gpu-pool-stage4-cutover.md` exactly
(wait for zero active legacy leases, freeze the old elastic decider, flip the circe controller
and the pool, then deploy this build). Consumers first: the gateway, cortex-exec, thought,
harness-governor, Hub and field-digester must already run stage 4.4 (they carry and honour the
hold ref); `CuriosityTurnRequestV1.gpu_lease` is `extra="forbid"` at Hub.

`compactor.digest` (2026-09-30) is additive on `extra="forbid"` contracts: deploy this service
before cortex-orch (orch submits the run and validates `DurableRunRequestV1`), and cortex-orch before
orion-actions (which settles schedule runs from the terminal row).

Set `HUB_CURIOSITY_DURABLE_ADMISSION_ENABLED=false` to retain the earlier durable
kickoff without resource admission. To return to Hub's direct in-process path,
set both that admission flag and `HUB_CURIOSITY_KICKOFF_VIA_CORTEX=false`.

`orion_day.letter` (additive `DurableWorkflowV1` value and journal `trigger_kind`/`source_kind`):
apply `manual_migration_orion_day_letter_v1.sql`, then deploy orion-sql-writer and
orion-actions (they parse `DurableRunStateV1` / `JournalEntryWriteV1`), orion-cortex-orch +
orion-cortex-exec (the two verbs and their budgets), this service, and only then the Hub build
that submits the workflow.

`situation.update` (2026-10-07, additive `DurableWorkflowV1` value): deploy orion-sql-writer
first (it validates `DurableRunStateV1` rows; an old one drops them), then this service. Hub and
orion-actions skip rows they cannot parse and filter by workflow, so their order does not matter.

## Checks

```bash
PYTHONPATH=. .venv/bin/python -m pytest services/orion-durable-runs/tests -q
python scripts/check_service_env_compose_parity.py orion-durable-runs
curl -fsS http://localhost:8124/health
```

## Admitted runs: one GPU pool hold per run (stage 4.5)

Spec: `docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md`
(Decision 1). The GPU pool is the only scheduler; this service no longer grants anything.

- **`resource_request`** asks the pool for a *hold* for the whole run
  (`orion.gpu_pool.client.acquire_hold`, `kind=hold`, holder `durable-runs:<run_id>`,
  request id `<run_id>:<seq>` -- idempotent, re-asked with the same id until that hold ends).
  Class comes from the run's route in `config/gpu_pool.yaml` `routes`; priority is the
  requirement's (`background`, or `urgent`); `requirements.minimum_context_tokens` rides as `min_ctx_tokens`.
  An unknown route fails the run with `gpu_pool_unknown_route:<route>` instead of waiting.
- **`resource_wait`** interrupts until the pool grants. A waiting run is woken by the pool's
  `granted` event for its holder on `orion:gpu_pool:event`; the missed-event fallback is one
  `status` read per `DURABLE_RUNS_HOLD_STATUS_POLL_SEC` (60 s).
- **Which pool refusals fail a run** (`app/pool_hold.py` `refusal_is_terminal`). Only a refusal
  that is a property of the run's own request: `deadline` (the hold's deadline is the run's, as
  `workflow_deadline`), `min_ctx_exceeds_class`, `backlog_max_age`, `replay_payload_too_large`,
  and `unknown_class` / `operator_only_class` *only when durable-runs' own `gpu_pool.yaml` agrees*.
  Everything else is the pool's trouble, not the run's -- an unreachable pool (RPC timeout), a
  version-skewed pool (`invalid:*`, `unknown_verb:*`; incident 2026-09-26, when an old pool
  refused the 4.5 request shape and 11 runs were failed), a config roll, any unclassified reason.
  Those keep the run `waiting_resource` under the same request id and ask again after
  `DURABLE_RUNS_POOL_RETRY_BASE_SEC * 2^(n-1)` s (capped at `DURABLE_RUNS_POOL_RETRY_MAX_SEC`),
  each one a `run.waiting_resource` event with `transient: true`, `pool_status`, `reason`,
  `refusals`, `retry_at`. A skew refusal to `status`/`heartbeat`/`release` is treated like an
  unanswered RPC (it says nothing about the hold). `backlogged` keeps waiting. A hold that was
  granted and later dead-lettered/aborted is ended and re-asked under a new request id.
- **Deploy order is not a boot gate.** durable-runs does not refuse to start or probe the pool
  first: the hold acquire *is* the probe, and a pool that is down, old, or mid-roll now only makes
  runs wait. A boot-time gate would add a way to hang the service without removing any failure.
- **Work nodes** (curiosity `harness_turn`, self-sense `ask_questions`, reflect `llm_call`) run
  under `execute`, which heartbeats the hold every `DURABLE_RUNS_LEASE_HEARTBEAT_SEC` (must be at
  most half of the pool's `hold_lease_ttl_sec`, checked at boot). A lost hold stops the turn
  (harness cancel) and the run waits for the SAME lease_id, which the pool re-queues.
- **Every LLM call of the run carries the hold's `GpuLeaseRefV1`** (`CuriosityTurnRequestV1.gpu_lease`;
  reflect's `options.gpu_lease`), so the gateway *attaches* it to the hold instead of queueing it
  behind the run. The hold's role (e.g. `agent-gpu2`) is never sent as a route or `assigned_lane`:
  held calls name the `agent` route.
- **Recall** (the pool wants the seat back) is honoured at the next node boundary, inside
  `hold_clawback_grace_sec`; the tail nodes need no GPU and continue without it. A work node still
  running when the grace runs out (gpu2's `max_hold_sec` seat limit, chat's owner reclaiming lent
  gpu0) is stopped when the pool aborts the hold: `HoldLost`, harness cancel, and the run waits for
  the SAME hold (any eligible role) and replays the node. **No `HoldLost` spends an attempt** in any
  admitted graph (2026-09-29; before, curiosity/reflect spent one of three per recall and self-sense
  failed on the first -- 19 live failures in three days). If the pool ended the hold instead
  (dead-lettered, released), the run asks afresh under a new request id. Only a refusal that is a
  property of the run (`deadline`, `pool_hold.refusal_is_terminal`) takes the failure path.
  `resource.lease_expired` records each take-back (reason `recall_grace_exceeded`). State
  `hold_takebacks` counts them; past `DURABLE_RUNS_HOLD_MAX_TAKEBACKS` (default 12, 0 = unbounded)
  the run fails with `hold_takeback_limit:<n>` so a step that never fits cannot replay forever.
- **Urgent preemption** (Plan 2, `docs/superpowers/plans/2026-09-28-urgent-curiosity-plan-2-pool-urgent.md`).
  The pool pauses a background/system hold for a waiting urgent run: recall, 5 s grace, then it
  puts the hold back in line in its original place (`queued`, reason `urgent_preempt`). The work
  node stops (`HoldPreempted`, a `HoldLost`: harness cancel), keeps the re-queued hold and replays
  on the next grant under the same lease_id -- never a failed attempt. A turn that fails on its
  own first (its next LLM call cannot attach to the aborted hold) is checked against the pool once:
  an exception in `execute`, a failed *result* in the reading / reflect / self-sense node
  (`AdmissionDeps.requeued`, which now covers any pool take-back, not only urgent). A hold recalled for urgent work *before* its step starts (execute's
  first beat, or `resource_wait`) is not released -- that would forfeit its place: the driver polls
  the pool (1 s) until the abort re-queues it, for at most grace + 3 s, then falls back to releasing
  it. Other recall reasons are released at once as before. `list_pending()` pages urgent rows
  first. Proof trace: `run.preempted` (lease_id, generation, lane) instead
  of `resource.lease_expired`. At a tail node boundary a preempted hold is simply ended (the tail
  needs no GPU). Reconcile drives urgent runs first and outside `MAX_CONCURRENT_DRIVERS` (4),
  capped at the pool's `defaults.urgent_max_concurrent`; `0` there drives urgent like background.
- **Restart**: the hold's ids live in the checkpoint. A restarted driver fences a turn still
  running under the same generation (harness cancel + `turn_fence` for a new turn identity) and
  replays it under the same hold; an expired hold is waited for by its lease_id.
- **Door-A**: `finish` with `reach_out` keeps the hold (finish detail `gpu_lease`), heartbeats it
  until Hub calls `POST /runs/{id}/release-outreach-lease` (a pool `release`) or
  `DURABLE_RUNS_OUTREACH_HOLD_MAX_SEC` passes; a restarted process adopts it from the outbox.
- Lifecycle events are unchanged for Hub's run views: `run.waiting_resource`,
  `run.resource_granted` + `run.lane_assigned` (detail `lane` = the hold's role),
  `run.started`, `resource.lease_released`, `resource.lease_expired`, `run.outreach_pending`
  (plus `run.preempted`, not yet rendered by Hub).
  `run.lane_swap_suppressed`, `run.resource_eligibility_expanded` and `resource.elastic_*` are
  no longer emitted.

Deleted in 4.5 (kill means kill, no fallback): the durable broker, lane policy and widening
(the old durable-admission `broker`/`policy`/`elastic` modules), `app/elastic_runtime.py`,
`/leases/validate`, `/admission`, `/elastic/status`, `/elastic/target`.

Deleted in 4.6: the durable lease token itself (`ResourceLeaseV1`, `X-Orion-Resource-Lease`,
`options.resource_lease`, the gateway's `LeaseGuard`). A run's GPU pool hold ref (`GpuLeaseRefV1`:
`X-Orion-Gpu-Lease`, `options.gpu_lease`) is the only run lease.

A resume that keeps failing is retried on the next reconcile tick, but once a run has at least
`DURABLE_RUNS_RESUME_MAX_FAILURES` (default 10) failures since its last real node progress AND
the first of them is `DURABLE_RUNS_RESUME_MIN_FAILURE_SPAN_SEC` (default 600) old, it is failed
terminally (`run.failed`, `error: "checkpoint_resume_failed: ..."`) and its hold is released.

Internal operator endpoints (host port 8124, container port 8121):

| Endpoint | Result |
| --- | --- |
| `POST /runs` | `DurableRunRequestV1` body; persisted receipt, HTTP 202 |
| `GET /runs/{run_id}` | Graph position, control, the hold (`hold`, granted `lease`, live `pool` status) and history |
| `POST /runs/{run_id}/pause` | Release the hold, stop the active attempt, retain checkpoint |
| `POST /runs/{run_id}/resume` | Continue checkpoint; cancellation stays terminal |
| `POST /runs/{run_id}/cancel` | Release the hold and durably cancel |
| `POST /runs/{run_id}/release-outreach-lease` | Hub finished Door-A composition: release the kept hold |

Deleted in 5.6 (GPU pool stage 5, Decision 5): the `/capacity` permit broker (`POST
/capacity/{acquire,renew,release}`, `GET /capacity`, `DURABLE_RUNS_CAPACITY_ENABLED`,
`DURABLE_RUNS_LEASE_SECONDS`, the `Capacity*V1` schemas and the capacity client). World-model takes a
pool `world` lease and the visual chain runs under its run's diffusion hold since 5.4;
`orion/gpu_pool/tests/test_stage5_4_no_capacity_callers.py` fails if a caller returns. The run
registry store moved to `orion/durable_runs/registry_store.py` (`DurableRunRegistryStore`). The four
dead tables (`durable_gateway_permits`, `durable_resource_demands`, `durable_resource_leases`,
`durable_elastic_slot`) are dropped by `services/orion-sql-db/manual_migration_gpu_pool_stage5_drop_legacy_tables.sql`,
run through `scripts/gpu_pool_stage5_snapshot_and_drop.sh` (snapshot first). `durable_admission_runs`
and `durable_resource_events` stay.

Run the real Postgres tests and evals with an explicitly disposable database (each creates a
fresh schema). They run the REAL GPU pool runtime in process (`tests/pool_fixture.py`):

```bash
python -m pip install -r services/orion-durable-runs/requirements.txt -r requirements-dev.txt -r services/orion-durable-runs/tests/requirements-acceptance.txt
ORION_ADMISSION_TEST_DSN=postgresql://user@127.0.0.1:55439/admission_test PYTHONPATH=. python -m pytest services/orion-durable-runs/tests -q
ORION_ADMISSION_TEST_DSN=postgresql://user@127.0.0.1:55439/admission_test PYTHONPATH=. python services/orion-durable-runs/evals/hold_fairness.py
```

`tests/test_durable_acceptance.py` exercises Cortex receipt, Postgres wait/restart, the pool's
grant event waking the run over the bus, Hub's real turn adapters and the gateway attaching
every call under the hold -- on the home agent card and on gpu2 loaded by the pool (fixture
actuator) -- through accepted drafts and conditional response repair. Model outputs, FCC
execution, llama.cpp servers and external knowledge reads are isolated fixtures; this is not
evidence of production cognition.

## Lent lane, alternatives, gpu2 elastic (retired in 4.5)

The `chat-burst` lane policy, policy-additive alternatives and the GPU2 elastic decider are gone.
Where a run runs is the pool's decision: its `agent` class may use `agent`, then `agent-gpu2`
when loaded (the pool loads it after a hold waits `swap.after_wait_sec`, 1200 s, with the thermal
and visual-baseline guards clear), then `chat` while gpu0 is lent. Unlending gpu0 recalls a hold
borrowing it (grace, then abort and re-queue). `ResourceRequirementV1.alternatives`,
`allow_elastic_activation`, `pinned_lane` and `operator_override` were deleted in 4.6
(`extra="forbid"`: a producer still sending them is refused). A duplicate receipt of a row stored
before 4.6 still ignores those keys when comparing (`IGNORED_ADMISSION_FIELDS`).

### Curiosity chronology (`curiosity.compaction`)

The broader dream/sleep arc admits closed weekly/monthly windows; this service owns
all generation. `prepare -> resource_request -> resource_wait -> compact (topics,
reduction, overall) -> finalize -> finish`. `compact` makes one cortex verb call under
`llm.route.agent` background admission and the run's GPU hold. Pool waits/recalls do
not spend model attempts. The shared `orion.curiosity.compaction_store` contract saves
each accepted answer before the node returns; replay reloads that cursor, including a
crash between artifact commit and LangGraph checkpoint. The checkpoint itself stays
small: source material is stored once in `curiosity_compaction`, not in every LangGraph
state. A SQL write checks the run generation to fence delayed old workers.

Each call sends a JSON schema for its answer as `structured_output_schema`
(`output_schema` in `orion/curiosity/compaction.py`; the gateway turns it into llama.cpp's
`response_format`): `record_topics` first, an object with one REQUIRED key per alias of the batch
(`s0`, `s1`, ...) and no other key, each mapped to a topic index; then at most 4 topics, body
`maxLength` 1200; the overall body `maxLength` 2400. The grammar therefore cannot skip or invent a
record (V1 had a free `refs` list per topic, and the model dropped refs on 13-record batches).
`record_answer` still checks every rule, and also that each written topic has a record. The
prompt asks for about 80-110 words per topic and 180-240 for the overall, leaving room under
the caps (live bodies run ~7 characters per word and overshoot the asked length); a body that
fills the cap without ending a sentence was cut by the grammar and is rejected.

The final journal contains content-derived topic summaries with dated sources plus an
overall synthesis. Missing/invented topic references, empty text and malformed JSON
are rejected (`ModelOutputRejected`: a bounded retry, each accepted batch already saved). Empty input skips the GPU. Finalization releases the hold before journal
publication; delivery failure retries without regeneration, and the dream-side
reconciler repairs missed journal delivery using the same ID. Every source survives.
See the dream README and the 2026-10-14 PR report for activation and migration order.

### Reading resonance

`reading.resonance` accepts `ReadingResonanceRunBriefV1` with frozen evidence and a background resource requirement. The checkpointed graph waits for a GPU hold, makes a tool-free gateway judgment, checks quotes/citations, stores `ReadingResonanceResultV1`, releases the hold and finishes. Invalid output is a terminal rejected receipt. Persistence retries reuse the saved answer; transport attempts are bounded. Hub must have bound the request in `reading_resonance` first; the first completed result wins.

The existing `reading.lifecycle.v1` rail carries resonance start/completion/failure. Apply `manual_migration_reading_resonance.sql` before deployment. Tests in `test_reading_resonance_pg.py` need `TEST_POSTGRES_DSN` pointing to an isolated test database; the runtime image gains no dependency. [Implementation and rollout report](../../docs/superpowers/pr-reports/2026-11-11-reading-resonance-runtime-pr.md).
