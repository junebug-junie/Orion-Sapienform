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
- **Hold:** route `memory_distill` (`config/gpu_pool.yaml`: agent class, the 27B lanes) at
  `priority: system`. System runs have their own driver slot (`MAX_CONCURRENT_SYSTEM_DRIVERS=1`), so a
  distill run is not stuck behind four background turns, and the pool ranks it above background holds.
- **distill:** ONE direct LLM gateway call (`orion:exec:request:LLMGatewayService`) with `options.gpu_lease`;
  not through cortex-orch (its recall step would mix retrieved memories into the evidence). JSON-object output,
  thinking off. An unparseable answer is a bounded, backed-off attempt.
- **persist:** releases the hold, validates deterministically (`orion/memory/episode/validate.py`: every quote
  checked against the turn's FULL text; voice downgraded when its evidence does not support it; internal channels
  never labelled as Juniper's words), then writes `episode_memory*`, `memory_tension_shadow` and
  `episode_distill_run` in one idempotent transaction. Rejected candidates go only to `episode_memory_event`.
- **Nothing live reads these tables.** The daily old-vs-new report (orion-memory-consolidation) does.
- **Env:** `MEMORY_EPISODE_WRITER_ENABLED` (kill switch), `CHANNEL_MEMORY_EPISODE_CLOSED`,
  `MEMORY_EPISODE_DISTILL_ROUTE`, `MEMORY_EPISODE_DISTILL_TIMEOUT_SEC` (<= 900), `MEMORY_EPISODE_DISTILL_MAX_TOKENS`,
  `MEMORY_EPISODE_DISTILL_DEADLINE_HOURS` (< 24), `CHANNEL_LLM_INTAKE`.
- **Deploy order:** apply `manual_migration_episode_memory_v1.sql`; deploy orion-sql-writer (parses
  `DurableRunStateV1` with the new workflow), orion-llm-gateway and orion-gpu-pool (they read the new
  `memory_distill` route from `config/gpu_pool.yaml`), then this service. orion-memory-consolidation (the
  producer of the close event) can go any time after.

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

### Every admitted terminal reaches `orion:durable:run:state` (2026-09-28)

`completed`, `failed` and `cancelled` admitted runs are all published as a
`DurableRunStateV1` (node `finish`, `failed`, `finish`) from the same outbox row as
their `run.<status>` lifecycle event (`entry_id = <run>:terminal:<status>:state`,
acked only after both publishes, deduped by sql-writer). Before this only
`completed` was, so a waiter such as cortex-exec's self-study reflect never saw an
admitted failure end the run.

### Admitted Orion's Day letter (`orion_day.letter`, 2026-09-30)

Orion's daily note about yesterday (America/Denver), for the 08:30 "Orion's Day" email.
Contract: `orion/schemas/orion_day.py`; Hub gathers the day and builds the brief with
`orion/orion_day/` (read-only SQL + a deterministic ~70k-token budget) and submits an
admitted request (`llm.route.agent`, background, `deadline_at` = end of the next local day).
Graph (`app/orion_day_graph.py`):

    resource_request -> resource_wait -> write_note -> write_carry_forward -> persist -> finish

- Two separate cortex verbs under the run's hold (`options.gpu_lease`, route `agent`):
  `orion_day_note_v1` (long first-person note, plain markdown; its prompt has no
  future-curiosity instruction) and `orion_day_carry_forward_v1` (threads for future curiosity,
  given the digest and the finished note). Each text is checkpointed; a restart never
  regenerates one. A lost/recalled hold replays the node (not an attempt). A transport error,
  or a completion `runner.strict_final_text` refuses (empty final text -- never the reasoning
  fields --, error text framed as prose, `finish_reason=length`) or below the node's floor (note
  < 400 chars; carry-forward with no list item) is one attempt
  (`DURABLE_RUNS_RETRY_MAX_ATTEMPTS`), and the next one waits out a backoff in `retry_wait`.
- The hold asks for `requirements.minimum_context_tokens` = digest + note + carry-forward, so a
  heavy day is never placed on the 65,536-token chat card (live `/props` 2026-09-30) when the
  pool spills agent work there.
- The hold is released as soon as the carry-forward text is checkpointed.
- `persist` (no GPU): `INSERT INTO orion_day_letter ... ON CONFLICT (letter_date) DO NOTHING`
  (`app/orion_day_store.py`, migration `services/orion-sql-db/manual_migration_orion_day_letter_v1.sql`),
  then publishes the STORED row's NOTE as `journal.entry.write` (id `uuid5(letter_date)`, the
  row's `created_at`, the writing run as correlation id, `trigger_kind=orion_day_letter`,
  journal email disabled: Hub sends the styled email) -- identical from every run, so a later
  run heals a journal the first never got out. An insert failure backs off and eventually fails
  the run; once the row exists the journal never fails it (own retry budget, then
  `journal_published=false`).
- The checkpoint keeps only a slim brief; the full brief (material + digest, ~1 MB on a heavy
  day) is read from `durable_admission_runs.request`, so the resume sweep's checkpoint scan
  does not load it once per checkpoint.
- Finish detail: `line=orion_day`, `letter_date`, `persisted`, `persist_outcome`
  (`written` / `already_written` -- key on this, not on `completed`), `existing_run_id`,
  `journal_entry_id`, `journal_published`, text lengths, `carry_forward_refs`, `llm_attempts`.
  The status API adds an `orion_day` block.
- A timed-out verb call is not cancelled at cortex (no harness turn to cancel): the RPC gives up
  30 s before the node's budget and the hold is released, but the model may finish that
  generation on the card. It is the same exposure `self_study.reflect` has.

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
