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

Set `HUB_CURIOSITY_DURABLE_ADMISSION_ENABLED=false` to retain the earlier durable
kickoff without resource admission. To return to Hub's direct in-process path,
set both that admission flag and `HUB_CURIOSITY_KICKOFF_VIA_CORTEX=false`.

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
  requirement's (`background`); `requirements.minimum_context_tokens` rides as `min_ctx_tokens`.
  An unknown route fails the run with `gpu_pool_unknown_route:<route>` instead of waiting.
- **`resource_wait`** interrupts until the pool grants. A waiting run is woken by the pool's
  `granted` event for its holder on `orion:gpu_pool:event`; the missed-event fallback is one
  `status` read per `DURABLE_RUNS_HOLD_STATUS_POLL_SEC` (60 s). A hold the pool refuses
  (dead-lettered, unavailable) fails the run with the pool's reason.
- **Work nodes** (curiosity `harness_turn`, self-sense `ask_questions`, reflect `llm_call`) run
  under `execute`, which heartbeats the hold every `DURABLE_RUNS_LEASE_HEARTBEAT_SEC` (must be at
  most half of the pool's `hold_lease_ttl_sec`, checked at boot). A lost hold stops the turn
  (harness cancel) and the run waits for the SAME lease_id, which the pool re-queues.
- **Every LLM call of the run carries the hold's `GpuLeaseRefV1`** (`CuriosityTurnRequestV1.gpu_lease`;
  reflect's `options.gpu_lease`), so the gateway *attaches* it to the hold instead of queueing it
  behind the run. The hold's role (e.g. `agent-gpu2`) is never sent as a route or `assigned_lane`:
  held calls name the `agent` route.
- **Recall** (the pool wants the seat back) is honoured at the next node boundary, inside
  `hold_clawback_grace_sec`; the tail nodes need no GPU and continue without it.
- **Restart**: the hold's ids live in the checkpoint. A restarted driver fences a turn still
  running under the same generation (harness cancel + `turn_fence` for a new turn identity) and
  replays it under the same hold; an expired hold is waited for by its lease_id.
- **Door-A**: `finish` with `reach_out` keeps the hold (finish detail `gpu_lease`), heartbeats it
  until Hub calls `POST /runs/{id}/release-outreach-lease` (a pool `release`) or
  `DURABLE_RUNS_OUTREACH_HOLD_MAX_SEC` passes; a restarted process adopts it from the outbox.
- Lifecycle events are unchanged for Hub's run views: `run.waiting_resource`,
  `run.resource_granted` + `run.lane_assigned` (detail `lane` = the hold's role),
  `run.started`, `resource.lease_released`, `resource.lease_expired`, `run.outreach_pending`.
  `run.lane_swap_suppressed`, `run.resource_eligibility_expanded` and `resource.elastic_*` are
  no longer emitted.

Deleted in 4.5 (kill means kill, no fallback): the durable broker, lane policy and widening
(`orion/durable_admission/{broker,policy,elastic}.py`), `app/elastic_runtime.py`,
`/leases/validate`, `/admission`, `/elastic/status`, `/elastic/target`. The frozen tables
`durable_resource_demands` / `durable_resource_leases` / `durable_elastic_slot` get no new rows;
they stay only because `/capacity` joins them until stage 5.

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

`DURABLE_RUNS_CAPACITY_ENABLED=true` keeps the world-model / visual-chain permit authority at
`/capacity` (NOT durable runs; stage 5 moves those onto pool leases). Since 4.5 it is built with
`reserve_waiting=False`: frozen pending demands no longer reserve a backend. An active legacy
durable lease still fences its backend, which is why the cutover waits for zero of them.

Run the real Postgres tests and evals with an explicitly disposable database (each creates a
fresh schema). They run the REAL GPU pool runtime in process (`tests/pool_fixture.py`):

```bash
python -m pip install -r services/orion-durable-runs/requirements.txt -r requirements-dev.txt -r services/orion-durable-runs/tests/requirements-acceptance.txt
ORION_ADMISSION_TEST_DSN=postgresql://user@127.0.0.1:55439/admission_test PYTHONPATH=. python -m pytest services/orion-durable-runs/tests -q
ORION_ADMISSION_TEST_DSN=postgresql://user@127.0.0.1:55439/admission_test PYTHONPATH=. python services/orion-durable-runs/evals/hold_fairness.py
ORION_ADMISSION_TEST_DSN=postgresql://user@127.0.0.1:55439/admission_test PYTHONPATH=. python services/orion-durable-runs/evals/gateway_capacity.py
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
`allow_elastic_activation`, `pinned_lane` and `operator_override` are accepted and ignored until
producers stop sending them (PR 4.6).
