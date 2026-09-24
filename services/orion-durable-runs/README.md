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
whose only real benefit is GPU2 elastic-burst eligibility on the ONE LLM call.

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

## Deploy order

The operator templates select admitted Curiosity. Before restarting, apply both
admission migrations named below. Then restart this authority, every LLM Gateway
replica, Thought, governor, Cortex Exec, Cortex Orch, and Hub, in that order.
Thought must carry the owning lease into stance execution before Hub submits an
admitted study; an older Thought can otherwise block a study on its own reservation.
`orion-sql-writer` must already contain the durable-run state route/table.

Set `HUB_CURIOSITY_DURABLE_ADMISSION_ENABLED=false` to retain the earlier durable
kickoff without resource admission. To return to Hub's direct in-process path,
set both that admission flag and `HUB_CURIOSITY_KICKOFF_VIA_CORTEX=false`.

## Checks

```bash
PYTHONPATH=. .venv/bin/python -m pytest services/orion-durable-runs/tests -q
python scripts/check_service_env_compose_parity.py orion-durable-runs
curl -fsS http://localhost:8124/health
```

## Optional resource admission

The admission path extends this graph with persisted demand, resource-wait and
retry-wait boundaries. Waiting uses a LangGraph interrupt and releases the run
task. The legacy maximum-age sweep excludes admitted threads: their queue wait
has no deadline unless the request explicitly supplies one. Inference timeout
starts after a validated lease, independently of queue age and lease renewal.
For admitted turns, the Hub RPC permits at least the declared inference budget;
the admission runtime owns the actual inference/overall deadline. Replies must
match the expected kind, run and attempt correlation before becoming graph state.
Legacy turns retain their configured RPC timeout.

Apply `services/orion-sql-db/manual_migration_durable_resource_admission_v1.sql`
to the same Postgres database as the existing checkpointer before enabling
`DURABLE_RUNS_ADMISSION_ENABLED`. Admission tables are operator-managed; startup
fails if they are missing. LangGraph continues to manage its checkpoint tables.
No migration is applied to production by this patch.

Internal operator endpoints (host port 8124, container port 8121):

| Endpoint | Result |
| --- | --- |
| `POST /runs` | `DurableRunRequestV1` body; persisted receipt, HTTP 202 |
| `GET /runs/{run_id}` | Graph position, control, lease, lane decision and history |
| `POST /runs/{run_id}/pause` | Revoke lease, stop active attempt, retain checkpoint |
| `POST /runs/{run_id}/resume` | Continue checkpoint; cancellation stays terminal |
| `POST /runs/{run_id}/cancel` | Revoke lease and durably cancel |
| `GET /admission` | Queue/lease counts, ages, first-admission wait histogram and event counts |
| `POST /leases/validate` | Authoritative fencing validation used by Hub/Gateway |

These follow the existing internal unauthenticated service API boundary; keep
them on the trusted service network. Submission via Cortex remains the normal
Curiosity entry point. Retry an ambiguous receipt with the same request/run ID.

`DURABLE_RUNS_CAPACITY_ENABLED=true` enables the shared Gateway request authority
at `/capacity` in the operator template. Apply the additive
`manual_migration_gateway_capacity_v1.sql` first. Its acquire/renew/release APIs
share the broker's transaction so an ordinary request and a durable lease cannot
both win the same capacity. Capacity can operate while cognition admission is
off. The service intentionally fails startup if an enabled authority lacks its
tables. See [API, rollout and limits](../../docs/architecture/durable-gateway-capacity.md).

Defaults, lane declarations, shadow mode, activation order, recovery limits,
metrics provenance and the actual Hub/FCC/Exec execution path are documented in
[the ADR](../../docs/architecture/durable-resource-admission.md).

Run the real Postgres tests and separate evals with an explicitly disposable
database (each creates a fresh schema). The acceptance suite additionally needs
the test-only dependencies below:

```bash
python -m pip install -r services/orion-durable-runs/requirements.txt -r requirements-dev.txt -r services/orion-durable-runs/tests/requirements-acceptance.txt
ORION_ADMISSION_TEST_DSN=postgresql://user@127.0.0.1:55439/admission_test PYTHONPATH=. python -m pytest services/orion-durable-runs/tests -q
ORION_ADMISSION_TEST_DSN=postgresql://user@127.0.0.1:55439/admission_test PYTHONPATH=. python services/orion-durable-runs/evals/admission_fairness.py
ORION_ADMISSION_TEST_DSN=postgresql://user@127.0.0.1:55439/admission_test PYTHONPATH=. python services/orion-durable-runs/evals/gateway_capacity.py
```

The [connected acceptance contract](../../docs/architecture/durable-run-acceptance.md)
exercises Cortex receipt, Postgres wait/recovery, grant-driven wakeup, Hub's real
turn adapters and Gateway ownership through accepted drafts and conditional
response repair. It also kills a runner process while an independently held
backend permit drains. Model outputs, FCC execution and external knowledge reads
are explicit isolated fixtures; this is not evidence of production cognition.


## Lent lane: `chat-burst`

`DURABLE_RUNS_LANE_POLICY_JSON` declares `chat-burst` as `compatible_with: ["agent"]`, so every
new `agent`-preferring run derives it as an alternative at submission. It needs no
`allow_elastic_activation`: nothing is physically borrowed. Eligibility follows the gateway
catalog: while the Hub gate is closed the gateway reports the route `operator_closed`, which
`refresh_lanes` maps to `healthy=False` (`health_unknown_or_unavailable` in the run's
`suppressed` map). Runs already queued before this policy change keep their frozen
alternatives and will not widen onto it. A run first granted `chat-burst` stays pinned to it
(`run_assignment_locked`). Closing the gate mid-run makes the run's next gateway call fail
(`route_operator_closed`), which `admitted_graph.py` records as one failed attempt: the lease
is released, the partial FCC turn is lost, and the run waits on the closed lane for its next
attempt (up to `DURABLE_RUNS_RETRY_MAX_ATTEMPTS`, then `failed`). Prefer closing the gate when
no chat-burst lease is active (`GET /admission` shows active leases). Follow-up: exempt
`route_operator_closed` from the attempt count.


## Alternatives are policy-additive

A demand's `alternatives` are frozen at submission (a duplicate receipt can never shrink a
demand). On every broker tick they are additionally unioned with whatever the current
`DURABLE_RUNS_LANE_POLICY_JSON` declares `compatible_with` the run's preferred lane
(`orion/durable_admission/policy.py::widen_alternatives`), so a lane declared after a run was
queued still reaches it. The stored row is not rewritten; the wider list shows in the run's
per-tick `admission.eligible_lanes`. Live 2026-09-22: 7 runs (oldest 14 h) sat with only
`agent-burst` while the newly opened `chat-burst` lane idled, which is what this closes.


## Optional GPU2 elastic admission

GPU2 diffusion/agent-burst borrowing is additive and defaults off. See the
[ownership ADR](../../docs/architecture/gpu2-elastic-admission.md),
[pre-edit repository/live evidence](../../docs/architecture/gpu2-elastic-evidence.md),
and [consumer-first rollout and rollback](../../docs/runbooks/gpu2-elastic-admission.md)
for this service's exact flags, HTTP contracts and operator commands.
No production env sync, migration, GPU transition or deployment was performed.

GPU2 control endpoints use the existing internal service/tailnet boundary without
bearer tokens. Durable intent, generation fencing, lease/permit protection, and
atomic diffusion draining remain enforced. The existing GPU1 API is unchanged.
