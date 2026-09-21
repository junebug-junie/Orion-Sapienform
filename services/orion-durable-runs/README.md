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
(`orion/schemas/durable_run.py`, currently `Literal["curiosity.investigate"]` --
deliberately not widened by this patch), and its own request/brief shape if it doesn't
fit `CuriosityRunBriefV1`.

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
