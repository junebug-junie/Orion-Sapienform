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

## Deploy order (consumer-first, learned the hard way on 2026-09-06)

1. `orion-sql-writer` (new route/table + the `durable_run` value on `AttentionSchemaV1.process`)
2. this service
3. `orion-cortex-orch` (dispatch branch)
4. `orion-hub` with `HUB_CURIOSITY_KICKOFF_VIA_CORTEX=true`

`HUB_CURIOSITY_KICKOFF_VIA_CORTEX=false` (the default) keeps Hub's direct path exactly as
before; this service then receives nothing.

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

Defaults, lane declarations, shadow mode, activation order, recovery limits,
metrics provenance and the actual Hub/FCC/Exec execution path are documented in
[the ADR](../../docs/architecture/durable-resource-admission.md).

Run the real Postgres tests and the separate fairness eval with an explicitly
disposable database (each creates a fresh schema):

```bash
ORION_ADMISSION_TEST_DSN=postgresql://user@127.0.0.1:55439/admission_test PYTHONPATH=. python -m pytest services/orion-durable-runs/tests -q
ORION_ADMISSION_TEST_DSN=postgresql://user@127.0.0.1:55439/admission_test PYTHONPATH=. python services/orion-durable-runs/evals/admission_fairness.py
```
