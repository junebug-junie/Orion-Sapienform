# orion-execution-dispatch-runtime

Layer 9 of the Orion cognition substrate: converts `PolicyDecisionFrameV1` + `ProposalFrameV1` + `SelfStateV1` into `ExecutionDispatchFrameV1` envelopes.

## Safety (v1: build, no send)

- Default mode: `EXECUTION_DISPATCH_MODE=dry_run`
- No bus publish and no cortex-exec calls in the default worker path
- Mutating dispatch is disabled in policy config (`allow_mutating_dispatch: false`)

## Real sends (P1: the motor nerve)

Real sends require **both** gates open:

1. `services/orion-execution-dispatch-runtime/.env`: `EXECUTION_DISPATCH_MODE=dispatch_read_only`
2. `config/execution_dispatch/execution_dispatch_policy.v1.yaml`: `mode.allow_dispatch_read_only: true`
   (this is the shipped default as of P1 — the runtime's own env mode is what actually gates
   live traffic; the policy flag alone does not turn on sending)

When both are open, the worker sends `prepared_for_dispatch` candidates to `orion-cortex-exec`
over `orion:cortex:exec:request:background` (via `orion.execution_dispatch.cortex_client
.ExecutionDispatchCortexClient`, one real RPC per candidate, bounded by
`EXECUTION_DISPATCH_RPC_TIMEOUT_SEC`), persists the result to `substrate_dispatch_results`, and
promotes the candidate to a real, evidenced `dispatched` status.

**Budgets**, both enforced per tick before any send happens:
- `config/execution_dispatch/execution_dispatch_policy.v1.yaml`'s `limits.max_dispatches_per_tick`
- `ORION_DISPATCH_MAX_PER_DAY` (rolling UTC calendar day, counted against `substrate_dispatch_results`)

**Theater tripwire**: if more than half of the trailing 10 real results have `status="empty"`
(a real send that produced no usable observation), the worker stops sending for the rest of
its process lifetime — visible via `GET /latest`'s `theater_tripwire_active` field and one
`orion-notify` warning event on the transition into tripped. Re-arm requires a restart; it does
not self-clear, by design (a self-clearing tripwire could resume sending on a coincidentally
non-empty sample without anyone deciding that was safe).

**Idempotency**: `dispatch_id` is deterministic per proposal+policy, so if this process dies
between a successful send and the frame being persisted, the next tick's rebuild of the same
candidate replays the stored `substrate_dispatch_results` row instead of resending — a real
cortex-exec call never fires twice for the same candidate.

**Rollback**: set `EXECUTION_DISPATCH_MODE=dry_run` and restart this one container. Single kill
switch for all real sending.

## Status vocabulary

`ExecutionDispatchCandidateV1.dispatch_status`:

- `prepared`, `dry_run`, `blocked`, `skipped` — no send involved.
- `prepared_for_dispatch` — cleared every gate for `dispatch_read_only` mode; the request
  envelope is built. Terminal state whenever real sending is off, or once per-tick/daily
  budgets are exhausted for this tick.
- `dispatched` — a real, evidenced send attempt. `ExecutionDispatchCandidateV1` enforces this
  at the schema level: `dispatch_status="dispatched"` requires `dispatched_at` plus one of
  `result_ref` (a `substrate_dispatch_results.result_id`) or `dispatch_error`.

## Prerequisites

1. `substrate_policy_decision_frames` populated (`orion-policy-runtime`, port 8120)
2. `substrate_proposal_frames` and `substrate_self_state` from Layers 7–6
3. Apply migrations:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_execution_dispatch_frame_v1.sql
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_substrate_dispatch_results_v1.sql
```

## Run

```bash
cd services/orion-execution-dispatch-runtime
cp -n .env_example .env
docker compose up -d --build
```

## Debug

- `GET http://localhost:8121/health`
- `GET http://localhost:8121/latest` (includes `theater_tripwire_active`)
- Hub: `GET http://localhost:8080/api/substrate/execution-dispatch/latest`

## Smoke

```bash
./scripts/smoke_execution_dispatch_v1.sh
```
