# orion-execution-dispatch-runtime

Layer 9 of the Orion cognition substrate: converts `PolicyDecisionFrameV1` + `ProposalFrameV1` into `ExecutionDispatchFrameV1` envelopes. `field_tick_id` is carried straight off `PolicyDecisionFrameV1.source_field_tick_id` -- 2026-07-22 (SelfStateV1 burn), no separate `SelfStateV1` dependency.

## Safety (v1: build, no send)

- Default mode: `EXECUTION_DISPATCH_MODE=dry_run`
- No bus publish and no cortex-exec calls in the default worker path
- Mutating dispatch is disabled in policy config (`allow_mutating_dispatch: false`)

Separately from dispatch traffic, this service always publishes a bus-native `SystemHealthV1`
heartbeat to `orion:system:health` every `HEARTBEAT_INTERVAL_SEC` (default 10s), on its own
independent bus connection -- process liveness telemetry, not dispatch behavior, and unaffected
by `EXECUTION_DISPATCH_MODE`.

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
- `ORION_DISPATCH_MAX_RISK_PER_DAY` (rolling UTC calendar day) -- a real cumulative risk-score
  budget, not a blind action count. Replaces the old `ORION_DISPATCH_MAX_PER_DAY` (2026-07-26):
  that was a hand-picked round number (checked-in 24, live-drifted to an unexplained 88) never
  validated against real behavior -- exactly the kind of un-measured hand-authored proxy the
  Sentience Striving Program exists to replace. Each dispatched candidate already carries a
  real, already-computed `risk_score` ([0,1]); the budget is spent against the sum of those
  scores (`ExecutionDispatchRuntimeStore.sum_risk_dispatched_today()`, reading
  `dispatched_candidates` off `substrate_execution_dispatch_frames`, not
  `substrate_dispatch_results`), so five trivial inspects no longer cost the same as five
  genuinely higher-risk candidates. The default (10.0) is anchored to real observed data (the
  first day this pipeline dispatched successfully spent a real cumulative risk of 4.4 across 88
  dispatches) rather than a fresh guess, but is still a disclosed starting judgment call --
  expect it to need re-derivation as more real history accumulates across a wider risk_score mix.
  **2026-07-28: `ORION_DISPATCH_RISK_CAP_ADVISORY_ONLY=true` (the default) means this cap does
  not actually block sends** -- the "real anchoring" only ever produced a multiplier (2x one
  day's total), not a derived ceiling, and every real candidate observed so far has had an
  identical `risk_score=0.05` (no real variance yet to derive one from). Reaching the cap logs
  `execution_dispatch_risk_budget_status ... cap_reached=true` and dispatch proceeds anyway
  (still bounded by `max_dispatches_per_tick`); set `ORION_DISPATCH_RISK_CAP_ADVISORY_ONLY=false`
  once enough real `risk_score` variance exists to justify actually enforcing a number.
  **Also fixed 2026-07-28**: `docker-compose.yml` had never been updated for the 2026-07-26
  rename -- it was still passing through the dead `ORION_DISPATCH_MAX_PER_DAY` and never passed
  `ORION_DISPATCH_MAX_RISK_PER_DAY` at all, so the live container was silently running on the
  Settings class default rather than anything the compose file actually wired.

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

## Experience loop (P2)

Every real send (success, empty observation, or RPC failure) also publishes an
`ActionOutcomeEmitV1` event onto `orion:autonomy:action:outcome`
(`BUS_ACTION_OUTCOME_OUT`, default `orion:autonomy:action:outcome`) — the same
always-on route `orion-spark-concept-induction` already produces onto for
curiosity-fetch outcomes, consumed by `orion-sql-writer` into the durable
`action_outcomes` table. `subject="orion"` (self-directed action, never
relationship-scoped). This is how a real Layer 9 dispatch becomes something
`load_action_outcomes()` — and therefore chat-turn stance context — can see;
see `services/orion-cortex-exec/README.md` for the read side.

The idempotent-replay path (see Idempotency above) re-emits on every replay,
not just the first attempt: `action_outcomes.action_id` is the SQL primary
key and sql-writer's route upserts by `merge()`, so a repeat emit for the
same `dispatch_id` overwrites the same row rather than duplicating it — this
is what makes replay-safe re-emission correct instead of risky.

A publish failure here is caught and logged, never raises out of the tick —
`substrate_dispatch_results` already durably recorded the result before the
emit is attempted, so an unreachable bus loses only chat-visible narration,
never the underlying record.

**`surprise` is a real signal, not the honest placeholder it started as
(2026-07-13).** Every other `ActionOutcomeEmitV1` producer in the repo
(`orion/autonomy/episode_fetch.py`, `policy_act.py`, `curiosity_reuse.py`)
still emits a binary success/fail proxy — see
`orion/autonomy/models.py::ActionOutcomeRefV1`'s docstring. This service is
the one exception: `surprise` is read live off `substrate_field_state`'s
`node:substrate.bus_synaptic` node (`ExecutionDispatchRuntimeStore.
latest_bus_synaptic_prediction_error()`), i.e.
`bus_synaptic_prediction_error()` (`orion/substrate/prediction_error.py`,
already generic across the whole bus mesh, already fixed once for a
calm-floor bias in PR #1391). Falls back to `0.0` if the node hasn't been
written yet, the field is older than `_BUS_SYNAPTIC_STALENESS_HORIZON_SEC`
(reuses `PressureConfig().prediction_error_decay_horizon_seconds`, 30 min —
`prediction_error` is a deliberately undecayed raw snapshot, see
`services/orion-field-digester/app/digestion/decay.py`, so a frozen value
from a stalled upstream tick must not be presented as live), or the read
fails — fail-open, never blocks emitting the outcome itself. See
`docs/superpowers/specs/2026-07-13-autonomy-experience-loop-p2-design.md`
(original placeholder) and
`docs/superpowers/specs/2026-07-26-transport-domain-retirement-bus-synaptic-successor-design.md`
(why `bus_synaptic` is the domain to build on) for the full history.

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
2. `substrate_proposal_frames` from Layer 7
3. Apply migrations:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_execution_dispatch_frame_v1.sql
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_execution_dispatch_frame_v2_drop_self_state.sql
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
- Hub: `GET http://localhost:8080/api/substrate/execution-dispatch/latest` (P6:
  also includes `status_summary` -- `dispatched_count` / `prepared_for_dispatch_count`
  / `dry_run_count`, derived purely from the frame's own candidate statuses.
  Does not surface `theater_tripwire_active`; that's in-process state on the
  execution-dispatch-runtime service itself, a different process than the hub
  route serving this summary.)

## Smoke

```bash
./scripts/smoke_execution_dispatch_v1.sh
```
