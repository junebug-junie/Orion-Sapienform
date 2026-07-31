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
.ExecutionDispatchCortexClient`, bounded by `EXECUTION_DISPATCH_RPC_TIMEOUT_SEC`), persists the
result to `substrate_dispatch_results`, and promotes the candidate to a real, evidenced
`dispatched` status.

**Concurrent sends (2026-07-31, Part 2 of docs/superpowers/specs/2026-07-30-execution-dispatch-
staleness-discard-design.md)**: up to `limits.max_dispatches_per_tick` (now **5**, was 1) real
cortex-exec RPCs per tick run concurrently via `asyncio.gather(..., return_exceptions=True)` in
`_send_prepared_candidates`, not sequentially. Fixes a real, measured steady-state gap: with
production of new proposals holding around ~12-17/min and each real dispatch bound by a ~7-13s
synchronous RPC, sequential (`max_dispatches_per_tick=1`) dispatch structurally could never process
more than ~4-5/min — roughly 65-70% of every real, current proposal was going stale and getting
discarded, forever, not as a temporary backlog artifact but as the permanent steady state (measured
live: ~4.6/min dispatched vs ~12.6/min produced once the backlog from Part 1 fully drained).

`return_exceptions=True` is load-bearing, not a style choice — verified empirically across two
adversarial passes before shipping (see the design doc's "Part 2, missing question 7" for the full
account with real test output): without it, one candidate's failure can cancel a still-in-flight
sibling mid-RPC via `asyncio.run()`'s own shutdown path (real production code invokes this via
`asyncio.run(self._send_prepared_candidates(frame))` inside `_tick()`), which could cancel a
candidate *after* its real cortex-exec call already succeeded but *before* the result gets recorded
— the exact double-send the idempotency guard in `_send_one` exists to prevent.
`_send_one`/`_send_one_inner` were split so the outer `_send_one` is a total function that can
never raise (any unexpected failure degrades to a `dispatch_error` candidate) — required
independently of `return_exceptions=True`, since a raw exception object in the gathered results
list would fail `ExecutionDispatchFrameV1.dispatched_candidates`'s schema validation outright.
Real headroom downstream was confirmed, not assumed, before raising the fan-out width: `orion-
cortex-exec`'s `background` lane already handles unbounded concurrent requests
(`concurrent_handlers=True`, no cap).

**Budgets**, both enforced per tick before any send happens (unchanged by the concurrency patch —
the whole `to_send` batch's risk budget is reserved synchronously, before any concurrent RPCs
fire, so making the *sending* step concurrent doesn't touch this reservation logic):
- `config/execution_dispatch/execution_dispatch_policy.v1.yaml`'s `limits.max_dispatches_per_tick`
- A **self-calibrating daily risk ceiling** (rolling UTC calendar day) -- a real cumulative
  risk-score budget, not a blind action count and, as of 2026-07-29, not a fixed hand-picked
  number either. Each dispatched candidate already carries a real, already-computed `risk_score`
  ([0,1]); the budget is spent against the sum of those scores
  (`ExecutionDispatchRuntimeStore.sum_risk_dispatched_today()`, reading `dispatched_candidates`
  off `substrate_execution_dispatch_frames`), so five trivial inspects no longer cost the same as
  five genuinely higher-risk candidates.

  **How the ceiling itself is derived (2026-07-29,
  `ExecutionDispatchRuntimeWorker._derive_daily_risk_cap` in `app/worker.py`)**: an EWMA baseline
  over real *uncapped* daily demand, mirroring the same `orion/bus/ewma.py::compute_ewma_update`
  mechanism already shipped for field-attention's `recent_perturbation` caps (PR #1433) and
  `execution_prediction_error`'s own per-tick baseline (PR #1434) -- new domain, not a new
  mechanism. Fed by `ExecutionDispatchRuntimeStore.sum_uncapped_risk_for_day()`, which sums
  `risk_score` across every candidate that existed a given day regardless of whether it actually
  got sent (`prepared_for_dispatch` candidates left unsent that tick, plus everything already in
  `dispatched_candidates`) -- deliberately **not** `sum_risk_dispatched_today()`, which is
  right-censored at whatever cap was enforced that day and therefore cannot report true demand
  back into the thing that sets the cap (see that method's own docstring for the fuller
  rationale). With `>=2` real daily samples the cap is `ewma + 3.0 * sqrt(max(var, 1.0))`; with
  exactly 1 sample it's `ewma * 2.0` (an interim margin, same shape as the old static default's
  own "2x one observed day" comment, now correctly anchored on real uncapped demand); with zero
  samples anywhere in history it falls back to the static `ORION_DISPATCH_MAX_RISK_PER_DAY`
  setting (last resort only -- never actually triggers against this repo's real history).

  **Why this replaced the old fixed `ORION_DISPATCH_MAX_RISK_PER_DAY=10.0` constant, and the
  real sequence that led here** (not "we always knew this"): that constant was ENFORCED from
  2026-07-26 through 2026-07-27 and worked exactly like a real ceiling -- clamping real
  dispatched-risk totals at exactly `10.00`/day both days (150 candidates on the 26th, 88 on the
  27th). It looked like a healthy, working cap. It wasn't: `ORION_DISPATCH_RISK_CAP_ADVISORY_ONLY
  =true` shipped 2026-07-28 specifically to observe what real *uncapped* demand looked like once
  the clamp was lifted, and the answer was `817.65`/day (15,099 candidates) and climbing -- ~80x
  the old enforced number, confirmed live via `substrate_execution_dispatch_frames.
  dispatch_frame_json`. That one real day of advisory-only data is exactly what now seeds the
  EWMA baseline above instead of a hand-picked multiplier.

  **Enforcement is back ON as of 2026-07-29**: `ORION_DISPATCH_RISK_CAP_ADVISORY_ONLY` now
  defaults `false`. Reaching the derived cap logs
  `execution_dispatch_risk_budget_status ... cap_reached=true` and blocks further sends for the
  rest of the day (still bounded independently by `max_dispatches_per_tick`). Set
  `ORION_DISPATCH_RISK_CAP_ADVISORY_ONLY=true` as an explicit operator override to return to
  log-only behavior without touching the derived-cap machinery itself.

  `ORION_DISPATCH_MAX_RISK_PER_DAY` itself is no longer the primary mechanism -- it's now only
  the fallback used when the EWMA baseline has never been seeded and no historical closed day
  with real candidate data exists at all.

**Staleness discard (2026-07-30)**: this service consumes `substrate_policy_decision_frames`
strictly FIFO, oldest-undispatched-first (`ExecutionDispatchRuntimeStore.
load_oldest_policy_frames_without_dispatch()`). Because a real dispatch is a synchronous
`cortex-exec` RPC (~7-11s measured live, one per real send, bounded by
`EXECUTION_DISPATCH_RPC_TIMEOUT_SEC`), this single-threaded consumer cannot keep pace with real
production of new policy decisions (~16/min produced vs ~6.8/min consumed, measured live
2026-07-30) — the queue grows without bound if nothing intervenes. Found live 2026-07-30: 46,617
policy frames backlogged, oldest 37h old, meaning real cortex actions were describing hours-old
field pressure as current.

Every `_tick()` now first fast-drains any policy frame older than a randomized
`[EXECUTION_DISPATCH_STALENESS_MIN_SEC, EXECUTION_DISPATCH_STALENESS_MAX_SEC]` threshold (default
120-300s) — no single fixed cutoff every candidate sits the same distance from — capped at
`MAX_STALE_DISCARDS_PER_TICK` (200) discards per tick so one deep-backlog catch-up tick doesn't
feed the discard-rate EWMA below one absurd outlier sample. Each discard is **materialized, never
silently dropped**: `build_stale_discard_execution_dispatch_frame` saves a real
`ExecutionDispatchFrameV1` (`dispatch_attempted=False`, `blocked_count=len(decisions)`) with a
`stale_backlog_discarded age_sec=... threshold_sec=... candidates=N` warning plus one
`stale_discard:{template_key}:{decision}` warning per discarded candidate — real, queryable
forensic content in `substrate_execution_dispatch_frames.warnings`, same table every real frame
lands in.

**Backlog-pressure signal**: `ExecutionDispatchFrameV1.staleness_discard_count_ewma` (`_var`/`_n`
alongside it, same carried-forward-on-every-frame convention as `daily_risk_baseline_*`) — an
EWMA over how many consecutive stale frames each `_tick()` call discarded before finding one
fresh enough to process, or running out. Near 0 in steady state (consumption keeping pace);
rising means the backlog is growing again. Read via
`ExecutionDispatchRuntimeStore.load_latest_staleness_discard_baseline()`, same read pattern as the
daily risk baseline. `STALENESS_DISCARD_EWMA_ALPHA`/`_MIN_VARIANCE` (`app/worker.py`) are a
disclosed, uncalibrated first-pass guess — no real history existed to size them against before
this shipped; revisit once real post-deploy discard-count data exists.

**Operator override**: `EXECUTION_DISPATCH_STALENESS_OVERRIDE_SEC` (unset by default) bypasses the
randomized window entirely and uses one fixed value every tick instead — set very high if a
deliberate deep-backlog catch-up ever becomes desirable again (e.g. Orion's own attention/dispatch
cadence changes later) instead of reverting this patch. Same "explicit override, don't touch the
derived machinery" shape as `ORION_DISPATCH_RISK_CAP_ADVISORY_ONLY` above.

**Fresh-priority fallback (2026-07-30, same-day follow-up)**: deployed and found live, within
minutes, that the FIFO-only drain above has a real failure mode — a deep pre-existing backlog
(the exact 46,617-row/37h one that motivated this feature) means every tick's entire
`MAX_STALE_DISCARDS_PER_TICK` budget goes to old garbage without ever reaching a frame recent
enough to dispatch. Confirmed live: **zero real dispatches for the first 6+ minutes** after this
shipped. Fixed the same day: whenever `_drain_stale_policy_frames` doesn't surface a candidate
(empty queue, or the cap was hit first), `_tick()` now also checks
`ExecutionDispatchRuntimeStore.load_freshest_policy_frame_without_dispatch()` — the single
*newest* unprocessed policy frame, `ORDER BY generated_at DESC` — and processes it directly if
it's within the staleness window. A genuinely current proposal is never gated behind however deep
the old backlog is; old backlog still drains steadily in the background via the unchanged FIFO
path. Returns correctly-empty only when even the newest available frame is already stale, i.e.
production itself has stalled, not backlog depth hiding something current.

**Query performance fix (2026-07-30, same-day follow-up to the follow-up)**: deployed the
fresh-priority fallback above and found, live, that real dispatch had resumed but was still only
happening ~1/minute — far below the ~6.8/min ceiling. Traced with `EXPLAIN ANALYZE` (not
guessed): both the FIFO drain's per-row lookup and the freshest-check both used a
`LEFT JOIN substrate_execution_dispatch_frames d ON ... WHERE d.frame_id IS NULL` anti-join.
Postgres was **not** using the available indexes for either — a full `Parallel Hash Left Join`
scanning both tables, ~280-300ms per call, *regardless of table size beyond a point*. The drain
loop called this **up to 200 times per tick** (`MAX_STALE_DISCARDS_PER_TICK`) — up to ~56 seconds
of pure query time per tick, confirmed as the real, dominant cause of the observed ~75s/tick
cadence (independently verified via `staleness_discard_count_ewma_n` growth over a precise
10-minute window: 8 real ticks, not an inferred number). Adversarially re-tested against stale
planner statistics as an alternative explanation (`ANALYZE` on both tables, re-ran `EXPLAIN
ANALYZE`) — plan and cost unchanged, ruled out.

Two different fixes for two different access patterns (confirmed via `EXPLAIN ANALYZE`, not
assumed symmetric):
- **`load_freshest_policy_frame_without_dispatch`** (DESC, newest-first): rewritten to
  `WHERE NOT EXISTS (SELECT 1 FROM substrate_execution_dispatch_frames d WHERE d.source_policy_
  frame_id = p.frame_id)`. Almost nothing near "now" has been processed yet, so Postgres's
  nested-loop anti-join plan terminates on the very first probe. Measured: ~294ms → ~0.19ms
  (~1500x). A `NOT EXISTS` rewrite for the *other* direction is measurably **worse** (~6+ seconds
  — a huge prefix of already-processed ancient history, predating the original backlog, means a
  nested loop has to walk hundreds of thousands of already-matched rows before finding the first
  true miss), so this rewrite is intentionally direction-specific, not applied everywhere.
- **`load_oldest_policy_frames_without_dispatch(limit)`** (ASC, FIFO drain): batches the whole
  per-tick fetch into ONE query (`LIMIT :limit`, kept as the `LEFT JOIN` shape — the cheaper
  available plan for this direction) instead of a while loop calling a `LIMIT 1` version up to 200
  times. The Hash Join's dominant cost is building the hash table, which happens once regardless
  of LIMIT size (measured: `LIMIT 1` ~280ms, `LIMIT 200` ~327ms, not 200x) — batching cuts real
  per-tick SELECT cost from ~56s to ~0.3s. Confirmed live against the real table: 200 frames in
  464ms (includes Python/pydantic overhead beyond the raw SQL execution time).

See `docs/superpowers/specs/2026-07-30-execution-dispatch-staleness-discard-design.md` for the
full live-data investigation (real backlog depth, throughput math, root cause) and the deferred
part-2 throughput redesign this was scoped alongside but does not itself implement.

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
