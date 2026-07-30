# execution-dispatch backlog: staleness discard (shipped) + throughput redesign (spec)

2026-07-30. Follow-up to the proposals `proposal_priority()` precision-weighting patch
(PR #1497). Verifying that patch against real post-deploy data surfaced something much bigger:
`orion-execution-dispatch-runtime` was 46,617 policy decisions behind, oldest from 2026-07-29
07:24 — over 37 hours — and the gap was growing, not shrinking. This doc has two parts: what
shipped today (staleness discard, a correctness fix), and a design-mode spec for the harder,
deferred follow-up (real throughput).

## Part 1: what was found, live, 2026-07-30

**Chain of discovery.** After merging PR #1497 (precision-weighted `proposal_priority()` +
the `proposal_confidence()` fallback fix that unblocks 5 previously-permanently-blocked
templates), real post-deploy data was pulled directly from Postgres to verify it. A specific
tick's persisted policy decision showed `confidence_score: 0.0` — but recomputing the *same*
shipped code against the *same* persisted field state gave `0.81`. That looked like a live
code/data inconsistency at first. It wasn't: the discrepancy resolved once the *age* of the
underlying policy decision was checked — `substrate_policy_decision_frames.generated_at` was
`2026-07-29 07:24:47`, a full day-plus old, while the `substrate_execution_dispatch_frames` row
built from it had `generated_at` of *right now*. `orion-execution-dispatch-runtime` was working
through a backlog, not processing live traffic.

**Root cause, precisely.** `ExecutionDispatchRuntimeStore.load_latest_policy_frame_without_
dispatch()` (`services/orion-execution-dispatch-runtime/app/store.py:42`) is strict FIFO,
oldest-first: `ORDER BY p.generated_at ASC LIMIT 1`. `_tick()`
(`services/orion-execution-dispatch-runtime/app/worker.py`) processes exactly one policy
decision frame per poll cycle, and when that frame has a real candidate to send, does a
**synchronous** cortex-exec RPC inline before the tick completes
(`await client.dispatch(...)`, `worker.py:527`, timeout 120s but real observed latency
~7-11s). Poll interval is 2s, but the real tick-to-tick cadence is bounded by that RPC, not the
interval — measured live via `action_outcomes.observed_at` deltas: **7.6-11.2 seconds between
consecutive real dispatches**, consistently.

**The math doesn't close.** Measured live, 5-minute windows:
- Production: `substrate_policy_decision_frames` — **16/min**
- Consumption: `substrate_execution_dispatch_frames` — **6.8/min**

Net backlog growth: ~9/minute, ~13,000/day — consistent with a 46,617-row backlog accumulating
since 2026-07-29 07:24. Confirmed **not** caused by the same-day `proposal_priority()` patch:
policy-decision-frame production is one frame per field tick regardless of how many candidates
are approved vs. blocked within it, and consumption cost is one LLM call per tick regardless of
how many candidates existed — unblocking 5 more templates changes *who wins* the per-tick
dispatch slot, not the throughput math on either side.

**Why this is a correctness bug, not just a performance one.** Strict oldest-first FIFO means
real cortex actions were being dispatched describing 37-hour-old field pressure as current —
a "no empty-shell cognition" violation in its own right (CLAUDE.md §0A), independent of the
throughput question.

## Part 1 (continued): what shipped

**Staleness discard**, `services/orion-execution-dispatch-runtime/app/worker.py`:

- `_tick()` now fast-drains any policy frame older than a randomized
  `[EXECUTION_DISPATCH_STALENESS_MIN_SEC, EXECUTION_DISPATCH_STALENESS_MAX_SEC]` threshold
  (default 120-300s, drawn fresh via `random.uniform` on every check — deliberately not one
  fixed constant, so there is no single sharp, predictable cliff every candidate sits the same
  distance from) before considering a real send. Capped at `MAX_STALE_DISCARDS_PER_TICK` (200)
  discards per `_tick()` call so the very first tick after this ships (which found the entire
  46,617-row backlog stale) doesn't feed the new discard-rate EWMA one absurd one-time outlier
  sample, and doesn't hold the poll loop for however long tens of thousands of real DB writes
  take in one go.
- **Materialized, never silently dropped.** `orion/execution_dispatch/builder.py::
  build_stale_discard_execution_dispatch_frame()` saves a real `ExecutionDispatchFrameV1`
  (`dispatch_attempted=False`, `blocked_count=len(decisions)`) with a
  `stale_backlog_discarded age_sec=... threshold_sec=... candidates=N` warning plus one
  `stale_discard:{template_key}:{decision}` entry per discarded candidate — real, queryable
  forensic detail in the same `substrate_execution_dispatch_frames.warnings` column every real
  frame already uses. No new table.
- **Backlog-pressure signal.** `ExecutionDispatchFrameV1.staleness_discard_count_ewma`
  (`_var`/`_n` alongside it) — an EWMA over "how many consecutive stale frames did this tick
  discard before finding one fresh enough, or running out," updated once per `_tick()` call that
  finds any policy frame at all (same "once per real cycle regardless of whether a real event
  landed" convention `DIMENSION_PRECISION_EWMA_ALPHA` already uses in field-digester), carried
  forward on every saved frame the same way `daily_risk_baseline_*` already is. **Disclosed gap
  found in review**: a tick where the queue is fully empty from the start has no frame left to
  carry a `value=0.0` sample on, so that tick's update is skipped rather than persisted —
  deliberately not patched with a synthetic no-op frame (would be inventing content to carry a
  metric, not recording something real); this only under-samples the true-idle case, never the
  backlog case this patch exists for. Near 0 in steady state; rising
  means the backlog is growing again — this is the thing a future health surface (see Part 1's
  own closing note below) should read, rather than requiring another 20-query manual psql
  investigation like this one.
  - **Disclosed, uncalibrated constants**: `STALENESS_DISCARD_EWMA_ALPHA=0.2`,
    `STALENESS_DISCARD_EWMA_MIN_VARIANCE=1.0` (`app/worker.py`). This is the first time this
    metric has ever existed — unlike `DIMENSION_PRECISION_MIN_VARIANCE` or
    `DAILY_RISK_BASELINE_MIN_VARIANCE` elsewhere in this repo, both seeded from a real measured
    population before shipping, there is no real history to calibrate against yet. Revisit once
    real post-deploy discard-count data exists, per this repo's metric-quality-gate discipline —
    do not treat "the EWMA update runs without error" as proof this floor is right.
- **Operator override shim.** `EXECUTION_DISPATCH_STALENESS_OVERRIDE_SEC` (unset by default)
  bypasses the randomized window entirely and uses one fixed value every tick instead — same
  "explicit override, don't touch the derived machinery" shape as
  `ORION_DISPATCH_RISK_CAP_ADVISORY_ONLY`. Exists because this service's consumption/production
  balance is not assumed permanent: if Orion's own attention/dispatch cadence changes later
  (faster real consumption, fewer but more deliberate proposals, etc.), a deliberate
  deep-backlog catch-up may become desirable again without a code change.

**Files changed**: `orion/schemas/execution_dispatch_frame.py` (3 new fields),
`orion/execution_dispatch/builder.py` (`build_stale_discard_execution_dispatch_frame`,
`_decision_template_key`), `services/orion-execution-dispatch-runtime/app/worker.py`
(`_staleness_threshold_sec`, `_drain_stale_policy_frames`, rewritten `_tick`),
`services/orion-execution-dispatch-runtime/app/store.py`
(`load_latest_staleness_discard_baseline`), `services/orion-execution-dispatch-runtime/app/
settings.py` (3 new env-backed settings), `.env`/`.env_example` (synced), tests (worker + builder).

**What this patch deliberately does not do**: fix the throughput gap itself. Once the backlog is
drained (which this patch does quickly — stale-skip is a cheap DB write, not a ~9s LLM call),
steady-state behavior is real but rate-limited: with production still exceeding raw dispatch
throughput, a rolling fraction of candidates will always go stale rather than dispatch, forever,
under this patch alone. That is arguably fine (the system dispatches what it can, in real time,
against current field state, and drops the rest instead of queuing it up as false-freshness
debt) — but it is a real, disclosed trade-off, not a full fix. Part 2 below specs the throughput
side.

## Part 1b: same-day follow-up — FIFO-only drain starved real-time dispatch entirely

Deployed this same day, and checked against real live data within minutes (not left to "should
work"). Found something the original patch got wrong: **zero real dispatches happened in the 6+
minutes following deploy.** `substrate_policy_decision_frames` backlog dropped steadily
(~46,617 → ~46,134, a consistent ~165/min drain rate, confirmed via real per-minute row counts)
— the discard mechanism itself worked exactly as designed. But `action_outcomes` had **zero** new
rows in that same window.

**Root cause**: the original design assumed the drain would "quickly clear backlog, letting real
dispatch resume." It doesn't work that way under strict FIFO. `_drain_stale_policy_frames` walks
oldest-first and only *stops* (handing back a candidate to process) once it finds a frame within
the freshness window. With a 37-hour-deep backlog and a 120-300s freshness window, essentially
every frame in that backlog is stale by definition — so every tick spends its entire
`MAX_STALE_DISCARDS_PER_TICK` (200) budget discarding ancient garbage and *never reaches* a frame
recent enough to actually dispatch. At the measured ~165/min drain rate, that's **~4.6 hours of
total dispatch silence** before the drain would naturally catch up to real time. The PR's own risk
section undersold this ("a rolling fraction of candidates will go stale") — in practice it was
100% stale, 0% dispatched, for hours, not a rolling fraction.

**Fix**: `ExecutionDispatchRuntimeStore.load_freshest_policy_frame_without_dispatch()` (new,
`ORDER BY generated_at DESC LIMIT 1`, same schema-validation-failure handling as the existing FIFO
lookup) — a direct "is there something current available right now" check. `_tick()` calls this
as a fallback whenever `_drain_stale_policy_frames` doesn't surface a candidate (empty queue, or
the cap was hit first). If the single newest unprocessed frame is within the staleness window, it
gets processed for real dispatch this tick regardless of how deep the old backlog behind it is.
Old backlog still drains steadily via the unchanged FIFO path — this only ensures real-time
dispatch is never additionally gated by backlog depth. Correctly returns nothing when even the
newest available frame is already stale (production itself has stalled, not a backlog-depth
artifact).

**Files touched**: `services/orion-execution-dispatch-runtime/app/store.py`
(`load_freshest_policy_frame_without_dispatch`), `app/worker.py` (extracted `_age_sec` shared
helper, new `_check_freshest_fallback`, `_tick` wiring), README, tests.

## Part 2: throughput redesign — design mode, not implemented

### Arsonist summary

Even with staleness discard live, this service can only ever really dispatch ~6.8 candidates/min
because `_send_one()` is awaited sequentially inside a loop bounded by `max_dispatches_per_tick`
(currently 1) and the whole tick blocks on that RPC. Raising `max_dispatches_per_tick` alone does
nothing today, because the `for candidate in to_send: newly_dispatched.append(await self._send_
one(...))` loop (`worker.py:410-411`) still awaits each one in series. Real concurrency requires
touching shared per-process state that assumes single-threaded-per-tick access.

### Current architecture

- `worker.py::_tick()` → `_send_prepared_candidates()` selects up to `max_dispatches_per_tick`
  candidates by cumulative risk budget, then sequentially `await`s `_send_one()` for each.
- `_send_one()` does: idempotency check (`load_dispatch_result_by_dispatch_id`), the real RPC
  (`client.dispatch(...)`), `save_dispatch_result()`, `_recent_dispatch_statuses.append(status)`
  (mutates `self.theater_tripwire_active`-adjacent in-process `deque`, not thread-safe), and
  `_emit_action_outcome()` (a bus publish).
- Risk budget accounting (`cumulative_risk`, `remaining_risk_budget`) is computed once per tick
  from `frame.candidates`, assuming nothing else spends from the same budget concurrently.
- Theater tripwire (`_check_theater_tripwire`) reads/writes `self._recent_dispatch_statuses`
  (a `deque(maxlen=10)`) and `self.theater_tripwire_active`, both plain instance attributes with
  no lock — safe today only because everything is strictly sequential.
- `max_dispatches_per_tick=1` (`config/execution_dispatch/execution_dispatch_policy.v1.yaml`) —
  already the real cap; the sequential-await loop is currently a distinction without a
  difference at that setting, but becomes the actual bottleneck the moment this value is raised.

### Missing questions

1. **Is raising throughput even the right goal**, or is staleness discard's "dispatch what's
   current, drop what isn't" already the intended long-run shape? Real production (16/min) may
   keep exceeding real consumption (6.8/min) even after concurrency work, if `orion-cortex-exec`
   itself has a real LLM-inference ceiling — concurrency inside this service doesn't help if the
   downstream cortex-exec route is itself the bottleneck. Needs tracing `orion-cortex-exec`'s
   own real concurrent-request capacity before assuming this service's serial loop is the only
   constraint.
2. **What does real concurrent risk-budget accounting look like?** `cumulative_risk` is computed
   assuming candidates are reserved in the order they're sent; genuine concurrency means multiple
   in-flight sends could all pass the same-tick budget check before any of them completes. Needs
   either a pre-reservation step (claim budget before dispatching, release/adjust on failure) or
   accepting some real over-spend risk and tightening the ceiling to compensate.
3. **Does the theater tripwire's `deque` need a lock, or does it need to become per-batch instead
   of per-process** (e.g., check tripwire status once per tick before firing a batch, not
   per-candidate-in-flight)? Real answer depends on how tightly the tripwire is meant to react —
   current behavior re-checks after every send; a batched version changes that granularity.
4. **Is per-tick concurrency (asyncio.gather within one `_tick()`) enough, or does the underlying
   one-poll-thread-per-process model need to change** (multiple worker processes/replicas)?
   Concurrency within a tick only closes the gap up to whatever `orion-cortex-exec` itself can
   sustain concurrently; if that's the real ceiling, no amount of restructuring this service's
   loop helps further.

### Proposed schema / API changes

None proposed yet — blocked on missing questions 1 and 4 (whether the bottleneck is even in this
service). If concurrency within a tick is the right direction: `_send_prepared_candidates` would
need to reserve risk budget synchronously before firing each RPC (not after, as today), and
`_check_theater_tripwire`'s deque would need either a lock or to move to a per-batch check.
Neither is scoped as a patch here.

### Files likely to touch (once a direction is chosen — not started)

- `services/orion-execution-dispatch-runtime/app/worker.py` (`_send_prepared_candidates`,
  `_send_one`, `_check_theater_tripwire`)
- `config/execution_dispatch/execution_dispatch_policy.v1.yaml` (`max_dispatches_per_tick`, if
  raised)
- `services/orion-cortex-exec/` (tracing real concurrent-capacity, possibly touched if it's the
  actual ceiling)
- `tests/test_execution_dispatch_runtime_worker.py`

### Non-goals

- Not touching staleness discard's own thresholds/constants here (Part 1 is shipped and
  separate).
- Not proposing a multi-process/replica model without first tracing whether `orion-cortex-exec`
  itself is the real ceiling (missing question 1) — no point building concurrency this service
  can't actually benefit from.
- Not building a health-surface UI in this doc (see Part 1's closing note — a script, not a
  dashboard, was the right-sized answer when this question came up for proposals scoring
  earlier the same day).

### Acceptance checks

N/A — no patch proposed yet.

### Recommended next patch

Trace `orion-cortex-exec`'s real concurrent-request capacity first (missing question 1) — cheap,
and determines whether any of the concurrency work below is worth doing at all. Only after that:
scope a minimal concurrent-send patch (missing questions 2-3) if the trace shows real headroom
downstream.
