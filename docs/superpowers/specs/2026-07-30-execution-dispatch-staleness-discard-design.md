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

## Part 1c: same-day follow-up #2 — the real bottleneck was a query plan, not LLM latency

Deployed Part 1b, checked live again (the same discipline that found 1b's own bug in the first
place). Real dispatch had resumed but was stuck at ~1/minute — far below the ~6.8/min ceiling Part
2 was scoped against. Instead of assuming the cause, ran `EXPLAIN ANALYZE` on the real live
queries.

**Root cause, precise**: `load_latest_policy_frame_without_dispatch` (ASC) and
`load_freshest_policy_frame_without_dispatch` (DESC) both used `LEFT JOIN
substrate_execution_dispatch_frames d ON ... WHERE d.frame_id IS NULL`. Despite real indexes
existing on both join columns, Postgres chose a `Parallel Hash Left Join` — full scans of both
tables, ~280-300ms per call, independent of the actual answer size. The FIFO drain called this
**up to 200 times per tick**. Real, independently-measured tick cadence (via
`staleness_discard_count_ewma_n`'s growth over an exact 10-minute window — a real counter, not an
inferred number): **8 ticks**, ~75s each. 1,542 frames processed across those 8 ticks (~193/tick)
confirms the drain was hitting its 200-cap almost every single tick. ~193 × ~280ms ≈ 54s of the
75s directly attributable to this one query pattern.

**Adversarial pass before accepting the conclusion** (explicitly requested, not skipped):
- *Stale planner statistics?* Ran `ANALYZE` on both tables live, re-ran `EXPLAIN ANALYZE`. Plan
  and cost unchanged (~302ms). Ruled out.
- *Was the ~75s/tick figure just arithmetic, not measured?* Re-derived independently via the
  `staleness_discard_count_ewma_n` counter (increments exactly once per real `_tick()` call) —
  confirmed 8 ticks in 10 real minutes, matching the inferred figure exactly.
- *Could the INSERT side (`save_dispatch_frame`) be a second hidden bottleneck?* Checked its SQL —
  a simple indexed single-row upsert on the primary key, not O(N) like the SELECT. A minor
  contributor via per-call transaction round-trip overhead, not the dominant cost.
- *Could my own diagnostic queries have inflated the live numbers?* Checked the direction of the
  bias — ad-hoc `EXPLAIN ANALYZE` runs warm Postgres's shared buffer cache, which if anything makes
  the measurement a *floor*, not an inflated worst case; and `EXPLAIN ANALYZE`'s reported execution
  time is server-side processing time, independent of which client submitted the query.

**Fix, two different shapes for two different access patterns** (confirmed via `EXPLAIN ANALYZE`,
not assumed symmetric):
- `load_freshest_policy_frame_without_dispatch` (DESC) → rewritten to `WHERE NOT EXISTS (...)`.
  Measured: ~294ms → ~0.19ms (~1500x). Correct because almost nothing near "now" is processed yet,
  so a nested-loop anti-join plan terminates on the first probe.
- `load_oldest_policy_frames_without_dispatch(limit)` (ASC, renamed from the old singular
  `load_latest_policy_frame_without_dispatch`) → same `LEFT JOIN` shape kept (a `NOT EXISTS`
  rewrite here measured **worse**, 6+ seconds — a large prefix of already-processed ancient
  history, predating the original backlog, means a nested loop walks hundreds of thousands of
  already-matched rows before the first true miss), but now fetches the whole `MAX_STALE_DISCARDS_
  PER_TICK`-sized batch in ONE query instead of calling the `LIMIT 1` version in a 200-iteration
  while loop. The Hash Join's dominant cost (building the hash table) is paid once regardless of
  LIMIT size (measured: `LIMIT 1` ~280ms, `LIMIT 200` ~327ms) — batching cuts real per-tick SELECT
  cost from ~56s to ~0.3s. Confirmed live against the real table (not just mocked): 200 real frames
  fetched, correctly ordered, in 464ms including Python/pydantic overhead.

**Files touched**: `services/orion-execution-dispatch-runtime/app/store.py`
(`load_oldest_policy_frames_without_dispatch` replacing `load_latest_policy_frame_without_
dispatch`, `load_freshest_policy_frame_without_dispatch` rewritten to `NOT EXISTS`,
`_validate_policy_frame_row` extracted as a shared validate-or-retire helper), `app/worker.py`
(`_drain_stale_policy_frames` rewritten to fetch-then-iterate instead of loop-and-fetch), README,
tests (worker + store, including two new store-level happy-path tests for the batch and `NOT
EXISTS` query shapes, plus a mixed-batch regression test added in review -- see below).

**Two real side effects of the batch rewrite, found in review, both real improvements but
disclosed here since neither was originally intentional or tested for on the first pass**:
- **A schema-incompatible row no longer stalls the whole tick.** The old single-row lookup
  returned `None` the instant it hit ANY incompatible row (a pre-2026-07-22 SelfStateV1-burn-era
  payload, say), which the old `while` loop treated identically to "queue empty" -- stopping the
  entire tick's drain even if perfectly valid rows sat right behind it in FIFO order. The new
  batch method (`load_oldest_policy_frames_without_dispatch`) retires the bad row (same stub-frame
  mechanism as before) and simply excludes it from the returned list, so the other valid rows in
  the same batch still get processed the same tick. Locked in by
  `test_load_oldest_policy_frames_without_dispatch_skips_only_the_bad_row_in_a_mixed_batch`.
- **The per-tick discard cap is now a SQL-only guarantee, not also enforced in Python.** The old
  `while discarded < MAX_STALE_DISCARDS_PER_TICK` loop was a real, independent backstop against
  ever discarding more than the cap in one tick, regardless of what the store returned. The new
  code trusts the store's bound `:limit` parameter entirely -- if `load_oldest_policy_frames_
  without_dispatch` ever returned more than `limit` rows (a bug, not an expected case), nothing in
  `worker.py` would stop it. `test_max_stale_discards_per_tick_caps_the_drain`'s docstring was
  tightened in review to say this explicitly rather than imply a worker-side cap still exists.

**What this does to Part 2**: doesn't eliminate the question, but changes its urgency and framing.
The concurrency work below was scoped against a ~9s-per-real-dispatch ceiling; with query cost no
longer dominating, real tick cadence should approach the ~2s poll interval, and real dispatch
throughput should approach the ~6.8/min ceiling Part 2 always assumed as the starting point — not
exceed it. Concurrent dispatch is still the only way past that ~6.8/min number itself (still
bounded by sequential `await self._send_one(...)` calls), so Part 2 remains real, valid future
work — it's just no longer entangled with what turned out to be a separate, bigger, already-fixed
problem.

## Part 2: throughput redesign — scoped, ready to implement

Status change from the original draft below: every missing question that gated this has now been
answered with real evidence (live trace + direct code reads), not assumption. This section is the
resolved scope; the original open-questions draft is kept below it for the record.

### Arsonist summary

Steady-state math, once the backlog fully clears (confirmed live, 2026-07-30, draining
~150-165/min from a 46,617 peak): real production of policy decisions holds steady at
**~17/min**; real dispatch throughput is capped at **~4-5/min** (one candidate at a time, each a
synchronous ~9-13s cortex-exec RPC). That means roughly **65-70% of every real, current proposal
will never get a dispatch attempt at all**, forever — not because they're bad candidates, but
because a proposal that arrives between two ticks gets superseded by an even-newer one before its
turn, ages past the staleness window, and gets swept into the discard path having never run. This
directly undercuts the point of PR #1497 (which specifically unblocked 5 previously-permanently-
blocked templates): unblocking a template's confidence gate doesn't cash out into real coverage if
steady-state dispatch can only ever process ~30% of what's produced.

### Missing questions from the original draft — resolved

1. **Is raising throughput even the right goal?** Yes — quantified above, not assumed. This isn't
   speculative performance polish; it's a measured, permanent coverage gap.
2. **Is `orion-cortex-exec` itself the ceiling?** No. Traced live: the `background` lane
   (`orion/core/bus/bus_service_chassis.py::Rabbit`, `concurrent_handlers=True` default, no
   override for this lane in `services/orion-cortex-exec/docker-compose.yml:226-230`) spawns an
   uncapped `asyncio.create_task` per incoming message — no semaphore, no worker-pool limit. Real
   headroom exists downstream.
3. **What does real concurrent risk-budget accounting look like?** Turns out to need no redesign
   at all — re-read `_send_prepared_candidates` (`worker.py:375-390`) carefully: the `to_send`
   list and its `cumulative_risk` are already built in a single synchronous pass, over
   `frame.candidates`, BEFORE any RPC fires. The whole batch's budget is reserved atomically up
   front today, regardless of how many candidates end up in `to_send`. Making the *sending* step
   concurrent (`asyncio.gather` instead of a `for` loop of sequential `await`s) doesn't touch this
   reservation logic at all — it already correctly handles a multi-candidate batch. Missing
   question 2 dissolves once you notice the reservation and the sending were already two separate
   steps.
4. **Does the theater tripwire's `deque` need a lock?** No, and not because of luck — because of
   what asyncio concurrency actually is. `asyncio.gather` runs coroutines cooperatively on ONE
   thread; two tasks can only interleave at `await` points, never mid-statement. `deque.append(x)`
   has no `await` inside it, so it is a single, non-preemptible operation regardless of how many
   concurrent `_send_one()` calls are in flight — this is not the same hazard as real OS-thread
   concurrency, and reaching for a lock here would be solving a problem that doesn't exist in this
   execution model. The one real, disclosed behavior change: `_check_theater_tripwire()` currently
   re-checks after every sequential send (`worker.py:429`, "lets the tripwire fire the same tick it
   actually crosses the threshold"); under `asyncio.gather`, there's no clean mid-batch checkpoint,
   so the recheck moves to once-per-batch (after `gather` returns) instead of once-per-candidate.
   Coarser, not unsafe — the tripwire's own 10-sample window makes this a small, acceptable
   granularity change, not a correctness regression.
5. **Is the bus/RPC layer itself safe for concurrent use on one shared connection?** Checked, not
   assumed — and an adversarial re-check (explicitly requested, given "bus RPC is the backbone to
   the entire mesh") **found and corrected a real error in this section's first draft**: the
   original claim was that `OrionBusAsync.rpc_request`'s worker path demultiplexes pending RPCs by
   `(reply_channel, corr)` key. That path only runs if `self._rpc_worker_task` is active — and
   `services/orion-execution-dispatch-runtime/app/worker.py:608` constructs `OrionBusAsync` with
   the plain constructor, never via `.fork(start_rpc_worker=True)` or any other path that starts
   that task. This service **never uses the worker path** — the original claim described code this
   service does not execute.

   What actually makes it safe (verified by reading `OrionBusAsync.subscribe()`,
   `async_service.py:325-336`, the path this service actually takes): the inline RPC path calls
   `self._create_pubsub_redis()` on **every single call**, which does `aioredis.from_url(self.url,
   ...)` — a brand-new, fully independent Redis connection, not shared or pooled with anything
   else, subscribed only to that call's own UUID-derived `reply_channel`
   (`ExecutionDispatchCortexClient.dispatch()`, `cortex_client.py:73-74`, still correctly generates
   a fresh `uuid4()` correlation ID and unique reply channel per call), torn down in a `finally`
   block when the RPC completes. This is *stronger* isolation than the multiplexed path would have
   given — full per-call connection separation, not shared-socket demuxing — but it is a different
   mechanism than originally claimed. Concurrent `dispatch()` calls cannot cross-talk, for the
   corrected reason above.

   Also checked while auditing this (same adversarial pass, since this genuinely is mesh-critical
   code): the shared *command* connection used for `publish()` (`self._redis`, distinct from the
   per-RPC pubsub connections above) is a standard `redis.asyncio.Redis` client, internally pooled
   by redis-py for concurrent use — not custom code, a well-established library guarantee.
   `OrionCodec.encode`/`decode` are stateless pure functions. `RpcHealthAggregator` (records
   RPC success/timeout stats) is shared, mutable state, but its own docstring already states "no
   lock is needed" under asyncio's cooperative scheduling — independent, pre-existing corroboration
   of the same reasoning used for the theater tripwire's deque in point 4 above, not something
   novel introduced by this patch. `enforcer` (`orion/core/bus/enforce.py`, channel-catalog
   validation) is a process-wide singleton with a lazy-init catalog load, shared across every
   `OrionBusAsync` instance in the process — real shared mutable state, but `_ensure_catalog()` has
   no `await` inside it, so the same no-interleaving-mid-mutation guarantee holds.

   **One real, newly-disclosed cost, not a correctness bug**: concurrency means N simultaneous new
   Redis connections per tick (one TCP handshake each) instead of one connection reused
   sequentially across N sends. Real, additional load on Redis that doesn't exist today — worth
   monitoring post-deploy, not a blocker at `max_dispatches_per_tick=5`.
6. **Is per-tick concurrency enough, or does the one-poll-thread-per-process model need to
   change** (multiple worker replicas)? Per-tick concurrency is enough. The fan-out target
   (cortex-exec's `background` lane) already handles unbounded concurrent requests (point 2
   above), so there's no need to scale execution-dispatch-runtime itself to multiple processes —
   one process firing N concurrent RPCs per tick already has real headroom to use on the other
   end.
7. **Is `_send_one()` actually safe to run under `asyncio.gather`?** No, on a second adversarial
   pass — and the fix from that pass alone turned out to be incomplete, caught on a third pass
   (both explicitly requested, both changing the patch's scope, not just its reasoning; the third
   pass's correction is folded directly into this entry rather than kept as a separate stale
   version). `_send_one()` is 147 lines (`worker.py:661-807`) with exactly ONE `try/except`,
   wrapping only the `client.dispatch()` RPC call (`worker.py:736-743`). Two other real operations
   are completely unguarded: the idempotency check (`self._store.load_dispatch_result_by_
   dispatch_id`, line 678) and the result save (`self._store.save_dispatch_result`, line 775) —
   both plain synchronous DB calls, no exception handling. (`_emit_action_outcome`, called from
   three places inside `_send_one`, was checked separately and confirmed already fully self-guarded
   — every real operation inside it has its own try/except, catches broadly, only logs. Not a risk.)

   Why this specifically matters for `asyncio.gather` and not the current sequential loop: if one
   of those two unguarded DB calls raises for *one* candidate in a concurrent batch (a transient
   connection drop, a query error — real, if rare, and this repo's own Postgres history includes a
   full host-disk death, so "the DB call sometimes fails" is not a hypothetical here), `asyncio.
   gather()`'s documented default behavior is easy to get wrong: it propagates the first exception
   to the caller immediately, but does **not** cancel the other in-flight coroutines in the group —
   they keep running, unawaited, detached from whatever now-exited scope was gathering them. `_tick
   ()` would exit via that propagated exception, `_poll_loop` sleeps 2s and starts a new tick, while
   the orphaned sends from the failed tick may still be mid-RPC or mid-DB-write. Two generations of
   dispatch work overlapping across a tick boundary is a shape nothing in this codebase was built
   to handle. In the current sequential code, the identical underlying exception just silently
   truncates `to_send` for that tick — loses that tick's frame-level bookkeeping (annoying,
   already true today, undisclosed until this pass), but nothing is ever orphaned, since nothing
   else was in flight to begin with.

   **Fix, revised on a third adversarial pass (explicitly requested again) — the second-pass fix
   alone was insufficient, confirmed empirically, not just reasoned about.** The second-pass fix
   (wrap `_send_one()` in `try/except Exception`) does not actually close this gap. Tested directly
   (`python3 -c "..."`, not just read about): bare `asyncio.gather(a, b, c)` with one coroutine
   raising does NOT leave the others silently running forever as originally described above — when
   invoked via `asyncio.run()` (which is how `_tick()` calls `_send_prepared_candidates`,
   `worker.py`'s real-send branch), `asyncio.run()`'s own shutdown path cancels every other
   still-pending task before closing its event loop. Confirmed live: a slow sibling task logged
   `cancelled`, not silent continuation. That's real, but it does not make the original concern
   go away — it changes it into a different, still-real one: those siblings are cancelled
   **mid-flight**, at whatever `await` point they happened to be suspended. If a candidate's real
   cortex-exec RPC had *already succeeded* and that candidate's coroutine was cancelled before it
   reached `self._store.save_dispatch_result(...)`, the real-world action already happened but
   Orion's own durable record of it never gets written. On the next tick (if the frame-level save
   was also lost to the same propagated exception), the identical policy frame would be
   re-selected, the idempotency check would find nothing, and the same candidate would be
   **dispatched a second time for real** — the exact double-send the idempotency mechanism exists
   to prevent.

   And `_send_one()`'s own `try/except Exception` (the second-pass fix) does not protect against
   this specific failure mode, for a precise, verified reason: `asyncio.CancelledError` has been a
   `BaseException` subclass, not an `Exception` subclass, since Python 3.8 — confirmed directly
   against this repo's actual interpreter (`issubclass(asyncio.CancelledError, Exception)` →
   `False`, Python 3.12.3). A bare `except Exception` inside `_send_one()` cannot catch a
   cancellation triggered by a *sibling's* failure elsewhere in the same `gather()` call, no matter
   how the body is wrapped.

   **The actual, complete fix**: `asyncio.gather(*aws, return_exceptions=True)`, not bare
   `asyncio.gather(*aws)`. Verified empirically, not assumed: with `return_exceptions=True`, the
   same slow siblings that were cancelled above instead run to full, real completion regardless of
   another candidate's failure — confirmed via the identical test with only that flag changed
   (`slow_task` logged `complete`, not `cancelled`; the failing task's exception came back as a
   `RuntimeError` *value* inside the results list, never raised, never triggering cancellation of
   anything else). This eliminates the cross-candidate cancellation risk structurally, not just in
   the cases already anticipated.

   `_send_one()`'s own `try/except Exception` hardening from the second pass is **still required**,
   but for a different, more mechanical reason than originally stated: with `return_exceptions=
   True`, any `_send_one()` call that still raised an ordinary `Exception` (not cancelled by a
   sibling — its own internal failure) would land in `newly_dispatched` as a raw exception object,
   not an `ExecutionDispatchCandidateV1`. `dispatched_candidates = list(frame.dispatched_
   candidates) + newly_dispatched` feeds directly into `frame.model_copy(update={...})`, which
   Pydantic would reject outright (a `list[ExecutionDispatchCandidateV1]` field cannot hold a raw
   `RuntimeError`). So both fixes ship together, and both are now load-bearing for different
   reasons: `return_exceptions=True` prevents any candidate from being cancelled mid-flight by an
   unrelated sibling's failure; `_send_one()`'s internal hardening keeps every item in the gathered
   list a valid `ExecutionDispatchCandidateV1`, which the schema requires regardless of concurrency.

   **Separate, smaller precision correction found while tracing this** (not a bug): the store calls
   inside `_send_one()` are synchronous/blocking (plain SQLAlchemy `conn.execute(...)`, no
   `await`), so `asyncio.gather` only genuinely parallelizes the ~9-13s RPC wait specifically — the
   ~5-20ms DB calls still execute one-at-a-time on the single event-loop thread underneath. Doesn't
   undermine the concurrency win (RPC latency dominates DB latency by roughly three orders of
   magnitude), but "concurrent" here should be understood precisely as "concurrent RPC wait," not
   "every operation runs in parallel."

### Proposed schema / API changes

None. This is a behavior change inside `_send_prepared_candidates`/`_send_one`, not a schema or
contract change — `ExecutionDispatchCandidateV1`/`ExecutionDispatchFrameV1` are unaffected.

- `config/execution_dispatch/execution_dispatch_policy.v1.yaml`: raise `limits.max_dispatches_per_
  tick` from `1` to an initial small number (recommend **5** — enough to meaningfully close the
  17/min-vs-4-5/min gap without a large first-patch blast radius; revisit once real post-deploy
  concurrent-dispatch data exists, same discipline as every other constant shipped this session).
- `services/orion-execution-dispatch-runtime/app/worker.py::_send_prepared_candidates`: replace
  the sequential `for candidate in to_send: newly_dispatched.append(await self._send_one(...))`
  loop with `newly_dispatched = await asyncio.gather(*(self._send_one(client, bus, frame, c) for c
  in to_send), return_exceptions=True)`. **`return_exceptions=True` is not optional** — verified
  empirically (missing question 7) that without it, one candidate's failure cancels its siblings
  mid-flight via `asyncio.run()`'s own shutdown path, risking a real-but-unrecorded dispatch. Move
  the `_check_theater_tripwire()` recheck to after the `gather` (already effectively true given the
  loop restructure).
- `services/orion-execution-dispatch-runtime/app/worker.py::_send_one`: **required companion
  change, still needed even with `return_exceptions=True`, for a different reason** (missing
  question 7) — wrap the entire body in a broad `try/except Exception`, converting any failure (not
  just an RPC failure) into the same `dispatch_status="dispatched", dispatch_error=str(exc)[:500]`
  shape already used for RPC failures. With `return_exceptions=True`, any candidate whose
  `_send_one()` call still raised an ordinary exception would otherwise land in `newly_dispatched`
  as a raw exception object, which `ExecutionDispatchFrameV1.dispatched_candidates: list[
  ExecutionDispatchCandidateV1]` cannot validate — Pydantic would reject the frame outright.
- No change needed to the risk-budget reservation loop (already correct, see missing question 3
  above) or to `_recent_dispatch_statuses` (already safe, see missing question 4 above).

### Files likely to touch

- `services/orion-execution-dispatch-runtime/app/worker.py` (`_send_prepared_candidates` AND
  `_send_one` — the latter added to scope by missing question 7's finding; `_check_theater_
  tripwire` and the risk-budget loop remain unchanged)
- `config/execution_dispatch/execution_dispatch_policy.v1.yaml` (`max_dispatches_per_tick: 1 → 5`)
- `tests/test_execution_dispatch_runtime_worker.py` (new tests: concurrent sends actually run
  concurrently — e.g. assert total wall time for N mocked-slow sends is closer to one send's
  duration than N times it; a `_send_one` failure — e.g. mock `load_dispatch_result_by_dispatch_id`
  to raise — degrades to a `dispatch_error` candidate instead of propagating out of `asyncio.
  gather`, and does not prevent the OTHER concurrent candidates in the same batch from completing
  and being reflected in the saved frame; theater tripwire still trips correctly when checked
  post-batch; risk budget still correctly caps `to_send` size pre-gather)

### Non-goals

- Not touching staleness discard's own thresholds/constants (Parts 1/1b/1c are shipped, separate).
- Not scaling execution-dispatch-runtime to multiple processes/replicas — per-tick concurrency
  within the existing single process is sufficient given confirmed downstream headroom (missing
  question 6).
- Not redesigning the risk-budget or theater-tripwire mechanisms — both already work correctly
  under concurrency once traced properly; this patch does not touch their logic, only the send
  loop's control flow. (`_send_one`'s exception-safety IS in scope — see missing question 7 — that
  is a correctness prerequisite for `asyncio.gather`, not a mechanism redesign.)
- Not building a health-surface UI (per Part 1's closing note — a script, not a dashboard, was
  the right-sized answer when this question came up for proposals scoring the same day).
- Not re-verifying `orion-cortex-exec`'s concurrent capacity beyond the code-level trace already
  done — a real live load test (N truly simultaneous dispatch RPCs) would be the honest next
  verification step once this ships, not a blocker to shipping it.

### Acceptance checks

- `max_dispatches_per_tick` raised to 5, `_send_prepared_candidates` uses
  `asyncio.gather(..., return_exceptions=True)` — bare `gather()` without that flag does not meet
  this acceptance check, per missing question 7's empirical finding.
- `_send_one` wrapped end-to-end in exception handling — a forced failure in the idempotency check
  or the result save degrades to a `dispatch_error` candidate, never propagates.
- New concurrency test proves sends actually overlap in wall-clock time, not just that the code
  compiles.
- New failure-isolation test proves one candidate's forced exception does NOT cancel or truncate a
  concurrent sibling mid-flight — the sibling must reach real completion (its own `save_dispatch_
  result` call actually happens), not just "the batch doesn't raise." A test that only asserts the
  batch completes without raising would pass even with the incomplete second-pass fix and must not
  be treated as sufficient on its own.
- Theater tripwire and risk-budget tests (existing + new) still pass unmodified in their
  assertions about *what* gets blocked, only updated for *when* the recheck happens.
- Post-deploy: real dispatch rate approaches (not necessarily hits) the ~17/min production rate
  over a real observation window — the actual metric this patch exists to move. `staleness_
  discard_count_ewma` should trend toward 0 once backlog is clear and concurrent dispatch is
  keeping pace.
- Post-deploy: watch real Redis connection count (`INFO clients`/`connected_clients`) for a
  sustained increase attributable to this service specifically — the disclosed per-RPC connection
  churn cost (missing question 5 above) should be small at `max_dispatches_per_tick=5`, but this is
  the actual check, not an assumption that it's fine.

### Recommended next patch

Implement directly — every blocking question above was resolved by reading real code and live
data, not by further scoping. Ship as ONE patch, not two: `max_dispatches_per_tick: 5` + the
`asyncio.gather` rewrite + the `_send_one` exception-safety hardening (missing question 7) —
the gather rewrite is not safe to ship without the hardening, so splitting them would mean
shipping a real, known risk on purpose. Reviewed and tested the same way every other patch in
this arc was, then watch `staleness_discard_count_ewma` and real dispatch rate post-deploy to
confirm the steady-state math above actually closes the gap it predicts.

---

<details>
<summary>Original open-questions draft (superseded by the resolved scope above, kept for the record)</summary>

Even with staleness discard live, this service can only ever really dispatch ~6.8 candidates/min
because `_send_one()` is awaited sequentially inside a loop bounded by `max_dispatches_per_tick`
(currently 1) and the whole tick blocks on that RPC. Raising `max_dispatches_per_tick` alone does
nothing today, because the `for candidate in to_send: newly_dispatched.append(await self._send_
one(...))` loop still awaits each one in series. Real concurrency requires touching shared
per-process state that assumes single-threaded-per-tick access.

Missing questions as originally posed, before being resolved above: (1) is raising throughput even
the right goal — needed real production-vs-consumption math, not assumption; (2) is
`orion-cortex-exec` itself the ceiling — needed a live trace of its concurrency model; (3) what
does real concurrent risk-budget accounting look like — needed a careful re-read of where
reservation actually happens relative to sending; (4) does the theater tripwire's deque need a
lock — needed to reason about asyncio's cooperative-scheduling model specifically, not assume
thread-safety concerns transfer directly; (5) is per-tick concurrency enough, or does the
one-poll-thread-per-process model need to change — needed to know whether the downstream target
could absorb more concurrent load before proposing to scale this service's own process count.

</details>
