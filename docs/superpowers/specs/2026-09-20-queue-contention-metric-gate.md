# queue_contention_score — metric quality gate

**Date:** 2026-09-20
**Parent:** `docs/superpowers/specs/2026-09-20-hire-handoff-and-queue-pressure-design.md`
**Plan Task 5:** blocking before digester wire (Task 6)

## Verdict

**PASS to implement** with all three sources retained, with documented caveats:

- `durable_demand_pending` and `gateway_waiting` show real calm and real variation.
- `world_pulse_seed_pending` is chronically elevated (stock ~121 pending) — EWMA will re-baseline around that stockpile; the score then answers “worse than *this backlog’s* recent normal,” not “is the backlog zero.” That is still useful for hire weather. Do **not** drop the source; do **not** disclose raw counts.
- Not a rebadge of `gpu_pressure`, `sustained_load_pressure`, or `cortex_exec_step_load` (different producers, different theory).

## 1. Provenance

| Source key | Producer | Live read (2026-09-20) |
| --- | --- | --- |
| `world_pulse_seed_pending` | `SELECT count(*) FROM world_pulse_read_seed WHERE status='pending'` | **121** pending (also 103 skipped, 15 done, 3 failed) |
| `durable_demand_pending` | `SELECT count(*) FROM durable_resource_demands WHERE status='pending'` | **3** pending; 48 withdrawn since 2026-09-14; 1 granted today |
| `gateway_waiting` | LLM gateway `GET /admission` → sum `upstreams[*].waiting` | **0** waiting; inflight_sum **3** across 4 upstreams |

Digester writer (Task 6): `services/orion-field-digester/app/digestion/queue_contention.py` → `FieldStateV1.queue_contention_*` fields. Must be named again in the impl PR when the file lands.

## 2. Independence

| Existing metric | Chain to queue contention | Independent? |
| --- | --- | --- |
| `gpu_pressure` | Node biometrics / strain hints → field channel | **Yes** — different sensors; may correlate under shared circe load but not a monotonic transform of queue counts |
| `sustained_load_pressure` | Field channel regime (`loaded_steady`) over a window | **Yes** — level+dispersion on field channels, not SQL/admission queue depth |
| `cortex_exec_step_load` | Cortex-exec step telemetry | **Yes** — execution step load, not seed/durable/gateway waiting |

**Cross-source correlation (open, acceptable for `max()`):** seed backlog, durable GPU waits, and gateway waiting can all rise when the agent lane is overloaded. Spec already allows `max()` over correlated sources. No requirement that the three be orthogonal — only that the *score* not be a rename of an existing pressure channel.

## 3. Theory anchor

Measures **shared agent/curiosity capacity queue contention**: how backed up the reading-seed pipeline, durable GPU lease waits, and LLM gateway admission waiting are relative to each source’s own recent EWMA baseline. Used to inform Orion’s Cursor-vs-local hire decision. Not GPU thermals, not tension change-detection, not sustained field-channel overload.

## 4. Live-data sanity

### world_pulse_seed_pending

- Snapshot: 121 pending (chronic stockpile).
- 72h create cadence (rows with `created_at` in window): only 4 hour-buckets with activity; pending_created per active hour ~5–9 — backlog is not exploding hour-to-hour, it is **not draining to zero** either.
- Calm-at-zero: **not observed** in this snapshot. After EWMA warm-up, score≈0 means “at backlog normal,” not “empty queue.”
- **Keep** — still the dominant shared-capacity backlog Juniper named; EWMA relative signal is the product.

### durable_demand_pending

- Snapshot: 3 pending; daily withdrawn churn 2–11/day over 14 days; granted appears.
- Calm: pending often near 0 historically (withdrawn dominates). **Can return to calm.** Keep.

### gateway_waiting

- Snapshot: waiting_sum **0**, inflight 3 on one upstream.
- Admitted totals large (17k+ on busiest upstream) with waiting 0 now → **can be calm.** Keep. Sparse spikes expected; floor=1 in score math avoids divide-by-near-zero.

### Degenerate-source check

None dropped. Seed is chronically high but still varies enough for relative EWMA; durable and gateway show calm. Revisit after soak if seed sub-score never leaves 0 after warm-up while hire decisions still need seed weather — then consider rate-of-change, not raw stock (separate proposal).

## 5. Existing-mechanism check

`rg` on `orion/field` + digester (2026-09-20): `gpu_pressure`, `sustained_load_pressure`, `cortex_exec_step_load` exist; **no** `queue_contention*` producer. No existing FieldState field for seed/durable/gateway queue depths. Proceed with new instrument.

## 6. Reversibility

Additive `FieldStateV1` fields + digester module + inner-state registry + metric lock (+ optional glossary). Kill = delete producer **and** registry/lock/glossary together; Hub disclosure fails open if fields absent. No Hub Redis EWMA.

## Half-life / floor (provisional pending soak)

- Floor: `1.0` (plan).
- Half-life: ~24h (plan); digester tick ~2s → alpha derived in Task 6 settings.
- Recalibrate after first soak if seed warm-up hides real spikes.

## Gate status for Task 6

**UNBLOCKED** — implement Task 6 with all three source keys:
`world_pulse_seed_pending`, `durable_demand_pending`, `gateway_waiting`.

## Re-point: `gateway_waiting` → `gpu_pool_waiting` (2026-09-24, Juniper-approved)

The LLM gateway's in-process admission ledger (`GET /admission`, the `gateway_waiting` source) is
deleted by the GPU pool gateway cutover (stage 3 of `2026-09-24-gpu-pool-design.md`). The metric is
re-pointed, not left to go silent. The gate, re-run in full:

1. **Provenance.** `gpu_pool_waiting` = `SELECT count(*) FROM gpu_pool_leases WHERE status IN
   ('queued','backlogged')` (`services/orion-field-digester/app/store.py::count_gpu_pool_waiting`).
   The rows are the pool's fenced projection, written by the single-writer runtime
   (`services/orion-gpu-pool/app/runtime.py::_row`) on every lease transition. `retry_wait` is
   excluded (cooling down after a failure, not waiting for capacity), and so are `granted` and
   `recalling` (holding a GPU).
2. **Independence.** It replaces `gateway_waiting` (same concept, the retired producer). It does
   **not** double-count `durable_demand_pending`: a durable run waits in the old broker until it is
   granted, and only then takes a pool lease for the slot it uses. Its wait is counted in one place,
   never both. When durable runs move onto the pool (stage 4), `durable_demand_pending` retires and
   its wait moves into `gpu_pool_waiting`; that is the next re-point.
3. **Theory anchor.** Unchanged from this gate: backlog of work waiting for a GPU, relative to its own
   EWMA baseline, is the queueing-pressure signal the hire decision consumes. The pool's queue is
   where that waiting now physically happens, because every LLM call leases before it runs.
4. **Live data.**
   - **The retired source was degenerate.** `substrate_field_state` at 2026-09-24 23:29Z shows
     `queue_contention_ewma.gateway_waiting = 0.0` over **163,577** observations, and it was never the
     driver in 3 days (drivers: `world_pulse_seed_pending` ×66,458, avg score 0.14;
     `durable_demand_pending` ×54,581, max 6.38).
   - **Why it was zero.** The gateway's semaphore cap was 8 per upstream, far above the real slot
     counts (chat 1, agent 1, metacog/fast 4), so nothing ever queued there. Oversubscription
     happened inside llama.cpp instead, invisible to the metric.
   - **Why the new source can move.** The pool grants against discovered real slots, so the queue
     forms exactly where the GPU is full. Its rest state is a genuine 0 (empty queue). It is not a
     decay artifact: the value is a fresh count every tick, never carried forward.
   - **UNVERIFIED live** until stage 3 is deployed: after deploy, confirm the new source rises under
     load (e.g. a metacog burst beyond 4 slots, or chat while a turn runs) and returns to 0.
5. **Existing mechanism.** This is the same metric; only one source changes. `SOURCE_KEYS` stays at
   three.
6. **Reversibility.** Cheap. The key rename starts a fresh EWMA (the first tick scores 0 by
   construction). The retired `gateway_waiting` baseline is dropped from state rather than carried
   (`score_queue_contention` keeps only `SOURCE_KEYS`). Disclosure text is updated
   (`orion/curiosity/queue_contention_disclosure.py`).

## Oldest-wait component (2026-09-25, Juniper-approved)

**Why.** The score only compared each queue's depth to its own recent average. A queue that stops
moving keeps the same depth, the average catches up, and the score reads calm. Live on 2026-09-25:
the seed queue held 144 pending items, the oldest from 2026-09-07, nothing finished since
2026-09-15 02:05Z, and the score sat at ~0.14 (driver `world_pulse_seed_pending`, count 144 vs
EWMA 136) on 1176/1176 ticks, sinking toward 0. Juniper, asked "do you want the backed-up signal to
count how long the oldest item has waited?", answered "fix all the things".

**What changed.** Each source gets a second sub-score from the age of its oldest waiting item:
`age_sub = clip(10 * (age / expected_wait - 1) / 4, 0, 10)` (same shape as the depth sub: 0 up to
1x, 10 at 5x). The score is the max over all six subs. When an age sub wins, the driver is
`<source>:oldest_wait`. The expected wait is fixed config, not an average, on purpose: an average
would learn "stuck" as normal, the exact failure being fixed.

### 1. Provenance

| Sub | Producer (`services/orion-field-digester/app/store.py`) | Live 2026-09-25 ~06:10Z |
| --- | --- | --- |
| seed oldest wait | `oldest_world_pulse_seed_pending_age_sec`: `EXTRACT(EPOCH FROM now() - min(created_at)) FROM world_pulse_read_seed WHERE status='pending'` | 1,555,830 s (432 h) |
| durable oldest wait | `oldest_durable_demand_pending_age_sec`: same over `durable_resource_demands WHERE status='pending'` | 798 s (one fresh demand) |
| GPU pool oldest wait | `oldest_gpu_pool_waiting_age_sec`: `min(coalesce(queued_since, created_at)) FROM gpu_pool_leases WHERE status IN ('queued','backlogged')` | NULL -> 0.0 (queue empty) |

Filters match each source's `count_*` sibling exactly, so depth and age describe the same set of
items. The subtraction runs inside Postgres against its own `now()`. `created_at` on seeds and
durable demands is a DB default; on `gpu_pool_leases` it is written by orion-gpu-pool
(`services/orion-gpu-pool/app/runtime.py::_row`), which runs on athena, the same host as the
database, so no cross-host clock enters. Readers go through `app/digestion/queue_contention.py::
default_queue_contention_age_readers` -> `read_queue_contention_oldest_waits` (fail-open: a failed
read is omitted, never written as 0.0) -> `orion/field/queue_contention.py::score_queue_contention`.
Every query uses an existing index (`EXPLAIN` on the pool query: `Index Scan using
gpu_pool_leases_live_idx`).

### 2. Independence

- **Against the depth sub of the same source.** Not a transform of it. Depth counts how many items
  wait; age is when the oldest one arrived. They split exactly where this fix matters: the seed
  queue on 2026-09-25 has depth at 1.06x its average (sub 0.15) and age at 9x expected (sub 10). A
  queue can also be deep and young (a burst just landed: depth high, age 0) or shallow and old (one
  stuck item: depth ~1x, age high).
- **Against each other.** Three different tables written by three different services; correlated
  only when one shared cause (e.g. the agent lane being saturated) backs up more than one. `max()`
  was already chosen for correlated sources (section 2 above).
- **Against `gpu_pressure`, `sustained_load_pressure`, `cortex_exec_step_load`.** Unchanged from
  section 2: different producers (node biometrics, field-channel regime, step telemetry); none reads
  a queue table.
- **Against orion-gpu-pool's own `waited_ms`** (`runtime.py`, emitted when a lease is granted).
  Not the same thing and cannot replace this: it is only computed on the grant event, so a lease that
  is never granted never reports a wait. That is the event-only blind spot this component exists to
  close.

### 3. Theory anchor

Queueing theory, specifically head-of-line / oldest-waiter age as the standard backlog-staleness
signal: under Little's law (L = lambda * W) a queue's length L can hold steady while its wait W grows
without bound if the service rate lambda drops to zero -- L alone cannot distinguish "steady and
flowing" from "frozen". The age of the oldest waiting item is a direct lower bound on the current
wait W of anything behind it, and it grows linearly with wall-clock time when nothing is served,
whatever the arrival rate. That is the claim the score makes: "work is waiting longer than it
normally does."

### 4. Live-data sanity, including the rest state

**Rest state by hand.** An empty queue gives `min(created_at)` = NULL -> the reader returns 0.0 ->
`age / expected = 0` -> `clip(10 * (0 - 1) / 4) = clip(-2.5) = 0.0`. Any oldest item younger than
the expected wait gives `age / expected < 1` -> a negative pre-clip value -> exactly 0.0. So the rest
point is a true 0, not a floor like `sqrt(2/pi)`. And it cannot decay into a fake 0: the age is a
fresh SQL read every tick, never carried forward or multiplied down, and it is not in
`NODE_DECAY_CHANNELS` (it is not a node vector at all). Tests:
`tests/test_queue_contention.py::test_empty_queue_rest_point_is_exact_zero`,
`::test_fresh_items_rest_point_is_exact_zero`.

**Replayed history** (reconstructed from each table's own timestamps; one sample per 10 min for
durable, per 10 s for the pool):

| Source | Window | Empty (0) | Waiting but under 1x (0) | Above 1x (nonzero) | Saturated (>= 5x) | Worst |
| --- | --- | --- | --- | --- | --- | --- |
| durable, expected 12 h | 2026-09-14..09-25, 1604 samples | 715 (45%) | 715 (45%) | 174 (11%) | 0 | 26 h -> sub 2.9 |
| GPU pool, expected 60 s | 2026-09-24 20:00..09-25 06:10, 3695 samples | 3615 (97.8%) | 62 (1.7%) | 18 (0.5%) | 0 | 231 s -> sub 7.1 |
| seed, expected 48 h | 2026-09-07..09-25, every 6 h | never empty | first ~2 days | since 2026-09-09 | since 2026-09-17 | 432 h -> sub 10 |

Durable and GPU pool both go quiet at a genuine 0 most of the time, rise when work really waits,
and never pinned. **The seed source is pinned at 10, and that is the truthful reading, not a metric
defect:** the seed queue has never once drained since it was created (296 seeds created, 15 ever
finished, the last on 2026-09-15; oldest-pending age has grown monotonically from 6 h to 432 h in
every sample). Its rest point is reachable (the math above), but the queue has never been in a rest
state to show it. **Consequence to know before deploying:** while the seed queue stays frozen, the
score reads 10 with driver `world_pulse_seed_pending:oldest_wait` on every tick, the hire
disclosure will say so every time it fires, and the `max()` will hide any durable or GPU-pool
contention behind it. It clears only when the seed backlog older than 48 h is claimed or skipped.
If that is unwanted before the seed pipeline is fixed, set
`FIELD_QUEUE_CONTENTION_SEED_EXPECTED_WAIT_SEC` very high to mute only the seed age sub.

**Expected waits -- knobs, anchored, not findings:**

- **Seeds, 48 h (172,800 s).** p90 claim wait over every seed ever finished = 175,596 s (48.8 h;
  n=15, p50 13.9 h). Small sample from the only period the queue ever moved; revisit once it moves
  again.
- **Durable, 12 h (43,200 s).** p90 time from demand to first grant over 138 granted demands =
  45,086 s (p50 2.7 h, max 26 h). Note the broker's own burst threshold is 1,200 s
  (`decision.threshold_seconds`); using it would have scored most real demands > 0, so the observed
  p90 is used instead.
- **GPU pool, 60 s.** 10x the worst grant wait seen over 1,275 leases (p99 1.8-3.2 s by class, max
  5.7 s; ~10 h of history since the pool went live). Scores 10 at 300 s, which is the `deadline_at`
  most callers put on a lease (fast/agent ~300 s, metacog up to 700 s): full pressure means the oldest
  waiter is about to give up.

### 5. Existing-mechanism check

`rg` for oldest/age signals over `orion/field`, the digester, orion-gpu-pool and
`orion/world_pulse_read`: nothing measures the age of waiting work on these queues. The digester's
`health_monitor` `field_state_oldest_age_hours` is the retention age of `substrate_field_state` rows
(unrelated). The pool's `waited_ms` is grant-time only (see section 2). World-pulse `/api/status`
reports retry counts, not age. Extending this metric beats a new one: same sources, same consumer,
same disclosure line.

### 6. Reversibility

Cheap. No schema change: `FieldStateV1` is `extra="forbid"`, and adding a field there broke two
readers for two days on 2026-09-20 (`orion/substrate_ladder_liveness.py`), so the raw ages are not
persisted; they surface in the score, the driver string, and a `queue_contention_driver_changed` log
line (raw counts + ages, once per driver change). An old Hub reading a new `...:oldest_wait` driver
falls back to its generic blurb, it does not fail. No EWMA state is added, so reverting leaves
nothing behind. Per-source mute without code: raise that source's `*_EXPECTED_WAIT_SEC`. Full
revert: drop the age readers and the six-sub max; the depth path is untouched.

**Expected live change on deploy (UNVERIFIED until deployed):** score ~0.14 -> 10.0, driver
`world_pulse_seed_pending` -> `world_pulse_seed_pending:oldest_wait`, on the first tick (no warm-up;
the expected wait is config, not learned).
